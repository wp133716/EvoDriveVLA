"""
sliding_window_inference.py — 滑窗 KV Cache 推理优化
=====================================================
每次推理只对新帧跑 ViT + backbone；历史帧的 K/V 直接从缓存取。

核心原理：
  - RoPE 使用绝对位置，attention score 体现相对位置差
  - 只要 token 的绝对位置不变，其 KV 向量永远有效
  - 新帧的位置 = max(已缓存位置) + 1 起，保证相对距离正确

使用方式：
    infer = SlidingWindowInference(model, window_size=5)

    # 首帧：冷启动，处理完整 5 帧序列
    waypoints = infer.cold_start(input_ids, attention_mask,
                                 pixel_values, image_grid_thw,
                                 n_cams_per_frame=4)

    # 后续帧：只传入新帧图像，其余从缓存读取
    waypoints = infer.step(new_pixel_values, new_grid_thw)
"""

import torch
from transformers.cache_utils import DynamicCache


# ─── KV Cache 扩展：支持按帧驱逐 ──────────────────────────────────────────
class FrameKVCache(DynamicCache):
    """
    在 DynamicCache 基础上增加：
      - 按帧记录 token 数量
      - 驱逐最旧帧（保留文字 token）
      - 驱逐末尾若干 token（用于清除 query token）
    """

    def __init__(self):
        super().__init__()
        # 每帧对应的图像 token 数量，FIFO 队列
        self.frame_img_token_counts: list[int] = []

    def register_frame(self, n_img_tokens: int):
        """注册一帧的图像 token 数，供后续驱逐使用。"""
        self.frame_img_token_counts.append(n_img_tokens)

    def evict_oldest_frame(self) -> int:
        """
        驱逐最旧帧的图像 KV。
        文字 token（指令、自车状态等）不在队列里，永远保留。
        返回实际驱逐的 token 数。
        """
        if not self.frame_img_token_counts:
            return 0
        n = self.frame_img_token_counts.pop(0)
        for i in range(len(self.key_cache)):
            # key_cache[i]: (bs, n_heads, seq_len, head_dim)
            self.key_cache[i] = self.key_cache[i][:, :, n:, :]
            self.value_cache[i] = self.value_cache[i][:, :, n:, :]
        self._seen_tokens -= n
        return n

    def evict_last_n(self, n: int):
        """
        驱逐末尾 n 个 token 的 KV（用于清除每次推理后的 query token）。
        """
        if n <= 0:
            return
        for i in range(len(self.key_cache)):
            self.key_cache[i] = self.key_cache[i][:, :, :-n, :]
            self.value_cache[i] = self.value_cache[i][:, :, :-n, :]
        self._seen_tokens -= n

    @property
    def n_cached_frames(self) -> int:
        return len(self.frame_img_token_counts)


# ─── 滑窗推理主类 ──────────────────────────────────────────────────────────
class SlidingWindowInference:
    """
    对 Qwen2_5_VLForLearnableQ 的滑窗 KV Cache 推理封装。

    帧窗口示意（window_size=5）:
      冷启动: [f1, f2, f3, f4, f5] → 全部计算 KV
      step 1: [f2, f3, f4, f5, f6] → f2-f5 复用缓存，只算 f6
      step 2: [f3, f4, f5, f6, f7] → f3-f6 复用缓存，只算 f7
    """

    def __init__(self, model, window_size: int = 5):
        self.model = model
        self.window_size = window_size
        self.cache: FrameKVCache = FrameKVCache()
        # 当前缓存中已分配的最大绝对位置（用于给新帧分配连续位置）
        self.max_cached_pos: int = -1
        # 文字 token 数量（冷启动后固定，不驱逐）
        self.n_text_tokens: int = 0

    # ── 工具：计算一组图像的 token 数量 ───────────────────────────────────
    def _count_img_tokens(self, image_grid_thw: torch.Tensor) -> list[int]:
        """
        每张图的图像 token 数 = T×H×W / spatial_merge_size²
        image_grid_thw: (N, 3)，每行是 (T, H, W)
        """
        s = self.model.config.vision_config.spatial_merge_size
        return [
            int(thw[0] * thw[1] * thw[2] / (s * s))
            for thw in image_grid_thw
        ]

    # ── 工具：给新帧计算位置 ID ───────────────────────────────────────────
    def _make_frame_position_ids(
        self,
        frame_input_ids: torch.Tensor,   # (1, seq_len_frame) 含图像占位 token
        frame_grid_thw: torch.Tensor,    # (n_cams, 3)
        frame_attention_mask: torch.Tensor,
        offset: int,
    ) -> torch.Tensor:
        """
        为单帧（含 n_cams 张图像）计算位置 ID，然后整体偏移 offset。

        get_rope_index() 对独立帧返回从 0 开始的位置。
        加上 offset 后，新帧位置紧接在已缓存的最大位置之后。
        返回: (3, 1, seq_len_frame)
        """
        pos, _ = self.model.get_rope_index(
            frame_input_ids,
            frame_grid_thw,
            None,
            None,
            frame_attention_mask,
        )
        pos = pos + offset
        return pos

    # ── 冷启动：处理完整初始窗口 ──────────────────────────────────────────
    @torch.no_grad()
    def cold_start(
        self,
        input_ids: torch.Tensor,         # (1, full_seq_len) 完整序列
        attention_mask: torch.Tensor,    # (1, full_seq_len)
        pixel_values: torch.Tensor,      # (N_total_imgs, C, H, W)
        image_grid_thw: torch.Tensor,    # (N_total_imgs, 3)
        n_cams_per_frame: int = 4,       # 每帧的摄像头数
    ) -> torch.Tensor:
        """
        处理完整初始窗口（通常 5 帧），建立初始 KV 缓存。
        返回: waypoints (1, num_waypoints, waypoint_dim)
        """
        model = self.model
        device = input_ids.device

        # ── Step 1: embed tokens + 合并图像特征 ──────────────────────────
        inputs_embeds = model.model.embed_tokens(input_ids)
        if pixel_values is not None:
            pv = pixel_values.type(model.visual.dtype)
            image_embeds = model.visual(pv, grid_thw=image_grid_thw)
            mask = (input_ids == model.config.image_token_id)
            img_mask = mask.unsqueeze(-1).expand_as(inputs_embeds).to(device)
            image_embeds = image_embeds.to(device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(img_mask, image_embeds)

        # ── Step 2: 计算原始序列 position_ids ────────────────────────────
        position_ids, _ = model.get_rope_index(
            input_ids, image_grid_thw, None, None, attention_mask
        )
        self.max_cached_pos = int(position_ids.max().item())

        # ── Step 3: 拼接 query token ──────────────────────────────────────
        bs = 1
        queries = model.traj_queries.unsqueeze(0).to(inputs_embeds.dtype)
        inputs_embeds_ext = torch.cat([inputs_embeds, queries], dim=1)

        q_offsets = torch.arange(1, model.num_waypoints + 1, device=device)
        q_pos = (self.max_cached_pos + q_offsets).view(1, -1).expand(bs, -1)
        q_pos_3d = q_pos.unsqueeze(0).expand(3, -1, -1)
        pos_ext = torch.cat([position_ids, q_pos_3d], dim=-1)

        q_mask = torch.ones(bs, model.num_waypoints, device=device, dtype=attention_mask.dtype)
        attn_ext = torch.cat([attention_mask, q_mask], dim=1)

        # ── Step 4: backbone forward，开启 KV cache ───────────────────────
        n_total = inputs_embeds_ext.shape[1]
        cache_position = torch.arange(n_total, device=device)
        self.cache = FrameKVCache()

        outputs = model.model(
            input_ids=None,
            inputs_embeds=inputs_embeds_ext,
            attention_mask=attn_ext,
            position_ids=pos_ext,
            past_key_values=self.cache,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
            cache_position=cache_position,
        )

        # ── Step 5: 取轨迹 ────────────────────────────────────────────────
        last_hidden = outputs.hidden_states[-1]
        traj_hidden = last_hidden[:, -model.num_waypoints:, :]
        waypoints = model.trajectory_head(traj_hidden) * model.waypoint_std + model.waypoint_mean

        # ── Step 6: 清理 query token 的 KV，注册各帧 ─────────────────────
        self.cache.evict_last_n(model.num_waypoints)

        # 统计每帧的图像 token 数
        img_token_counts = self._count_img_tokens(image_grid_thw)
        n_frames = len(img_token_counts) // n_cams_per_frame
        for f in range(n_frames):
            frame_img_tokens = sum(img_token_counts[f * n_cams_per_frame:(f + 1) * n_cams_per_frame])
            self.cache.register_frame(frame_img_tokens)

        # 记录文字 token 数（非图像 token）
        total_img_tokens = sum(img_token_counts)
        self.n_text_tokens = int(attention_mask.sum().item()) - total_img_tokens

        return waypoints

    # ── 滑动步：只处理新帧 ────────────────────────────────────────────────
    @torch.no_grad()
    def step(
        self,
        new_pixel_values: torch.Tensor,   # (n_cams, C, H, W)  新帧图像
        new_grid_thw: torch.Tensor,       # (n_cams, 3)
        new_frame_input_ids: torch.Tensor,      # (1, n_img_tokens_new_frame) 只含图像占位 token
        new_frame_attention_mask: torch.Tensor, # (1, n_img_tokens_new_frame)
    ) -> torch.Tensor:
        """
        滑动一帧：ViT 只处理新帧，LLM backbone 复用历史 KV。
        返回: waypoints (1, num_waypoints, waypoint_dim)
        """
        model = self.model
        device = new_pixel_values.device

        # ── Step 1: 只对新帧跑 ViT ───────────────────────────────────────
        pv = new_pixel_values.type(model.visual.dtype)
        image_embeds = model.visual(pv, grid_thw=new_grid_thw)
        # image_embeds: (n_img_tokens_new_frame, hidden)

        # 构建新帧的 inputs_embeds（纯图像 token，无文字）
        n_img = image_embeds.shape[0]
        inputs_embeds = image_embeds.unsqueeze(0)  # (1, n_img, hidden)

        # ── Step 2: 计算新帧的 position_ids，偏移到已缓存最大位置之后 ────
        offset = self.max_cached_pos + 1
        pos_new = self._make_frame_position_ids(
            new_frame_input_ids, new_grid_thw,
            new_frame_attention_mask, offset
        )
        # pos_new: (3, 1, n_img)
        new_frame_max_pos = int(pos_new.max().item())

        # ── Step 3: 拼接 query token ──────────────────────────────────────
        queries = model.traj_queries.unsqueeze(0).to(inputs_embeds.dtype)
        inputs_embeds_ext = torch.cat([inputs_embeds, queries], dim=1)

        q_offsets = torch.arange(1, model.num_waypoints + 1, device=device)
        q_pos = (new_frame_max_pos + q_offsets).view(1, -1)
        q_pos_3d = q_pos.unsqueeze(0).expand(3, 1, -1)
        pos_ext = torch.cat([pos_new, q_pos_3d], dim=-1)
        # pos_ext: (3, 1, n_img + num_waypoints)

        # ── Step 4: backbone forward（新帧 + query，历史帧从 cache 取）───
        n_cached = self.cache.get_seq_length()
        n_new_total = inputs_embeds_ext.shape[1]
        cache_position = torch.arange(n_cached, n_cached + n_new_total, device=device)

        outputs = model.model(
            input_ids=None,
            inputs_embeds=inputs_embeds_ext,
            attention_mask=None,   # None → is_causal=True，新 token 自动 attend 所有已缓存 token
            position_ids=pos_ext,
            past_key_values=self.cache,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
            cache_position=cache_position,
        )

        # ── Step 5: 取轨迹 ────────────────────────────────────────────────
        last_hidden = outputs.hidden_states[-1]
        traj_hidden = last_hidden[:, -model.num_waypoints:, :]
        waypoints = model.trajectory_head(traj_hidden) * model.waypoint_std + model.waypoint_mean

        # ── Step 6: 更新缓存状态 ──────────────────────────────────────────
        # 6a. 清除 query token 的 KV（ephemeral，不应缓存）
        self.cache.evict_last_n(model.num_waypoints)

        # 6b. 注册新帧
        new_frame_img_tokens = sum(self._count_img_tokens(new_grid_thw))
        self.cache.register_frame(new_frame_img_tokens)
        self.max_cached_pos = new_frame_max_pos

        # 6c. 驱逐最旧帧（如果超出窗口）
        if self.cache.n_cached_frames > self.window_size:
            self.cache.evict_oldest_frame()

        return waypoints

    def reset(self):
        """清空缓存，准备全新推理序列。"""
        self.cache = FrameKVCache()
        self.max_cached_pos = -1
        self.n_text_tokens = 0
