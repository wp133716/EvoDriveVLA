"""
Qwen2.5-VL Learnable Query 轨迹输出
=====================================
支持两种输出模式（通过 use_lm_head 选择）：

模式 A — 回归头 (use_lm_head=False，默认)
  num_waypoints 个 query token 拼接到序列末尾，
  取末尾 num_waypoints 个 hidden states → trajectory_head(Linear) → (bs, 6, 3)。
  损失：Smooth L1（归一化空间）。

模式 B — lm_head + CE (use_lm_head=True)
  num_waypoints × waypoint_dim = 18 个 query token，
  每个 query 独立预测对应坐标值的词表 token。
  损失：交叉熵（直接对词表）。
  优点：无需维护 waypoint_stats，无额外参数（复用 lm_head）。
  代价：精度受词表对浮点数覆盖粒度限制；decode 时取 argmax token 文本解析为 float。

数据流（模式 A）：
  input_ids + pixel_values
    → embed_tokens + visual encoder → inputs_embeds (bs, N, c)
    → cat(inputs_embeds, traj_queries[6]) → (bs, N+6, c)
    → Qwen2_5_VLModel backbone
    → last_hidden[:, -6:, :] (bs, 6, c)
    → trajectory_head Linear → waypoints (bs, 6, 3)

数据流（模式 B）：
  ...同上，但 traj_queries 数量为 18...
    → last_hidden[:, -18:, :] (bs, 18, c)
    → lm_head → logits (bs, 18, vocab_size)
    → argmax → token_ids → decode → waypoints (bs, 6, 3)
"""

import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen2_5_VLForConditionalGeneration
from transformers.modeling_outputs import ModelOutput
from dataclasses import dataclass
from typing import Optional


@dataclass
class Qwen2_5_VLLearnableQOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    waypoints: torch.FloatTensor = None
    logits: torch.FloatTensor = None  # 保持与 Trainer 的兼容性


class Qwen2_5_VLForLearnableQ(Qwen2_5_VLForConditionalGeneration):
    """
    Learnable Query 轨迹预测模型。

    新增参数（相对于父类）：
      traj_queries   : nn.Parameter (n_query_tokens, hidden_size)

      [use_lm_head=False]
      trajectory_head: nn.Linear    (hidden_size → waypoint_dim)
      waypoint_mean  : buffer (waypoint_dim,)
      waypoint_std   : buffer (waypoint_dim,)

      [use_lm_head=True]
      无额外参数，复用父类 lm_head
      n_query_tokens = num_waypoints × waypoint_dim = 18
    """

    def __init__(self, config, num_waypoints=6, waypoint_dim=3,
                 waypoint_mean=None, waypoint_std=None,
                 use_lm_head=False):
        super().__init__(config)

        self.num_waypoints = num_waypoints
        self.waypoint_dim = waypoint_dim
        self.use_lm_head = use_lm_head

        # CE 模式：每个坐标分量对应一个独立 query token（共 18 个）
        # 回归模式：每个 waypoint 对应一个 query token（共 6 个）
        self.n_query_tokens = num_waypoints * waypoint_dim if use_lm_head else num_waypoints

        hidden_size = config.hidden_size

        self.traj_queries = nn.Parameter(torch.zeros(self.n_query_tokens, hidden_size))
        nn.init.normal_(self.traj_queries, std=0.02)

        if not use_lm_head:
            # 回归头：每个 query → (x, y, z) waypoint
            self.trajectory_head = nn.Linear(hidden_size, waypoint_dim)
            nn.init.normal_(self.trajectory_head.weight, std=0.02)
            nn.init.zeros_(self.trajectory_head.bias)

            # 归一化参数（registered buffer，随模型保存/加载）
            if waypoint_mean is None:
                waypoint_mean = [0.0] * waypoint_dim
            if waypoint_std is None:
                waypoint_std = [1.0] * waypoint_dim
            self.register_buffer(
                'waypoint_mean', torch.tensor(waypoint_mean, dtype=torch.float32)
            )
            self.register_buffer(
                'waypoint_std', torch.tensor(waypoint_std, dtype=torch.float32)
            )

    # ------------------------------------------------------------------
    # 辅助（CE 模式）：label token ids → 18 个目标词表 id
    # ------------------------------------------------------------------
    def _parse_waypoint_token_ids(self, labels: torch.Tensor) -> torch.Tensor:
        """
        直接从 label token id 序列中取前 n_query_tokens 个有效 token 作为目标。

        不做二次 encode，避免浮点数 tokenization 碎片化（如 "0.10" → ["0",".","10"]）
        导致 target 全为 "0" token 的 loss collapse 问题。

        返回: (bs, n_query_tokens) LongTensor
        """
        bs = labels.shape[0]
        device = labels.device
        all_ids = []

        for i in range(bs):
            if labels.dim() == 2:
                valid_ids = labels[i][labels[i] != -100]
            else:
                valid_ids = labels[i]

            n = len(valid_ids)
            if n >= self.n_query_tokens:
                token_ids = valid_ids[:self.n_query_tokens].tolist()
            elif n > 0:
                # 用最后一个有效 token 填充（比填 0 更有意义）
                token_ids = valid_ids.tolist() + [valid_ids[-1].item()] * (self.n_query_tokens - n)
            else:
                # 完全没有有效 label（数据问题），返回 0 并打印警告
                token_ids = [0] * self.n_query_tokens
                print("[WARNING] _parse_waypoint_token_ids: all labels are -100, CE loss will be meaningless.")

            all_ids.append(token_ids)

        return torch.tensor(all_ids, dtype=torch.long, device=device)
        # (bs, n_query_tokens)

    # ------------------------------------------------------------------
    # 辅助（CE 模式）：推理时 token ids → waypoints tensor
    # ------------------------------------------------------------------
    def _decode_waypoints_from_token_ids(
        self, token_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        将 n_query_tokens 个预测 token 拼成文本，再用 parse_waypoints_from_text 解析。
        token_ids: (bs, n_query_tokens) LongTensor
        返回: (bs, num_waypoints, waypoint_dim) FloatTensor
        """
        bs = token_ids.shape[0]
        results = []
        for i in range(bs):
            partial_text = self.tokenizer.decode(
                token_ids[i].tolist(), skip_special_tokens=True
            )
            floats = self.parse_waypoints_from_text(partial_text)
            results.append(floats)

        return torch.tensor(
            results, dtype=torch.float32, device=token_ids.device
        ).view(bs, self.num_waypoints, self.waypoint_dim)

    # ------------------------------------------------------------------
    # 辅助：构造 Image Chunk Mask（帧内双向 + 帧间因果）
    # ------------------------------------------------------------------
    @staticmethod
    def build_image_chunk_mask(
        input_ids: torch.Tensor,          # (bs, seq_len)
        image_grid_thw: torch.Tensor,     # (N_total_imgs, 3)
        spatial_merge_size: int,
        image_token_id: int,
        n_cams_per_frame: int,
        n_query_tokens: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        构造 Image Chunk Mask，shape = (bs, 1, full_seq_len, full_seq_len)。
        full_seq_len = input_ids.shape[1] + n_query_tokens（含末尾 query token）。

        规则:
          - 帧内所有图像 token 之间：双向可见（打开上三角）
          - 帧间：因果（新帧可看旧帧，反之不行）
          - 文字 token：标准因果
          - query token（末尾 n_query_tokens 个）：因果 mask 已允许它们看所有前序 token

        需要 attn_implementation="sdpa" 或 "eager"（flash_attention_2 不支持 4D 浮点 mask）。

        注意：必须在拼接 query token 之后调用（此时 inputs_embeds 已经是 full_seq_len）。
        """
        bs, orig_seq_len = input_ids.shape
        full_seq_len = orig_seq_len + n_query_tokens  # 含 query token 的完整长度
        device = input_ids.device
        min_val = torch.finfo(dtype).min

        # ── 1. 计算每帧图像 token 在序列中的位置范围 ────────────────────
        s = spatial_merge_size
        tokens_per_img = [
            int(thw[0] * thw[1] * thw[2] / (s * s))
            for thw in image_grid_thw
        ]
        n_frames = len(tokens_per_img) // n_cams_per_frame

        img_positions = (input_ids[0] == image_token_id).nonzero(as_tuple=True)[0]

        frame_ranges = []
        pos_cursor = 0
        for f in range(n_frames):
            frame_n_tokens = sum(tokens_per_img[f * n_cams_per_frame:(f + 1) * n_cams_per_frame])
            start = img_positions[pos_cursor].item()
            end   = img_positions[pos_cursor + frame_n_tokens - 1].item() + 1
            frame_ranges.append((start, end))
            pos_cursor += frame_n_tokens

        # ── 2. 以标准因果 mask 为基底（full_seq_len × full_seq_len）────────
        # query token 行在下三角末尾，自然可以 attend 所有前序 token
        causal = torch.tril(torch.ones(full_seq_len, full_seq_len, device=device, dtype=torch.bool))
        mask = torch.where(causal,
                           torch.zeros(full_seq_len, full_seq_len, device=device, dtype=dtype),
                           torch.full((full_seq_len, full_seq_len), min_val, device=device, dtype=dtype))

        # ── 3. 帧内打开上三角（双向，仅在 orig_seq_len 范围内）────────────
        for (start, end) in frame_ranges:
            mask[start:end, start:end] = 0.0

        # ── 4. 扩展到 4D ─────────────────────────────────────────────────
        return mask.unsqueeze(0).unsqueeze(0).expand(bs, 1, -1, -1)
        # (bs, 1, full_seq_len, full_seq_len)

    # ------------------------------------------------------------------
    # 辅助：从文本 label 解析 waypoints
    # ------------------------------------------------------------------
    @staticmethod
    def parse_waypoints_from_text(text: str):
        """'[(x,y,z), ...]' → flat list of 18 floats"""
        if isinstance(text, str):
            pattern = r"\(([+-]?\d+\.?\d*)\s*,\s*([+-]?\d+\.?\d*)\s*,\s*([+-]?\d+\.?\d*)\)"
            matches = re.findall(pattern, text)
            waypoints = []
            for x, y, z in matches[:6]:
                waypoints.extend([float(x), float(y), float(z)])
            while len(waypoints) < 18:
                waypoints.append(0.0)
            return waypoints[:18]
        return [0.0] * 18

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    # ── NaN 调试开关（训练时设为 True，找到原因后关掉）──────────────
    _DEBUG_NAN: bool = False

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=True,
        return_dict=None,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        use_image_chunk_mask: bool = False,   # 帧内双向注意力开关
        n_cams_per_frame: int = 4,             # 每帧摄像头数，构造 mask 时需要
        **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        dbg = self._DEBUG_NAN

        # ── Step 0: 检查输入基本信息 ──────────────────────────────────────
        if dbg:
            if labels is not None:
                n_valid = (labels != -100).sum().item()
                n_total = labels.numel()
                print(f"[NaN-DBG] labels: shape={labels.shape} valid_tokens={n_valid}/{n_total}"
                      f" all_ignored={n_valid==0}")
                if n_valid > 0:
                    sample_valid = labels[0][labels[0] != -100][:20].tolist()
                    print(f"[NaN-DBG] labels[0] first valid tokens: {sample_valid}")
            else:
                print("[NaN-DBG] labels=None (no loss computed)")
            if input_ids is not None:
                print(f"[NaN-DBG] input_ids: shape={input_ids.shape}"
                      f" n_img_tokens={(input_ids==self.config.image_token_id).sum().item()}")

        # ── Step 1: embed tokens + 合并图像特征 ────────────────────────
        # 复用父类在 Qwen2_5_VLForConditionalGeneration.forward() 里的
        # pixel_values 处理逻辑（embed_tokens + visual encoder merge）
        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
            if dbg:
                print(f"[NaN-DBG] after embed_tokens: nan={torch.isnan(inputs_embeds).any().item()}"
                      f" inf={torch.isinf(inputs_embeds).any().item()}"
                      f" shape={inputs_embeds.shape} dtype={inputs_embeds.dtype}")

            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.dtype)
                if dbg:
                    print(f"[NaN-DBG] pixel_values: nan={torch.isnan(pixel_values).any().item()}"
                          f" min={pixel_values.float().min().item():.4f}"
                          f" max={pixel_values.float().max().item():.4f}"
                          f" shape={pixel_values.shape}")
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                if dbg:
                    print(f"[NaN-DBG] after visual encoder: nan={torch.isnan(image_embeds).any().item()}"
                          f" inf={torch.isinf(image_embeds).any().item()}"
                          f" shape={image_embeds.shape}")
                mask = (input_ids == self.config.image_token_id)
                image_mask = mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
                if dbg:
                    print(f"[NaN-DBG] after visual merge: nan={torch.isnan(inputs_embeds).any().item()}"
                          f" inf={torch.isinf(inputs_embeds).any().item()}")

            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                mask = (input_ids == self.config.video_token_id)
                video_mask = mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        bs = inputs_embeds.shape[0]

        # ── Step 2: 计算原始序列的 position_ids ────────────────────────
        # 始终从 input_ids 重算，确保 seq_len 与 inputs_embeds 严格对齐。
        # collator 预计算的 position_ids 经 pad_and_cat 后 seq_len 可能长于
        # model_max_length（truncation bug），直接使用会导致 RoPE 维度不匹配 → NaN。
        if input_ids is not None and (attention_mask is None or attention_mask.ndim == 2):
            position_ids, _ = self.get_rope_index(
                input_ids, image_grid_thw, video_grid_thw, None, attention_mask
            )
            # position_ids: (3, bs, orig_seq_len)  orig_seq_len == inputs_embeds.shape[1]
            if dbg:
                print(f"[NaN-DBG] position_ids: shape={position_ids.shape}"
                      f" max={position_ids.max().item()} min={position_ids.min().item()}")

        # ── Step 3: 拼接可学习 query token ─────────────────────────────
        queries = self.traj_queries.unsqueeze(0).expand(bs, -1, -1).to(inputs_embeds.dtype)
        inputs_embeds = torch.cat([inputs_embeds, queries], dim=1)
        # inputs_embeds: (bs, orig_seq_len + n_query_tokens, hidden)

        # ── Step 4: 延伸 attention_mask ─────────────────────────────────
        if attention_mask is not None:
            if attention_mask.ndim == 2:
                query_mask = torch.ones(
                    bs, self.n_query_tokens,
                    device=attention_mask.device,
                    dtype=attention_mask.dtype,
                )
                attention_mask = torch.cat([attention_mask, query_mask], dim=1)
            # 4D mask（image chunk mask）已在 Step 6 覆盖，此处跳过

        # ── Step 5: 延伸 position_ids ───────────────────────────────────
        if position_ids is not None:
            max_pos = position_ids[0].max(dim=-1).values  # (bs,)
            q_offsets = torch.arange(
                1, self.n_query_tokens + 1, device=position_ids.device
            )
            query_pos = max_pos.unsqueeze(-1) + q_offsets.unsqueeze(0)
            query_pos_3d = query_pos.unsqueeze(0).expand(3, -1, -1)
            position_ids = torch.cat([position_ids, query_pos_3d], dim=-1)
            if dbg:
                print(f"[NaN-DBG] position_ids after query extension: shape={position_ids.shape}"
                      f" max={position_ids.max().item()}")

        # ── Step 6: 可选 Image Chunk Mask ──────────────────────────────
        if use_image_chunk_mask and image_grid_thw is not None and input_ids is not None:
            attention_mask = self.build_image_chunk_mask(
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                spatial_merge_size=self.config.vision_config.spatial_merge_size,
                image_token_id=self.config.image_token_id,
                n_cams_per_frame=n_cams_per_frame,
                n_query_tokens=self.n_query_tokens,
                dtype=inputs_embeds.dtype,
            )

        # ── Step 7: backbone ────────────────────────────────────────────
        # output_hidden_states=False：只取 last_hidden_state，避免
        # gradient_checkpointing + 全量中间态 → NaN 梯度的冲突。
        if dbg:
            print(f"[NaN-DBG] inputs_embeds before backbone: nan={torch.isnan(inputs_embeds).any().item()}"
                  f" shape={inputs_embeds.shape}"
                  f" attn_mask shape={attention_mask.shape if attention_mask is not None else None}")
        model_outputs = self.model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=False,
            output_attentions=output_attentions,
            output_hidden_states=False,
            return_dict=True,
        )

        # ── Step 8: 取最后 n_query_tokens 个位置的 hidden states ────────
        last_hidden = model_outputs.last_hidden_state          # (bs, N+n_q, hidden)
        traj_hidden = last_hidden[:, -self.n_query_tokens:, :] # (bs, n_q, hidden)
        if dbg:
            print(f"[NaN-DBG] last_hidden_state: nan={torch.isnan(last_hidden).any().item()}"
                  f" traj_hidden nan={torch.isnan(traj_hidden).any().item()}"
                  f" shape={last_hidden.shape}")

        # ── Step 9: 输出头（按模式分支）────────────────────────────────
        if self.use_lm_head:
            # ── CE 模式 ────────────────────────────────────────────────
            # traj_hidden: (bs, 18, hidden) → lm_head → (bs, 18, vocab_size)
            head_logits = self.lm_head(traj_hidden)
            if dbg:
                print(f"[NaN-DBG] CE head_logits: nan={torch.isnan(head_logits).any().item()}"
                      f" shape={head_logits.shape}")

            loss = None
            if labels is not None:
                target_ids = self._parse_waypoint_token_ids(labels)  # (bs, 18)
                if dbg:
                    print(f"[NaN-DBG] CE target_ids: {target_ids[0].tolist()}")
                loss = F.cross_entropy(
                    head_logits.reshape(-1, head_logits.size(-1)),
                    target_ids.reshape(-1),
                )
                if dbg:
                    print(f"[NaN-DBG] CE loss: {loss.item()}")

            # 推理时解码 token → float
            token_ids = head_logits.argmax(-1)            # (bs, 18)
            waypoints = self._decode_waypoints_from_token_ids(token_ids)

            if not return_dict:
                return ((loss, waypoints) if loss is not None else (waypoints,))

            return Qwen2_5_VLLearnableQOutput(
                loss=loss,
                waypoints=waypoints,
                logits=head_logits.reshape(bs, -1),
            )

        else:
            # ── 回归模式 ───────────────────────────────────────────────
            normalized_waypoints = self.trajectory_head(traj_hidden)  # (bs, 6, 3)
            if dbg:
                print(f"[NaN-DBG] normalized_waypoints: nan={torch.isnan(normalized_waypoints).any().item()}"
                      f" values={normalized_waypoints[0].float().tolist()}")
            waypoints = normalized_waypoints * self.waypoint_std.to(normalized_waypoints.dtype) \
                      + self.waypoint_mean.to(normalized_waypoints.dtype)

            loss = None
            if labels is not None:
                target_list = []
                for i in range(bs):
                    if labels.dim() == 2:
                        valid_ids = labels[i][labels[i] != -100]
                        text = self.tokenizer.decode(valid_ids, skip_special_tokens=True)
                    else:
                        text = str(labels[i])
                    if dbg and i == 0:
                        print(f"[NaN-DBG] label text[0]: '{text[:120]}'")
                    target_list.append(self.parse_waypoints_from_text(text))

                target_tensor = torch.tensor(
                    target_list,
                    dtype=torch.float32,
                    device=normalized_waypoints.device,
                ).view(bs, self.num_waypoints, self.waypoint_dim)

                # waypoint_std 在 float32，确保不出现 fp16/bf16 精度截断
                mean_f32 = self.waypoint_mean.float().view(1, 1, -1)
                std_f32  = self.waypoint_std.float().view(1, 1, -1)
                normalized_target = (target_tensor - mean_f32) / std_f32.clamp(min=1e-6)

                if dbg:
                    print(f"[NaN-DBG] waypoint_std: {self.waypoint_std.tolist()}")
                    print(f"[NaN-DBG] target_tensor[0]: {target_tensor[0].tolist()}")
                    print(f"[NaN-DBG] normalized_target: nan={torch.isnan(normalized_target).any().item()}"
                          f" values={normalized_target[0].tolist()}")

                loss = F.smooth_l1_loss(
                    normalized_waypoints.float(),
                    normalized_target,
                    beta=0.1,
                )
                if dbg:
                    print(f"[NaN-DBG] regression loss: {loss.item()}")

            if not return_dict:
                return ((loss, waypoints) if loss is not None else (waypoints,))

            return Qwen2_5_VLLearnableQOutput(
                loss=loss,
                waypoints=waypoints,
                logits=normalized_waypoints.reshape(bs, -1),
            )

    @torch.no_grad()
    def generate(self, **kwargs):
        """
        模拟 generate 接口，使推理代码无需修改。
        """
        self.eval()
        outputs = self.forward(**kwargs)
        waypoints_list = outputs.waypoints[0].cpu().tolist()
        text = "[" + ", ".join(
            f"({x:.2f},{y:.2f},{z:.2f})" for x, y, z in waypoints_list
        ) + "] These are the future waypoints. \n"
        return [[text]]
