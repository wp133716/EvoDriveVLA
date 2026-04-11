"""
Qwen2.5-VL 回归版本 V2 - 最小改动方案
=====================================
复用生成方案的所有数据流，只修改：
1. 添加 trajectory_head
2. 修改 loss 计算

数据流：
  input_ids + pixel_values -> model -> hidden_states
  -> trajectory_head -> waypoints [batch, 6, 3]

labels: 文本格式 "[(x1,y1,z1), ...]" (和生成方案一样)
内部解析为张量计算 loss
"""

import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen2_5_VLForConditionalGeneration
from transformers.modeling_outputs import ModelOutput
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class Qwen2_5_VLRegressionOutput(ModelOutput):
    """回归模型输出"""
    loss: Optional[torch.FloatTensor] = None
    waypoints: torch.FloatTensor = None
    logits: torch.FloatTensor = None  # 保持兼容


class Qwen2_5_VLForRegressionV2(Qwen2_5_VLForConditionalGeneration):
    """
    回归版本 V2：最小改动，复用父类所有功能
    """

    def __init__(self, config, num_waypoints=6, waypoint_dim=3):
        # 先调用父类初始化
        super().__init__(config)

        self.num_waypoints = num_waypoints
        self.waypoint_dim = waypoint_dim
        self.output_dim = num_waypoints * waypoint_dim

        # 添加回归头
        hidden_size = config.hidden_size
        self.trajectory_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, self.output_dim),
        )

        # 初始化回归头
        self._init_weights(self.trajectory_head)

    @staticmethod
    def _init_weights(module):
        """初始化权重"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()

    @staticmethod
    def parse_waypoints_from_text(text):
        """从文本解析 waypoints"""
        if isinstance(text, str):
            pattern = r"\(([+-]?\d+\.?\d*)\s*,\s*([+-]?\d+\.?\d*)\s*,\s*([+-]?\d+\.?\d*)\)"
            matches = re.findall(pattern, text)
            waypoints = []
            for x, y, z in matches[:6]:
                waypoints.extend([float(x), float(y), float(z)])
            # 补齐
            while len(waypoints) < 18:
                waypoints.append(0.0)
            return waypoints[:18]
        return [0.0] * 18

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,  # 文本格式的 labels: "[(x,y,z), ...]"
        use_cache=None,
        output_attentions=None,
        output_hidden_states=True,  # 必须输出 hidden_states
        return_dict=None,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        **kwargs,
    ):
        """
        前向传播

        关键：labels 仍然是文本格式（和生成方案一样），
        内部解析为 waypoints 张量计算回归 loss
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 1. 调用父类的完整 forward 来处理视觉输入
        # 父类会将 pixel_values 转换为 inputs_embeds，然后传给 self.model
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=None,  # 不传递 labels，我们自己计算回归 loss
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,  # 必须输出 hidden_states
            return_dict=True,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
        )

        hidden_states = outputs.hidden_states[-1]  # [batch, seq_len, hidden]

        # 2. 取最后一个 token 的特征
        trajectory_feature = hidden_states[:, -1, :]  # [batch, hidden]

        # 3. 回归得到 waypoints
        waypoints_flat = self.trajectory_head(trajectory_feature)  # [batch, 18]
        waypoints = waypoints_flat.view(-1, self.num_waypoints, self.waypoint_dim)

        # 4. 计算 loss（如果提供了 labels）
        loss = None
        if labels is not None:
            # labels 是文本，需要解析为张量
            batch_size = waypoints.shape[0]
            target_waypoints = []

            for i in range(batch_size):
                # 获取第 i 个样本的 label
                if labels.dim() == 2:
                    # labels: [batch, seq_len] - token ids
                    # 需要解码为文本
                    label_ids = labels[i]
                    # 过滤掉 -100 (ignore_index)
                    valid_ids = label_ids[label_ids != -100]
                    text = self.tokenizer.decode(valid_ids, skip_special_tokens=True)
                else:
                    text = str(labels[i])

                coords = self.parse_waypoints_from_text(text)
                target_waypoints.append(coords)

            target_tensor = torch.tensor(
                target_waypoints,
                dtype=waypoints.dtype,
                device=waypoints.device
            ).view(-1, self.num_waypoints, self.waypoint_dim)

            # Smooth L1 loss
            loss = F.smooth_l1_loss(waypoints, target_tensor, beta=0.1)

        if not return_dict:
            return ((loss, waypoints) if loss is not None else (waypoints,))

        return Qwen2_5_VLRegressionOutput(
            loss=loss,
            waypoints=waypoints,
            logits=waypoints_flat,  # 保持兼容
        )

    @torch.no_grad()
    def generate(self, **kwargs):
        """
        模拟 generate 接口，返回 waypoints
        这样推理代码可以最小改动
        """
        self.eval()

        # 获取输出
        outputs = self.forward(**kwargs)
        waypoints = outputs.waypoints  # [1, 6, 3]

        # 转换为文本格式（和生成方案兼容）
        waypoints_list = waypoints[0].cpu().tolist()
        text = "[" + ", ".join(
            [f"({x:.2f},{y:.2f},{z:.2f})" for x, y, z in waypoints_list]
        ) + "] These are the future waypoints. \n"

        # 模拟 generate 返回格式
        from transformers import GenerationMixin
        class FakeGenerationOutput:
            def __init__(self, text):
                # tokenize 文本得到 fake ids
                self.text = text

        return [[text]]
