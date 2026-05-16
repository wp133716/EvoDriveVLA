"""
Qwen2.5-VL action-token waypoint policy.

This module follows the LaST-R1-style action-token design without changing the
existing regression or learnable-query model files:

  image/text prompt -> Qwen2.5-VL backbone -> action placeholder positions
  -> lm_head over <action_0> ... <action_N> -> discretized waypoints

The class preserves the common `outputs.waypoints` inference interface used by
the existing regression deployment path.
"""

import re
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers.modeling_outputs import ModelOutput

from .modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration


IGNORE_INDEX = -100


@dataclass
class Qwen2_5_VLActionTokenOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    waypoints: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    action_token_ids: Optional[torch.LongTensor] = None


class Qwen2_5_VLForActionToken(Qwen2_5_VLForConditionalGeneration):
    """Qwen2.5-VL policy that predicts waypoints as discrete action tokens."""

    def __init__(
        self,
        config,
        num_action_steps=6,
        action_dim=3,
        action_bins=256,
        action_token_prefix="<action_",
        waypoint_mean=None,
        waypoint_std=None,
    ):
        super().__init__(config)
        self.num_action_steps = int(num_action_steps)
        self.action_dim = int(action_dim)
        self.action_bins = int(action_bins)
        self.action_token_prefix = action_token_prefix
        self.action_token_len = self.num_action_steps * self.action_dim
        self.action_0_id = None

        if waypoint_mean is None:
            waypoint_mean = [0.0] * self.action_dim
        if waypoint_std is None:
            waypoint_std = [1.0] * self.action_dim
        self.register_buffer("waypoint_mean", torch.tensor(waypoint_mean, dtype=torch.float32))
        self.register_buffer("waypoint_std", torch.tensor(waypoint_std, dtype=torch.float32))

    @staticmethod
    def action_special_tokens(action_bins=256):
        return [f"<action_{i}>" for i in range(int(action_bins))]

    def set_action_tokenizer(self, tokenizer):
        """Attach tokenizer and cache the start id for the contiguous action-token slice."""
        self.tokenizer = tokenizer
        action_0 = f"{self.action_token_prefix}0>"
        action_0_id = tokenizer.convert_tokens_to_ids(action_0)
        if action_0_id is None or action_0_id == tokenizer.unk_token_id:
            raise ValueError(
                f"{action_0} is missing from tokenizer. Add action_special_tokens() "
                "and resize model embeddings before training/inference."
            )

        ids = tokenizer.convert_tokens_to_ids(self.action_special_tokens(self.action_bins))
        expected = list(range(action_0_id, action_0_id + self.action_bins))
        if ids != expected:
            raise ValueError(
                "Action tokens must be contiguous in tokenizer ids because the model "
                "uses a compact lm_head slice for action logits."
            )
        self.action_0_id = int(action_0_id)

    def _ensure_action_tokens_ready(self):
        if self.action_0_id is None:
            if not hasattr(self, "tokenizer"):
                raise ValueError("Call set_action_tokenizer(tokenizer) before forward().")
            self.set_action_tokenizer(self.tokenizer)

    @staticmethod
    def parse_waypoints_from_text(text, num_action_steps, action_dim):
        target_len = int(num_action_steps) * int(action_dim)
        if isinstance(text, str):
            group_pattern = r"\(([^()]*)\)"
            number_pattern = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
            values_out = []
            for group in re.findall(group_pattern, text):
                values = re.findall(number_pattern, group)
                if len(values) != int(action_dim):
                    continue
                values_out.extend(float(v) for v in values)
                if len(values_out) >= target_len:
                    break
            while len(values_out) < target_len:
                values_out.append(0.0)
            return values_out[:target_len]
        return [0.0] * target_len

    def _labels_to_target_action_ids(self, labels, dtype, device):
        if labels is None:
            return None
        if not hasattr(self, "tokenizer"):
            raise ValueError("labels require self.tokenizer for waypoint text decoding.")

        flat_targets = []
        for i in range(labels.shape[0]):
            valid_ids = labels[i][labels[i] != IGNORE_INDEX]
            text = self.tokenizer.decode(valid_ids, skip_special_tokens=True)
            flat_targets.append(
                self.parse_waypoints_from_text(text, self.num_action_steps, self.action_dim)
            )

        target = torch.tensor(flat_targets, dtype=torch.float32, device=device).view(
            -1, self.num_action_steps, self.action_dim
        )
        mean = self.waypoint_mean.float().view(1, 1, -1).to(device)
        std = self.waypoint_std.float().view(1, 1, -1).to(device).clamp(min=1e-6)
        normalized = ((target - mean) / std).clamp(-1.0, 1.0)

        # Map [-1, 1] to integer bins [0, action_bins - 1].
        bin_ids = torch.round((normalized + 1.0) * 0.5 * (self.action_bins - 1)).long()
        return bin_ids.view(-1, self.action_token_len)

    def _action_ids_to_waypoints(self, action_ids):
        normalized = action_ids.float() / max(self.action_bins - 1, 1)
        normalized = normalized * 2.0 - 1.0
        normalized = normalized.view(-1, self.num_action_steps, self.action_dim)
        mean = self.waypoint_mean.to(normalized.device, normalized.dtype).view(1, 1, -1)
        std = self.waypoint_std.to(normalized.device, normalized.dtype).view(1, 1, -1)
        return normalized * std + mean

    def _strip_supervised_answer_tokens(self, input_ids, attention_mask, labels):
        """Keep prompt/system/assistant-prefix tokens and drop visible answer tokens."""
        if labels is None or input_ids is None:
            return input_ids, attention_mask

        if attention_mask is None:
            visible = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            visible = attention_mask.bool()
        visible = visible & labels.eq(IGNORE_INDEX)

        pad_id = self.config.pad_token_id if self.config.pad_token_id is not None else 0
        compact_ids = []
        compact_masks = []
        max_len = 1
        for i in range(input_ids.shape[0]):
            ids = input_ids[i][visible[i]]
            if ids.numel() == 0:
                ids = input_ids[i, :1]
            compact_ids.append(ids)
            max_len = max(max_len, ids.numel())

        for ids in compact_ids:
            pad_len = max_len - ids.numel()
            if pad_len:
                ids = F.pad(ids, (0, pad_len), value=pad_id)
            compact_masks.append(ids.ne(pad_id))

        return torch.stack(compact_ids, dim=0), torch.stack(compact_masks, dim=0)

    def _merge_visual_inputs(self, input_ids, inputs_embeds, pixel_values, pixel_values_videos, image_grid_thw, video_grid_thw):
        if pixel_values is not None:
            pixel_values = pixel_values.type(self.visual.dtype)
            image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
            n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
            if n_image_tokens != image_embeds.shape[0]:
                raise ValueError(
                    f"Image features and image tokens do not match: tokens={n_image_tokens}, "
                    f"features={image_embeds.shape[0]}"
                )
            image_mask = (input_ids == self.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(
                image_mask.to(inputs_embeds.device),
                image_embeds.to(inputs_embeds.device, inputs_embeds.dtype),
            )

        if pixel_values_videos is not None:
            pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
            video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
            n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
            if n_video_tokens != video_embeds.shape[0]:
                raise ValueError(
                    f"Video features and video tokens do not match: tokens={n_video_tokens}, "
                    f"features={video_embeds.shape[0]}"
                )
            video_mask = (input_ids == self.config.video_token_id).unsqueeze(-1).expand_as(inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(
                video_mask.to(inputs_embeds.device),
                video_embeds.to(inputs_embeds.device, inputs_embeds.dtype),
            )
        return inputs_embeds

    def _extend_position_ids(self, position_ids, attention_mask):
        if position_ids is None:
            return None
        max_pos = position_ids[0].masked_fill(~attention_mask.bool(), 0).max(dim=-1).values
        offsets = torch.arange(1, self.action_token_len + 1, device=position_ids.device)
        action_pos = max_pos.unsqueeze(-1) + offsets.unsqueeze(0)
        action_pos = action_pos.unsqueeze(0).expand(3, -1, -1)
        return torch.cat([position_ids, action_pos], dim=-1)

    @staticmethod
    def build_mixed_action_mask(attention_mask, action_token_len, dtype):
        """Causal prompt + full-attention action block."""
        batch_size, prompt_len = attention_mask.shape
        full_len = prompt_len + action_token_len
        device = attention_mask.device
        min_val = torch.finfo(dtype).min

        causal = torch.tril(torch.ones(full_len, full_len, device=device, dtype=torch.bool))
        causal[-action_token_len:, -action_token_len:] = True

        key_valid = torch.cat(
            [
                attention_mask.bool(),
                torch.ones(batch_size, action_token_len, device=device, dtype=torch.bool),
            ],
            dim=1,
        )
        query_valid = key_valid
        allowed = causal.unsqueeze(0) & key_valid[:, None, :] & query_valid[:, :, None]
        # Avoid fully masked query rows for right-padded prompt positions. These
        # rows are not used by the action head, but SDPA can produce NaNs if an
        # entire query row is masked out.
        allowed = allowed | torch.eye(full_len, device=device, dtype=torch.bool).unsqueeze(0)

        mask = torch.full((batch_size, 1, full_len, full_len), min_val, device=device, dtype=dtype)
        mask = mask.masked_fill(allowed.unsqueeze(1), 0.0)
        return mask

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
        output_hidden_states=False,
        return_dict=None,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        do_sample: bool = False,
        temperature: float = 1.0,
        **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        self._ensure_action_tokens_ready()

        if input_ids is not None and labels is not None and inputs_embeds is None:
            input_ids, attention_mask = self._strip_supervised_answer_tokens(input_ids, attention_mask, labels)
            position_ids = None

        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
            inputs_embeds = self._merge_visual_inputs(
                input_ids, inputs_embeds, pixel_values, pixel_values_videos, image_grid_thw, video_grid_thw
            )
            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        batch_size = inputs_embeds.shape[0]
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, inputs_embeds.shape[1], device=inputs_embeds.device, dtype=torch.bool)

        if position_ids is None and input_ids is not None:
            position_ids, _ = self.get_rope_index(
                input_ids,
                image_grid_thw,
                video_grid_thw,
                None,
                attention_mask,
            )

        action_embeds = torch.zeros(
            batch_size,
            self.action_token_len,
            inputs_embeds.shape[-1],
            device=inputs_embeds.device,
            dtype=inputs_embeds.dtype,
        )
        inputs_embeds = torch.cat([inputs_embeds, action_embeds], dim=1)
        position_ids = self._extend_position_ids(position_ids, attention_mask)
        mixed_attention_mask = self.build_mixed_action_mask(
            attention_mask=attention_mask,
            action_token_len=self.action_token_len,
            dtype=inputs_embeds.dtype,
        )

        outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=mixed_attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=False,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        action_hidden = outputs.last_hidden_state[:, -self.action_token_len:, :]
        full_logits = self.lm_head(action_hidden)
        action_logits = full_logits[..., self.action_0_id:self.action_0_id + self.action_bins]

        target_ids = self._labels_to_target_action_ids(labels, action_logits.dtype, action_logits.device)
        loss = None
        if target_ids is not None:
            loss = F.cross_entropy(
                action_logits.float().reshape(-1, self.action_bins),
                target_ids.reshape(-1),
            )

        if do_sample:
            probs = torch.softmax(action_logits / max(float(temperature), 1e-6), dim=-1)
            action_ids = torch.multinomial(probs.reshape(-1, self.action_bins), 1).view(batch_size, self.action_token_len)
        else:
            action_ids = action_logits.argmax(dim=-1)
        waypoints = self._action_ids_to_waypoints(action_ids)

        if not return_dict:
            return ((loss, waypoints, action_logits) if loss is not None else (waypoints, action_logits))

        return Qwen2_5_VLActionTokenOutput(
            loss=loss,
            waypoints=waypoints,
            logits=action_logits,
            action_token_ids=action_ids,
        )

    @torch.no_grad()
    def generate(self, **kwargs):
        self.eval()
        outputs = self.forward(**kwargs)
        waypoints_list = outputs.waypoints[0].cpu().tolist()
        text = "[" + ", ".join(
            f"({x:.2f},{y:.2f},{z:.2f})" for x, y, z in waypoints_list
        ) + "] These are the future waypoints. \n"
        return [[text]]
