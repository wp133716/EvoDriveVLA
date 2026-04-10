import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import ModelOutput
from transformers import Qwen2_5_VLForConditionalGeneration as Qwen2Base
from transformers import Qwen2_5_VLModel
from ..vision_models import Qwen2_5_VisionTransformerPretrainedModel
from model.language_models import Qwen2_5_VLForConditionalGeneration
import copy
import random
import torch.distributed as dist

@dataclass
class Qwen2_5_VLCausalLMOutputWithPast(ModelOutput):
    """
    Base class for Qwen2_5_VL causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
            The rope index difference between sequence length and multimodal rope.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    rope_deltas: Optional[torch.LongTensor] = None

class AnchorFormer(nn.Module):
    def __init__(self, base_model, num_layers = 1):
        super().__init__()

        self.layers = nn.ModuleList([copy.deepcopy(base_model.layers[i]) for i in range(num_layers)])
        self.rotary_emb = copy.deepcopy(base_model.rotary_emb)

        self.query = nn.Parameter(
            torch.randn(1, base_model.config.hidden_size)
        )

        self.ranker = nn.Linear(
            base_model.config.hidden_size, 1, bias=False
        )

    def forward(
        self,
        hidden_states,
        attention_mask,
        position_ids,
        past_key_values,
        output_attentions,
        use_cache,
        cache_position,
        position_embeddings,
        image_mask,
    ):
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                output_attentions=False,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )[0]  

        # query token output
        query_hidden = hidden_states[:, -1, :]  # (B, C)
        
        num_visual = image_mask[0].sum().item()  
        B, _, C = hidden_states.shape
        visual_hidden_states = torch.zeros(B, num_visual, C, device=hidden_states.device, dtype=hidden_states.dtype)

        for b in range(B):
            mask_b = image_mask[b]  # [N]
            hidden_b = hidden_states[b]  # [N, C]
            # 直接提取掩码位置的特征
            visual_hidden_states[b] = hidden_b[mask_b]

        # visual weights
        visual_weights = self.ranker(
            visual_hidden_states * query_hidden.unsqueeze(1)
        ).squeeze(-1) 

        temperature = 2.0
        visual_weights = torch.sigmoid(visual_weights / temperature)

        return visual_weights.reshape(-1)

class Qwen2_5_VLForConditionalGeneration_KD(Qwen2Base):
    def __init__(
        self, 
        config, 
        encoder_kd=False, 
        encoder_loss_weight=0.5,
        llm_kd=False, 
        logits_loss=False,
        logits_loss_weight=0.5,
        logits_loss_temperature=3.0,
        hs_loss=False,
        hs_loss_weight=0.5,
        feature_loss=False,
        feature_loss_weight=0.5,
        feature_loss_layer=4,
        train_teacher=False,
        tokenizer=None,
    ):
        super().__init__(config)
        self.visual = Qwen2_5_VisionTransformerPretrainedModel._from_config(config.vision_config)
        self.model = Qwen2_5_VLModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.rope_deltas = None  # cache rope_deltas here
        # Initialize weights and apply final processing
        self.post_init()

        self.train_teacher = train_teacher

        self.encoder_kd = encoder_kd
        self.encoder_loss_weight = encoder_loss_weight

        self.llm_kd = llm_kd

        self.logits_loss = logits_loss
        self.logits_loss_weight = logits_loss_weight
        self.logits_loss_temperature = logits_loss_temperature

        self.hs_loss = hs_loss
        self.hs_loss_weight = hs_loss_weight

        self.teacher = None
        self.teacher_rope_deltas = None   

        self.tokenizer = tokenizer

    # Initialize vision encoder teachers
    def init_teacher(
        self, 
        teacher_model_name_or_path=None, 
        teacher_cache_dir=None, 
        teacher_attn_implementation=None,
    ):
        if self.llm_kd:
            self.teacher = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                teacher_model_name_or_path,
                cache_dir=teacher_cache_dir,
                attn_implementation=teacher_attn_implementation,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                device_map={"": torch.cuda.current_device()},
            )
            self.teacher.requires_grad_(False)
            self.teacher.eval()

            # 保存模型时忽略教师模型
            if self._keys_to_ignore_on_save is None:
                self._keys_to_ignore_on_save = set()
            else:
                self._keys_to_ignore_on_save = set(self._keys_to_ignore_on_save)

            for name, _ in self.teacher.named_parameters():
                self._keys_to_ignore_on_save.add("teacher." + name)
            
            for name, _ in self.teacher.named_buffers():
                self._keys_to_ignore_on_save.add("teacher." + name)
                
            print(f"Ignored {len(self._keys_to_ignore_on_save)} teacher keys from saving process.")

            self.drop = torch.nn.Dropout(p=0.1)

        if self.encoder_kd:
            self.encoder_teacher = copy.deepcopy(self.visual)
            self.anchor_former = AnchorFormer(self.model)
        else:
            self.encoder_teacher = None
            self.anchor_former = None


    def get_input_embeds(
        self,
        inputs_embeds=None, 
        input_ids=None, 
        pixel_values=None, 
        image_grid_thw=None, 
        pixel_values_videos=None, 
        video_grid_thw=None, 
        attention_mask=None,
        teacher=False,
    ):
        if teacher:
            inputs_embeds = self.teacher.model.embed_tokens(input_ids)
        else:
            inputs_embeds = self.model.embed_tokens(input_ids)
        if pixel_values is not None:
            if teacher:
                pixel_values = pixel_values.type(self.teacher.visual.dtype)
                image_embeds = self.teacher.visual(pixel_values, grid_thw=image_grid_thw)
            else:
                pixel_values = pixel_values.type(self.visual.dtype) # [b*patch_nums, temporal_patch_size*patch_size**2*3]
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw) # [b*patch_nums//4, 2048]
            n_image_tokens = (input_ids == self.config.image_token_id).sum().item() 
            n_image_features = image_embeds.shape[0]
            if n_image_tokens != n_image_features:
                raise ValueError(
                    f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                )

            mask = input_ids == self.config.image_token_id
            mask_unsqueezed = mask.unsqueeze(-1)
            mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
            image_mask = mask_expanded.to(inputs_embeds.device)

            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            if teacher:
                pixel_values_videos = pixel_values_videos.type(self.teacher.visual.dtype)
                video_embeds = self.teacher.visual(pixel_values_videos, grid_thw=video_grid_thw)
            else:
                pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
            n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
            n_video_features = video_embeds.shape[0]
            if n_video_tokens != n_video_features:
                raise ValueError(
                    f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                )

            mask = input_ids == self.config.video_token_id
            mask_unsqueezed = mask.unsqueeze(-1)
            mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
            video_mask = mask_expanded.to(inputs_embeds.device)

            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        if attention_mask is not None:
            attention_mask = attention_mask.to(inputs_embeds.device)

        return inputs_embeds, attention_mask, image_embeds, image_mask

    def mc_dropout_trajectory(
        self,
        teacher_hidden_states,
        teacher_mask,
        labels,
        student_mask,
        n_samples: int = 10
    ):
        """
        对每个 batch 样本多次 dropout，返回每个样本 loss 最小对应的 valid_teacher_logits 并拼接
        """
        batch_size = teacher_hidden_states.shape[0]
        best_logits_list = []
        best_hidden_states_list = []
        best_loss_list = []
        loss_fct = CrossEntropyLoss()

        for i in range(batch_size):
            best_loss = None
            best_logits = None
            best_hidden_states = None

            hs_i = teacher_hidden_states[i:i+1]       
            mask_i = teacher_mask[i]                  
            labels_i = labels[i]                
            student_mask_i = student_mask[i]          

            for _ in range(n_samples):
                dropped_hs = self.drop(hs_i)

                teacher_logits = self.lm_head(dropped_hs).float()         
                shift_logits_teacher = teacher_logits[..., :-1, :].contiguous()  
                valid_teacher_logits = shift_logits_teacher[0][mask_i]          

                loss = loss_fct(valid_teacher_logits, labels_i[student_mask_i.view(-1)])

                if best_loss is None or loss.item() < best_loss:
                    best_loss = loss.item()
                    best_logits = valid_teacher_logits.clone()
                    best_hidden_states = dropped_hs.clone()

            best_logits_list.append(best_logits)
            best_hidden_states_list.append(best_hidden_states)
            best_loss_list.append(best_loss)

        batch_logits = torch.stack(best_logits_list, dim=0)
        batch_hidden_states = torch.cat(best_hidden_states_list, dim=0)
        return batch_logits, batch_hidden_states, best_loss_list

    def refine_trajectory(
            self,
            teacher_inputs_embeds, 
            teacher_position_ids, 
            teacher_labels,
            shift_logits,
            mask,
            teacher=False,
        ):
            batch_size = teacher_labels.shape[0]
            seq_len = teacher_labels.shape[1]
            hidden_dim = teacher_inputs_embeds.shape[2]

            valid_teacher_logits = []

            for i in range(batch_size):
                logits_i = shift_logits[i][mask[i]]
                valid_teacher_logits.append(logits_i)

            valid_teacher_logits = torch.stack(valid_teacher_logits, dim=0)  

            teacher_token_ids = valid_teacher_logits.argmax(dim=-1)  

            if teacher:
                embedding_layer = self.teacher.model.get_input_embeddings()
            else:
                embedding_layer = self.model.get_input_embeddings()  # nn.Embedding

            teacher_embeddings = embedding_layer(teacher_token_ids)  # [batch_size, seq_len, hidden_dim]

            prompt_text = "I can provide the trajectory predictions from the student model here, and you can also refine them based on this:"
            prompt_ids = self.tokenizer(prompt_text, return_tensors="pt").input_ids.to(teacher_embeddings.device)  # [1, prompt_len]
            prompt_embeddings = embedding_layer(prompt_ids)  # [1, prompt_len, hidden_dim]
            prompt_embeddings = prompt_embeddings.expand(batch_size, -1, -1)  # [batch_size, prompt_len, hidden_dim]

            add_embeds = torch.cat([prompt_embeddings, teacher_embeddings], dim=1)  # [batch_size, prompt_len + seq_len, hidden_dim]
            add_len = add_embeds.shape[1]

            refine_inputs_embeds = []
            refine_attention_mask = []
            refine_labels = []
            refine_position_ids = []

            for i in range(batch_size):
                answer_mask = (teacher_labels[i] != -100)
                start = answer_mask.float().argmax().item()

                prefix_embeds = teacher_inputs_embeds[i, :start]
                suffix_embeds = teacher_inputs_embeds[i, start:]
                embeds = torch.cat([prefix_embeds, add_embeds[i], suffix_embeds], dim=0)
                refine_inputs_embeds.append(embeds)

                attn_mask = torch.ones(embeds.shape[0], device=teacher_inputs_embeds.device, dtype=torch.long)
                refine_attention_mask.append(attn_mask)

                labels = torch.cat([
                    teacher_labels[i, :start],
                    torch.full((add_len,), -100, device=teacher_labels.device, dtype=teacher_labels.dtype),
                    teacher_labels[i, start:]
                ], dim=0)
                refine_labels.append(labels)

                prefix_pos = teacher_position_ids[:, i, :start] 
                suffix_len = suffix_embeds.shape[0] 
                start_val = prefix_pos[0, -1].item()
                suffix_pos = torch.arange(start_val + 1, start_val + 1 + suffix_len + add_len, device=teacher_inputs_embeds.device)
                suffix_pos = suffix_pos.unsqueeze(0).expand(3, -1)
                pos_ids = torch.cat([prefix_pos, suffix_pos], dim=1)
                refine_position_ids.append(pos_ids)

            refine_inputs_embeds = torch.stack(refine_inputs_embeds, dim=0) 
            refine_attention_mask = torch.stack(refine_attention_mask, dim=0)
            refine_labels = torch.stack(refine_labels, dim=0)
            refine_position_ids = torch.stack(refine_position_ids, dim=1) 

            return refine_inputs_embeds, refine_attention_mask, refine_position_ids, refine_labels

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        batch_img_metas: Optional[List[Dict]] = None,

        teacher_input_ids: torch.LongTensor = None,
        teacher_attention_mask: Optional[torch.Tensor] = None,
        teacher_position_ids: Optional[torch.LongTensor] = None,
        teacher_inputs_embeds: Optional[torch.FloatTensor] = None,
        teacher_labels: Optional[torch.LongTensor] = None,
        teacher_pixel_values: Optional[torch.Tensor] = None,
        teacher_pixel_values_videos: Optional[torch.FloatTensor] = None,
        teacher_image_grid_thw: Optional[torch.LongTensor] = None,
        teacher_video_grid_thw: Optional[torch.LongTensor] = None,
        teacher_cache_position: Optional[torch.LongTensor] = None,
        teacher_rope_deltas: Optional[torch.LongTensor] = None,
        teacher_past_key_values: Optional[List[torch.FloatTensor]] = None,
        teacher_second_per_grid_ts: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[Tuple, Qwen2_5_VLCausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        >>> model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
        >>> processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

        >>> messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What is shown in this image?"},
                ],
            },
        ]
        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        >>> inputs = processor(text=[text], images=[image], vision_infos=[vision_infos])

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "The image shows a street scene with a red stop sign in the foreground. In the background, there is a large red gate with Chinese characters ..."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds, attention_mask, image_embeds, image_mask = self.get_input_embeds(
                inputs_embeds=inputs_embeds,
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                pixel_values_videos=pixel_values_videos,
                video_grid_thw=video_grid_thw,
                attention_mask=attention_mask,
            )

        # if we get 4D attention mask we cannot calculate rope deltas anymore. TODO @raushan fixme
        if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
            # calculate RoPE index once per generation in the pre-fill stage only
            if (
                (cache_position is not None and cache_position[0] == 0)
                or self.rope_deltas is None
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            ):
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    second_per_grid_ts,
                    attention_mask,
                ) # [3,b,token_nums], [b, 1]
                self.rope_deltas = rope_deltas
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                    if cache_position is not None
                    else 0
                )
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        refine_key = random.randint(0, 1)
        if self.train_teacher and refine_key == 0:
            if labels is not None:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                mask = (shift_labels != -100)

                refined_inputs_embeds, refined_attention_mask, refined_position_ids, refined_labels = self.refine_trajectory(
                    teacher_inputs_embeds=inputs_embeds,
                    teacher_position_ids=position_ids,
                    teacher_labels=labels,
                    shift_logits=shift_logits,
                    mask=mask,
                )
                labels = refined_labels
            
            refined_outputs = self.model(
                input_ids=None,
                position_ids=refined_position_ids,
                attention_mask=refined_attention_mask,
                past_key_values=past_key_values,
                inputs_embeds=refined_inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
            )

            hidden_states = refined_outputs[0]
            logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Upcast to float if we need to compute the loss to avoid potential precision issues
            logits = logits.float()
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

            # === DEBUG: 检测 base CE loss 异常 ===
            if torch.isnan(loss) or torch.isinf(loss):
                _rank = dist.get_rank() if dist.is_initialized() else 0
                _sample_ids = kwargs.get("id", ["unknown"])
                _valid = (shift_labels != -100).sum().item()
                _has_nan_logits = torch.isnan(shift_logits).any().item()
                _has_inf_logits = torch.isinf(shift_logits).any().item()
                _logits_max = shift_logits.max().item() if not _has_nan_logits else "NaN"
                _logits_min = shift_logits.min().item() if not _has_nan_logits else "NaN"
                print(f"[Rank {_rank}] DEBUG CE loss={loss.item()}, sample_ids={_sample_ids}, valid_tokens={_valid}, "
                      f"logits_has_nan={_has_nan_logits}, logits_has_inf={_has_inf_logits}, "
                      f"logits_range=[{_logits_min}, {_logits_max}]")
                if pixel_values is not None:
                    print(f"[Rank {_rank}] DEBUG pixel_values: has_nan={torch.isnan(pixel_values).any().item()}, "
                          f"has_inf={torch.isinf(pixel_values).any().item()}, shape={pixel_values.shape}")
                if image_embeds is not None:
                    print(f"[Rank {_rank}] DEBUG image_embeds: has_nan={torch.isnan(image_embeds).any().item()}, "
                          f"has_inf={torch.isinf(image_embeds).any().item()}, shape={image_embeds.shape}")
                print(f"[Rank {_rank}] DEBUG inputs_embeds: has_nan={torch.isnan(inputs_embeds).any().item()}, "
                      f"has_inf={torch.isinf(inputs_embeds).any().item()}")

        if self.llm_kd:
            with torch.no_grad():
                if self.teacher is None:
                    raise ValueError("Teacher model is not initialized for LLM KD.")
                if teacher_inputs_embeds is None:
                    teacher_inputs_embeds, teacher_attention_mask, _, teacher_image_mask = self.get_input_embeds(
                        inputs_embeds=teacher_inputs_embeds,
                        input_ids=teacher_input_ids,
                        pixel_values=teacher_pixel_values,
                        image_grid_thw=teacher_image_grid_thw,
                        pixel_values_videos=teacher_pixel_values_videos,
                        video_grid_thw=teacher_video_grid_thw,
                        attention_mask=teacher_attention_mask,
                        teacher=True,
                    )

                if teacher_position_ids is None and (teacher_attention_mask is None or teacher_attention_mask.ndim == 2):
                    # calculate RoPE index once per generation in the pre-fill stage only
                    if (
                        (teacher_cache_position is not None and teacher_cache_position[0] == 0)
                        or self.teacher_rope_deltas is None
                        or (teacher_past_key_values is None or teacher_past_key_values.get_seq_length() == 0)
                    ):
                        teacher_position_ids, teacher_rope_deltas = self.get_rope_index(
                            teacher_input_ids,
                            teacher_image_grid_thw,
                            teacher_video_grid_thw,
                            teacher_second_per_grid_ts,
                            teacher_attention_mask,
                        ) # [3,b,token_nums], [b, 1]
                        self.teacher_rope_deltas = teacher_rope_deltas
                    # then use the prev pre-calculated rope-deltas to get the correct position ids
                    else:
                        teacher_batch_size, teacher_seq_length, _ = teacher_inputs_embeds.shape
                        teacher_delta = (
                            (teacher_cache_position[0] + self.teacher_rope_deltas).to(teacher_inputs_embeds.device)
                            if teacher_cache_position is not None
                            else 0
                        )
                        teacher_position_ids = torch.arange(teacher_seq_length, device=teacher_inputs_embeds.device)
                        teacher_position_ids = teacher_position_ids.view(1, -1).expand(teacher_batch_size, -1)
                        if teacher_cache_position is not None:  # otherwise `deltas` is an int `0`
                            teacher_delta = teacher_delta.repeat_interleave(teacher_batch_size // teacher_delta.shape[0], dim=0)
                        teacher_position_ids = teacher_position_ids.add(teacher_delta)
                        teacher_position_ids = teacher_position_ids.unsqueeze(0).expand(3, -1, -1)

                teacher_outputs = self.teacher.model(
                    input_ids=None,
                    position_ids=teacher_position_ids,
                    attention_mask=teacher_attention_mask,
                    past_key_values=None,
                    inputs_embeds=teacher_inputs_embeds,
                    use_cache=False,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=True,
                )
                
                teacher_hidden_states = teacher_outputs[0]
                teacher_logits = self.teacher.lm_head(teacher_hidden_states).float()

                shift_student_labels = labels[..., 1:].contiguous()
                shift_student_mask = (shift_student_labels != -100)

                shift_teacher_labels = teacher_labels[..., 1:].contiguous()
                shift_teacher_mask = (shift_teacher_labels != -100)

                if (shift_student_mask is not None) and (shift_teacher_mask is not None):
                    shift_logits_student = logits[..., :-1, :].contiguous()
                    valid_student_logits = shift_logits_student[shift_student_mask]
                    shift_logits_teacher = teacher_logits[..., :-1, :].contiguous()
                    valid_teacher_logits = shift_logits_teacher[shift_teacher_mask]
                    
                    teacher_drop_logits, teacher_drop_hidden_states, teacher_drop_loss = self.mc_dropout_trajectory(
                        teacher_hidden_states, 
                        shift_teacher_mask,
                        shift_student_labels, 
                        shift_student_mask, 
                        n_samples = 10
                    )

                    refine_inputs_embeds, refine_attention_mask, refine_position_ids, refine_labels = self.refine_trajectory(
                        teacher_inputs_embeds, 
                        teacher_position_ids, 
                        teacher_labels,
                        shift_logits_teacher, 
                        shift_teacher_mask,
                        teacher=True,
                    )

                    refine_outputs = self.teacher.model(
                        input_ids=None,
                        position_ids=refine_position_ids,
                        attention_mask=refine_attention_mask,
                        past_key_values=None,
                        inputs_embeds=refine_inputs_embeds,
                        use_cache=False,
                        output_attentions=output_attentions,
                        output_hidden_states=output_hidden_states,
                        return_dict=True,
                    )

                    refine_hidden_states = refine_outputs[0]
                    refine_logits = self.teacher.lm_head(refine_hidden_states).float()

                    shift_refine_labels = refine_labels[..., 1:].contiguous()
                    shift_refine_mask = (shift_refine_labels != -100)
                    shift_logits_refine = refine_logits[..., :-1, :].contiguous()
                    refine_drop_logits, refine_drop_hidden_states, refine_drop_loss = self.mc_dropout_trajectory(
                        refine_hidden_states, shift_refine_mask,
                        shift_student_labels,  
                        shift_student_mask, 
                        n_samples = 10, 
                    )

                    batch_size = teacher_hidden_states.shape[0]                     

                    valid_teacher_logit_new = []
                    valid_teacher_hs_new = []

                    teacher_mask = (teacher_labels != -100)
                    refine_mask = (refine_labels != -100)

                    for i in range(batch_size):
                        logits_teacher = shift_logits_teacher[i][shift_teacher_mask[i]]
                        logits_teacher_drop = teacher_drop_logits[i]
                        logits_refine = shift_logits_refine[i][shift_refine_mask[i]]
                        logits_refine_drop = refine_drop_logits[i]
                        
                        hs_teacher = teacher_hidden_states[i].contiguous()[teacher_mask[i]]
                        hs_teacher_drop = teacher_drop_hidden_states[i].contiguous()[teacher_mask[i]]
                        hs_refine = refine_hidden_states[i].contiguous()[refine_mask[i]]
                        hs_refine_drop = refine_drop_hidden_states[i].contiguous()[refine_mask[i]]

                        labels_i = shift_student_labels[i][shift_student_mask[i].view(-1)]

                        loss_teacher = loss_fct(logits_teacher, labels_i)
                        loss_teacher_drop = torch.tensor(teacher_drop_loss[i], device=loss_teacher.device)
                        loss_refine = loss_fct(logits_refine, labels_i)
                        loss_refine_drop = torch.tensor(refine_drop_loss[i], device=loss_teacher.device)

                        losses = torch.stack([
                            loss_teacher,
                            loss_teacher_drop,
                            loss_refine,
                            loss_refine_drop
                        ])

                        logits_candidates = ([
                            logits_teacher,
                            logits_teacher_drop,
                            logits_refine,
                            logits_refine_drop
                        ])

                        hs_candidates = [
                            hs_teacher,
                            hs_teacher_drop,
                            hs_refine,
                            hs_refine_drop
                        ]

                        best_id = losses.argmin().item()
                        best_logits = logits_candidates[best_id]
                        best_hs = hs_candidates[best_id]
                        valid_teacher_logit_new.append(best_logits)
                        valid_teacher_hs_new.append(best_hs)

                    valid_teacher_logits = torch.cat(valid_teacher_logit_new, dim=0)
                    valid_teacher_hidden_states = torch.cat(valid_teacher_hs_new, dim=0)

                else:
                    raise NotImplementedError(
                        "KD without labels is not supported when student and teacher sequences have different lengths."
                    )

            if self.logits_loss:
                if valid_student_logits.shape[0] != valid_teacher_logits.shape[0]:
                    raise ValueError(
                        f"KD FAILED: Mismatched number of valid output tokens. "
                        f"Student has {valid_student_logits.shape[0]} valid tokens, "
                        f"but Teacher has {valid_teacher_logits.shape[0]} valid tokens. "
                        "Check your DataCollator and your assumption that output lengths are equal."
                    )

                loss_fct_kd = nn.KLDivLoss(reduction="batchmean")
                log_softmax_student = F.log_softmax(valid_student_logits / self.logits_loss_temperature, dim=-1)
                log_softmax_teacher = F.softmax(valid_teacher_logits / self.logits_loss_temperature, dim=-1)

                loss_logits = loss_fct_kd(
                    log_softmax_student,
                    log_softmax_teacher
                ) * (self.logits_loss_temperature * self.logits_loss_temperature)

                # === DEBUG: 检测 KD logits loss 异常 ===
                if torch.isnan(loss_logits) or torch.isinf(loss_logits):
                    _rank = dist.get_rank() if dist.is_initialized() else 0
                    print(f"[Rank {_rank}] DEBUG logits_loss={loss_logits.item()}, "
                          f"valid_tokens={valid_student_logits.shape[0]}, "
                          f"student_logits_nan={torch.isnan(valid_student_logits).any().item()}, "
                          f"teacher_logits_nan={torch.isnan(valid_teacher_logits).any().item()}, "
                          f"student_logits_range=[{valid_student_logits.min().item():.4f}, {valid_student_logits.max().item():.4f}], "
                          f"teacher_logits_range=[{valid_teacher_logits.min().item():.4f}, {valid_teacher_logits.max().item():.4f}]")

                if loss is None:
                    loss = self.logits_loss_weight * loss_logits
                else:
                    loss += self.logits_loss_weight * loss_logits
            if self.hs_loss:
                student_mask = (labels != -100)
                traj_student_hidden_states = hidden_states[student_mask]
                if traj_student_hidden_states.shape[0] != valid_teacher_hidden_states.shape[0]:
                    raise ValueError(
                        f"KD FAILED: Mismatched number of valid output tokens. "
                        f"Student has {traj_student_hidden_states.shape[0]} valid tokens, "
                        f"but Teacher has {valid_teacher_hidden_states.shape[0]} valid tokens. "
                        "Check your DataCollator and your assumption that output lengths are equal."
                    )
                loss_traj_L1 = nn.SmoothL1Loss()
                loss_traj = loss_traj_L1(
                    traj_student_hidden_states,
                    valid_teacher_hidden_states,
                )

                # === DEBUG: 检测 hidden state loss 异常 ===
                if torch.isnan(loss_traj) or torch.isinf(loss_traj):
                    _rank = dist.get_rank() if dist.is_initialized() else 0
                    print(f"[Rank {_rank}] DEBUG hs_loss={loss_traj.item()}, "
                          f"student_hs_nan={torch.isnan(traj_student_hidden_states).any().item()}, "
                          f"teacher_hs_nan={torch.isnan(valid_teacher_hidden_states).any().item()}, "
                          f"student_hs_range=[{traj_student_hidden_states.min().item():.4f}, {traj_student_hidden_states.max().item():.4f}], "
                          f"teacher_hs_range=[{valid_teacher_hidden_states.min().item():.4f}, {valid_teacher_hidden_states.max().item():.4f}]")

                if loss is None:
                    loss = self.hs_loss_weight * loss_traj
                else:
                    loss += self.hs_loss_weight * loss_traj
            
        if self.encoder_kd:
            if self.encoder_teacher is None:
                raise ValueError("Teacher model is not initialized for encoder KD.")
            loss_kd = nn.MSELoss(reduction='none')
            
            with torch.no_grad():
                kd_embeds = self.encoder_teacher(pixel_values, grid_thw=image_grid_thw)
                kd_embeds = kd_embeds.to(image_embeds.device, image_embeds.dtype)

                if isinstance(kd_embeds, tuple) and kd_embeds is not None:
                    kd_embeds = kd_embeds[0]
            
            B, N, C = inputs_embeds.shape
            anchor_embeds = inputs_embeds.clone()
            anchor_embeds[image_mask] = kd_embeds.reshape(-1)

            query = self.anchor_former.query.expand(B, -1, -1).to(kd_embeds.dtype)
            anchor_embeds = torch.cat([anchor_embeds, query], dim=1)

            anchor_position_ids = torch.arange(N+1).to(inputs_embeds.device, inputs_embeds.dtype)
            anchor_cache_position = anchor_position_ids
            anchor_position_ids = anchor_position_ids.view(1, -1).expand(B, -1)
            anchor_position_ids = anchor_position_ids.unsqueeze(0).expand(3, -1, -1)
            anchor_position_embeddings = self.anchor_former.rotary_emb(anchor_embeds, anchor_position_ids)
            anchor_attention_mask = torch.ones(anchor_embeds.shape[:2]).to(inputs_embeds.device, inputs_embeds.dtype)

            false_pad = torch.zeros((B, 1), dtype=image_mask.dtype, device=image_mask.device)
            anchor_image_mask = torch.cat([image_mask[:, :, 0], false_pad], dim=1)
            weight = self.anchor_former(
                    anchor_embeds,
                    attention_mask=anchor_attention_mask,
                    position_ids=anchor_position_ids,
                    past_key_values=past_key_values,
                    output_attentions=False,
                    use_cache=use_cache,
                    cache_position=anchor_cache_position,
                    position_embeddings=anchor_position_embeddings,
                    image_mask=anchor_image_mask
            )

            loss_encoder = loss_kd(kd_embeds, image_embeds)
            loss_encoder  = (loss_encoder.mean(dim=1)*weight).sum()/(weight.sum() + 1e-8)

            # === DEBUG: 检测 encoder KD loss 异常 ===
            if torch.isnan(loss_encoder) or torch.isinf(loss_encoder):
                _rank = dist.get_rank() if dist.is_initialized() else 0
                print(f"[Rank {_rank}] DEBUG encoder_loss={loss_encoder.item()}, "
                      f"weight_sum={weight.sum().item()}, weight_min={weight.min().item():.6f}, weight_max={weight.max().item():.6f}, "
                      f"kd_embeds_nan={torch.isnan(kd_embeds).any().item()}, "
                      f"image_embeds_nan={torch.isnan(image_embeds).any().item()}, "
                      f"kd_embeds_range=[{kd_embeds.min().item():.4f}, {kd_embeds.max().item():.4f}], "
                      f"image_embeds_range=[{image_embeds.min().item():.4f}, {image_embeds.max().item():.4f}]")

            if loss is None:
                loss = self.encoder_loss_weight * loss_encoder
            else:
                loss += self.encoder_loss_weight * loss_encoder

        # NaN loss 保护：用零替代，跳过本 step 的梯度更新，防止模型参数损坏
        if loss is not None and (torch.isnan(loss) or torch.isinf(loss)):
            _rank = dist.get_rank() if dist.is_initialized() else 0
            _sample_ids = kwargs.get("id", ["unknown"])
            print(f"[Rank {_rank}] WARNING: loss is NaN/Inf, replacing with 0 to skip this step. "
                  f"sample_ids={_sample_ids}")
            #loss = torch.zeros_like(loss, requires_grad=True)
            loss = 0.0 * self.lm_head.weight.sum()

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return Qwen2_5_VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,

        )
