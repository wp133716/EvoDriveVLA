"""
Train Qwen2.5-VL with LaST-R1-style discrete action tokens.

This is an additive training entrypoint. Existing regression/KD training files
are intentionally left untouched.
"""

import json
import logging
import os
import pathlib
import sys
from dataclasses import dataclass, field
from typing import Optional

import torch
import transformers
from transformers import AutoProcessor, Trainer

project_root = pathlib.Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from model.language_models.modeling_qwen2_5_vl_action_token import (  # noqa: E402
    Qwen2_5_VLForActionToken,
)
from qwenvl.data.data_qwen import make_supervised_data_module  # noqa: E402
from qwenvl.train.argument import DataArguments, ModelArguments, TrainingArguments  # noqa: E402


@dataclass
class ActionTokenModelArguments(ModelArguments):
    num_action_steps: int = field(
        default=6,
        metadata={"help": "Number of future action/waypoint steps to predict."},
    )
    action_dim: int = field(
        default=3,
        metadata={"help": "Number of dimensions per action step."},
    )
    action_bins: int = field(
        default=256,
        metadata={"help": "Number of discrete bins per action dimension."},
    )


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)


def add_action_tokens(tokenizer, action_bins: int) -> int:
    action_tokens = Qwen2_5_VLForActionToken.action_special_tokens(action_bins)
    existing = set(tokenizer.additional_special_tokens)
    to_add = [tok for tok in action_tokens if tok not in existing]
    if to_add:
        tokenizer.add_special_tokens({"additional_special_tokens": to_add})
    return len(to_add)


def set_model_trainability(model_args: ActionTokenModelArguments, model: Qwen2_5_VLForActionToken):
    if model_args.tune_mm_vision:
        for _, p in model.visual.named_parameters():
            p.requires_grad = True
    else:
        for _, p in model.visual.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_mlp:
        for _, p in model.visual.merger.named_parameters():
            p.requires_grad = True
    else:
        for _, p in model.visual.merger.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_llm:
        for _, p in model.model.named_parameters():
            p.requires_grad = True
    else:
        for _, p in model.model.named_parameters():
            p.requires_grad = False

    # The action-token policy is trained through lm_head over <action_i>.
    for p in model.lm_head.parameters():
        p.requires_grad = True


def load_waypoint_stats(path: Optional[str], action_dim: int):
    if not path:
        return [0.0] * action_dim, [1.0] * action_dim
    if not os.path.exists(path):
        raise FileNotFoundError(f"waypoint_stats_path does not exist: {path}")
    with open(path, "r") as f:
        stats = json.load(f)
    return stats["global"]["mean"], stats["global"]["std"]


def train():
    parser = transformers.HfArgumentParser(
        (ActionTokenModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    os.makedirs(training_args.output_dir, exist_ok=True)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    num_added = add_action_tokens(tokenizer, model_args.action_bins)
    print(f"Action tokens added: {num_added}; tokenizer size: {len(tokenizer)}")

    waypoint_mean, waypoint_std = load_waypoint_stats(
        model_args.waypoint_stats_path,
        model_args.action_dim,
    )
    print(f"Waypoint mean: {waypoint_mean}")
    print(f"Waypoint std:  {waypoint_std}")

    model = Qwen2_5_VLForActionToken.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        attn_implementation=training_args.attn_implementation,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map={"": training_args.local_rank} if training_args.local_rank != -1 else "auto",
        num_action_steps=model_args.num_action_steps,
        action_dim=model_args.action_dim,
        action_bins=model_args.action_bins,
        waypoint_mean=waypoint_mean,
        waypoint_std=waypoint_std,
    )
    model.resize_token_embeddings(len(tokenizer))
    model.set_action_tokenizer(tokenizer)
    model.config.use_cache = False

    for _, param in model.named_parameters():
        if param.dtype == torch.float32:
            param.data = param.data.to(torch.bfloat16)

    data_args.image_processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
        use_fast=True,
    ).image_processor
    data_args.model_type = "qwen2.5vl"

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, inputs, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    set_model_trainability(model_args, model)

    if training_args.local_rank in [-1, 0]:
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(
            f"trainable params: {trainable_params:,} || all params: {total_params:,} "
            f"|| trainable%: {100 * trainable_params / total_params:.4f}"
        )
        print(
            f"action_token_len={model.action_token_len}, action_bins={model.action_bins}, "
            f"action_0_id={model.action_0_id}"
        )

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        **data_module,
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        logging.info("checkpoint found, resume training")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()
    data_args.image_processor.save_pretrained(training_args.output_dir)

    model.config.use_cache = True
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    trainer.model.config.save_pretrained(training_args.output_dir)
    print(f"Training complete. Model saved to: {training_args.output_dir}")


if __name__ == "__main__":
    train()

