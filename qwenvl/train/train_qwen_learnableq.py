"""
Qwen2.5-VL Learnable Query 训练脚本
=====================================
与 train_qwen_regression_v2.py 的唯一区别：
  1. 加载 Qwen2_5_VLForLearnableQ
  2. set_model() 额外确保 traj_queries 可训练

用法:
    torchrun --nproc_per_node=4 \\
        qwenvl/train/train_qwen_learnableq.py \\
        --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \\
        --data_path data/nuscenes/Drive_KD_train_his_ego.json \\
        --output_dir ./output/learnableq
"""

import os
import sys
import logging
import pathlib
import torch
import transformers
import json

project_root = pathlib.Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from model.language_models.modeling_qwen2_5_vl_learnableq import (
    Qwen2_5_VLForLearnableQ,
)
from qwenvl.data.data_qwen import make_supervised_data_module
from qwenvl.train.argument import ModelArguments, DataArguments, TrainingArguments
from transformers import AutoTokenizer, AutoProcessor, Trainer


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {k: v.cpu() for k, v in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)


def set_model(model_args, model):
    """设置可训练参数。traj_queries 始终训练；trajectory_head 仅回归模式存在。"""
    # 视觉编码器
    for n, p in model.visual.named_parameters():
        p.requires_grad = model_args.tune_mm_vision
    # MLP merger
    for n, p in model.visual.merger.named_parameters():
        p.requires_grad = model_args.tune_mm_mlp
    # LLM backbone + lm_head
    for n, p in model.model.named_parameters():
        p.requires_grad = model_args.tune_mm_llm
    model.lm_head.requires_grad_(model_args.tune_mm_llm)

    # 始终训练：learnable query token
    model.traj_queries.requires_grad_(True)

    # 回归模式额外训练 trajectory_head；CE 模式无此参数
    if hasattr(model, 'trajectory_head'):
        for p in model.trajectory_head.parameters():
            p.requires_grad = True
    # CE 模式：lm_head 必须可训练（无论 tune_mm_llm 设置如何）
    if model.use_lm_head:
        model.lm_head.requires_grad_(True)


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    os.makedirs(training_args.output_dir, exist_ok=True)

    use_lm_head = getattr(training_args, 'use_lm_head', False)
    print(
        f"Trajectory config: num_waypoints={model_args.num_waypoints}, "
        f"waypoint_dim={model_args.waypoint_dim}"
    )
    print(f"Loss mode: {'lm_head + CE' if use_lm_head else 'regression head + Smooth L1'}")

    # 回归模式才需要归一化参数
    waypoint_mean = waypoint_std = None
    if not use_lm_head:
        if model_args.waypoint_stats_path and os.path.exists(model_args.waypoint_stats_path):
            with open(model_args.waypoint_stats_path) as f:
                stats = json.load(f)
            waypoint_mean = stats['global']['mean']
            waypoint_std = stats['global']['std']
            print(f"Waypoint mean: {waypoint_mean}")
            print(f"Waypoint std:  {waypoint_std}")

    # 加载模型
    print(f"Loading LearnableQ model from {model_args.model_name_or_path}...")
    model = Qwen2_5_VLForLearnableQ.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        attn_implementation=training_args.attn_implementation,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map={"": training_args.local_rank} if training_args.local_rank != -1 else "auto",
        num_waypoints=model_args.num_waypoints,
        waypoint_dim=model_args.waypoint_dim,
        waypoint_mean=waypoint_mean,
        waypoint_std=waypoint_std,
        use_lm_head=use_lm_head,
    )

    # 统一转 bf16
    for name, param in model.named_parameters():
        if param.dtype == torch.float32:
            param.data = param.data.to(torch.bfloat16)

    # 重新初始化新增参数（不在预训练权重中）
    # low_cpu_mem_usage=True 使用 init_empty_weights 上下文，导致 __init__ 中的
    # nn.init.normal_ 等初始化调用被跳过（参数留为未初始化/NaN 内存）。
    # 必须在 from_pretrained 之后、训练之前显式重新初始化这些参数。
    with torch.no_grad():
        torch.nn.init.normal_(model.traj_queries, std=0.02)
        print(f"Re-initialized traj_queries: "
              f"nan={torch.isnan(model.traj_queries).any().item()} "
              f"mean={model.traj_queries.float().mean().item():.4f} "
              f"std={model.traj_queries.float().std().item():.4f}")
        if hasattr(model, 'trajectory_head'):
            torch.nn.init.normal_(model.trajectory_head.weight, std=0.02)
            torch.nn.init.zeros_(model.trajectory_head.bias)
            print("Re-initialized trajectory_head")

    # image processor
    data_args.image_processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path, use_fast=True
    ).image_processor
    data_args.model_type = "qwen2.5vl"
    model.config.use_cache = False

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    model.tokenizer = tokenizer

    set_model(model_args, model)

    if training_args.local_rank in [-1, 0]:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"trainable: {trainable:,} / total: {total:,} ({100*trainable/total:.4f}%)")
        tq_params = model.traj_queries.numel()
        print(f"  traj_queries: {tq_params:,} params  (n_query_tokens={model.n_query_tokens})")
        if hasattr(model, 'trajectory_head'):
            head_params = sum(p.numel() for p in model.trajectory_head.parameters())
            print(f"  trajectory_head: {head_params:,} params")
        else:
            print(f"  trajectory_head: N/A (CE mode, using lm_head)")

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        **data_module,
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        logging.info("checkpoint found, resuming training")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()
    data_args.image_processor.save_pretrained(training_args.output_dir)
    model.config.use_cache = True
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    trainer.model.config.save_pretrained(training_args.output_dir)
    print(f"Training complete. Model saved to {training_args.output_dir}")


if __name__ == "__main__":
    train()
