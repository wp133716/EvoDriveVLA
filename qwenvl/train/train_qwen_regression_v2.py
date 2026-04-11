"""
Qwen2.5-VL 回归训练脚本 V2
==========================
完全复用现有训练结构，只修改：
1. 使用回归模型 Qwen2_5_VLForRegressionV2
2. 复用 data_qwen.py (不需要修改)
3. 复用所有训练参数和逻辑

用法:
    torchrun --nproc_per_node=4 \
        qwenvl/train/train_qwen_regression_v2.py \
        --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
        --data_path data/nuscenes/Drive_KD_train_his_ego.json \
        --output_dir ./output/regression_v2
"""

import os
import sys
import logging
import pathlib
import torch
import transformers

project_root = pathlib.Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# 复用现有导入
from model.language_models.modeling_qwen2_5_vl_regression_v2 import (
    Qwen2_5_VLForRegressionV2,
)
from qwenvl.data.data_qwen import make_supervised_data_module
from qwenvl.train.argument import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
)
from transformers import AutoTokenizer, AutoProcessor
from transformers import Trainer

# 复用现有函数
def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)


def set_model(model_args, model):
    """设置模型可训练参数"""
    # 视觉编码器
    if model_args.tune_mm_vision:
        for n, p in model.visual.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.named_parameters():
            p.requires_grad = False

    # MLP
    if model_args.tune_mm_mlp:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = False

    # LLM
    if model_args.tune_mm_llm:
        for n, p in model.model.named_parameters():
            p.requires_grad = True
        for p in model.lm_head.parameters():
            p.requires_grad = True
    else:
        for n, p in model.model.named_parameters():
            p.requires_grad = False
        model.lm_head.requires_grad = False

    # 回归头必须训练
    for p in model.trajectory_head.parameters():
        p.requires_grad = True


def train():
    """训练函数 - 复用现有结构"""
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    os.makedirs(training_args.output_dir, exist_ok=True)

    # 加载回归模型 (唯一改动点1)
    print(f"Loading regression model from {model_args.model_name_or_path}...")
    model = Qwen2_5_VLForRegressionV2.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        attn_implementation=training_args.attn_implementation,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map={"": training_args.local_rank} if training_args.local_rank != -1 else "auto",
        num_waypoints=6,
        waypoint_dim=3,
    )

    # 转为 bf16
    for name, param in model.named_parameters():
        if param.dtype == torch.float32:
            param.data = param.data.to(torch.bfloat16)

    # 加载 processor
    data_args.image_processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
        use_fast=True
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

    # 加载 tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    model.tokenizer = tokenizer

    # 设置模型
    set_model(model_args, model)

    # 打印可训练参数
    if training_args.local_rank in [-1, 0]:
        model.model.print_trainable_parameters()

    # 复用现有数据加载 (唯一改动点2：不需要KD相关)
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    # 创建 Trainer (复用现有)
    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        **data_module
    )

    # 训练
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

    print(f"Training complete! Model saved to {training_args.output_dir}")


if __name__ == "__main__":
    train()
