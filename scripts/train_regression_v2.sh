#!/bin/bash
# 回归方案 V2 训练脚本 - 对齐生成方案参数

set -e

export PYTHONWARNINGS="ignore::DeprecationWarning"
export WANDB_PROJECT="EvoDriveVLA"
export ALBUMENTATIONS_DISABLE_VERSION_CHECK=1
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$HOME/.triton}"
mkdir -p "$TRITON_CACHE_DIR/autotune"

export OMP_NUM_THREADS=1
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export MASTER_ADDR=localhost
MASTER_PORT=$((10000 + RANDOM % 50000))
export MASTER_PORT=$MASTER_PORT

DEFAULT_NUM_NODES=1
DEFAULT_NUM_GPUS=8
DEFAULT_MIN_IMAGE_SIZE="100352"
DEFAULT_MAX_IMAGE_SIZE="200704"
DEFAULT_MODEL_SIZE="3B"
DEFAULT_MAX_TOKEN=4096

NUM_EPOCH=5

TUNE_MM_LLM=True
TUNE_MM_MLP=True
TUNE_MM_VISION=True

NUM_NODES=${1:-$DEFAULT_NUM_NODES}
NUM_GPUS=${2:-$DEFAULT_NUM_GPUS}

TIMESTAMP=$(date "+%Y-%m-%d_%H-%M-%S")
CHECKPOINT="${TIMESTAMP}_${DEFAULT_MODEL_SIZE}_${NUM_EPOCH}epoch_regression_v2"

OUTPUT_DIR="./result/evodrivevla/nuscenes/${CHECKPOINT}"
train_data="./data/nuscenes/Drive_KD_train_his_ego.json"
img_dir="./data/nuscenes"
model="Qwen/Qwen2.5-VL-${DEFAULT_MODEL_SIZE}-Instruct"

echo "=========================================="
echo "回归方案 V2 训练"
echo "GPU数量: ${NUM_GPUS}, 节点数: ${NUM_NODES}"
echo "=========================================="

deepspeed --master_port $MASTER_PORT \
  --num_gpus ${NUM_GPUS} \
  --num_nodes ${NUM_NODES} \
  --module qwenvl.train.train_qwen_regression_v2 \
  --deepspeed ./configs/deepspeed_regression.json \
  --model_name_or_path ${model} \
  --output_dir ${OUTPUT_DIR} \
  --img_dir $img_dir \
  --num_train_epochs ${NUM_EPOCH} \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --eval_strategy no \
  --save_strategy "epoch" \
  --learning_rate 2e-5 \
  --weight_decay 0. \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --dataloader_num_workers 8 \
  --save_total_limit 2 \
  --dataset_use $train_data \
  --tune_mm_vision $TUNE_MM_VISION \
  --tune_mm_mlp $TUNE_MM_MLP \
  --tune_mm_llm $TUNE_MM_LLM \
  --max_pixels ${DEFAULT_MAX_IMAGE_SIZE} \
  --min_pixels ${DEFAULT_MIN_IMAGE_SIZE} \
  --model_max_length ${DEFAULT_MAX_TOKEN} \
  --report_to tensorboard \
  --logging_steps 10 \
  --run_name ${CHECKPOINT} \
  --save_safetensors False \
  --gradient_checkpointing True \
  --bf16

echo ""
echo "训练完成! 模型保存在: ${OUTPUT_DIR}"
