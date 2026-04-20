#!/bin/bash
# Learnable Query 训练脚本
# 与 run.sh 的唯一区别：--module qwenvl.train.train_qwen_learnableq

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
export CUDA_VISIBLE_DEVICES=0

export MASTER_ADDR=localhost
MASTER_PORT=$((10000 + RANDOM % 50000))
export MASTER_PORT=$MASTER_PORT

DEFAULT_NUM_NODES=1
DEFAULT_NUM_GPUS=1
DEFAULT_MIN_IMAGE_SIZE="100352"
DEFAULT_MAX_IMAGE_SIZE="200704"
DEFAULT_MODEL_SIZE="3B"
DEFAULT_MAX_TOKEN=6144

RESUME_FROM_CHECKPOINT=""

#=============================================
NUM_EPOCH=5
TUNE_MM_LLM=True
TUNE_MM_MLP=True
TUNE_MM_VISION=False

# 损失模式
# False → regression head + Smooth L1（默认，6 query tokens）
# True  → lm_head + CE（18 query tokens，无需 waypoint_stats）
USE_LM_HEAD=False

# 帧内双向注意力（Image Chunk Mask）
# True  → 帧内双向，需配合 attn_implementation=sdpa
# False → 标准因果掩码（默认，兼容 flash_attention_2）
IMAGE_CHUNK_MASK=False
#=============================================

NUM_NODES=${1:-$DEFAULT_NUM_NODES}
NUM_GPUS=${2:-$DEFAULT_NUM_GPUS}

TIMESTAMP=$(date "+%Y-%m-%d_%H-%M-%S")
LOSS_SUFFIX=$([ "$USE_LM_HEAD" = "True" ] && echo "ce" || echo "l1")
CHECKPOINT="${TIMESTAMP}_${DEFAULT_MODEL_SIZE}_${NUM_EPOCH}epoch_learnableq_${LOSS_SUFFIX}"

OUTPUT_DIR="./result/evodrivevla/nuscenes/${CHECKPOINT}"
train_data="./data/nuscenes/Drive_KD_train_his_ego_history5f.json"
test_data="./data/nuscenes/Drive_KD_val_his_ego_history5f.json"
img_dir="./data/"
model="Qwen/Qwen2.5-VL-${DEFAULT_MODEL_SIZE}-Instruct"
waypoint_stats="./data/nuscenes/waypoint_stats.json"

deepspeed --master_port $MASTER_PORT \
  --num_gpus ${NUM_GPUS} \
  --num_nodes ${NUM_NODES} \
  --module qwenvl.train.train_qwen_learnableq \
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
  --save_total_limit 1 \
  --data_packing False \
  --dataset_use $train_data \
  --tune_mm_vision $TUNE_MM_VISION \
  --tune_mm_mlp $TUNE_MM_MLP \
  --tune_mm_llm $TUNE_MM_LLM \
  --max_pixels ${DEFAULT_MAX_IMAGE_SIZE} \
  --min_pixels ${DEFAULT_MIN_IMAGE_SIZE} \
  --model_max_length ${DEFAULT_MAX_TOKEN} \
  --report_to wandb \
  --logging_steps 10 \
  --run_name ${CHECKPOINT} \
  --save_safetensors False \
  --waypoint_stats_path ${waypoint_stats} \
  --gradient_checkpointing True \
  --use_lm_head $USE_LM_HEAD \
  $([ "$IMAGE_CHUNK_MASK" = "True" ] && echo "--attn_implementation sdpa" || echo "--attn_implementation flash_attention_2") \
  ${RESUME_FROM_CHECKPOINT:+--resume_from_checkpoint $RESUME_FROM_CHECKPOINT}

# if [ -d "${OUTPUT_DIR}" ]; then
#   python -m inference_scripts.infer_multi \
#       --model_name_or_path ${OUTPUT_DIR} \
#       --img_dir $img_dir \
#       --dataset_use $test_data \
#       --eval_save_path ${OUTPUT_DIR}/result.json \
#       --max_pixels ${DEFAULT_MAX_IMAGE_SIZE} \
#       --min_pixels ${DEFAULT_MIN_IMAGE_SIZE} \
#       --model_max_length ${DEFAULT_MAX_TOKEN} \
#       --inference True \
#       --random False

#   python ./eval_planning/evaluation/evaluation.py \
#       --result_file ${OUTPUT_DIR}/result.json \
#       --save_file ${OUTPUT_DIR}/eval_result.json
# fi
