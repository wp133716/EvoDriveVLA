#!/bin/bash
# 回归方案 V2 训练脚本 - 对齐生成方案参数

set -e

NUM_GPUS=${1:-4}

#=========================================
# 模型微调配置 (对齐生成方案)
TUNE_MM_LLM=True
TUNE_MM_MLP=True
TUNE_MM_VISION=True

# 图像尺寸配置
MIN_IMAGE_SIZE="100352"
MAX_IMAGE_SIZE="200704"
MAX_TOKEN=4096

# 学习率和优化配置
LEARNING_RATE=2e-5
WEIGHT_DECAY=0.0
WARMUP_RATIO=0.03
NUM_EPOCH=5

# DeepSpeed配置
USE_DEEPSPEED=True
DEEPSPEED_CONFIG="./configs/deepspeed_regression.json"
#=========================================

echo "=========================================="
echo "回归方案 V2 训练 (参数对齐生成方案)"
echo "复用数据流: data_qwen.py"
echo "复用训练流: train_qwen_regression_v2.py"
echo "GPU数量: ${NUM_GPUS}"
echo "=========================================="
echo ""
echo "微调配置:"
echo "  TUNE_MM_LLM: ${TUNE_MM_LLM}"
echo "  TUNE_MM_MLP: ${TUNE_MM_MLP}"
echo "  TUNE_MM_VISION: ${TUNE_MM_VISION}"
echo "  USE_DEEPSPEED: ${USE_DEEPSPEED}"
echo ""

DATA_PATH="./data/nuscenes/Drive_KD_train_his_ego.json"
MODEL_NAME="Qwen/Qwen2.5-VL-3B-Instruct"

TIMESTAMP=$(date "+%Y-%m-%d_%H-%M-%S")
CHECKPOINT="${TIMESTAMP}_3B_${NUM_EPOCH}epoch_regression_v2"
OUTPUT_DIR="./result/evodrivevla/nuscenes/${CHECKPOINT}"

# 创建输出目录
mkdir -p ${OUTPUT_DIR}

# 构建启动命令
if [ "$USE_DEEPSPEED" = "True" ]; then
    echo "使用 DeepSpeed ZeRO-2 训练"
    LAUNCHER="deepspeed --master_port $((10000 + RANDOM % 50000)) --num_gpus ${NUM_GPUS}"
    DEEPSPEED_ARGS="--deepspeed ${DEEPSPEED_CONFIG}"
else
    echo "使用 torchrun 训练"
    LAUNCHER="torchrun --nnodes=1 --nproc_per_node=${NUM_GPUS}"
    DEEPSPEED_ARGS=""
fi

${LAUNCHER} \
    --nnodes=1 --nproc_per_node=${NUM_GPUS} \
    qwenvl/train/train_qwen_regression_v2.py \
    --model_name_or_path ${MODEL_NAME} \
    --data_path ${DATA_PATH} \
    --dataset_use ${DATA_PATH} \
    --img_dir ./data/nuscenes \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs ${NUM_EPOCH} \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate ${LEARNING_RATE} \
    --weight_decay ${WEIGHT_DECAY} \
    --warmup_ratio ${WARMUP_RATIO} \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --eval_strategy no \
    --save_strategy "epoch" \
    --dataloader_num_workers 8 \
    --save_total_limit 2 \
    --dataset_use ${DATA_PATH} \
    --tune_mm_vision ${TUNE_MM_VISION} \
    --tune_mm_mlp ${TUNE_MM_MLP} \
    --tune_mm_llm ${TUNE_MM_LLM} \
    --max_pixels ${MAX_IMAGE_SIZE} \
    --min_pixels ${MIN_IMAGE_SIZE} \
    --model_max_length ${MAX_TOKEN} \
    --report_to tensorboard \
    --logging_steps 10 \
    --run_name ${CHECKPOINT} \
    --save_safetensors False \
    --gradient_checkpointing True \
    --bf16 \
    ${DEEPSPEED_ARGS}

echo ""
echo "训练完成! 模型保存在: ${OUTPUT_DIR}"
echo ""
echo "如需推理，请运行:"
echo "  python deploy_regression_v2.py --model_path ${OUTPUT_DIR}"
