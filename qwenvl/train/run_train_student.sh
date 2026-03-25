export PYTHONWARNINGS="ignore::DeprecationWarning"
export ALBUMENTATIONS_DISABLE_VERSION_CHECK=1
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$HOME/.triton}"
mkdir -p "$TRITON_CACHE_DIR/autotune"

# 一些常见 CUDA/NCCL 环境变量（无 IB 时关闭 RDMA，避免 NCCL 超时）
export OMP_NUM_THREADS=1
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#export CUDA_VISIBLE_DEVICES=0

export MASTER_ADDR=localhost
MASTER_PORT=$((10000 + RANDOM % 50000))
export MASTER_PORT=$MASTER_PORT

DEFAULT_NUM_NODES=1
DEFAULT_NUM_GPUS=8
DEFAULT_MIN_IMAGE_SIZE="100352"
#DEFAULT_MIN_IMAGE_SIZE="50176"
DEFAULT_MAX_IMAGE_SIZE="200704"
#DEFAULT_MAX_IMAGE_SIZE="100352"
DEFAULT_MODEL_SIZE="3B"
DEFAULT_MAX_TOKEN=4096

#=============================================
NUM_EPOCH=5

TRAIN_TEACHER=False

ENCODER_KD=True
ENCODER_WEIGHT=0.05

LLM_KD=True

LOGITS=True
LOGITS_WEIGHT=0.1
LOGITS_TEMP=5

HS=True
HS_WEIGHT=0.2
#=============================================

TUNE_MM_LLM=True
TUNE_MM_MLP=True
TUNE_MM_VISION=True

NUM_NODES=${1:-$DEFAULT_NUM_NODES}
NUM_GPUS=${2:-$DEFAULT_NUM_GPUS}

TIMESTAMP=$(date "+%Y-%m-%d_%H-%M-%S")
CHECKPOINT="${TIMESTAMP}_${DEFAULT_MODEL_SIZE}_${NUM_EPOCH}epoch"
PROMPT=""
if [ "$LLM_KD" = "True" ]; then
  PROMPT="${PROMPT}_llm_kd"
fi
if [ "$LOGITS" = "True" ]; then
  CHECKPOINT="${CHECKPOINT}_LOGITS${LOGITS_WEIGHT}_TEMP${LOGITS_TEMP}"
fi
if [ "$HS" = "True" ]; then
  CHECKPOINT="${CHECKPOINT}_HS${HS_WEIGHT}"
fi
if [ "$ENCODER_KD" = "True" ]; then
  CHECKPOINT="${CHECKPOINT}_ENCODER_KD${ENCODER_WEIGHT}"
fi

OUTPUT_DIR="./result/evodrivevla/nuscenes/${CHECKPOINT}"
train_data="./data/nuscenes/Drive_KD_train_his_ego${PROMPT}.json"
test_data="./data/nuscenes/Drive_KD_val_his_ego${PROMPT}.json"
img_dir="./data"
model="Qwen/Qwen2.5-VL-${DEFAULT_MODEL_SIZE}-Instruct"
teacher_model="./result/drivevla-kd/checkpoints/teacher_model_2026-03-24_09-53-02_3B_5epoch/checkpoint-5848"

if [ "$TRAIN_TEACHER" = "True" ]; then
  train_data="./data/nuscenes/Drive_KD_train_his_ego_future.json"
  test_data="./data/nuscenes/Drive_KD_val_his_ego_future.json"
  OUTPUT_DIR="./result/drivevla-kd/checkpoints/teacher_model_${TIMESTAMP}_${DEFAULT_MODEL_SIZE}_${NUM_EPOCH}epoch"
fi

deepspeed --master_port $MASTER_PORT \
  --num_gpus ${NUM_GPUS} \
  --num_nodes ${NUM_NODES} \
  --module qwenvl.train.train_qwen_kd \
  --deepspeed ./train/zero2.json \
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
  --data_packing False \
  --dataset_use $train_data \
  --tune_mm_vision $TUNE_MM_VISION \
  --tune_mm_mlp $TUNE_MM_MLP \
  --tune_mm_llm $TUNE_MM_LLM \
  --max_pixels ${DEFAULT_MAX_IMAGE_SIZE} \
  --min_pixels ${DEFAULT_MIN_IMAGE_SIZE} \
  --model_max_length ${DEFAULT_MAX_TOKEN} \
  --report_to tensorboard \
  --logging_steps 10 \
  --logging_dir ${OUTPUT_DIR}/logs \
  --save_safetensors False \
  --train_teacher ${TRAIN_TEACHER} \
  --teacher_model_name_or_path ${teacher_model} \
  --encoder_kd ${ENCODER_KD} \
  --encoder_loss_weight ${ENCODER_WEIGHT} \
  --llm_kd ${LLM_KD} \
  --kd_data ${LLM_KD} \
  --logits_loss ${LOGITS} \
  --logits_loss_weight ${LOGITS_WEIGHT} \
  --logits_loss_temperature ${LOGITS_TEMP} \
  --hs_loss ${HS} \
  --hs_loss_weight ${HS_WEIGHT} \
  --gradient_checkpointing True \

if [ -d "${OUTPUT_DIR}" ]; then

  python -m inference_scripts.infer_multi \
      --model_name_or_path ${OUTPUT_DIR} \
      --img_dir $img_dir \
      --dataset_use $test_data \
      --eval_save_path ${OUTPUT_DIR}/result.json \
      --max_pixels ${DEFAULT_MAX_IMAGE_SIZE} \
      --min_pixels ${DEFAULT_MIN_IMAGE_SIZE} \
      --model_max_length ${DEFAULT_MAX_TOKEN} \
      --inference True \
      --random False \

  python ./eval_planning/evaluation/evaluation.py \
      --result_file ${OUTPUT_DIR}/result.json \
      --save_file ${OUTPUT_DIR}/eval_result.json \

fi
