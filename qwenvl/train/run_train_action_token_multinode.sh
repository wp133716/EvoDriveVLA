#!/bin/bash
# Multi-node action-token training for EvoDriveVLA.
# Run ONLY on the master node (rank 0); deepspeed will SSH into the rest.
#
# Usage:
#   bash run_train_action_token_multinode.sh [NUM_NODES] [GPUS_PER_NODE]
#   e.g. bash run_train_action_token_multinode.sh 4 8   # 4 servers x 8 GPUs = 32
#
# Prerequisites:
#   1. Passwordless SSH from master to every worker listed in HOSTFILE.
#   2. Identical code / data / conda env paths on all nodes (shared FS or synced).
#   3. MASTER_ADDR reachable from all nodes; MASTER_PORT open between them.

set -e

export PYTHONWARNINGS="ignore::DeprecationWarning"
export WANDB_PROJECT="EvoDriveVLA"
export ALBUMENTATIONS_DISABLE_VERSION_CHECK=1
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$HOME/.triton}"
mkdir -p "$TRITON_CACHE_DIR/autotune"

# NCCL config for the MASTER node's own process. Worker nodes get these from
# the .deepspeed_env file (deepspeed propagates it over SSH) -- keep them in sync.
export OMP_NUM_THREADS=1
export NCCL_DEBUG=INFO           # confirm "NET/IB" in logs, then drop back to WARN
export NCCL_IB_DISABLE=0         # nodes have InfiniBand (HCA mlx5_0)
export NCCL_IB_HCA=mlx5_0
export NCCL_P2P_DISABLE=0
export NCCL_SOCKET_IFNAME=^docker0,lo,virbr0   # bootstrap NIC; exclude virtual ifaces

# -----------------------------------------------------------------------------
# Multi-node config -- EDIT THESE
# -----------------------------------------------------------------------------
# hostfile lists every node and its GPU count, e.g.
#   192.168.1.10 slots=8
#   192.168.1.11 slots=8
# The FIRST entry is the master node.
HOSTFILE="${HOSTFILE:-./train/hostfile}"
# Master node IP (must match the first hostfile entry; NOT localhost).
export MASTER_ADDR="${MASTER_ADDR:-192.168.81.90}"
MASTER_PORT=$((10000 + RANDOM % 50000))
export MASTER_PORT=$MASTER_PORT

DEFAULT_NUM_NODES=4
DEFAULT_NUM_GPUS=8               # GPUs PER node
DEFAULT_MIN_IMAGE_SIZE="100352"
DEFAULT_MAX_IMAGE_SIZE="200704"
DEFAULT_MODEL_SIZE="3B"
# 18 images * ~128 tokens/img (max_pixels=100352) ~= 2304 vision tokens; plus
# system + user prompt + 6 action tokens fits comfortably in 8192.
DEFAULT_MAX_TOKEN=8192

NUM_EPOCH=${NUM_EPOCH:-5}
# Drone action vector: (pixel_dx, pixel_dy, gimbal_pitch, gimbal_yaw, zoom, grid_idx)
NUM_ACTION_STEPS=${NUM_ACTION_STEPS:-1}
ACTION_DIM=${ACTION_DIM:-6}
ACTION_BINS=${ACTION_BINS:-256}

# Per-device batch and grad-accum. With 32 GPUs the global batch grows 4x vs a
# single 8-GPU node; bump LR or grad-accum to keep an equivalent effective batch.
PER_DEVICE_BATCH=${PER_DEVICE_BATCH:-1}
GRAD_ACCUM=${GRAD_ACCUM:-1}
LEARNING_RATE=${LEARNING_RATE:-2e-5}

TUNE_MM_LLM=${TUNE_MM_LLM:-True}
TUNE_MM_MLP=${TUNE_MM_MLP:-True}
TUNE_MM_VISION=${TUNE_MM_VISION:-True}

NUM_NODES=${1:-$DEFAULT_NUM_NODES}
NUM_GPUS=${2:-$DEFAULT_NUM_GPUS}

TIMESTAMP=$(date "+%Y-%m-%d_%H-%M-%S")
CHECKPOINT="${TIMESTAMP}_${DEFAULT_MODEL_SIZE}_${NUM_EPOCH}epoch_action_token_${NUM_ACTION_STEPS}x${ACTION_DIM}_${ACTION_BINS}bins"

OUTPUT_DIR="./result/evodrivevla/drone_search/${CHECKPOINT}"
DRONE_ROOT="/home/admin/wp_workspace/gimbal_data"
train_data="${DRONE_ROOT}/drone_vla_train.json"
val_data="${DRONE_ROOT}/drone_vla_val.json"
img_dir="${DRONE_ROOT}"
model="Qwen/Qwen2.5-VL-${DEFAULT_MODEL_SIZE}-Instruct"
waypoint_stats="${DRONE_ROOT}/waypoint_stats.json"
GRID_COLS=${GRID_COLS:-7}

if [ ! -f "${HOSTFILE}" ]; then
  echo "ERROR: hostfile not found at ${HOSTFILE}" >&2
  echo "Create it with one line per node, e.g.:" >&2
  echo "  ${MASTER_ADDR} slots=${NUM_GPUS}" >&2
  exit 1
fi

echo "=========================================="
echo "Multi-node action-token training"
echo "节点数: ${NUM_NODES}, 每节点GPU: ${NUM_GPUS}, 总GPU: $((NUM_NODES * NUM_GPUS))"
echo "master: ${MASTER_ADDR}:${MASTER_PORT}, hostfile: ${HOSTFILE}"
echo "action shape: ${NUM_ACTION_STEPS} x ${ACTION_DIM}, bins=${ACTION_BINS}"
echo "=========================================="

deepspeed --hostfile=${HOSTFILE} \
  --master_addr ${MASTER_ADDR} \
  --master_port ${MASTER_PORT} \
  --num_gpus ${NUM_GPUS} \
  --num_nodes ${NUM_NODES} \
  --module qwenvl.train.train_qwen_action_token \
  --deepspeed ./train/zero2.json \
  --model_name_or_path ${model} \
  --output_dir ${OUTPUT_DIR} \
  --img_dir $img_dir \
  --num_train_epochs ${NUM_EPOCH} \
  --per_device_train_batch_size ${PER_DEVICE_BATCH} \
  --gradient_accumulation_steps ${GRAD_ACCUM} \
  --eval_strategy no \
  --save_strategy "epoch" \
  --learning_rate ${LEARNING_RATE} \
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
  --attn_implementation sdpa \
  --report_to wandb \
  --logging_steps 10 \
  --run_name ${CHECKPOINT} \
  --save_safetensors False \
  --gradient_checkpointing True \
  --bf16 \
  --waypoint_stats_path ${waypoint_stats} \
  --num_action_steps ${NUM_ACTION_STEPS} \
  --action_dim ${ACTION_DIM} \
  --action_bins ${ACTION_BINS}

echo ""
echo "训练完成! 模型保存在: ${OUTPUT_DIR}"

# -----------------------------------------------------------------------------
# Post-training validation (single-process, runs on the master node only).
# -----------------------------------------------------------------------------
if [ -d "${OUTPUT_DIR}" ] && [ -f "${val_data}" ]; then
  echo "=========================================="
  echo "Running validation on ${val_data}"
  echo "=========================================="

  python -m inference_scripts.infer_drone_action_token \
      --model_name_or_path ${OUTPUT_DIR} \
      --img_dir ${img_dir} \
      --dataset_use ${val_data} \
      --eval_save_path ${OUTPUT_DIR}/val_predictions.json \
      --max_pixels ${DEFAULT_MAX_IMAGE_SIZE} \
      --min_pixels ${DEFAULT_MIN_IMAGE_SIZE} \
      --model_max_length ${DEFAULT_MAX_TOKEN} \
      --num_action_steps ${NUM_ACTION_STEPS} \
      --action_dim ${ACTION_DIM} \
      --action_bins ${ACTION_BINS} \
      --inference False \
      --random False

  python ./eval_planning/evaluation/eval_drone_action.py \
      --result_file ${OUTPUT_DIR}/val_predictions.json \
      --gt_file ${val_data} \
      --save_file ${OUTPUT_DIR}/val_metrics.json \
      --grid_cols ${GRID_COLS}

  echo "验证完成! 指标: ${OUTPUT_DIR}/val_metrics.json"
fi
