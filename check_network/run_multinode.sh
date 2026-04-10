#!/bin/bash
# 多节点训练启动脚本
# ==================
# 使用前：
# 1. 修改 NODE1_IP 为 node1 的实际 IP
# 2. 确保两个节点都能通过 SSH 免密登录
# 3. 确保代码和数据在两个节点上的路径一致

set -e

# ========== 配置 ==========
NODE1_IP="10.0.0.1"  # <-- 修改为你的 node1 IP
MASTER_PORT=29500
NUM_NODES=2
NUM_GPUS_PER_NODE=8
WORLD_SIZE=$((NUM_NODES * NUM_GPUS_PER_NODE))

# 模型和数据路径 (两个节点必须一致)
MODEL_PATH="/path/to/your/model"  # <-- 修改为你的模型路径
DATA_PATH="/path/to/cached_nuscenes_info.pkl"  # <-- 修改为你的数据路径
OUTPUT_DIR="./output_multinode"

# DeepSpeed 配置
DS_CONFIG="ds_config_zero2.json"

# ========== 环境变量 (优化 NCCL) ==========
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=eth0  # 根据你的网络接口修改
export NCCL_TREE_THRESHOLD=0
export NCCL_ALGO=RING  # 或者 TREE

# RoCE 网络优化 (如果是 RoCE 而非 InfiniBand)
# export NCCL_IB_GID_INDEX=3
# export NCCL_IB_TC=106

# PyTorch 分布式
export PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS=8

# ========== 函数 ==========

# 获取当前节点的主机名
current_hostname=$(hostname)

# 测试通信
test_communication() {
    echo "========================================"
    echo "Step 1: Testing multi-node communication"
    echo "========================================"

    # 创建 hostfile
    cat > hostfile.txt << EOF
$(hostname -s) slots=8
EOF
    # 注意：实际多节点测试需要手动在两个节点上分别运行

    echo "Running communication test..."

    # 单节点测试
    python test_multinode_comm.py

    echo ""
    echo "Single-node test completed."
    echo "To test multi-node communication, run the following on both nodes:"
    echo ""
    echo "Node 1:"
    echo "  torchrun --nnodes=2 --nproc_per_node=8 --node_rank=0 \\"
    echo "           --master_addr=$NODE1_IP --master_port=$MASTER_PORT \\"
    echo "           test_multinode_comm.py"
    echo ""
    echo "Node 2:"
    echo "  torchrun --nnodes=2 --nproc_per_node=8 --node_rank=1 \\"
    echo "           --master_addr=$NODE1_IP --master_port=$MASTER_PORT \\"
    echo "           test_multinode_comm.py"
    echo ""
    read -p "Press enter to continue to training (after confirming communication works)..."
}

# 生成 DeepSpeed 配置
generate_ds_config() {
    cat > $DS_CONFIG << 'EOF'
{
  "bf16": {
    "enabled": true
  },
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {
      "device": "none",
      "pin_memory": true
    },
    "allgather_partitions": true,
    "allgather_bucket_size": 5e8,
    "overlap_comm": true,
    "reduce_scatter": true,
    "reduce_bucket_size": 5e8,
    "contiguous_gradients": true
  },
  "gradient_accumulation_steps": 1,
  "gradient_clipping": 1.0,
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "wall_clock_breakdown": false,
  "memory_breakdown": false
}
EOF
    echo "Generated $DS_CONFIG"
}

# 启动训练 (单节点测试)
run_single_node() {
    echo "========================================"
    echo "Step 2: Single-node training test"
    echo "========================================"

    mkdir -p $OUTPUT_DIR

    torchrun \
        --nnodes=1 \
        --nproc_per_node=$NUM_GPUS_PER_NODE \
        --master_addr=localhost \
        --master_port=$MASTER_PORT \
        qwenvl/train/train_qwen_kd.py \
        --model_name_or_path $MODEL_PATH \
        --data_path $DATA_PATH \
        --bf16 True \
        --output_dir $OUTPUT_DIR \
        --num_train_epochs 3 \
        --per_device_train_batch_size 2 \
        --per_device_eval_batch_size 2 \
        --gradient_accumulation_steps 4 \
        --evaluation_strategy "no" \
        --save_strategy "steps" \
        --save_steps 500 \
        --save_total_limit 3 \
        --learning_rate 2e-5 \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type "cosine" \
        --logging_steps 10 \
        --tf32 True \
        --model_max_length 2048 \
        --gradient_checkpointing True \
        --dataloader_num_workers 4 \
        --report_to wandb \
        --deepspeed $DS_CONFIG
}

# 启动训练 (多节点 - Node 1)
run_multinode_node1() {
    echo "========================================"
    echo "Step 3: Multi-node training (Node 1)"
    echo "========================================"

    mkdir -p $OUTPUT_DIR

    # 等待 node2 连接
    echo "Waiting for Node 2 to connect..."
    echo "Run the following command on Node 2 now:"
    echo ""
    echo "  cd $(pwd) && bash run_multinode.sh node2"
    echo ""

    torchrun \
        --nnodes=$NUM_NODES \
        --nproc_per_node=$NUM_GPUS_PER_NODE \
        --node_rank=0 \
        --master_addr=$NODE1_IP \
        --master_port=$MASTER_PORT \
        qwenvl/train/train_qwen_kd.py \
        --model_name_or_path $MODEL_PATH \
        --data_path $DATA_PATH \
        --bf16 True \
        --output_dir $OUTPUT_DIR \
        --num_train_epochs 10 \
        --per_device_train_batch_size 2 \
        --per_device_eval_batch_size 2 \
        --gradient_accumulation_steps 2 \
        --evaluation_strategy "no" \
        --save_strategy "steps" \
        --save_steps 500 \
        --save_total_limit 2 \
        --learning_rate 2e-5 \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type "cosine" \
        --logging_steps 10 \
        --tf32 True \
        --model_max_length 2048 \
        --gradient_checkpointing True \
        --dataloader_num_workers 4 \
        --report_to wandb \
        --deepspeed $DS_CONFIG
}

# 启动训练 (多节点 - Node 2)
run_multinode_node2() {
    echo "========================================"
    echo "Step 3: Multi-node training (Node 2)"
    echo "========================================"

    torchrun \
        --nnodes=$NUM_NODES \
        --nproc_per_node=$NUM_GPUS_PER_NODE \
        --node_rank=1 \
        --master_addr=$NODE1_IP \
        --master_port=$MASTER_PORT \
        qwenvl/train/train_qwen_kd.py \
        --model_name_or_path $MODEL_PATH \
        --data_path $DATA_PATH \
        --bf16 True \
        --output_dir $OUTPUT_DIR \
        --num_train_epochs 10 \
        --per_device_train_batch_size 2 \
        --per_device_eval_batch_size 2 \
        --gradient_accumulation_steps 2 \
        --evaluation_strategy "no" \
        --save_strategy "steps" \
        --save_steps 500 \
        --save_total_limit 2 \
        --learning_rate 2e-5 \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type "cosine" \
        --logging_steps 10 \
        --tf32 True \
        --model_max_length 2048 \
        --gradient_checkpointing True \
        --dataloader_num_workers 4 \
        --report_to wandb \
        --deepspeed $DS_CONFIG
}

# ========== 主逻辑 ==========

# 生成配置文件
generate_ds_config

# 解析参数
case "${1:-test}" in
    test)
        test_communication
        ;;
    single)
        run_single_node
        ;;
    node1)
        run_multinode_node1
        ;;
    node2)
        run_multinode_node2
        ;;
    *)
        echo "Usage: bash run_multinode.sh [test|single|node1|node2]"
        echo ""
        echo "Commands:"
        echo "  test    - 测试通信 (单节点)"
        echo "  single  - 单节点训练测试"
        echo "  node1   - 多节点训练 (在 node1 上运行)"
        echo "  node2   - 多节点训练 (在 node2 上运行)"
        exit 1
        ;;
esac
