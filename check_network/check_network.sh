#!/bin/bash
# 多节点网络连通性检查
# =====================

set -e

# 配置
NODE1_IP="10.0.0.1"      # <-- 修改为你的 node1 IP
NODE2_IP="10.0.0.2"      # <-- 修改为你的 node2 IP
MASTER_PORT=29500

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================"
echo "多节点网络连通性检查"
echo "========================================"
echo ""

# 获取当前节点信息
current_ip=$(hostname -I | awk '{print $1}')
current_hostname=$(hostname)

echo "当前节点:"
echo "  主机名: $current_hostname"
echo "  IP: $current_ip"
echo ""

# 检查 NCCL 环境变量
echo "1. 检查 NCCL 环境变量..."
if [ -z "$NCCL_SOCKET_IFNAME" ]; then
    echo -e "${YELLOW}  警告: NCCL_SOCKET_IFNAME 未设置${NC}"
    echo "  可用网卡:"
    ip link show | grep "^[0-9]" | awk '{print $2}' | sed 's/://'
else
    echo -e "${GREEN}  NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME${NC}"
fi

# 检查 GPU
echo ""
echo "2. 检查 GPU..."
if command -v nvidia-smi &> /dev/null; then
    gpu_count=$(nvidia-smi -L | wc -l)
    echo -e "${GREEN}  检测到 $gpu_count 个 GPU${NC}"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo -e "${RED}  错误: nvidia-smi 不可用${NC}"
    exit 1
fi

# 检查 PyTorch 和 CUDA
echo ""
echo "3. 检查 PyTorch..."
python3 -c "
import torch
print(f'PyTorch 版本: {torch.__version__}')
print(f'CUDA 可用: {torch.cuda.is_available()}')
print(f'CUDA 版本: {torch.version.cuda}')
print(f'GPU 数量: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
"

# 检查节点间连通性
echo ""
echo "4. 检查节点间连通性..."

# 确定对方节点
if [ "$current_ip" == "$NODE1_IP" ]; then
    other_ip=$NODE2_IP
    other_name="Node 2"
else
    other_ip=$NODE1_IP
    other_name="Node 1"
fi

echo "  检查到 $other_name ($other_ip) 的连通性..."
if ping -c 1 -W 5 $other_ip &> /dev/null; then
    echo -e "${GREEN}  ICMP ping 成功${NC}"
else
    echo -e "${RED}  错误: ICMP ping 失败${NC}"
fi

# 检查端口连通性
echo ""
echo "5. 检查端口 $MASTER_PORT 连通性..."
echo "  在 $other_name 上启动监听..."
echo ""

# 提示用户在另一节点上运行监听
echo "请在 $other_name 上运行以下命令:"
echo "  nc -l $MASTER_PORT"
echo ""
read -p "在另一节点启动监听后，按回车继续..."

# 测试连接
if nc -zv $other_ip $MASTER_PORT -w 5 &> /dev/null; then
    echo -e "${GREEN}  端口 $MASTER_PORT 连接成功${NC}"
else
    echo -e "${RED}  错误: 端口 $MASTER_PORT 连接失败${NC}"
    echo "  请检查防火墙设置"
fi

# 测试 NCCL
echo ""
echo "6. 运行 NCCL 测试..."
python3 test_multinode_comm.py

echo ""
echo "========================================"
echo "网络检查完成"
echo "========================================"
echo ""
echo "如果以上测试都通过，可以开始多节点训练:"
echo ""
echo "Node 1:"
echo "  bash run_multinode.sh node1"
echo ""
echo "Node 2:"
echo "  bash run_multinode.sh node2"
