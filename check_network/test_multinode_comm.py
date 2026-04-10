#!/usr/bin/env python3
"""
多节点通信测试脚本
====================
验证 NCCL 和多节点通信是否正常

用法:
    # 单节点测试
    python test_multinode_comm.py

    # 多节点测试 (node1)
    torchrun --nnodes=2 --nproc_per_node=8 --node_rank=0 \
             --master_addr=NODE1_IP --master_port=29500 \
             test_multinode_comm.py

    # 多节点测试 (node2)
    torchrun --nnodes=2 --nproc_per_node=8 --node_rank=1 \
             --master_addr=NODE1_IP --master_port=29500 \
             test_multinode_comm.py
"""

import os
import torch
import torch.distributed as dist
from datetime import datetime


def print_rank_info():
    """打印当前进程信息"""
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    node_rank = int(os.environ.get('GROUP_RANK', 0))

    hostname = os.popen('hostname').read().strip()

    print(f"\n{'='*60}")
    print(f"Rank Information:")
    print(f"  Hostname: {hostname}")
    print(f"  Global Rank: {rank}/{world_size-1}")
    print(f"  Local Rank: {local_rank}")
    print(f"  Node Rank: {node_rank}")
    print(f"  CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA Device: {torch.cuda.get_device_name(local_rank)}")
        print(f"  CUDA Memory: {torch.cuda.get_device_properties(local_rank).total_memory / 1e9:.1f} GB")
    print(f"{'='*60}\n")

    return rank, local_rank, world_size


def test_broadcast():
    """测试广播通信"""
    rank = dist.get_rank()
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = dist.get_world_size()

    # 只在 rank 0 创建张量
    if rank == 0:
        tensor = torch.tensor([1.0, 2.0, 3.0, 4.0], device=f'cuda:{local_rank}')
        print(f"[Rank {rank}] Broadcasting tensor: {tensor}")
    else:
        tensor = torch.zeros(4, device=f'cuda:{local_rank}')
        print(f"[Rank {rank}] Waiting to receive broadcast...")

    # 广播
    dist.broadcast(tensor, src=0)

    print(f"[Rank {rank}] After broadcast: {tensor}")

    # 验证
    expected = torch.tensor([1.0, 2.0, 3.0, 4.0], device=f'cuda:{local_rank}')
    if torch.allclose(tensor, expected):
        print(f"[Rank {rank}] ✅ Broadcast test PASSED\n")
        return True
    else:
        print(f"[Rank {rank}] ❌ Broadcast test FAILED\n")
        return False


def test_allreduce():
    """测试 AllReduce 通信"""
    rank = dist.get_rank()
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = dist.get_world_size()

    # 每个 rank 创建不同的张量
    tensor = torch.ones(4, device=f'cuda:{local_rank}') * (rank + 1)
    print(f"[Rank {rank}] Before allreduce: {tensor}")

    # AllReduce (求和)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    print(f"[Rank {rank}] After allreduce: {tensor}")

    # 验证：每个元素应该是 1+2+...+world_size = world_size*(world_size+1)/2
    expected_sum = world_size * (world_size + 1) // 2
    expected = torch.ones(4, device=f'cuda:{local_rank}') * expected_sum

    if torch.allclose(tensor, expected):
        print(f"[Rank {rank}] ✅ AllReduce test PASSED (sum={expected_sum})\n")
        return True
    else:
        print(f"[Rank {rank}] ❌ AllReduce test FAILED (expected {expected}, got {tensor})\n")
        return False


def test_allgather():
    """测试 AllGather 通信"""
    rank = dist.get_rank()
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = dist.get_world_size()

    # 每个 rank 创建不同大小的张量
    local_tensor = torch.tensor([rank * 10 + i for i in range(4)],
                                 dtype=torch.float32, device=f'cuda:{local_rank}')
    print(f"[Rank {rank}] Local tensor: {local_tensor}")

    # AllGather
    gathered = [torch.zeros(4, device=f'cuda:{local_rank}') for _ in range(world_size)]
    dist.all_gather(gathered, local_tensor)

    gathered_cat = torch.cat(gathered)
    print(f"[Rank {rank}] Gathered tensor: {gathered_cat}")

    # 验证
    expected = []
    for r in range(world_size):
        expected.extend([r * 10 + i for i in range(4)])
    expected = torch.tensor(expected, dtype=torch.float32, device=f'cuda:{local_rank}')

    if torch.allclose(gathered_cat, expected):
        print(f"[Rank {rank}] ✅ AllGather test PASSED\n")
        return True
    else:
        print(f"[Rank {rank}] ❌ AllGather test FAILED\n")
        return False


def test_bandwidth():
    """测试带宽"""
    rank = dist.get_rank()
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = dist.get_world_size()

    # 测试不同大小的张量
    sizes = [1024*1024, 10*1024*1024, 100*1024*1024]  # 4MB, 40MB, 400MB

    print(f"[Rank {rank}] Bandwidth Test:")

    for size in sizes:
        tensor = torch.randn(size, device=f'cuda:{local_rank}')

        # 预热
        for _ in range(5):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

        torch.cuda.synchronize()

        # 正式测试
        num_iters = 10
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(num_iters):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        end.record()

        torch.cuda.synchronize()
        elapsed_ms = start.elapsed_time(end) / num_iters

        # 计算带宽 (AllReduce 通信量大约是 2*data_size)
        data_size_mb = size * 4 / 1024 / 1024  # float32 = 4 bytes
        bandwidth_gb_s = 2 * data_size_mb / (elapsed_ms / 1000) / 1024

        if rank == 0:
            print(f"  Size: {data_size_mb:.1f} MB, Latency: {elapsed_ms:.2f} ms, "
                  f"Bandwidth: {bandwidth_gb_s:.2f} GB/s")

    print()
    return True


def test_single_node():
    """单节点测试 - 使用大数据量测试真实带宽"""
    print("\n" + "="*60)
    print("Single Node Test")
    print("="*60)

    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return

    # 测试 GPU 间通信
    num_gpus = torch.cuda.device_count()
    print(f"\nTesting {num_gpus} GPUs...")
    print("Note: Using large tensors (500MB) to measure sustained bandwidth\n")

    # 使用更大的数据量 (500MB = 125M floats)
    tensor_size = 125_000_000  # 500MB
    num_iterations = 5

    for i in range(num_gpus):
        for j in range(i+1, num_gpus):
            # 在每个GPU上创建不同的数据，避免缓存命中
            torch.cuda.manual_seed(i * 1000 + j)

            # 预热 - 确保CUDA上下文初始化
            torch.cuda.synchronize(i)
            torch.cuda.synchronize(j)
            warmup_i = torch.randn(tensor_size, device=f'cuda:{i}')
            warmup_j = torch.randn(tensor_size, device=f'cuda:{j}')
            _ = warmup_i.to(f'cuda:{j}')
            torch.cuda.synchronize()
            del warmup_i, warmup_j

            # 创建源数据
            tensor_i = torch.randn(tensor_size, device=f'cuda:{i}')
            torch.cuda.synchronize(i)
            torch.cuda.synchronize(j)

            # 多次迭代测试 - 每次都创建新的目标张量
            times = []
            for _ in range(num_iterations):
                # 清空目标GPU缓存
                torch.cuda.empty_cache()
                torch.cuda.synchronize(j)

                # 记录时间
                torch.cuda.synchronize(i)
                torch.cuda.synchronize(j)
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)

                start.record()
                tensor_j_copy = tensor_i.to(f'cuda:{j}')
                end.record()

                torch.cuda.synchronize(j)
                elapsed_ms = start.elapsed_time(end)
                times.append(elapsed_ms)

                # 删除复制结果，避免缓存
                del tensor_j_copy

            avg_time_ms = sum(times) / len(times)
            bandwidth_gb_s = (tensor_size * 4 / 1e9) / (avg_time_ms / 1000)

            print(f"  GPU{i} -> GPU{j}: {bandwidth_gb_s:.2f} GB/s ({avg_time_ms:.2f} ms)")

            del tensor_i
            torch.cuda.empty_cache()

    print("\n✅ Single node test completed")


def test_multinode():
    """多节点测试"""
    dist.init_process_group('nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # 设置当前设备
    if torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', 0)))

    # 打印信息
    print_rank_info()

    # 同步
    if world_size > 1:
        dist.barrier()

    results = []

    # 测试1: Broadcast
    if rank == 0:
        print("\n" + "="*60)
        print("Test 1: Broadcast")
        print("="*60)
    if world_size > 1:
        dist.barrier()
    results.append(test_broadcast())

    # 测试2: AllReduce
    if rank == 0:
        print("="*60)
        print("Test 2: AllReduce")
        print("="*60)
    if world_size > 1:
        dist.barrier()
    results.append(test_allreduce())

    # 测试3: AllGather
    if rank == 0:
        print("="*60)
        print("Test 3: AllGather")
        print("="*60)
    if world_size > 1:
        dist.barrier()
    results.append(test_allgather())

    # 测试4: Bandwidth
    if rank == 0:
        print("="*60)
        print("Test 4: Bandwidth")
        print("="*60)
    if world_size > 1:
        dist.barrier()
    results.append(test_bandwidth())

    # 总结
    if world_size > 1:
        dist.barrier()

    if rank == 0:
        print("="*60)
        print("Summary")
        print("="*60)
        if all(results):
            print("✅ All tests PASSED! Multi-node communication is working.")
        else:
            print("❌ Some tests FAILED. Check the output above.")
        print("="*60)

    dist.destroy_process_group()


def main():
    # 检测运行模式
    if 'RANK' in os.environ:
        # 多节点模式
        test_multinode()
    else:
        # 单节点模式
        print("Running single node test...")
        print("\nFor multi-node test, use:")
        print("  torchrun --nnodes=2 --nproc_per_node=8 --node_rank=0 \\")
        print("           --master_addr=10.42.14.40 --master_port=29500 \\")
        print("           test_multinode_comm.py")
        test_single_node()


if __name__ == '__main__':
    main()
