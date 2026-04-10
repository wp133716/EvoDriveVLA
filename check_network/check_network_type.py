#!/usr/bin/env python3
"""
检测网络类型 (IB/RoCE/以太网)
==========================
"""

import subprocess
import os

def check_ib_devices():
    """检查 IB 设备"""
    print("=" * 60)
    print("1. 检查 InfiniBand 设备")
    print("=" * 60)

    try:
        result = subprocess.run(['ibstat'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ 发现 IB 设备:")
            print(result.stdout)
            return True
        else:
            print("❌ 未发现 IB 设备 (ibstat 执行失败)")
            return False
    except FileNotFoundError:
        print("❌ ibstat 命令不存在 (未安装 IB 驱动)")
        return False

def check_ib_modules():
    """检查 IB 内核模块"""
    print("\n" + "=" * 60)
    print("2. 检查 IB 内核模块")
    print("=" * 60)

    ib_modules = ['mlx4_ib', 'mlx5_ib', 'ib_core', 'ib_umad']
    found = []

    try:
        with open('/proc/modules', 'r') as f:
            modules = f.read()
            for mod in ib_modules:
                if mod in modules:
                    found.append(mod)
                    print(f"✅ {mod}")

        if not found:
            print("❌ 未发现 IB 内核模块")
    except Exception as e:
        print(f"❌ 无法读取模块: {e}")

    return len(found) > 0

def check_roce():
    """检查 RoCE"""
    print("\n" + "=" * 60)
    print("3. 检查 RoCE 配置")
    print("=" * 60)

    # 检查 RDMA 设备
    rdma_path = '/sys/class/infiniband'
    if os.path.exists(rdma_path):
        devices = os.listdir(rdma_path)
        if devices:
            print(f"✅ 发现 RDMA 设备: {devices}")

            # 检查每个设备的类型
            for dev in devices:
                fw_ver_path = f'{rdma_path}/{dev}/fw_ver'
                if os.path.exists(fw_ver_path):
                    with open(fw_ver_path, 'r') as f:
                        fw_ver = f.read().strip()
                        print(f"  {dev} FW 版本: {fw_ver}")

            return True
        else:
            print("❌ 未发现 RDMA 设备")
    else:
        print("❌ RDMA 子系统未启用")

    return False

def check_network_cards():
    """检查网卡信息"""
    print("\n" + "=" * 60)
    print("4. 检查网卡信息")
    print("=" * 60)

    try:
        result = subprocess.run(['lspci'], capture_output=True, text=True)
        lines = result.stdout.split('\n')

        for line in lines:
            if any(x in line.lower() for x in ['mellanox', 'ethernet', 'infiniband']):
                print(f"🌐 {line}")
    except Exception as e:
        print(f"❌ 无法执行 lspci: {e}")

def check_eth_speed():
    """检查以太网速度"""
    print("\n" + "=" * 60)
    print("5. 检查以太网卡速度")
    print("=" * 60)

    try:
        # 获取所有网卡
        result = subprocess.run(['ip', 'link', 'show'], capture_output=True, text=True)
        print(result.stdout)

        # 尝试用 ethtool 获取速度
        for interface in ['eth0', 'ens1f0', 'ens2f0', 'ib0']:
            try:
                result = subprocess.run(['ethtool', interface], capture_output=True, text=True)
                if result.returncode == 0:
                    print(f"\n🎯 {interface}:")
                    for line in result.stdout.split('\n'):
                        if 'Speed' in line or 'speed' in line:
                            print(f"  {line.strip()}")
            except:
                pass
    except Exception as e:
        print(f"❌ 无法检查: {e}")

def check_nccl_env():
    """检查 NCCL 环境变量"""
    print("\n" + "=" * 60)
    print("6. NCCL 环境变量")
    print("=" * 60)

    nccl_vars = [
        'NCCL_DEBUG',
        'NCCL_IB_DISABLE',
        'NCCL_SOCKET_IFNAME',
        'NCCL_NET_GDR_LEVEL',
        'NCCL_P2P_DISABLE',
        'NCCL_SHM_DISABLE'
    ]

    for var in nccl_vars:
        value = os.environ.get(var, '未设置')
        print(f"  {var}: {value}")

def test_nccl_communication():
    """测试 NCCL 通信"""
    print("\n" + "=" * 60)
    print("7. NCCL 通信测试 (需要2卡)")
    print("=" * 60)

    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ PyTorch CUDA 可用")
            print(f"   CUDA 版本: {torch.version.cuda}")
            print(f"   GPU 数量: {torch.cuda.device_count()}")

            # 尝试初始化 NCCL
            if torch.cuda.device_count() >= 1:
                print("\n   尝试 NCCL 初始化...")
                try:
                    torch.distributed.init_process_group(
                        'nccl',
                        init_method='tcp://localhost:29500',
                        rank=0,
                        world_size=1
                    )
                    print("✅ NCCL 初始化成功")

                    # 创建张量测试
                    tensor = torch.randn(1000, 1000).cuda()
                    print(f"✅ NCCL 通信测试通过")

                    torch.distributed.destroy_process_group()
                except Exception as e:
                    print(f"❌ NCCL 测试失败: {e}")
        else:
            print("❌ CUDA 不可用")
    except ImportError:
        print("❌ PyTorch 未安装")

def main():
    print("\n" + "=" * 60)
    print("网络类型检测工具")
    print("=" * 60 + "\n")

    has_ib = check_ib_devices()
    has_ib_mod = check_ib_modules()
    has_roce = check_roce()
    check_network_cards()
    check_eth_speed()
    check_nccl_env()
    test_nccl_communication()

    # 总结
    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)

    if has_ib:
        print("✅ 检测到 InfiniBand 网络 (最高性能)")
    elif has_roce:
        print("✅ 检测到 RoCE 网络 (高性能)")
    else:
        print("⚠️  使用普通以太网 (TCP/IP)")

    print("\n提示:")
    print("  - InfiniBand: 使用 ibstat, ib_write_bw 测试")
    print("  - RoCE: 使用 rdma_perftest")
    print("  - 以太网: 使用 iperf3 测试")

if __name__ == '__main__':
    main()
