#!/usr/bin/env python3
"""
飞行日志可视化脚本
==================
加载 deploy_airsim_vla.py 生成的飞行日志，可视化世界坐标轨迹

用法:
    python visualize_log.py --log_path flight_log.json
    python visualize_log.py --log_path flight_log.json --save_plot trajectory.png
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def load_log(log_path: str) -> list:
    """加载日志文件"""
    with open(log_path, "r") as f:
        return json.load(f)


def extract_trajectory(log_data: list) -> tuple:
    """
    从日志中提取轨迹数据
    返回: (positions, targets, waypoints_ego_list)
    """
    positions = []  # 实际飞行位置 (世界坐标)
    targets = []    # 目标航点 (世界坐标)
    waypoints_ego_list = []  # 模型输出的原始航点 (自车坐标)

    for entry in log_data:
        # 实际位置 [x, y, z]
        pos = entry["pos"]
        positions.append([pos[0], pos[1], pos[2]])

        # 目标位置 [x, y, z]
        tgt = entry["target"]
        targets.append([tgt[0], tgt[1], tgt[2]])

        # 原始航点列表
        waypoints_ego_list.append(entry.get("waypoints_ego", []))

    return np.array(positions), np.array(targets), waypoints_ego_list


def visualize_3d(positions: np.ndarray, targets: np.ndarray, save_path: str = None):
    """3D 轨迹可视化"""
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制实际飞行轨迹
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2],
            'b-', linewidth=2, label='Actual Trajectory', marker='o', markersize=2)

    # 绘制目标航点轨迹
    ax.plot(targets[:, 0], targets[:, 1], targets[:, 2],
            'r--', linewidth=1.5, label='Target Waypoints', marker='x', markersize=3)

    # 标记起点和终点
    ax.scatter(*positions[0], color='green', s=150, marker='^',
               label='Start', edgecolors='black', linewidths=1)
    ax.scatter(*positions[-1], color='red', s=150, marker='s',
               label='End', edgecolors='black', linewidths=1)

    # 设置坐标轴标签 (NED 坐标系)
    ax.set_xlabel('X (North) [m]', fontsize=12)
    ax.set_ylabel('Y (East) [m]', fontsize=12)
    ax.set_zlabel('Z (Down) [m]', fontsize=12)
    ax.set_title('Drone Flight Trajectory (World Coordinates)', fontsize=14, fontweight='bold')

    # 添加图例
    ax.legend(loc='upper left', fontsize=10)

    # 添加网格
    ax.grid(True, alpha=0.3)

    # 设置等比例
    max_range = np.array([
        positions[:, 0].max() - positions[:, 0].min(),
        positions[:, 1].max() - positions[:, 1].min(),
        positions[:, 2].max() - positions[:, 2].min()
    ]).max() / 2.0

    mid_x = (positions[:, 0].max() + positions[:, 0].min()) * 0.5
    mid_y = (positions[:, 1].max() + positions[:, 1].min()) * 0.5
    mid_z = (positions[:, 2].max() + positions[:, 2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"3D 轨迹图已保存: {save_path}")
    else:
        plt.show()


def visualize_2d_projections(positions: np.ndarray, targets: np.ndarray, save_path: str = None):
    """2D 投影可视化 (俯视图、侧视图、正视图)"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 俯视图 (XY 平面)
    ax = axes[0, 0]
    ax.plot(positions[:, 0], positions[:, 1], 'b-', linewidth=2, label='Actual')
    ax.plot(targets[:, 0], targets[:, 1], 'r--', linewidth=1.5, label='Target')
    ax.scatter(*positions[0, :2], color='green', s=100, marker='^', label='Start')
    ax.scatter(*positions[-1, :2], color='red', s=100, marker='s', label='End')
    ax.set_xlabel('X (North) [m]')
    ax.set_ylabel('Y (East) [m]')
    ax.set_title('Top View (XY Plane)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')

    # 侧视图 (XZ 平面)
    ax = axes[0, 1]
    ax.plot(positions[:, 0], -positions[:, 2], 'b-', linewidth=2, label='Actual')
    ax.plot(targets[:, 0], -targets[:, 2], 'r--', linewidth=1.5, label='Target')
    ax.scatter(positions[0, 0], -positions[0, 2], color='green', s=100, marker='^', label='Start')
    ax.scatter(positions[-1, 0], -positions[-1, 2], color='red', s=100, marker='s', label='End')
    ax.set_xlabel('X (North) [m]')
    ax.set_ylabel('Altitude [m]')
    ax.set_title('Side View (XZ Plane)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 正视图 (YZ 平面)
    ax = axes[1, 0]
    ax.plot(positions[:, 1], -positions[:, 2], 'b-', linewidth=2, label='Actual')
    ax.plot(targets[:, 1], -targets[:, 2], 'r--', linewidth=1.5, label='Target')
    ax.scatter(positions[0, 1], -positions[0, 2], color='green', s=100, marker='^', label='Start')
    ax.scatter(positions[-1, 1], -positions[-1, 2], color='red', s=100, marker='s', label='End')
    ax.set_xlabel('Y (East) [m]')
    ax.set_ylabel('Altitude [m]')
    ax.set_title('Front View (YZ Plane)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 统计信息
    ax = axes[1, 1]
    ax.axis('off')

    # 计算统计信息
    total_distance = np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))
    avg_altitude = -np.mean(positions[:, 2])
    max_altitude = -np.min(positions[:, 2])
    min_altitude = -np.max(positions[:, 2])

    # 目标与实际位置的偏差
    deviations = np.linalg.norm(positions - targets, axis=1)
    mean_deviation = np.mean(deviations)
    max_deviation = np.max(deviations)

    info_text = f"""
    Flight Statistics:
    =================
    Total Steps: {len(positions)}
    Total Distance: {total_distance:.2f} m

    Altitude:
      Average: {avg_altitude:.2f} m
      Max: {max_altitude:.2f} m
      Min: {min_altitude:.2f} m

    Tracking Error:
      Mean: {mean_deviation:.2f} m
      Max: {max_deviation:.2f} m

    Position Range:
      X: [{positions[:, 0].min():.1f}, {positions[:, 0].max():.1f}] m
      Y: [{positions[:, 1].min():.1f}, {positions[:, 1].max():.1f}] m
      Z: [{positions[:, 2].min():.1f}, {positions[:, 2].max():.1f}] m
    """

    ax.text(0.1, 0.5, info_text, fontsize=11, family='monospace',
            verticalalignment='center')

    plt.tight_layout()

    if save_path:
        # 修改保存路径，添加 _2d 后缀
        save_path_2d = save_path.replace('.png', '_2d.png')
        plt.savefig(save_path_2d, dpi=300, bbox_inches='tight')
        print(f"2D 投影图已保存: {save_path_2d}")
    else:
        plt.show()


def print_summary(log_data: list):
    """打印日志摘要"""
    print(f"\n{'='*50}")
    print("Flight Log Summary")
    print(f"{'='*50}")
    print(f"Total entries: {len(log_data)}")

    if len(log_data) == 0:
        return

    # 推理时间统计
    infer_times = [entry.get("infer_time", 0) for entry in log_data]
    print(f"\nInference Time:")
    print(f"  Mean: {np.mean(infer_times):.3f}s")
    print(f"  Max: {np.max(infer_times):.3f}s")
    print(f"  Min: {np.min(infer_times):.3f}s")

    # 航点数量统计
    waypoint_counts = [len(entry.get("waypoints_ego", [])) for entry in log_data]
    print(f"\nWaypoints per step:")
    print(f"  Mean: {np.mean(waypoint_counts):.1f}")
    print(f"  Min: {np.min(waypoint_counts)}")

    # 检查是否有解析失败的步骤
    failed_steps = [i for i, entry in enumerate(log_data) if len(entry.get("waypoints_ego", [])) == 0]
    if failed_steps:
        print(f"\nWarning: {len(failed_steps)} steps with no waypoints parsed")
        print(f"  Failed steps: {failed_steps[:10]}{'...' if len(failed_steps) > 10 else ''}")

    print(f"{'='*50}\n")


def main():
    parser = argparse.ArgumentParser(description="可视化飞行日志")
    parser.add_argument("--log_path", type=str, required=True, help="日志文件路径 (JSON)")
    parser.add_argument("--save_plot", type=str, default=None, help="保存图表路径 (PNG)")
    parser.add_argument("--no_3d", action="store_true", help="跳过 3D 可视化")
    parser.add_argument("--no_2d", action="store_true", help="跳过 2D 投影")
    args = parser.parse_args()

    # 加载日志
    print(f"加载日志: {args.log_path}")
    log_data = load_log(args.log_path)

    # 打印摘要
    print_summary(log_data)

    # 提取轨迹
    positions, targets, waypoints_ego = extract_trajectory(log_data)

    print(f"提取到 {len(positions)} 个位置点")

    # 3D 可视化
    if not args.no_3d:
        print("生成 3D 轨迹图...")
        visualize_3d(positions, targets, args.save_plot)

    # 2D 投影
    if not args.no_2d:
        print("生成 2D 投影图...")
        visualize_2d_projections(positions, targets, args.save_plot)

    print("完成!")


if __name__ == "__main__":
    main()
