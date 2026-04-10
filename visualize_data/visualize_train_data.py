#!/usr/bin/env python3
"""
训练数据可视化脚本
====================
加载 cached_nuscenes_info.pkl 并可视化轨迹数据
功能：
- 显示历史轨迹和未来航点
- 分析轨迹形状（左转/右转/直线）
- 估算曲率半径
- ASCII图形可视化
- 数据集整体统计（命令分布、转向分布）
用法:
  # 分析数据集整体统计
  python visualize_train_data.py --data_path /path/to/cached_nuscenes_info.pkl

  # 查看5个随机样本的详细信息
  python visualize_train_data.py --data_path /path/to/cached_nuscenes_info.pkl --num_samples 5

  # 启用ASCII轨迹可视化
  python visualize_train_data.py --data_path /path/to/cached_nuscenes_info.pkl --num_samples 3 --visualize

  # 查看特定样本
  python visualize_train_data.py --data_path /path/to/cached_nuscenes_info.pkl --token <token_id>

"""

import argparse
import pickle
import json
import numpy as np
from pathlib import Path


def load_train_data(data_path: str):
    """加载训练数据"""
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    return data


def analyze_sample(data_dict: dict, token: str = None):
    """分析单个样本"""
    sample = data_dict if token is None else data_dict[token]

    print(f"\n{'='*60}")
    print(f"Sample Token: {token or 'N/A'}")
    print(f"{'='*60}")

    # 基本信息
    print(f"Scene Token: {sample.get('scene_token', 'N/A')}")
    print(f"Timestamp: {sample.get('timestamp', 'N/A')}")

    # 相机信息
    cams = sample.get('cams', {})
    print(f"\nCameras ({len(cams)}):")
    for cam_name, cam_info in cams.items():
        print(f"  {cam_name}: {cam_info.get('data_path', 'N/A')}")

    # 历史轨迹
    his_traj = sample.get('gt_ego_his_trajs', [])
    print(f"\nHistorical Trajectory ({len(his_traj)} points):")
    for i, point in enumerate(his_traj):
        time_label = -2.0 + i * 0.5
        print(f"  ({time_label:+.1f}s): ({point[0]:7.2f}, {point[1]:7.2f}, {point[2]:7.2f})")

    # 未来轨迹
    fut_traj = sample.get('gt_ego_fut_trajs', [])
    print(f"\nFuture Trajectory ({len(fut_traj)} points):")
    for i, point in enumerate(fut_traj):
        time_label = (i + 1) * 0.5
        print(f"  ({time_label:+.1f}s): ({point[0]:7.2f}, {point[1]:7.2f}, {point[2]:7.2f})")

    # 分析未来轨迹的形状
    if len(fut_traj) >= 3:
        analyze_trajectory_shape(fut_traj)

    # can_bus 信息
    can_bus = sample.get('can_bus', {})
    print(f"\nCan Bus (Ego State):")
    print(f"  Velocity: {can_bus.get('vel', 'N/A'):.2f} m/s")
    print(f"  Acc X: {can_bus.get('acc_x', 'N/A'):.2f} m/s^2")
    print(f"  Acc Y: {can_bus.get('acc_y', 'N/A'):.2f} m/s^2")
    print(f"  Steer/Yaw Rate: {can_bus.get('steer', 'N/A'):.2f}")

    # Command
    cmd = sample.get('gt_ego_fut_cmd', [])
    cmd_labels = ['RIGHT', 'LEFT', 'FORWARD']
    if len(cmd) == 3:
        active_cmd = cmd_labels[np.argmax(cmd)]
        print(f"\nMission Goal: {active_cmd} ({cmd})")

    return his_traj, fut_traj


def analyze_trajectory_shape(trajectory: list):
    """分析轨迹的形状特征"""
    traj = np.array(trajectory)

    print(f"\nTrajectory Shape Analysis:")

    # 计算每段的长度和方向
    segments = []
    angles = []
    for i in range(len(traj) - 1):
        dx = traj[i+1][0] - traj[i][0]
        dy = traj[i+1][1] - traj[i][1]
        length = np.sqrt(dx**2 + dy**2)
        angle = np.arctan2(dy, dx) * 180 / np.pi
        segments.append(length)
        angles.append(angle)

    print(f"  Segment lengths: {[f'{s:.2f}' for s in segments]}")
    print(f"  Segment angles: {[f'{a:.1f}°' for a in angles]}")

    # 计算转向
    if len(angles) >= 2:
        turns = [angles[i+1] - angles[i] for i in range(len(angles) - 1)]
        print(f"  Turn angles: {[f'{t:.1f}°' for t in turns]}")
        avg_turn = np.mean(turns)
        print(f"  Average turn: {avg_turn:.1f}° ({'LEFT' if avg_turn > 5 else 'RIGHT' if avg_turn < -5 else 'STRAIGHT'})")

    # 计算曲率 (使用三点法)
    if len(traj) >= 3:
        curvatures = []
        for i in range(len(traj) - 2):
            p1 = traj[i][:2]
            p2 = traj[i+1][:2]
            p3 = traj[i+2][:2]

            # 计算曲率
            v1 = p2 - p1
            v2 = p3 - p2

            cross = v1[0] * v2[1] - v1[1] * v2[0]
            dot = v1[0] * v2[0] + v1[1] * v2[1]

            # 转向方向
            direction = 'LEFT' if cross > 0 else 'RIGHT' if cross < 0 else 'STRAIGHT'

            # 估算曲率半径
            chord1 = np.linalg.norm(v1)
            chord2 = np.linalg.norm(v2)
            avg_chord = (chord1 + chord2) / 2

            # 使用叉积估算转角
            sin_theta = cross / (chord1 * chord2 + 1e-6)
            theta = np.arcsin(np.clip(sin_theta, -1, 1))

            if abs(theta) > 0.01:
                radius = avg_chord / (2 * np.sin(theta/2) + 1e-6)
                curvatures.append((direction, abs(radius), theta * 180 / np.pi))

        if curvatures:
            print(f"\n  Curvature Analysis:")
            for i, (dir, r, theta) in enumerate(curvatures[:3]):
                print(f"    Segment {i}: {dir}, radius≈{r:.1f}m, angle={theta:.1f}°")


def visualize_trajectory_ascii(his_traj: list, fut_traj: list, width: int = 60, height: int = 20):
    """使用ASCII字符可视化轨迹"""
    all_points = np.array(his_traj + fut_traj)

    # 归一化到显示区域
    x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
    y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()

    # 添加边距
    margin = 0.1
    x_range = x_max - x_min
    y_range = y_max - y_min

    if x_range < 1: x_range = 1
    if y_range < 1: y_range = 1

    x_min -= x_range * margin
    x_max += x_range * margin
    y_min -= y_range * margin
    y_max += y_range * margin

    # 创建画布
    canvas = [[' ' for _ in range(width)] for _ in range(height)]

    # 坐标转换函数
    def to_canvas(x, y):
        cx = int((x - x_min) / (x_max - x_min) * (width - 1))
        cy = height - 1 - int((y - y_min) / (y_max - y_min) * (height - 1))
        return max(0, min(width-1, cx)), max(0, min(height-1, cy))

    # 绘制原点
    ox, oy = to_canvas(0, 0)
    canvas[oy][ox] = '+'

    # 绘制历史轨迹 (用 '.')
    for point in his_traj:
        cx, cy = to_canvas(point[0], point[1])
        canvas[cy][cx] = '.'

    # 绘制未来轨迹 (用数字)
    for i, point in enumerate(fut_traj):
        cx, cy = to_canvas(point[0], point[1])
        if i < 6:
            canvas[cy][cx] = str(i + 1)
        else:
            canvas[cy][cx] = '*'

    print(f"\n  Trajectory Visualization (X: {x_min:.1f}~{x_max:.1f}, Y: {y_min:.1f}~{y_max:.1f}):")
    print(f"  '+' = origin (current ego position)")
    print(f"  '.' = historical trajectory")
    print(f"  '1-6' = future waypoints (0.5s intervals)")
    print()
    for row in canvas:
        print('  ' + ''.join(row))


def analyze_dataset(data: dict):
    """分析整个数据集"""
    print(f"\n{'='*60}")
    print(f"Dataset Analysis")
    print(f"{'='*60}")
    print(f"Total samples: {len(data)}")

    # 统计命令分布
    cmd_counts = {'RIGHT': 0, 'LEFT': 0, 'FORWARD': 0}

    # 统计轨迹长度
    traj_lengths = []

    # 统计转向
    left_turns = 0
    right_turns = 0
    straight = 0

    for token, sample in data.items():
        cmd = sample.get('gt_ego_fut_cmd', [])
        if len(cmd) == 3:
            if cmd[0] > 0: cmd_counts['RIGHT'] += 1
            elif cmd[1] > 0: cmd_counts['LEFT'] += 1
            else: cmd_counts['FORWARD'] += 1

        fut_traj = sample.get('gt_ego_fut_trajs', [])
        if len(fut_traj) > 0:
            traj_lengths.append(len(fut_traj))

            # 分析转向
            if len(fut_traj) >= 3:
                p1 = np.array(fut_traj[0][:2])
                p2 = np.array(fut_traj[1][:2])
                p3 = np.array(fut_traj[2][:2])

                v1 = p2 - p1
                v2 = p3 - p2
                cross = v1[0] * v2[1] - v1[1] * v2[0]

                if cross > 0.1: left_turns += 1
                elif cross < -0.1: right_turns += 1
                else: straight += 1

    print(f"\nCommand Distribution:")
    for cmd, count in cmd_counts.items():
        print(f"  {cmd}: {count} ({count/len(data)*100:.1f}%)")

    if traj_lengths:
        print(f"\nFuture Trajectory Lengths:")
        print(f"  Min: {min(traj_lengths)}")
        print(f"  Max: {max(traj_lengths)}")
        print(f"  Avg: {np.mean(traj_lengths):.1f}")

    print(f"\nTurn Direction Distribution:")
    print(f"  LEFT: {left_turns} ({left_turns/len(data)*100:.1f}%)")
    print(f"  RIGHT: {right_turns} ({right_turns/len(data)*100:.1f}%)")
    print(f"  STRAIGHT: {straight} ({straight/len(data)*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="可视化训练数据")
    parser.add_argument("--data_path", type=str, required=True, help="cached_nuscenes_info.pkl 路径")
    parser.add_argument("--num_samples", type=int, default=5, help="查看的样本数量")
    parser.add_argument("--token", type=str, default=None, help="查看特定 token")
    parser.add_argument("--visualize", action="store_true", help="启用 ASCII 可视化")
    args = parser.parse_args()

    # 加载数据
    print(f"Loading data from: {args.data_path}")
    data = load_train_data(args.data_path)
    print(f"Loaded {len(data)} samples")

    # 数据集整体分析
    analyze_dataset(data)

    # 查看特定样本或随机样本
    tokens = list(data.keys())

    if args.token:
        if args.token in data:
            his_traj, fut_traj = analyze_sample(data, args.token)
            if args.visualize:
                visualize_trajectory_ascii(his_traj, fut_traj)
        else:
            print(f"Error: Token {args.token} not found!")
    else:
        # 随机选择样本
        import random
        random.seed(42)
        selected_tokens = random.sample(tokens, min(args.num_samples, len(tokens)))

        for token in selected_tokens:
            his_traj, fut_traj = analyze_sample(data, token)
            if args.visualize:
                visualize_trajectory_ascii(his_traj, fut_traj)


if __name__ == "__main__":
    main()
