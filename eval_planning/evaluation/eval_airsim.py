#!/usr/bin/env python3
"""
AirSim 数据评估脚本
===================
计算预测轨迹与真实轨迹的 L2 误差
无需占用地图，只评估轨迹精度

用法:
    python eval_airsim.py --result_file result.json --gt_file ground_truth.json
    python eval_airsim.py --result_file result.json --gt_pkl cached_nuscenes_info.pkl
"""

import argparse
import json
import pickle
import re
import numpy as np
from pathlib import Path
from collections import defaultdict


def parse_trajectory(text: str) -> np.ndarray:
    """从文本解析轨迹点 [(x1,y1,z1), (x2,y2,z2), ...] 或 [(x1,y1), (x2,y2), ...]"""
    import re

    # 尝试匹配 (x,y,z) 3D格式
    pattern3d = r'\(\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\)'
    matches3d = re.findall(pattern3d, text)

    if matches3d:
        # 3D格式，只取 (x, y)
        points = [(float(x), float(y)) for x, y, z in matches3d]
        return np.array(points)

    # 尝试匹配 (x,y) 2D格式
    pattern2d = r'\(\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\)'
    matches2d = re.findall(pattern2d, text)

    if matches2d:
        points = [(float(x), float(y)) for x, y in matches2d]
        return np.array(points)

    # 尝试匹配 [x,y,z] 3D格式
    pattern3d_b = r'\[\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\]'
    matches3d_b = re.findall(pattern3d_b, text)

    if matches3d_b:
        points = [(float(x), float(y)) for x, y, z in matches3d_b]
        return np.array(points)

    # 尝试匹配 [x,y] 2D格式
    pattern2d_b = r'\[\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\]'
    matches2d_b = re.findall(pattern2d_b, text)

    if matches2d_b:
        points = [(float(x), float(y)) for x, y in matches2d_b]
        return np.array(points)

    return None


def compute_l2_error(pred_traj: np.ndarray, gt_traj: np.ndarray) -> dict:
    """计算 L2 误差"""
    # 确保形状一致
    min_len = min(len(pred_traj), len(gt_traj))
    pred = pred_traj[:min_len]
    gt = gt_traj[:min_len]

    # 计算每点的 L2 距离
    distances = np.sqrt(np.sum((pred - gt) ** 2, axis=1))

    return {
        'l2_0.5s': distances[0] if len(distances) > 0 else None,
        'l2_1s': distances[1] if len(distances) > 1 else None,
        'l2_1.5s': distances[2] if len(distances) > 2 else None,
        'l2_2s': distances[3] if len(distances) > 3 else None,
        'l2_2.5s': distances[4] if len(distances) > 4 else None,
        'l2_3s': distances[5] if len(distances) > 5 else None,
        'l2_avg': np.mean(distances) if len(distances) > 0 else None,
    }


def load_ground_truth_from_pkl(pkl_path: str) -> dict:
    """从 cached_nuscenes_info.pkl 加载真实轨迹"""
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    gt_dict = {}
    for token, sample in data.items():
        fut_traj = sample.get('gt_ego_fut_trajs', [])
        if len(fut_traj) > 0:
            # 转换为 (x,y) 格式，取索引 1-6（对应 0.5s, 1.0s, 1.5s, 2.0s, 2.5s, 3.0s）
            # 注意：训练标签使用 fut_traj[1:7]，跳过索引 0 的近未来点
            traj = [(p[0], p[1]) for p in fut_traj[1:7]]
            gt_dict[token] = np.array(traj)

    return gt_dict


def load_ground_truth_from_json(json_path: str) -> dict:
    """从 JSON 加载真实轨迹

    支持两种格式:
    1. 标准格式: {"id": "token", "trajectory": "[(x,y,z), ...]"}
    2. 训练数据格式: {"id": "token", "messages": [{"role": "assistant", "content": "..."}]}
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    gt_dict = {}
    for item in data:
        token = item.get('id') or item.get('token')
        if not token:
            continue

        # 尝试标准格式
        traj_text = item.get('trajectory') or item.get('gt') or item.get('future_traj')

        # 如果是训练数据格式，从 assistant 的 response 中提取轨迹
        if not traj_text and 'messages' in item:
            for msg in item['messages']:
                if msg.get('role') == 'assistant':
                    content = msg.get('content', '')
                    # 提取轨迹文本 (格式: "[(x,y,z), ...] These are the future waypoints.")
                    traj_match = re.search(r'\[[\(\)\[\]\d.,\s]+\]', content)
                    if traj_match:
                        traj_text = traj_match.group(0)
                    break

        if traj_text:
            traj = parse_trajectory(traj_text)
            if traj is not None:
                gt_dict[token] = traj

    return gt_dict


def evaluate(result_file: str, gt_dict: dict, print_details: bool = True) -> dict:
    """评估预测结果"""
    with open(result_file, 'r') as f:
        results = json.load(f)

    errors = defaultdict(list)
    valid_count = 0
    invalid_count = 0

    if print_details:
        print(f"\n{'='*110}")
        print(f"{'ID':<20} {'预测值 (前3个点)':<35} {'GT值 (前3个点)':<35} {'平均误差':>10}")
        print(f"{'-'*110}")

    for i, item in enumerate(results):
        token = item.get('id')
        predict_text = item.get('predict', '')

        if token not in gt_dict:
            continue

        gt_traj = gt_dict[token]
        pred_traj = parse_trajectory(predict_text)

        if pred_traj is None or len(pred_traj) == 0:
            invalid_count += 1
            continue

        valid_count += 1
        l2_errors = compute_l2_error(pred_traj, gt_traj)

        for key, value in l2_errors.items():
            if value is not None:
                errors[key].append(value)

        # 打印每个样本的预测值、GT值和误差
        if print_details:
            pred_str = str([(round(p[0], 2), round(p[1], 2)) for p in pred_traj[:3]])
            gt_str = str([(round(g[0], 2), round(g[1], 2)) for g in gt_traj[:3]])
            avg_err = l2_errors.get('l2_avg', 0)
            print(f"{token:<20} {pred_str:<35} {gt_str:<35} {avg_err:>8.2f}m")

    if print_details:
        print(f"{'='*110}")
        print(f"共评估 {valid_count} 个样本\n")

    # 计算平均误差
    metrics = {}
    for key, values in errors.items():
        if values:
            metrics[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
            }

    return metrics, valid_count, invalid_count


def print_results(metrics: dict, valid_count: int, invalid_count: int):
    """打印评估结果"""
    print("\n" + "=" * 60)
    print("AirSim Trajectory Evaluation Results")
    print("=" * 60)
    print(f"\nValid samples: {valid_count}")
    print(f"Invalid samples: {invalid_count}")

    print("\nL2 Error (meters):")
    print("-" * 60)
    print(f"{'Time':<12} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
    print("-" * 60)

    time_labels = [
        ('l2_0.5s', '0.5s'),
        ('l2_1s', '1.0s'),
        ('l2_1.5s', '1.5s'),
        ('l2_2s', '2.0s'),
        ('l2_2.5s', '2.5s'),
        ('l2_3s', '3.0s'),
        ('l2_avg', 'Avg.'),
    ]

    for key, label in time_labels:
        if key in metrics:
            m = metrics[key]
            print(f"{label:<12} {m['mean']:<10.4f} {m['std']:<10.4f} {m['min']:<10.4f} {m['max']:<10.4f}")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='AirSim 轨迹评估')
    parser.add_argument('--result_file', type=str, required=True, help='预测结果 JSON 文件')
    parser.add_argument('--gt_file', type=str, default=None, help='真实轨迹 JSON 文件')
    parser.add_argument('--gt_pkl', type=str, default=None, help='cached_nuscenes_info.pkl 路径')
    parser.add_argument('--save_metrics', type=str, default=None, help='保存评估结果的路径')
    args = parser.parse_args()

    # 加载真实轨迹
    if args.gt_pkl:
        print(f"Loading ground truth from: {args.gt_pkl}")
        gt_dict = load_ground_truth_from_pkl(args.gt_pkl)
    elif args.gt_file:
        print(f"Loading ground truth from: {args.gt_file}")
        gt_dict = load_ground_truth_from_json(args.gt_file)
    else:
        print("Error: Please provide --gt_file or --gt_pkl")
        return

    print(f"Loaded {len(gt_dict)} ground truth trajectories")

    # 评估
    print(f"Evaluating: {args.result_file}")
    metrics, valid_count, invalid_count = evaluate(args.result_file, gt_dict, print_details=True)

    # 打印结果
    print_results(metrics, valid_count, invalid_count)

    # 保存结果
    if args.save_metrics:
        output = {
            'metrics': metrics,
            'valid_samples': valid_count,
            'invalid_samples': invalid_count,
        }
        with open(args.save_metrics, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nMetrics saved to: {args.save_metrics}")


if __name__ == '__main__':
    main()
