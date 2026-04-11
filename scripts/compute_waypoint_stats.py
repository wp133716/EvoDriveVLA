#!/usr/bin/env python3
"""
统计训练集中 waypoints 的坐标均值和标准差
用于回归模型的归一化

用法:
    python scripts/compute_waypoint_stats.py \
        --data_path ./data/nuscenes/Drive_KD_train_his_ego.json \
        --output ./model/language_models/waypoint_stats.json
"""

import json
import re
import argparse
import numpy as np
from pathlib import Path


def parse_waypoints_from_text(text):
    """从文本解析 waypoints"""
    pattern = r"\(([+-]?\d+\.?\d*)\s*,\s*([+-]?\d+\.?\d*)\s*,\s*([+-]?\d+\.?\d*)\)"
    matches = re.findall(pattern, text)
    waypoints = []
    for x, y, z in matches[:6]:
        waypoints.append([float(x), float(y), float(z)])
    # 补齐到6个点
    while len(waypoints) < 6:
        waypoints.append([0.0, 0.0, 0.0])
    return waypoints[:6]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True,
                        help="训练数据 JSON 文件路径")
    parser.add_argument("--output", type=str, default="./waypoint_stats.json",
                        help="统计结果输出路径")
    args = parser.parse_args()

    print(f"Loading data from {args.data_path}...")
    with open(args.data_path, 'r') as f:
        data = json.load(f)

    all_waypoints = []  # [N, 6, 3]

    print("Parsing waypoints...")
    for item in data:
        conversations = item.get("conversations", [])
        for conv in conversations:
            if conv.get("from") == "gpt":
                value = conv.get("value", "")
                waypoints = parse_waypoints_from_text(value)
                all_waypoints.append(waypoints)

    all_waypoints = np.array(all_waypoints)  # [N, 6, 3]
    print(f"Total samples: {len(all_waypoints)}")

    # 统计每个 waypoint 的均值和标准差
    mean = np.mean(all_waypoints, axis=0)  # [6, 3]
    std = np.std(all_waypoints, axis=0)    # [6, 3]
    min_val = np.min(all_waypoints, axis=0)
    max_val = np.max(all_waypoints, axis=0)

    # 全局统计（所有 waypoints 一起）
    global_mean = np.mean(all_waypoints.reshape(-1, 3), axis=0)  # [3]
    global_std = np.std(all_waypoints.reshape(-1, 3), axis=0)    # [3]

    print("\n" + "="*60)
    print("Waypoints Statistics")
    print("="*60)

    print("\nPer-waypoint statistics:")
    for i in range(6):
        print(f"\nWaypoint {i+1}:")
        print(f"  Mean: [{mean[i, 0]:.4f}, {mean[i, 1]:.4f}, {mean[i, 2]:.4f}]")
        print(f"  Std:  [{std[i, 0]:.4f}, {std[i, 1]:.4f}, {std[i, 2]:.4f}]")
        print(f"  Min:  [{min_val[i, 0]:.4f}, {min_val[i, 1]:.4f}, {min_val[i, 2]:.4f}]")
        print(f"  Max:  [{max_val[i, 0]:.4f}, {max_val[i, 1]:.4f}, {max_val[i, 2]:.4f}]")

    print("\n" + "-"*60)
    print("\nGlobal statistics (all waypoints):")
    print(f"  Mean: [{global_mean[0]:.4f}, {global_mean[1]:.4f}, {global_mean[2]:.4f}]")
    print(f"  Std:  [{global_std[0]:.4f}, {global_std[1]:.4f}, {global_std[2]:.4f}]")
    print(f"  Min:  [{np.min(min_val, axis=0)[0]:.4f}, {np.min(min_val, axis=0)[1]:.4f}, {np.min(min_val, axis=0)[2]:.4f}]")
    print(f"  Max:  [{np.max(max_val, axis=0)[0]:.4f}, {np.max(max_val, axis=0)[1]:.4f}, {np.max(max_val, axis=0)[2]:.4f}]")

    # 保存结果
    stats = {
        "per_waypoint": {
            "mean": mean.tolist(),
            "std": std.tolist(),
            "min": min_val.tolist(),
            "max": max_val.tolist(),
        },
        "global": {
            "mean": global_mean.tolist(),
            "std": global_std.tolist(),
        },
        "num_samples": len(all_waypoints),
    }

    with open(args.output, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\n\nStatistics saved to: {args.output}")

    # 输出生成代码片段
    print("\n" + "="*60)
    print("Code snippet for model initialization:")
    print("="*60)
    print(f"""
# 在 Qwen2_5_VLForRegressionV2.__init__ 中添加:
self.register_buffer('waypoint_mean', torch.tensor({global_mean.tolist()}))
self.register_buffer('waypoint_std', torch.tensor({global_std.tolist()}))

# 在 forward 中使用:
# 反归一化: waypoints = normalized_waypoints * self.waypoint_std + self.waypoint_mean
# 归一化 label: normalized_target = (target_tensor - self.waypoint_mean) / self.waypoint_std
""")


if __name__ == "__main__":
    main()
