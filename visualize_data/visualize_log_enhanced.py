#!/usr/bin/env python3
"""
飞行日志增强可视化脚本
======================
更直观地展示飞机飞行过程，包括：
1. 时序动画展示飞行轨迹演变
2. 实际位置 vs 目标路径对比
3. 推理时刻标记
4. 航向指示
5. 速度热力图

用法:
    python visualize_log_enhanced.py --log_path flight_log.json
    python visualize_log_enhanced.py --log_path flight_log.json --save_animation flight.gif
    python visualize_log_enhanced.py --log_path flight_log.json --show_inference_points
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d


class Arrow3D(FancyArrowPatch):
    """3D箭头"""
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)


def load_log(log_path: str) -> list:
    """加载日志文件"""
    with open(log_path, "r") as f:
        return json.load(f)


def extract_trajectory(log_data: list) -> tuple:
    """
    从日志中提取轨迹数据
    返回: (positions, targets, infer_indices, infer_waypoints)
    """
    positions = []
    targets = []
    infer_indices = []  # 推理发生的步骤索引
    infer_waypoints = []  # 每次推理的waypoints
    timestamps = []

    for idx, entry in enumerate(log_data):
        # 实际位置
        if "pos" in entry:
            pos = entry["pos"]
            positions.append([pos[0], pos[1], pos[2]])
            timestamps.append(idx)

            # 目标位置
            if "target" in entry:
                tgt = entry["target"]
                targets.append([tgt[0], tgt[1], tgt[2]])
            else:
                targets.append([pos[0], pos[1], pos[2]])

        # 记录推理时刻（有waypoints_ego的是推理记录）
        if "waypoints_world" in entry:
            infer_indices.append(len(positions) - 1)
            waypoints = entry.get("waypoints_world", [])
            infer_waypoints.append(waypoints)

    return (np.array(positions) if positions else np.array([]),
            np.array(targets) if targets else np.array([]),
            infer_indices, infer_waypoints, timestamps)


def visualize_3d_trajectory(log_data: list, save_path: str = None, show_inference: bool = True):
    """
    3D轨迹可视化 - 显示飞行过程
    """
    positions, targets, infer_indices, infer_waypoints, timestamps = extract_trajectory(log_data)

    if len(positions) == 0:
        print("没有位置数据可可视化")
        return

    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制完整实际轨迹（半透明）
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2],
            'b-', linewidth=1, alpha=0.3, label='Full Trajectory')

    # 绘制目标路径（虚线）
    if len(targets) > 0:
        ax.plot(targets[:, 0], targets[:, 1], targets[:, 2],
                'g--', linewidth=1, alpha=0.4, label='Target Path')

    # 标记推理时刻和路径规划
    if show_inference and infer_indices:
        colors = plt.cm.rainbow(np.linspace(0, 1, len(infer_indices)))

        for idx, (infer_idx, waypoints, color) in enumerate(zip(infer_indices, infer_waypoints, colors)):
            if infer_idx < len(positions):
                pos = positions[infer_idx]

                # 绘制推理点（星形）
                ax.scatter(*pos, color=color, s=200, marker='*',
                          edgecolors='black', linewidths=1.5,
                          label=f'Inference #{idx+1}' if idx < 5 else '')

                # 绘制规划的waypoints
                if waypoints:
                    wp_array = np.array(waypoints)
                    ax.plot(wp_array[:, 0], wp_array[:, 1], wp_array[:, 2],
                           'o-', color=color, linewidth=2, markersize=4,
                           alpha=0.7)

                # 连接线：推理位置 -> 第一个waypoint
                if waypoints:
                    ax.plot([pos[0], waypoints[0][0]],
                           [pos[1], waypoints[0][1]],
                           [pos[2], waypoints[0][2]],
                           'k:', alpha=0.3, linewidth=1)

    # 标记起点和终点
    ax.scatter(*positions[0], color='green', s=300, marker='^',
              label='Start', edgecolors='black', linewidths=2)
    ax.scatter(*positions[-1], color='red', s=300, marker='s',
              label='End', edgecolors='black', linewidths=2)

    # 添加当前位置高亮（最后一个点）
    ax.scatter(*positions[-1], color='yellow', s=150, marker='o',
              alpha=0.8, edgecolors='red', linewidths=2)

    # 设置坐标轴
    ax.set_xlabel('X (North) [m]', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y (East) [m]', fontsize=12, fontweight='bold')
    ax.set_zlabel('Z (Down) [m]', fontsize=12, fontweight='bold')
    ax.set_title('Drone Flight Trajectory with Inference Points',
                fontsize=14, fontweight='bold', pad=20)

    # 添加图例
    ax.legend(loc='upper left', fontsize=9, ncol=2)

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


def visualize_2d_with_time(log_data: list, save_path: str = None):
    """
    2D可视化 - 时间序列展示
    显示XY平面上的飞行过程，带时间热力图
    """
    positions, targets, infer_indices, infer_waypoints, timestamps = extract_trajectory(log_data)

    if len(positions) == 0:
        print("没有位置数据可可视化")
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # 1. XY平面 - 时间热力图
    ax = axes[0, 0]
    scatter = ax.scatter(positions[:, 0], positions[:, 1],
                        c=np.arange(len(positions)), cmap='viridis',
                        s=20, alpha=0.6)
    ax.plot(positions[:, 0], positions[:, 1], 'k-', alpha=0.2, linewidth=0.5)

    # 标记推理点
    if infer_indices:
        infer_positions = positions[infer_indices]
        ax.scatter(infer_positions[:, 0], infer_positions[:, 1],
                  c='red', s=200, marker='*',
                  edgecolors='black', linewidths=1.5,
                  label='Inference Points', zorder=5)

    ax.scatter(*positions[0, :2], color='green', s=200, marker='^',
              label='Start', edgecolors='black', linewidths=2, zorder=5)
    ax.scatter(*positions[-1, :2], color='red', s=200, marker='s',
              label='End', edgecolors='black', linewidths=2, zorder=5)

    ax.set_xlabel('X (North) [m]', fontsize=11)
    ax.set_ylabel('Y (East) [m]', fontsize=11)
    ax.set_title('Top View (Time Heatmap: Purple->Yellow)', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    plt.colorbar(scatter, ax=ax, label='Step')

    # 2. 高度变化
    ax = axes[0, 1]
    time_axis = np.arange(len(positions))
    ax.plot(time_axis, -positions[:, 2], 'b-', linewidth=2, label='Actual Altitude')
    if len(targets) > 0:
        ax.plot(time_axis, -targets[:, 2], 'g--', linewidth=1.5, label='Target Altitude', alpha=0.7)

    # 标记推理时刻
    for idx in infer_indices:
        ax.axvline(x=idx, color='r', linestyle=':', alpha=0.5)

    ax.set_xlabel('Step', fontsize=11)
    ax.set_ylabel('Altitude [m]', fontsize=11)
    ax.set_title('Altitude over Time', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. 跟踪误差
    ax = axes[1, 0]
    if len(targets) > 0:
        errors = np.linalg.norm(positions - targets, axis=1)
        ax.plot(time_axis, errors, 'r-', linewidth=1.5, label='Tracking Error')
        ax.fill_between(time_axis, errors, alpha=0.3)

        # 统计
        mean_error = np.mean(errors)
        max_error = np.max(errors)
        ax.axhline(y=mean_error, color='g', linestyle='--',
                  label=f'Mean: {mean_error:.2f}m')

        # 标记推理时刻
        for idx in infer_indices:
            ax.axvline(x=idx, color='b', linestyle=':', alpha=0.5)

        ax.set_xlabel('Step', fontsize=11)
        ax.set_ylabel('Error [m]', fontsize=11)
        ax.set_title(f'Tracking Error (Max: {max_error:.2f}m)', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # 4. 速度分析
    ax = axes[1, 1]
    if len(positions) > 1:
        velocities = np.linalg.norm(np.diff(positions, axis=0), axis=1) * 10  # 假设10Hz
        ax.plot(time_axis[1:], velocities, 'b-', linewidth=1.5, label='Speed')
        ax.fill_between(time_axis[1:], velocities, alpha=0.3)

        mean_speed = np.mean(velocities)
        ax.axhline(y=mean_speed, color='g', linestyle='--',
                  label=f'Mean: {mean_speed:.2f}m/s')

        # 标记推理时刻
        for idx in infer_indices:
            ax.axvline(x=idx, color='r', linestyle=':', alpha=0.5)

        ax.set_xlabel('Step', fontsize=11)
        ax.set_ylabel('Speed [m/s]', fontsize=11)
        ax.set_title('Flight Speed', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        save_path_2d = save_path.replace('.png', '_2d_analysis.png')
        plt.savefig(save_path_2d, dpi=300, bbox_inches='tight')
        print(f"2D 分析图已保存: {save_path_2d}")
    else:
        plt.show()


def create_animation(log_data: list, save_path: str = None, fps: int = 10):
    """
    创建飞行过程动画
    """
    positions, targets, infer_indices, infer_waypoints, timestamps = extract_trajectory(log_data)

    if len(positions) == 0:
        print("没有位置数据可创建动画")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    # 左图：XY轨迹
    ax1.set_xlim(positions[:, 0].min() - 10, positions[:, 0].max() + 10)
    ax1.set_ylim(positions[:, 1].min() - 10, positions[:, 1].max() + 10)
    ax1.set_xlabel('X (North) [m]', fontsize=11)
    ax1.set_ylabel('Y (East) [m]', fontsize=11)
    ax1.set_title('Flight Trajectory Animation', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')

    # 右图：高度
    ax2.set_xlim(0, len(positions))
    ax2.set_ylim(-positions[:, 2].min() - 5, -positions[:, 2].max() + 5)
    ax2.set_xlabel('Step', fontsize=11)
    ax2.set_ylabel('Altitude [m]', fontsize=11)
    ax2.set_title('Altitude Profile', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # 初始化线条
    line1, = ax1.plot([], [], 'b-', linewidth=2, label='Path')
    point1, = ax1.plot([], [], 'ro', markersize=10)
    trail1, = ax1.plot([], [], 'r-', alpha=0.3, linewidth=1)

    line2, = ax2.plot([], [], 'g-', linewidth=2)
    point2, = ax2.plot([], [], 'ro', markersize=8)

    # 标记推理点
    infer_markers = []
    for idx in infer_indices:
        if idx < len(positions):
            marker, = ax1.plot([positions[idx, 0]], [positions[idx, 1]],
                              'g*', markersize=15)
            infer_markers.append(marker)

    info_text = ax1.text(0.02, 0.98, '', transform=ax1.transAxes,
                        fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    def init():
        line1.set_data([], [])
        point1.set_data([], [])
        trail1.set_data([], [])
        line2.set_data([], [])
        point2.set_data([], [])
        info_text.set_text('')
        return line1, point1, trail1, line2, point2, info_text

    def update(frame):
        # XY轨迹
        line1.set_data(positions[:frame, 0], positions[:frame, 1])
        point1.set_data([positions[frame, 0]], [positions[frame, 1]])

        # 轨迹尾迹
        trail_start = max(0, frame - 20)
        trail1.set_data(positions[trail_start:frame, 0],
                       positions[trail_start:frame, 1])

        # 高度
        line2.set_data(range(frame), -positions[:frame, 2])
        point2.set_data([frame], [-positions[frame, 2]])

        # 信息
        speed = 0
        if frame > 0:
            dist = np.linalg.norm(positions[frame] - positions[frame-1])
            speed = dist * 10  # 假设10Hz

        info_text.set_text(f'Step: {frame}\n'
                          f'Position: ({positions[frame, 0]:.1f}, '
                          f'{positions[frame, 1]:.1f}, '
                          f'{-positions[frame, 2]:.1f})\n'
                          f'Speed: {speed:.2f} m/s')

        return line1, point1, trail1, line2, point2, info_text

    anim = animation.FuncAnimation(fig, update, init_func=init,
                                   frames=len(positions), interval=1000//fps,
                                   blit=True)

    if save_path:
        try:
            anim.save(save_path, writer='pillow', fps=fps)
            print(f"动画已保存: {save_path}")
        except Exception as e:
            print(f"保存动画失败: {e}")
            print("尝试保存为MP4...")
            try:
                anim.save(save_path.replace('.gif', '.mp4'), writer='ffmpeg', fps=fps)
                print(f"动画已保存为MP4")
            except:
                print("保存失败，请安装ffmpeg或pillow")
    else:
        plt.show()


def print_flight_summary(log_data: list):
    """打印飞行摘要"""
    positions, targets, infer_indices, infer_waypoints, timestamps = extract_trajectory(log_data)

    print(f"\n{'='*60}")
    print("飞行过程摘要")
    print(f"{'='*60}")

    if len(positions) == 0:
        print("没有位置数据")
        return

    # 基本信息
    total_distance = np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))
    duration = len(positions) / 10.0  # 假设10Hz

    print(f"总步数: {len(positions)}")
    print(f"飞行距离: {total_distance:.2f} m")
    print(f"飞行时间: {duration:.1f} s")
    print(f"平均速度: {total_distance/duration:.2f} m/s")

    # 推理统计
    print(f"\n推理次数: {len(infer_indices)}")
    if infer_indices:
        print(f"推理间隔: {duration/len(infer_indices):.1f} s/次")

    # 高度统计
    altitudes = -positions[:, 2]
    print(f"\n高度统计:")
    print(f"  平均: {np.mean(altitudes):.2f} m")
    print(f"  最大: {np.max(altitudes):.2f} m")
    print(f"  最小: {np.min(altitudes):.2f} m")

    # 跟踪误差
    if len(targets) > 0:
        errors = np.linalg.norm(positions - targets, axis=1)
        print(f"\n跟踪误差:")
        print(f"  平均: {np.mean(errors):.2f} m")
        print(f"  最大: {np.max(errors):.2f} m")

    # 范围
    print(f"\n飞行范围:")
    print(f"  X: [{positions[:, 0].min():.1f}, {positions[:, 0].max():.1f}] m")
    print(f"  Y: [{positions[:, 1].min():.1f}, {positions[:, 1].max():.1f}] m")

    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="增强版飞行日志可视化")
    parser.add_argument("--log_path", type=str, required=True, help="日志文件路径")
    parser.add_argument("--save_plot", type=str, default=None, help="保存3D图路径")
    parser.add_argument("--save_2d", type=str, default=None, help="保存2D分析图路径")
    parser.add_argument("--save_animation", type=str, default=None, help="保存动画路径(.gif)")
    parser.add_argument("--show_inference_points", action="store_true",
                       help="显示推理时刻标记")
    parser.add_argument("--fps", type=int, default=10, help="动画帧率")
    parser.add_argument("--no_3d", action="store_true", help="跳过3D可视化")
    parser.add_argument("--no_summary", action="store_true", help="跳过摘要")
    args = parser.parse_args()

    # 加载日志
    print(f"加载日志: {args.log_path}")
    log_data = load_log(args.log_path)

    # 打印摘要
    if not args.no_summary:
        print_flight_summary(log_data)

    # 3D可视化
    if not args.no_3d:
        print("生成3D轨迹图...")
        visualize_3d_trajectory(log_data, args.save_plot, args.show_inference_points)

    # 2D分析
    print("生成2D分析图...")
    visualize_2d_with_time(log_data, args.save_2d or args.save_plot)

    # 动画
    if args.save_animation:
        print("创建动画...")
        create_animation(log_data, args.save_animation, args.fps)

    print("完成!")


if __name__ == "__main__":
    main()
