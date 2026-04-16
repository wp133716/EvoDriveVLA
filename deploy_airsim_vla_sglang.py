#!/usr/bin/env python3
"""
AirSim VLA 部署推理脚本 - SGLang/vLLM 版本
===========================================
推理后端换为 SGLang 或 vLLM 服务（OpenAI 兼容 API），decode 速度大幅提升。
其余逻辑（路径跟踪、坐标转换、日志、AirSim 控制）与 deploy_airsim_vla_path.py 完全一致。

启动 SGLang 服务（单独终端）：
    python -m sglang.launch_server \
        --model-path <checkpoint_dir> \
        --port 30000 \
        --tp 1 \
        --dtype bfloat16

启动 vLLM 服务（单独终端）：
    vllm serve <checkpoint_dir> \
        --port 30000 \
        --dtype bfloat16 \
        --max-model-len 4096

然后运行本脚本：
    python deploy_airsim_vla_sglang.py \
        --sglang_url http://localhost:30000 \
        --instruction "Fly a circle counter-clockwise ..."
"""

import argparse
import base64
import math
import os
import re
import time
from collections import deque
from dataclasses import dataclass
from io import BytesIO
from typing import List, Tuple, Optional

import airsim
import cv2
import numpy as np
import requests
from PIL import Image

# ==================== 配置 ====================

VEHICLE_NAME = "UAV0"
CAMERAS = ["front", "left", "right", "down"]
CAM_TO_CHANNEL = {
    "front": "CAM_FRONT",
    "left": "CAM_LEFT",
    "right": "CAM_RIGHT",
    "down": "CAM_DOWN",
}

REPLAN_INTERVAL = 6.0
NUM_WAYPOINTS = 6
WAYPOINT_INTERVAL = 0.5

HISTORY_LEN = 5
HISTORY_INTERVAL = 0.4

SYSTEM_PROMPT = (
    "You're an autonomous drone's brain. "
    "Coordinates: X-axis is perpendicular, Y-axis is parallel to the direction you're facing, "
    "and Z-axis points upward. You're at point (0,0,0). Units: meters. "
    "Based on the provided particulars, please output the 3D plan waypoints (0.5s intervals) for the next 3 seconds."
)


# ==================== 数据类 ====================

@dataclass
class DroneState:
    pos_x: float
    pos_y: float
    pos_z: float
    vel_x: float
    vel_y: float
    vel_z: float
    roll: float
    pitch: float
    yaw: float
    imu_acc_x: float
    imu_acc_y: float
    imu_ang_z: float
    timestamp: float


# ==================== AirSim 交互 ====================

def get_drone_state(client: airsim.MultirotorClient) -> DroneState:
    state = client.getMultirotorState(vehicle_name=VEHICLE_NAME)
    pos = state.kinematics_estimated.position
    vel = state.kinematics_estimated.linear_velocity
    ori = state.kinematics_estimated.orientation
    import math as _math
    pitch, roll, yaw = airsim.to_eularian_angles(ori)
    imu = client.getImuData(vehicle_name=VEHICLE_NAME)

    return DroneState(
        pos_x=pos.x_val, pos_y=pos.y_val, pos_z=pos.z_val,
        vel_x=vel.x_val, vel_y=vel.y_val, vel_z=vel.z_val,
        roll=_math.degrees(roll), pitch=_math.degrees(pitch), yaw=_math.degrees(yaw),
        imu_acc_x=imu.linear_acceleration.x_val,
        imu_acc_y=imu.linear_acceleration.y_val,
        imu_ang_z=imu.angular_velocity.z_val,
        timestamp=time.time(),
    )


def capture_images(client: airsim.MultirotorClient) -> dict:
    requests_list = [
        airsim.ImageRequest(cam, airsim.ImageType.Scene, False, False)
        for cam in CAMERAS
    ]
    responses = client.simGetImages(requests_list, vehicle_name=VEHICLE_NAME)

    images = {}
    for cam, resp in zip(CAMERAS, responses):
        if resp.width > 0 and len(resp.image_data_uint8) > 0:
            img = np.frombuffer(resp.image_data_uint8, dtype=np.uint8)
            img = img.reshape(resp.height, resp.width, 3)
            images[CAM_TO_CHANNEL[cam]] = Image.fromarray(img)
    return images


# ==================== 坐标转换 ====================

def world_to_ego(pos_x, pos_y, pos_z, ref_x, ref_y, ref_z, yaw_rad):
    dx, dy = pos_x - ref_x, pos_y - ref_y
    cos_yaw = math.cos(-yaw_rad)
    sin_yaw = math.sin(-yaw_rad)
    local_x = dx * cos_yaw - dy * sin_yaw
    local_y = dx * sin_yaw + dy * cos_yaw
    local_z = -(pos_z - ref_z)
    return local_x, local_y, local_z


def ego_to_world(local_x, local_y, local_z, ref_x, ref_y, ref_z, yaw_rad):
    """自车坐标 → 世界坐标 (NED: X=北, Y=东, Z=下)

    当前匹配旧训练数据（XY bug版本）：world_to_ego 用 cos(-yaw), sin(-yaw)，
    ego_to_world 用其逆矩阵 R^T，即 cos(yaw), sin(yaw)。

    TODO: 训练数据修复并重训后，换用以下反射公式（与 system prompt 一致）：
        dx = -local_x * sin_yaw + local_y * cos_yaw
        dy =  local_x * cos_yaw + local_y * sin_yaw
    """
    cos_yaw = math.cos(yaw_rad)
    sin_yaw = math.sin(yaw_rad)
    dx = local_x * cos_yaw - local_y * sin_yaw
    dy = local_x * sin_yaw + local_y * cos_yaw
    world_x = ref_x + dx
    world_y = ref_y + dy
    world_z = ref_z - local_z
    return world_x, world_y, world_z


def waypoints_ego_to_world(waypoints_ego, ref_pos, ref_yaw_deg):
    ref_x, ref_y, ref_z = ref_pos
    yaw_rad = math.radians(ref_yaw_deg)
    waypoints_world = []
    for local_x, local_y, local_z in waypoints_ego:
        world_x, world_y, world_z = ego_to_world(
            local_x, local_y, local_z, ref_x, ref_y, ref_z, yaw_rad
        )
        waypoints_world.append((world_x, world_y, world_z))
    return waypoints_world


def calculate_path_length(path):
    total_length = 0.0
    for i in range(1, len(path)):
        dx = path[i].x_val - path[i-1].x_val
        dy = path[i].y_val - path[i-1].y_val
        dz = path[i].z_val - path[i-1].z_val
        total_length += math.sqrt(dx*dx + dy*dy + dz*dz)
    return total_length


# ==================== Prompt 构建 ====================

def compute_ego_state(state: DroneState) -> dict:
    vel = math.sqrt(state.vel_x**2 + state.vel_y**2 + state.vel_z**2)
    return {
        "vel": round(vel, 2),
        "acc_x": round(state.imu_acc_x, 2),
        "acc_y": round(state.imu_acc_y, 2),
        "steer": round(state.imu_ang_z, 2),
    }


def build_history_traj_text(history: deque, current: DroneState) -> str:
    yaw_rad = math.radians(current.yaw)
    points = []
    for state in history:
        lx, ly, lz = world_to_ego(
            state.pos_x, state.pos_y, state.pos_z,
            current.pos_x, current.pos_y, current.pos_z, yaw_rad,
        )
        points.append((lx, ly, lz))

    while len(points) < HISTORY_LEN:
        points.insert(0, (0.0, 0.0, 0.0))
    points = points[-HISTORY_LEN:]

    time_labels = ["-2.0s", "-1.5s", "-1.0s", "-0.5s", "-0.0s"]
    parts = [f"({t}):({p[0]:.2f},{p[1]:.2f},{p[2]:.2f})" for t, p in zip(time_labels, points)]
    return "Historical Trajectory (last 2 seconds): [" + ", ".join(parts) + "]\n"


def build_history_ego_text(ego_history: deque, current_ego: dict) -> str:
    egos = list(ego_history)
    while len(egos) < 4:
        egos.insert(0, {"vel": 0.0, "acc_x": 0.0, "acc_y": 0.0, "steer": 0.0})
    egos = egos[-4:]

    time_labels = ["-2.0s", "-1.5s", "-1.0s", "-0.5s"]
    parts = []
    for t, e in zip(time_labels, egos):
        parts.append(f"({t}):({e['vel']:.2f} m/s, {e['acc_x']:.2f} m/s^2, {e['acc_y']:.2f} m/s^2, {e['steer']:.2f})")
    parts.append(f"(-0.0s):({current_ego['vel']:.2f} m/s, {current_ego['acc_x']:.2f} m/s^2, {current_ego['acc_y']:.2f} m/s^2, {current_ego['steer']:.2f})")

    return "Historical ego (last 2 seconds), Format: (Velocity, Acceleration_x, Acceleration_y, yaw_rate): [" + ", ".join(parts) + "]\n"


def build_user_content_text(instruction: str, history: deque, ego_history: deque,
                             current: DroneState, mission_goal: str = "FORWARD") -> str:
    """构建纯文本部分（不含图像占位符）"""
    current_ego = compute_ego_state(current)

    content = "Here's some information you'll need:\n"
    if instruction:
        content += f"Instruction: {instruction}\n"
    content += build_history_traj_text(history, current)
    content += build_history_ego_text(ego_history, current_ego)
    content += f"Mission Goal: {mission_goal}\n"
    content += "Flight Rules: Avoid collision with obstacles and terrain.\n"
    content += "- Maintain safe altitude above ground.\n"
    content += "- Stay within designated flight area.\n"
    content += "Based on the provided particulars, please output the 3D plan waypoints (0.5s intervals) for the next 3 seconds.\n"
    return content


# ==================== SGLang/vLLM 推理 ====================

def pil_to_base64(img: Image.Image, quality: int = 85) -> str:
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=100)
    return base64.b64encode(buf.getvalue()).decode()


def run_inference(images: dict, instruction: str, history: deque,
                  ego_history: deque, current: DroneState,
                  mission_goal: str, sglang_url: str,
                  timeout: int = 30) -> str:
    """调用 SGLang/vLLM OpenAI 兼容 API 进行推理。

    content 格式：图像在前（4张），文字信息在后。
    SGLang/vLLM 会自动处理图像 token 插入。
    """
    image_order = ["CAM_FRONT", "CAM_LEFT", "CAM_RIGHT", "CAM_DOWN"]

    # 构建 content list：图像描述 + 4张图 + 文字信息
    content = []
    content.append({
        "type": "text",
        "text": "Here are current four images from the drone: 'CAM_FRONT', 'CAM_LEFT', 'CAM_RIGHT', 'CAM_DOWN':\n"
    })
    for cam in image_order:
        if cam in images:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{pil_to_base64(images[cam])}"
                }
            })

    text_info = build_user_content_text(instruction, history, ego_history, current, mission_goal)
    content.append({"type": "text", "text": text_info})

    payload = {
        "model": "default",   # SGLang 默认；vLLM 用模型路径最后一段，也可传 "default"
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": content},
        ],
        "max_tokens": 256,
        "temperature": 0,
        "top_p": 1.0,
    }

    resp = requests.post(
        f"{sglang_url}/v1/chat/completions",
        json=payload,
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def parse_waypoints(text: str) -> List[Tuple[float, float, float]]:
    """解析模型输出的 3D waypoints（与 deploy_airsim_vla_path.py 一致）"""
    pattern = r'\(([+-]?\d+\.?\d*)\s*,\s*([+-]?\d+\.?\d*)\s*,\s*([+-]?\d+\.?\d*)\)'
    matches = re.findall(pattern, text)
    waypoints = [(float(x), float(y), float(z)) for x, y, z in matches]
    return waypoints


def determine_mission_goal(instruction: str) -> str:
    inst_lower = instruction.lower()
    if "clockwise" in inst_lower and "counter" not in inst_lower:
        return "RIGHT"
    elif "counter-clockwise" in inst_lower or "counter clockwise" in inst_lower:
        return "LEFT"
    return "FORWARD"


# ==================== 主函数 ====================

def main():
    parser = argparse.ArgumentParser(description="AirSim VLA 部署推理 - SGLang/vLLM 版本")
    parser.add_argument("--sglang_url", type=str, default="http://localhost:30000",
                        help="SGLang 或 vLLM 服务地址")
    parser.add_argument("--instruction", type=str,
                        default="Fly a circle clockwise with radius 50m at 200m altitude with 8m/s speed")
    parser.add_argument("--takeoff_altitude", type=float, default=200)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--base_speed", type=float, default=6.3)
    parser.add_argument("--replan_interval", type=float, default=6.0)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--save_log", type=str, default=None)
    parser.add_argument("--save_image_dir", type=str, default=None)
    args = parser.parse_args()

    # 验证服务可用
    print(f"检查 SGLang/vLLM 服务: {args.sglang_url} ...")
    try:
        r = requests.get(f"{args.sglang_url}/health", timeout=5)
        print(f"服务状态: {r.status_code}")
    except Exception as e:
        print(f"警告: 服务健康检查失败 ({e})，继续尝试...")

    # 连接 AirSim
    print("连接 AirSim...")
    client = airsim.MultirotorClient()
    client.confirmConnection()

    try:
        vehicles = client.listVehicles()
        if vehicles:
            global VEHICLE_NAME
            VEHICLE_NAME = vehicles[0]
            print(f"使用无人机: {VEHICLE_NAME}")
    except:
        pass

    client.enableApiControl(True, vehicle_name=VEHICLE_NAME)
    client.armDisarm(True, vehicle_name=VEHICLE_NAME)

    # 起飞
    altitude = -args.takeoff_altitude
    print(f"起飞到 {args.takeoff_altitude}m...")
    client.takeoffAsync(vehicle_name=VEHICLE_NAME).join()
    client.moveToPositionAsync(100, 200, altitude, 5.0, timeout_sec=60,
                               vehicle_name=VEHICLE_NAME).join()
    print("起飞完成")
    time.sleep(2.0)

    # 初始化历史缓存
    state_history = deque(maxlen=20)
    traj_history = deque(maxlen=HISTORY_LEN)
    ego_history = deque(maxlen=4)

    current_path = []
    path_start_time = 0
    path_following = False
    current_waypoints_world = []

    mission_goal = determine_mission_goal(args.instruction)
    print(f"Instruction: {args.instruction}")
    print(f"Mission Goal: {mission_goal}")
    print(f"重新推理间隔: {args.replan_interval}s")
    print(f"开始自主飞行 (最多 {args.max_steps} 步)...")

    flight_log = []
    last_traj_update = 0
    last_infer_time = 0
    infer_count = 0
    step = 0

    try:
        while step < args.max_steps:
            loop_start = time.time()

            # 1. 获取当前状态
            current = get_drone_state(client)
            state_history.append(current)

            if time.time() - last_traj_update >= HISTORY_INTERVAL:
                traj_history.append(current)
                ego_history.append(compute_ego_state(current))
                last_traj_update = time.time()

            # 2. 判断是否需要重新推理
            need_replan = (
                not path_following or
                time.time() - last_infer_time >= args.replan_interval
            )

            if need_replan:
                print(f"\n{'='*60}")
                print(f"[推理 #{infer_count + 1}] 获取新路径...")

                images = capture_images(client)
                if len(images) < 4:
                    print("相机采集不完整，跳过本次推理")
                    time.sleep(0.1)
                    continue

                ref_position = (current.pos_x, current.pos_y, current.pos_z)
                ref_yaw = current.yaw

                t0 = time.time()
                try:
                    output_text = run_inference(
                        images, args.instruction, traj_history, ego_history,
                        current, mission_goal, args.sglang_url
                    )
                except Exception as e:
                    print(f"推理失败: {e}")
                    time.sleep(0.5)
                    continue
                infer_time = time.time() - t0

                # 推理完成后再取消旧路径，避免推理期间无人机漂移
                if path_following:
                    client.cancelLastTask(vehicle_name=VEHICLE_NAME)
                    print("取消上一路径")

                waypoints_ego = parse_waypoints(output_text)

                if len(waypoints_ego) >= NUM_WAYPOINTS:
                    print(f"\n[调试] Mission Goal: {mission_goal}")
                    print(f"[调试] 参考位置: ({ref_position[0]:.1f}, {ref_position[1]:.1f}), 朝向: {ref_yaw:.1f}°")
                    print(f"[调试] 模型输出的自车坐标 waypoints (前3个):")
                    for i, wp in enumerate(waypoints_ego[:3]):
                        print(f"  WP[{i}]: ({wp[0]:.2f}, {wp[1]:.2f}, {wp[2]:.2f})")

                    avg_x = sum(wp[0] for wp in waypoints_ego[:NUM_WAYPOINTS]) / NUM_WAYPOINTS
                    print(f"[调试] Waypoints X坐标平均值: {avg_x:.2f}")
                    if mission_goal == "RIGHT" and avg_x < 0:
                        print("  ⚠️ 警告: Mission Goal=RIGHT 但 waypoints X < 0")
                    elif mission_goal == "LEFT" and avg_x > 0:
                        print("  ⚠️ 警告: Mission Goal=LEFT 但 waypoints X > 0")

                    waypoints_world = waypoints_ego_to_world(
                        waypoints_ego[:NUM_WAYPOINTS], ref_position, ref_yaw
                    )

                    print(f"[调试] 转换后的世界坐标 waypoints (前3个):")
                    for i, wp in enumerate(waypoints_world[:3]):
                        print(f"  WP[{i}]: ({wp[0]:.1f}, {wp[1]:.1f}, {wp[2]:.1f})")

                    # 推理完成后重新获取位置，避免用推理前的旧位置作为路径起点
                    # （推理期间无人机已移动 ~20-35m，旧起点在身后会导致掉头折返）
                    fresh_state = get_drone_state(client)
                    path = [airsim.Vector3r(fresh_state.pos_x, fresh_state.pos_y, fresh_state.pos_z)]
                    for wp in waypoints_world:
                        path.append(airsim.Vector3r(wp[0], wp[1], wp[2]))

                    path_length = calculate_path_length(path)

                    # 固定速度匹配训练数据（训练时约6.3m/s）
                    # 不用 path_length/replan_interval，否则路径越长速度越高，
                    # 历史轨迹大位移又使模型预测更大路径，形成正反馈循环
                    target_speed = args.base_speed

                    print(f"生成路径: {len(path)} 个点, 长度: {path_length:.2f}m")
                    print(f"目标速度: {target_speed:.2f}m/s (飞行时间≈{args.replan_interval}s)")
                    print(f"Waypoints: {waypoints_world}")

                    client.moveOnPathAsync(
                        path,
                        velocity=target_speed,
                        timeout_sec=args.replan_interval + 5,
                        drivetrain=airsim.DrivetrainType.ForwardOnly,
                        yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=0),
                        vehicle_name=VEHICLE_NAME
                    )

                    current_path = path
                    current_waypoints_world = waypoints_world
                    path_following = True
                    path_start_time = time.time()
                    last_infer_time = time.time()
                    infer_count += 1

                    log_entry = {
                        "step": step,
                        "infer_count": infer_count,
                        "pos": [current.pos_x, current.pos_y, current.pos_z],
                        "target": [waypoints_world[0][0], waypoints_world[0][1], waypoints_world[0][2]],
                        "ref_position": ref_position,
                        "ref_yaw": ref_yaw,
                        "waypoints_ego": waypoints_ego[:NUM_WAYPOINTS],
                        "waypoints_world": waypoints_world,
                        "path_length": path_length,
                        "target_speed": target_speed,
                        "infer_time": infer_time,
                        "model_output": output_text,
                    }
                    flight_log.append(log_entry)

                    print(f"推理耗时: {infer_time:.2f}s")
                    print(f"{'='*60}")

                else:
                    print(f"解析 waypoints 不足 ({len(waypoints_ego)}个): {output_text[:100]}")

            # 3. 检查路径状态
            if path_following:
                elapsed = time.time() - path_start_time

                pos = client.getMultirotorState(vehicle_name=VEHICLE_NAME).kinematics_estimated.position
                current_pos = [pos.x_val, pos.y_val, pos.z_val]

                if step % 5 == 0 or elapsed >= args.replan_interval:
                    progress = min(1.0, elapsed / args.replan_interval)
                    target_idx = min(int(progress * len(current_waypoints_world)), len(current_waypoints_world) - 1)
                    target_pos = list(current_waypoints_world[target_idx]) if current_waypoints_world else current_pos
                    flight_log.append({
                        "step": step,
                        "pos": current_pos,
                        "target": target_pos,
                        "infer_count": infer_count,
                        "path_elapsed": elapsed,
                    })

                if step % 20 == 0:
                    print(f"  [Step {step}] 位置: ({pos.x_val:.1f}, {pos.y_val:.1f}, {-pos.z_val:.1f}), "
                          f"路径耗时: {elapsed:.1f}s")

                if elapsed >= args.replan_interval:
                    path_following = False
                    print(f"  路径执行完成 (耗时: {elapsed:.1f}s)")

            # 4. 可视化
            if args.visualize and step % 5 == 0:
                vis_images = capture_images(client)
                if "CAM_FRONT" in vis_images and args.save_image_dir:
                    front_cv = cv2.cvtColor(np.array(vis_images["CAM_FRONT"]), cv2.COLOR_RGB2BGR)
                    os.makedirs(args.save_image_dir, exist_ok=True)
                    cv2.imwrite(os.path.join(args.save_image_dir, f"step_{step:04d}.jpg"), front_cv)

            step += 1

            elapsed = time.time() - loop_start
            if elapsed < 0.05:
                time.sleep(0.05 - elapsed)

    except KeyboardInterrupt:
        print("\n用户中断")

    finally:
        print("\n降落...")
        client.cancelLastTask(vehicle_name=VEHICLE_NAME)
        client.landAsync(timeout_sec=10, vehicle_name=VEHICLE_NAME).join()
        client.armDisarm(False, vehicle_name=VEHICLE_NAME)
        client.enableApiControl(False, vehicle_name=VEHICLE_NAME)

        if args.save_log:
            import json
            with open(args.save_log, "w") as f:
                json.dump(flight_log, f, indent=2)
            print(f"飞行日志已保存: {args.save_log}")

        print(f"\n飞行完成! 共执行 {step} 步, 推理 {infer_count} 次")


if __name__ == "__main__":
    main()
