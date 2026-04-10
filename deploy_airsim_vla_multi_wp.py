#!/usr/bin/env python3
"""
AirSim VLA 部署推理脚本 - 多Waypoints跟踪版本
==============================
一次推理生成6个waypoints，依次跟踪执行，3秒后重新推理

关键修改：
- CONTROL_HZ = 2 (2Hz控制，每0.5s一个waypoint)
- waypoints_buffer 缓存6个预测点
- 每3秒或buffer用完时重新推理
"""

import argparse
import math
import os
import re
import time
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Optional

import airsim
import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoConfig, AutoTokenizer
import copy

# ==================== 配置 ====================

VEHICLE_NAME = "UAV0"
CAMERAS = ["front", "left", "right", "down"]
CAM_TO_CHANNEL = {
    "front": "CAM_FRONT",
    "left": "CAM_LEFT",
    "right": "CAM_RIGHT",
    "down": "CAM_DOWN",
}
CONTROL_HZ = 1          # 控制频率 (Hz)，每0.5s执行一个waypoint
REPLAN_INTERVAL = 3.0   # 重新推理间隔 (秒)，6个点 × 0.5s = 3s
HISTORY_LEN = 5         # 历史轨迹点数
HISTORY_INTERVAL = 0.4  # 历史点间隔 (秒)

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


# ==================== 模型加载 ====================

def load_model(model_path: str, device: str = "cuda:0"):
    """加载训练好的 VLA模型"""
    import sys
    import os
    evo_root = os.path.join(os.path.dirname(__file__), "..", "EvoDriveVLA")
    if evo_root not in sys.path:
        sys.path.insert(0, os.path.abspath(evo_root))

    from model.language_models import Qwen2_5_VLForConditionalGeneration

    print(f"加载模型: {model_path}")
    config = AutoConfig.from_pretrained(model_path)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=device,
    ).eval()

    processor = AutoProcessor.from_pretrained(
        model_path, use_fast=True,
        min_pixels=28 * 28 * 128,
        max_pixels=28 * 28 * 256,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, padding_side="right", use_fast=False,
    )

    print("模型加载完成")
    return model, processor, tokenizer


# ==================== AirSim 交互 ====================

def get_drone_state(client: airsim.MultirotorClient) -> DroneState:
    """获取无人机当前状态"""
    state = client.getMultirotorState(vehicle_name=VEHICLE_NAME)
    pos = state.kinematics_estimated.position
    vel = state.kinematics_estimated.linear_velocity
    ori = state.kinematics_estimated.orientation
    pitch, roll, yaw = airsim.to_eularian_angles(ori)
    imu = client.getImuData(vehicle_name=VEHICLE_NAME)

    return DroneState(
        pos_x=pos.x_val, pos_y=pos.y_val, pos_z=pos.z_val,
        vel_x=vel.x_val, vel_y=vel.y_val, vel_z=vel.z_val,
        roll=math.degrees(roll), pitch=math.degrees(pitch), yaw=math.degrees(yaw),
        imu_acc_x=imu.linear_acceleration.x_val,
        imu_acc_y=imu.linear_acceleration.y_val,
        imu_ang_z=imu.angular_velocity.z_val,
        timestamp=time.time(),
    )


def capture_images(client: airsim.MultirotorClient) -> dict:
    """采集4路相机图像"""
    requests = [
        airsim.ImageRequest(cam, airsim.ImageType.Scene, False, False)
        for cam in CAMERAS
    ]
    responses = client.simGetImages(requests, vehicle_name=VEHICLE_NAME)

    images = {}
    for cam, resp in zip(CAMERAS, responses):
        if resp.width > 0 and len(resp.image_data_uint8) > 0:
            img = np.frombuffer(resp.image_data_uint8, dtype=np.uint8)
            img = img.reshape(resp.height, resp.width, 3)
            pil_img = Image.fromarray(img)
            images[CAM_TO_CHANNEL[cam]] = pil_img
    return images


# ==================== Prompt 构建 ====================

def compute_ego_state(state: DroneState) -> dict:
    vel = math.sqrt(state.vel_x**2 + state.vel_y**2 + state.vel_z**2)
    return {
        "vel": round(vel, 2),
        "acc_x": round(state.imu_acc_x, 2),
        "acc_y": round(state.imu_acc_y, 2),
        "steer": round(state.imu_ang_z, 2),
    }


def world_to_ego(pos_x, pos_y, pos_z, ref_x, ref_y, ref_z, yaw_rad):
    """世界坐标 → 自车坐标"""
    dx, dy = pos_x - ref_x, pos_y - ref_y
    cos_yaw = math.cos(-yaw_rad)
    sin_yaw = math.sin(-yaw_rad)
    local_x = dx * cos_yaw - dy * sin_yaw
    local_y = dx * sin_yaw + dy * cos_yaw
    local_z = -(pos_z - ref_z)
    return local_x, local_y, local_z


def ego_to_world(local_x, local_y, local_z, ref_x, ref_y, ref_z, yaw_rad):
    """自车坐标 → 世界坐标"""
    cos_yaw = math.cos(yaw_rad)
    sin_yaw = math.sin(yaw_rad)
    dx = local_x * cos_yaw - local_y * sin_yaw
    dy = local_x * sin_yaw + local_y * cos_yaw
    world_x = ref_x + dx
    world_y = ref_y + dy
    world_z = ref_z - local_z
    return world_x, world_y, world_z


def build_history_traj_text(history: deque, current: DroneState) -> str:
    """构建历史轨迹文本"""
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
    """构建历史ego状态文本"""
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


def build_user_content(instruction: str, history: deque, ego_history: deque,
                       current: DroneState, mission_goal: str = "FORWARD") -> str:
    """构建完整的user message content"""
    current_ego = compute_ego_state(current)

    content = "Here are current four images from the drone: "
    content += "'CAM_FRONT': <image>\n,'CAM_LEFT': <image>\n, 'CAM_RIGHT': <image>\n,'CAM_DOWN': <image>\n"
    content += "Here's some information you'll need:\n"

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


# ==================== 模型推理 ====================

def run_inference(model, processor, tokenizer, images: dict,
                  user_content: str, device: str = "cuda:0") -> str:
    """运行模型推理"""

    image_list = [images["CAM_FRONT"], images["CAM_LEFT"],
                  images["CAM_RIGHT"], images["CAM_DOWN"]]
    pixel_values_list = []
    grid_thw_list = []

    for img in image_list:
        img_processor = copy.deepcopy(processor.image_processor)
        img_processor.size["longest_edge"] = img_processor.max_pixels
        img_processor.size["shortest_edge"] = img_processor.min_pixels
        visual_processed = img_processor.preprocess(img, return_tensors="pt")
        image_tensor = visual_processed["pixel_values"]
        if isinstance(image_tensor, list):
            image_tensor = image_tensor[0]
        pixel_values_list.append(image_tensor)
        grid_thw_list.append(visual_processed["image_grid_thw"][0])

    pixel_values = torch.cat(pixel_values_list, dim=0).to(device)
    image_grid_thw = torch.cat([g.unsqueeze(0) for g in grid_thw_list], dim=0)

    merge_size = processor.image_processor.merge_size
    grid_thw_merged = [
        int(g.prod().item() // (merge_size ** 2)) for g in grid_thw_list
    ]

    content_processed = user_content
    visual_replicate_index = 0
    parts = content_processed.split("<image>")
    new_parts = []
    for i in range(len(parts) - 1):
        new_parts.append(parts[i])
        replacement = (
            "<|vision_start|>"
            + "<|image_pad|>" * grid_thw_merged[visual_replicate_index]
            + "<|vision_end|>"
        )
        new_parts.append(replacement)
        visual_replicate_index += 1
    new_parts.append(parts[-1])
    content_processed = "".join(new_parts)

    tokenizer.chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content_processed},
    ]

    input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    add_id = tokenizer.apply_chat_template(
        [{"role": "assistant", "content": ""}],
        add_generation_prompt=True,
        tokenize=True,
    )
    input_ids = input_ids + add_id
    inputs = {
        "input_ids": torch.tensor([input_ids], dtype=torch.long, device=device),
        "attention_mask": torch.tensor([[1] * len(input_ids)], dtype=torch.long, device=device),
    }

    inputs["pixel_values"] = pixel_values.to(device)
    inputs["image_grid_thw"] = image_grid_thw.to(device)

    for k, v in inputs.items():
        if isinstance(v, torch.Tensor) and v.is_floating_point():
            inputs[k] = v.to(torch.bfloat16)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=256,  # 减少到256足够6个waypoints
            do_sample=False,
            num_beams=1,
        )

    trimmed_ids = generated_ids[0][inputs["input_ids"].shape[1]:]
    output_text = tokenizer.decode(trimmed_ids, skip_special_tokens=True)
    return output_text


def parse_waypoints(text: str) -> List[Tuple[float, float, float]]:
    """解析模型输出的3D waypoints"""
    pattern = r'\(([+-]?\d+\.?\d*)\s*,\s*([+-]?\d+\.?\d*)\s*,\s*([+-]?\d+\.?\d*)\)'
    matches = re.findall(pattern, text)
    waypoints = [(float(x), float(y), float(z)) for x, y, z in matches]
    return waypoints


def determine_mission_goal(instruction: str) -> str:
    """从instruction中提取mission goal"""
    inst_lower = instruction.lower()
    if "clockwise" in inst_lower and "counter" not in inst_lower:
        return "RIGHT"
    elif "counter-clockwise" in inst_lower or "counter clockwise" in inst_lower:
        return "LEFT"
    return "FORWARD"


# ==================== 主函数 ====================

def main():
    parser = argparse.ArgumentParser(description="AirSim VLA 部署推理 - 多Waypoints版本")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--instruction", type=str,
                        default="Fly a circle clockwise with radius 50m at 200m altitude with 8m/s speed")
    parser.add_argument("--takeoff_altitude", type=float, default=200)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--speed", type=float, default=6.3)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--save_log", type=str, default=None)
    parser.add_argument("--save_image_dir", type=str, default=None)
    args = parser.parse_args()

    # 加载模型
    model, processor, tokenizer = load_model(args.model_path, args.device)

    # 连接AirSim
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
    client.moveToPositionAsync(0, 0, altitude, 5.0, timeout_sec=60,
                               vehicle_name=VEHICLE_NAME).join()
    print("起飞完成")
    time.sleep(2.0)

    # 初始化历史缓存
    state_history = deque(maxlen=20)
    traj_history = deque(maxlen=HISTORY_LEN)
    ego_history = deque(maxlen=4)

    # ========== 多Waypoints跟踪关键变量 ==========
    waypoints_buffer = []   # 缓存6个waypoints
    buffer_idx = 0          # 当前执行到第几个
    last_infer_time = 0     # 上次推理时间戳
    infer_count = 0         # 推理次数统计
    # ==========================================

    mission_goal = determine_mission_goal(args.instruction)
    print(f"Instruction: {args.instruction}")
    print(f"Mission Goal: {mission_goal}")
    print(f"控制频率: {CONTROL_HZ} Hz (每0.5s一个waypoint)")
    print(f"重新推理间隔: {REPLAN_INTERVAL}s")
    print(f"开始自主飞行 (最多 {args.max_steps} 步)...\n")

    flight_log = []
    control_interval = 1.0 / CONTROL_HZ  # 0.5s
    last_traj_update = 0

    for step in range(args.max_steps):
        step_start = time.time()

        # 1. 获取状态
        current = get_drone_state(client)
        state_history.append(current)

        # 更新历史轨迹
        if time.time() - last_traj_update >= HISTORY_INTERVAL:
            traj_history.append(current)
            ego_history.append(compute_ego_state(current))
            last_traj_update = time.time()

        # ========== 2. 判断是否需要重新推理 ==========
        need_replan = (
            len(waypoints_buffer) == 0 or           # 首次
            buffer_idx >= len(waypoints_buffer) or  # buffer用完
            time.time() - last_infer_time >= REPLAN_INTERVAL  # 超时
        )

        if need_replan:
            # 采集图像
            images = capture_images(client)
            if len(images) < 4:
                print(f"  步骤 {step}: 相机采集不完整，使用旧waypoints")
            else:
                # 构建prompt并推理
                user_content = build_user_content(
                    args.instruction, traj_history, ego_history, current, mission_goal
                )

                t0 = time.time()
                output_text = run_inference(model, processor, tokenizer,
                                            images, user_content, args.device)
                infer_time = time.time() - t0

                # 解析waypoints
                waypoints = parse_waypoints(output_text)

                if len(waypoints) >= 6:
                    waypoints_buffer = waypoints[:6]  # 取前6个
                    buffer_idx = 0
                    last_infer_time = time.time()
                    infer_count += 1

                    print(f"\n[推理 #{infer_count}] 生成 {len(waypoints)} waypoints, "
                          f"用时 {infer_time:.2f}s, 跟踪3秒")
                    print(f"  Waypoints: {waypoints_buffer}")
                else:
                    print(f"  步骤 {step}: 解析waypoints不足 ({len(waypoints)}个): {output_text[:100]}")
        # ==========================================

        # 3. 使用buffer中的当前waypoint
        if buffer_idx < len(waypoints_buffer):
            local_x, local_y, local_z = waypoints_buffer[buffer_idx]
            current_wp_idx = buffer_idx  # 记录用于显示
            buffer_idx += 1

            # 转换到世界坐标
            yaw_rad = math.radians(current.yaw)
            target_x, target_y, target_z = ego_to_world(
                local_x, local_y, local_z,
                current.pos_x, current.pos_y, current.pos_z, yaw_rad
            )

            # 发送运动指令 (给0.6s执行时间，留0.1s余量)
            client.moveToPositionAsync(
                target_x, target_y, target_z, args.speed,
                timeout_sec=0.6,
                drivetrain=airsim.DrivetrainType.ForwardOnly,
                yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=0),
                vehicle_name=VEHICLE_NAME,
            )
        else:
            # 没有可用waypoint，悬停等待
            target_x, target_y, target_z = current.pos_x, current.pos_y, current.pos_z
            current_wp_idx = -1
            local_x = local_y = local_z = 0.0

        # 4. 日志
        log_entry = {
            "step": step,
            "pos": [current.pos_x, current.pos_y, current.pos_z],
            "target": [target_x, target_y, target_z],
            "waypoints_ego": waypoints_buffer.copy() if waypoints_buffer else [],
            "current_wp_idx": current_wp_idx,
            "infer_count": infer_count,
        }
        flight_log.append(log_entry)

        if step % 1 == 0:  # 每1秒打印一次
            wp_status = f"WP[{current_wp_idx+1}/6]=({local_x:.1f},{local_y:.1f},{local_z:.1f})" if current_wp_idx >= 0 else "WAITING"
            print(f"步骤 {step:03d} | "
                  f"位置 ({current.pos_x:.1f}, {current.pos_y:.1f}, {-current.pos_z:.1f}m) | "
                  f"{wp_status}")

        # 5. 可视化
        if args.visualize and "CAM_FRONT" in images:
            front_cv = cv2.cvtColor(np.array(images["CAM_FRONT"]), cv2.COLOR_RGB2BGR)
            info = f"Step {step} | {wp_status}"
            cv2.putText(front_cv, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if args.save_image_dir:
                os.makedirs(args.save_image_dir, exist_ok=True)
                cv2.imwrite(os.path.join(args.save_image_dir, f"vla_step_{step:03d}.jpg"), front_cv)

        # 6. 控制频率 (0.5s间隔)
        elapsed = time.time() - step_start
        if elapsed < control_interval:
            time.sleep(control_interval - elapsed)

    # 降落
    print("\n降落...")
    client.cancelLastTask(vehicle_name=VEHICLE_NAME)
    client.landAsync(timeout_sec=5, vehicle_name=VEHICLE_NAME).join()
    client.armDisarm(False, vehicle_name=VEHICLE_NAME)
    client.enableApiControl(False, vehicle_name=VEHICLE_NAME)

    if args.visualize:
        cv2.destroyAllWindows()

    # 保存日志
    if args.save_log:
        import json
        with open(args.save_log, "w") as f:
            json.dump(flight_log, f, indent=2)
        print(f"飞行日志已保存: {args.save_log}")

    print(f"\n飞行完成! 共执行 {len(flight_log)} 步, 推理 {infer_count} 次")
    print(f"推理频率: {infer_count / (len(flight_log) * control_interval) * 60:.1f} 次/分钟")


if __name__ == "__main__":
    main()
