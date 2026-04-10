#!/usr/bin/env python3
"""
AirSim VLA 部署推理脚本 - Path-based版本
========================================
使用 moveOnPathAsync 实现平滑轨迹跟踪：
1. VLA 推理输出 6 个未来 waypoints (3秒，0.5s间隔)
2. 转换为世界坐标路径
3. 计算路径长度，设置速度使得飞行时间≈6秒
4. 使用 moveOnPathAsync 平滑跟踪路径
5. 路径完成后或每 6 秒重新推理

与原版区别：
- 原版：逐点 moveToPositionAsync，高频控制
- 本版：moveOnPathAsync 整体路径跟踪，低频推理
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

# 推理参数
REPLAN_INTERVAL = 6.0   # 重新推理间隔 (秒)，与路径飞行时间匹配
NUM_WAYPOINTS = 6       # VLA输出的waypoints数量
WAYPOINT_INTERVAL = 0.5 # waypoints时间间隔 (秒)

# 历史轨迹
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


# ==================== 模型加载 ====================

def load_model(model_path: str, device: str = "cuda:0", use_int8: bool = False):
    """加载训练好的 VLA模型"""
    import sys
    import os
    evo_root = os.path.join(os.path.dirname(__file__), "..", "EvoDriveVLA")
    if evo_root not in sys.path:
        sys.path.insert(0, os.path.abspath(evo_root))

    from model.language_models import Qwen2_5_VLForConditionalGeneration

    print(f"加载模型: {model_path}")

    load_kwargs = {
        "config": AutoConfig.from_pretrained(model_path),
        "attn_implementation": "flash_attention_2",
        "device_map": device,
    }

    if use_int8:
        print("启用 INT8 量化...")
        try:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_skip_modules=[
                    "vision_tower", "visual", "merger", "image_newline", "rotary_emb",
                ],
                bnb_8bit_compute_dtype=torch.bfloat16,
            )
            load_kwargs["quantization_config"] = quantization_config
            load_kwargs["torch_dtype"] = torch.bfloat16
        except ImportError:
            print("警告: 未安装 bitsandbytes，回退到BF16模式")
            load_kwargs["torch_dtype"] = torch.bfloat16
    else:
        load_kwargs["torch_dtype"] = torch.bfloat16

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        **load_kwargs
    ).eval()

    processor = AutoProcessor.from_pretrained(
        model_path, use_fast=True,
        min_pixels=28 * 28 * 128,
        max_pixels=28 * 28 * 256,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, padding_side="right", use_fast=False,
    )

    mode_str = '(INT8量化)' if use_int8 else '(BF16)'
    print(f"模型加载完成 {mode_str}")

    try:
        if hasattr(torch, 'compile') and not use_int8:
            print("启用 torch.compile 优化...")
            model = torch.compile(model, mode="reduce-overhead")
            print("torch.compile 启用成功")
    except Exception as e:
        print(f"torch.compile 启用失败: {e}")

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


# ==================== 坐标转换 ====================

def world_to_ego(pos_x, pos_y, pos_z, ref_x, ref_y, ref_z, yaw_rad):
    """世界坐标 → 自车坐标 (X=右, Y=前, Z=上)"""
    dx, dy = pos_x - ref_x, pos_y - ref_y
    cos_yaw = math.cos(-yaw_rad)
    sin_yaw = math.sin(-yaw_rad)
    local_x = dx * cos_yaw - dy * sin_yaw
    local_y = dx * sin_yaw + dy * cos_yaw
    local_z = -(pos_z - ref_z)
    return local_x, local_y, local_z


def ego_to_world(local_x, local_y, local_z, ref_x, ref_y, ref_z, yaw_rad):
    """自车坐标 → 世界坐标 (NED: X=北, Y=东, Z=下)"""
    cos_yaw = math.cos(yaw_rad)
    sin_yaw = math.sin(yaw_rad)
    dx = local_x * cos_yaw - local_y * sin_yaw
    dy = local_x * sin_yaw + local_y * cos_yaw
    world_x = ref_x + dx
    world_y = ref_y + dy
    world_z = ref_z - local_z
    return world_x, world_y, world_z


def waypoints_ego_to_world(waypoints_ego: List[Tuple[float, float, float]],
                           ref_pos: Tuple[float, float, float],
                           ref_yaw_deg: float) -> List[Tuple[float, float, float]]:
    """将自车坐标waypoints批量转换为世界坐标"""
    ref_x, ref_y, ref_z = ref_pos
    yaw_rad = math.radians(ref_yaw_deg)

    waypoints_world = []
    for local_x, local_y, local_z in waypoints_ego:
        world_x, world_y, world_z = ego_to_world(
            local_x, local_y, local_z, ref_x, ref_y, ref_z, yaw_rad
        )
        waypoints_world.append((world_x, world_y, world_z))

    return waypoints_world


def calculate_path_length(path: List[airsim.Vector3r]) -> float:
    """计算路径总长度"""
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
            max_new_tokens=256,
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
    parser = argparse.ArgumentParser(description="AirSim VLA 部署推理 - Path-based版本")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--instruction", type=str,
                        default="Fly a circle clockwise with radius 50m at 200m altitude with 8m/s speed")
    parser.add_argument("--takeoff_altitude", type=float, default=200)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--base_speed", type=float, default=6.0, help="基础速度，实际速度会根据路径长度调整")
    parser.add_argument("--replan_interval", type=float, default=6.0, help="重新推理间隔(秒)")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--save_log", type=str, default=None)
    parser.add_argument("--save_image_dir", type=str, default=None)
    parser.add_argument("--use_int8", action="store_true")
    args = parser.parse_args()

    # 加载模型
    model, processor, tokenizer = load_model(args.model_path, args.device, use_int8=args.use_int8)

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

    # Path-based控制关键变量
    current_path = []  # 当前执行的路径 (Vector3r列表)
    path_start_time = 0  # 路径开始时间
    path_following = False  # 是否正在执行路径

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

            # 更新历史轨迹
            if time.time() - last_traj_update >= HISTORY_INTERVAL:
                traj_history.append(current)
                ego_history.append(compute_ego_state(current))
                last_traj_update = time.time()

            # 2. 判断是否需要重新推理
            need_replan = (
                not path_following or  # 没有在执行路径
                time.time() - last_infer_time >= args.replan_interval  # 超时
            )

            if need_replan:
                print(f"\n{'='*60}")
                print(f"[推理 #{infer_count + 1}] 获取新路径...")

                # 取消当前路径（如果有）
                if path_following:
                    client.cancelLastTask(vehicle_name=VEHICLE_NAME)
                    print("取消上一路径")

                # 采集图像
                images = capture_images(client)
                if len(images) < 4:
                    print("相机采集不完整，跳过本次推理")
                    time.sleep(0.1)
                    continue

                # 记录推理参考位置
                ref_position = (current.pos_x, current.pos_y, current.pos_z)
                ref_yaw = current.yaw

                # 构建prompt并推理
                user_content = build_user_content(
                    args.instruction, traj_history, ego_history, current, mission_goal
                )

                t0 = time.time()
                output_text = run_inference(model, processor, tokenizer,
                                            images, user_content, args.device)
                infer_time = time.time() - t0

                # 解析waypoints
                waypoints_ego = parse_waypoints(output_text)

                if len(waypoints_ego) >= NUM_WAYPOINTS:
                    # 转换到世界坐标
                    waypoints_world = waypoints_ego_to_world(
                        waypoints_ego[:NUM_WAYPOINTS], ref_position, ref_yaw
                    )

                    # 构建路径 (添加当前位置作为起点)
                    path = [airsim.Vector3r(current.pos_x, current.pos_y, current.pos_z)]
                    for wp in waypoints_world:
                        path.append(airsim.Vector3r(wp[0], wp[1], wp[2]))

                    # 计算路径长度
                    path_length = calculate_path_length(path)

                    # 计算速度使得飞行时间≈replan_interval
                    if path_length > 0.1:
                        target_speed = path_length / args.replan_interval
                        # 限制速度范围
                        target_speed = max(2.0, min(target_speed, 15.0))
                    else:
                        target_speed = args.base_speed

                    print(f"生成路径: {len(path)} 个点, 长度: {path_length:.2f}m")
                    print(f"目标速度: {target_speed:.2f}m/s (飞行时间≈{args.replan_interval}s)")
                    print(f"Waypoints: {waypoints_world}")

                    # 发送路径跟踪指令
                    client.moveOnPathAsync(
                        path,
                        velocity=target_speed,
                        timeout_sec=args.replan_interval + 5,  # 稍长超时
                        drivetrain=airsim.DrivetrainType.ForwardOnly,
                        yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=0),
                        vehicle_name=VEHICLE_NAME
                    )

                    # 更新状态
                    current_path = path
                    path_following = True
                    path_start_time = time.time()
                    last_infer_time = time.time()
                    infer_count += 1

                    # 记录日志
                    log_entry = {
                        "step": step,
                        "infer_count": infer_count,
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
                    print(f"解析waypoints不足 ({len(waypoints_ego)}个): {output_text[:100]}")

            # 3. 检查路径是否完成或超时
            if path_following:
                elapsed = time.time() - path_start_time

                # 获取当前状态打印
                if step % 20 == 0:
                    pos = client.getMultirotorState(vehicle_name=VEHICLE_NAME).kinematics_estimated.position
                    print(f"  [Step {step}] 位置: ({pos.x_val:.1f}, {pos.y_val:.1f}, {-pos.z_val:.1f}), "
                          f"路径耗时: {elapsed:.1f}s")

                # 如果路径执行时间超过replan_interval，标记为完成
                if elapsed >= args.replan_interval:
                    path_following = False
                    print(f"  路径执行完成 (耗时: {elapsed:.1f}s)")

            # 4. 可视化
            if args.visualize and step % 5 == 0:
                images = capture_images(client)
                if "CAM_FRONT" in images and args.save_image_dir:
                    front_cv = cv2.cvtColor(np.array(images["CAM_FRONT"]), cv2.COLOR_RGB2BGR)
                    os.makedirs(args.save_image_dir, exist_ok=True)
                    cv2.imwrite(os.path.join(args.save_image_dir, f"step_{step:04d}.jpg"), front_cv)

            step += 1

            # 控制循环频率（不需要太高，因为路径跟踪是异步的）
            elapsed = time.time() - loop_start
            if elapsed < 0.05:  # 20Hz 足够
                time.sleep(0.05 - elapsed)

    except KeyboardInterrupt:
        print("\n用户中断")

    finally:
        # 清理
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
