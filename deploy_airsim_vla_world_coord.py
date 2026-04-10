#!/usr/bin/env python3
"""
AirSim VLA 部署推理脚本 - 世界坐标转换版本
============================================
解决推理延迟导致的状态不一致问题：
1. 推理时记录当前位置和朝向
2. 将预测的自车坐标waypoints转换为世界坐标存储
3. 执行时将世界坐标waypoints重新转换回当前自车坐标

关键修改：
- waypoints_buffer存储世界坐标而非自车坐标
- 每次执行时动态转换到当前自车坐标
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
CONTROL_HZ = 2          # 控制频率 (Hz)
REPLAN_INTERVAL = 3.0   # 重新推理间隔 (秒)
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
    """加载训练好的 VLA模型

    Args:
        model_path: 模型路径
        device: 设备
        use_int8: 是否使用INT8量化（默认False）
    """
    import sys
    import os
    evo_root = os.path.join(os.path.dirname(__file__), "..", "EvoDriveVLA")
    if evo_root not in sys.path:
        sys.path.insert(0, os.path.abspath(evo_root))

    from model.language_models import Qwen2_5_VLForConditionalGeneration

    print(f"加载模型: {model_path}")

    # 构建加载参数
    load_kwargs = {
        "config": AutoConfig.from_pretrained(model_path),
        "attn_implementation": "flash_attention_2",
        "device_map": device,
    }

    if use_int8:
        print("启用 INT8 量化...")
        print("注意: Qwen-VL 模型的INT8支持可能不稳定，如遇错误请改用 --use_compile")
        try:
            from transformers import BitsAndBytesConfig
            # 更保守的量化配置：只量化语言模型部分，跳过所有视觉相关模块
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_skip_modules=[
                    "vision_tower",
                    "visual",
                    "merger",
                    "image_newline",
                    "rotary_emb",
                ],
                bnb_8bit_compute_dtype=torch.bfloat16,
            )
            load_kwargs["quantization_config"] = quantization_config
            load_kwargs["torch_dtype"] = torch.bfloat16  # 保持这个以兼容某些层
        except ImportError:
            print("警告: 未安装 bitsandbytes，回退到BF16模式")
            print("安装命令: pip install bitsandbytes accelerate")
            use_int8 = False
            load_kwargs["torch_dtype"] = torch.bfloat16
        except Exception as e:
            print(f"INT8配置出错: {e}，回退到BF16模式")
            use_int8 = False
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
    
    # 可选: torch.compile 加速 (PyTorch 2.0+)
    try:
        if hasattr(torch, 'compile') and not use_int8:
            print("启用 torch.compile 优化...")
            model = torch.compile(model, mode="reduce-overhead")
            print("torch.compile 启用成功")
    except Exception as e:
        print(f"torch.compile 启用失败: {e}")
    if use_int8:
        print("INT8模式: 显存减半，速度提升1.5-2x")
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
    """
    将自车坐标waypoints批量转换为世界坐标

    Args:
        waypoints_ego: 自车坐标下的waypoints [(x1,y1,z1), (x2,y2,z2), ...]
        ref_pos: 参考位置 (ref_x, ref_y, ref_z) - 推理时的无人机位置
        ref_yaw_deg: 参考朝向 (度) - 推理时的无人机朝向

    Returns:
        waypoints_world: 世界坐标下的waypoints
    """
    ref_x, ref_y, ref_z = ref_pos
    yaw_rad = math.radians(ref_yaw_deg)

    waypoints_world = []
    for local_x, local_y, local_z in waypoints_ego:
        world_x, world_y, world_z = ego_to_world(
            local_x, local_y, local_z, ref_x, ref_y, ref_z, yaw_rad
        )
        waypoints_world.append((world_x, world_y, world_z))

    return waypoints_world


def waypoints_world_to_ego(waypoints_world: List[Tuple[float, float, float]],
                           current_pos: Tuple[float, float, float],
                           current_yaw_deg: float) -> List[Tuple[float, float, float]]:
    """
    将世界坐标waypoints批量转换为当前自车坐标

    Args:
        waypoints_world: 世界坐标下的waypoints
        current_pos: 当前位置 (cur_x, cur_y, cur_z)
        current_yaw_deg: 当前朝向 (度)

    Returns:
        waypoints_ego: 当前自车坐标下的waypoints
    """
    cur_x, cur_y, cur_z = current_pos
    yaw_rad = math.radians(current_yaw_deg)

    waypoints_ego = []
    for world_x, world_y, world_z in waypoints_world:
        local_x, local_y, local_z = world_to_ego(
            world_x, world_y, world_z, cur_x, cur_y, cur_z, yaw_rad
        )
        waypoints_ego.append((local_x, local_y, local_z))

    return waypoints_ego


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
    parser = argparse.ArgumentParser(description="AirSim VLA 部署推理 - 世界坐标版本")
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
    parser.add_argument("--use_int8", action="store_true",
                        help="启用INT8量化，可减少显存占用并加速推理（需要安装bitsandbytes）")
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
    client.moveToPositionAsync(100, 200, altitude, 5.0, timeout_sec=60,
                               vehicle_name=VEHICLE_NAME).join()
    print("起飞完成")
    time.sleep(2.0)

    # 初始化历史缓存
    state_history = deque(maxlen=20)
    traj_history = deque(maxlen=HISTORY_LEN)
    ego_history = deque(maxlen=4)

    # ========== 世界坐标版本关键变量 ==========
    # waypoints_buffer 存储世界坐标 [(wx1,wy1,wz1), (wx2,wy2,wz2), ...]
    waypoints_buffer_world = []
    buffer_idx = 0
    last_infer_time = 0
    infer_count = 0

    # 记录推理时的参考位置和朝向
    ref_position = None  # (x, y, z) 推理时的位置
    ref_yaw = None       # 推理时的朝向(度)
    # ==========================================

    mission_goal = determine_mission_goal(args.instruction)
    print(f"Instruction: {args.instruction}")
    print(f"Mission Goal: {mission_goal}")
    print(f"控制频率: {CONTROL_HZ} Hz")
    print(f"重新推理间隔: {REPLAN_INTERVAL}s")
    print(f"开始自主飞行 (最多 {args.max_steps} 步)...\n")

    flight_log = []
    control_interval = 1.0 / CONTROL_HZ
    last_traj_update = 0

    for step in range(args.max_steps):
        step_start = time.time()

        # 1. 获取当前状态
        current = get_drone_state(client)
        state_history.append(current)

        # 更新历史轨迹
        if time.time() - last_traj_update >= HISTORY_INTERVAL:
            traj_history.append(current)
            ego_history.append(compute_ego_state(current))
            last_traj_update = time.time()

        # ========== 2. 判断是否需要重新推理 ==========
        need_replan = (
            len(waypoints_buffer_world) == 0 or
            buffer_idx >= len(waypoints_buffer_world) or
            time.time() - last_infer_time >= REPLAN_INTERVAL
        )

        if need_replan:
            # 采集图像
            images = capture_images(client)
            if len(images) < 4:
                print(f"  步骤 {step}: 相机采集不完整，使用旧waypoints")
            else:
                # 记录推理时的参考位置和朝向！
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

                # 解析自车坐标waypoints
                waypoints_ego = parse_waypoints(output_text)

                if len(waypoints_ego) >= 6:
                    # 关键：将自车坐标转换为世界坐标存储！
                    waypoints_buffer_world = waypoints_ego_to_world(
                        waypoints_ego[:6], ref_position, ref_yaw
                    )
                    buffer_idx = 0
                    last_infer_time = time.time()
                    infer_count += 1

                    print(f"\n[推理 #{infer_count}] 生成 {len(waypoints_ego)} waypoints, 用时 {infer_time:.2f}s")
                    print(f"  参考位置: ({ref_position[0]:.1f}, {ref_position[1]:.1f}, {ref_position[2]:.1f}), "
                          f"朝向: {ref_yaw:.1f}°")
                    print(f"  世界坐标waypoints: {waypoints_buffer_world}")
                else:
                    print(f"  步骤 {step}: 解析waypoints不足 ({len(waypoints_ego)}个): {output_text[:100]}")
        # ==========================================

        # 3. 使用buffer中的当前waypoint
        if buffer_idx < len(waypoints_buffer_world):
            # 关键：将世界坐标转换回当前自车坐标！
            current_pos = (current.pos_x, current.pos_y, current.pos_z)
            current_yaw = current.yaw

            # 转换整个buffer到当前自车坐标
            waypoints_ego_current = waypoints_world_to_ego(
                waypoints_buffer_world, current_pos, current_yaw
            )

            # 取当前要执行的waypoint
            local_x, local_y, local_z = waypoints_ego_current[buffer_idx]
            current_wp_idx = buffer_idx
            buffer_idx += 1

            # 转换到世界坐标用于执行（AirSim需要世界坐标）
            target_x, target_y, target_z = ego_to_world(
                local_x, local_y, local_z,
                current.pos_x, current.pos_y, current.pos_z,
                math.radians(current.yaw)
            )

            # 发送运动指令
            client.moveToPositionAsync(
                target_x, target_y, target_z, args.speed,
                timeout_sec=0.6,
                drivetrain=airsim.DrivetrainType.ForwardOnly,
                yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=0),
                vehicle_name=VEHICLE_NAME,
            )

            # 打印调试信息
            if step % 10 == 0:
                print(f"  步骤 {step}: 执行WP[{current_wp_idx+1}/6] "
                      f"自车坐标=({local_x:.2f}, {local_y:.2f}, {local_z:.2f}), "
                      f"世界坐标=({target_x:.1f}, {target_y:.1f}, {target_z:.1f})")
        else:
            # 没有可用waypoint，悬停
            target_x, target_y, target_z = current.pos_x, current.pos_y, current.pos_z
            current_wp_idx = -1
            local_x = local_y = local_z = 0.0

        # 4. 日志
        log_entry = {
            "step": step,
            "pos": [current.pos_x, current.pos_y, current.pos_z],
            "target": [target_x, target_y, target_z],
            "waypoints_world": waypoints_buffer_world.copy() if waypoints_buffer_world else [],
            "ref_position": ref_position if ref_position else None,
            "ref_yaw": ref_yaw if ref_yaw else None,
            "current_wp_idx": current_wp_idx,
            "infer_count": infer_count,
        }
        flight_log.append(log_entry)

        # 5. 可视化
        if args.visualize and "CAM_FRONT" in images:
            front_cv = cv2.cvtColor(np.array(images["CAM_FRONT"]), cv2.COLOR_RGB2BGR)
            info = f"Step {step} | WP[{current_wp_idx+1}/6]"
            cv2.putText(front_cv, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if args.save_image_dir:
                os.makedirs(args.save_image_dir, exist_ok=True)
                cv2.imwrite(os.path.join(args.save_image_dir, f"vla_step_{step:03d}.jpg"), front_cv)

        # 6. 控制频率
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


if __name__ == "__main__":
    main()
