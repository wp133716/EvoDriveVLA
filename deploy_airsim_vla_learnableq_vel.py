#!/usr/bin/env python3
"""
AirSim VLA 部署推理脚本 - LearnableQ 速度控制版本
===================================================
在 deploy_airsim_vla_learnableq.py 的基础上，将飞机控制接口从
moveOnPathAsync（路径跟踪）改为 moveByVelocityAsync（速度控制）。

控制逻辑：
  1. 模型预测 6 个 ego 坐标 waypoints（0.5s 间隔）
  2. 将相邻 waypoint 差除以 dt=0.5s 得到速度向量序列
  3. 主循环每隔 0.5s 发出一个 moveByVelocityAsync 指令（duration=0.5s）
  4. 6 段速度执行完毕（3s）或超过 replan_interval 后重新推理
"""

import argparse
import math
import os
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import airsim
import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoConfig, AutoTokenizer
import copy

# ==================== 路径设置 ====================

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_EVO_ROOT = os.path.join(_SCRIPT_DIR)
if _EVO_ROOT not in sys.path:
    sys.path.insert(0, _EVO_ROOT)

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
REPLAN_INTERVAL = 6.0    # 重新推理间隔 (秒)
NUM_WAYPOINTS = 6        # VLA 输出的 waypoints 数量
WAYPOINT_INTERVAL = 0.5  # waypoints 时间间隔 (秒)

# 历史轨迹（位置/ego状态）
HISTORY_LEN = 5
HISTORY_INTERVAL = 0.4

# 历史图像帧
IMAGE_HISTORY_FRAMES = 5
IMAGE_HISTORY_INTERVAL = 0.5

SYSTEM_PROMPT = (
    "You're an autonomous drone's brain. "
    "Coordinates: X-axis is perpendicular, Y-axis is parallel to the direction you're facing, "
    "and Z-axis points upward. You're at point (0,0,0). Units: meters. "
    "Based on the provided particulars, please output the 3D plan waypoints (0.5s intervals) for the next 3 seconds."
)


# ==================== 数据类 ====================

@dataclass
class CachedFrame:
    frame_id: int
    images: dict
    image_embeds: Optional[torch.Tensor] = field(default=None)
    grid_thw: Optional[torch.Tensor] = field(default=None)


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

def load_model(model_path: str, device: str = "cuda:0",
               use_lm_head: bool = False, use_int8: bool = False):
    from model.language_models.modeling_qwen2_5_vl_learnableq import Qwen2_5_VLForLearnableQ

    print(f"加载 LearnableQ 模型: {model_path}")
    print(f"输出模式: {'lm_head + CE' if use_lm_head else 'regression head + Smooth L1'}")

    load_kwargs = {
        "config": AutoConfig.from_pretrained(model_path),
        "attn_implementation": "flash_attention_2",
        "torch_dtype": torch.bfloat16,
        "device_map": device,
        "num_waypoints": NUM_WAYPOINTS,
        "waypoint_dim": 3,
        "use_lm_head": use_lm_head,
    }

    if use_int8:
        print("启用 INT4(NF4) 量化...")
        try:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        except ImportError:
            print("警告: 未安装 bitsandbytes，回退到 BF16 模式")

    model = Qwen2_5_VLForLearnableQ.from_pretrained(
        model_path, **load_kwargs
    ).eval()

    processor = AutoProcessor.from_pretrained(
        model_path, use_fast=True,
        min_pixels=28 * 28 * 128,
        max_pixels=28 * 28 * 256,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, padding_side="right", use_fast=False,
    )

    model.tokenizer = tokenizer

    mode_str = "(INT4-NF4 量化)" if use_int8 else "(BF16)"
    print(f"模型加载完成 {mode_str}")
    if not use_lm_head and hasattr(model, "waypoint_mean"):
        print(f"  waypoint_mean: {model.waypoint_mean.tolist()}")
        print(f"  waypoint_std:  {model.waypoint_std.tolist()}")

    return model, processor, tokenizer


# ==================== AirSim 交互 ====================

def get_drone_state(client: airsim.MultirotorClient) -> DroneState:
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
            images[CAM_TO_CHANNEL[cam]] = Image.fromarray(img)
    return images


_CAM_ORDER = ["CAM_FRONT", "CAM_LEFT", "CAM_RIGHT", "CAM_DOWN"]

def encode_frame_images(model, processor, frame_images: dict,
                        device: str = "cuda:0") -> Tuple[torch.Tensor, torch.Tensor]:
    pixel_values_list = []
    grid_thw_list = []

    for cam in _CAM_ORDER:
        img = frame_images[cam]
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
    grid_thw = torch.cat([g.unsqueeze(0) for g in grid_thw_list], dim=0).to(device)

    with torch.no_grad():
        image_embeds = model.visual(
            pixel_values.type(model.visual.dtype),
            grid_thw=grid_thw,
        )

    return image_embeds.cpu(), grid_thw.cpu()


# ==================== 坐标转换 ====================

def world_to_ego(pos_x, pos_y, pos_z, ref_x, ref_y, ref_z, yaw_rad):
    dx, dy = pos_x - ref_x, pos_y - ref_y
    cos_yaw = math.cos(-yaw_rad)
    sin_yaw = math.sin(-yaw_rad)
    local_x = -dx * sin_yaw + dy * cos_yaw
    local_y =  dx * cos_yaw + dy * sin_yaw
    local_z = -(pos_z - ref_z)
    return local_x, local_y, local_z


def ego_to_world(local_x, local_y, local_z, ref_x, ref_y, ref_z, yaw_rad):
    cos_yaw = math.cos(yaw_rad)
    sin_yaw = math.sin(yaw_rad)
    dx = -local_x * sin_yaw + local_y * cos_yaw
    dy =  local_x * cos_yaw + local_y * sin_yaw
    world_x = ref_x + dx
    world_y = ref_y + dy
    world_z = ref_z - local_z
    return world_x, world_y, world_z


def waypoints_ego_to_world(waypoints_ego: List[Tuple[float, float, float]],
                            ref_pos: Tuple[float, float, float],
                            ref_yaw_deg: float) -> List[Tuple[float, float, float]]:
    ref_x, ref_y, ref_z = ref_pos
    yaw_rad = math.radians(ref_yaw_deg)
    return [
        ego_to_world(lx, ly, lz, ref_x, ref_y, ref_z, yaw_rad)
        for lx, ly, lz in waypoints_ego
    ]


def compute_waypoint_velocities(
    actual_position: Tuple[float, float, float],
    waypoints_world: List[Tuple[float, float, float]],
    dt: float = WAYPOINT_INTERVAL,
    max_speed: float = 15.0,
) -> List[Tuple[float, float, float]]:
    """将 waypoints 转换为速度向量序列。
    每段速度 = 位移 / dt；超出 max_speed 时等比缩放。
    """
    velocities = []
    prev = actual_position
    for wp in waypoints_world:
        vx = (wp[0] - prev[0]) / dt
        vy = (wp[1] - prev[1]) / dt
        vz = (wp[2] - prev[2]) / dt
        speed = math.sqrt(vx**2 + vy**2 + vz**2)
        if speed > max_speed:
            scale = max_speed / speed
            vx, vy, vz = vx * scale, vy * scale, vz * scale
        velocities.append((vx, vy, vz))
        prev = wp
    return velocities


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
    parts = [
        f"({t}):({e['vel']:.2f} m/s, {e['acc_x']:.2f} m/s^2, {e['acc_y']:.2f} m/s^2, {e['steer']:.2f})"
        for t, e in zip(time_labels, egos)
    ]
    parts.append(
        f"(-0.0s):({current_ego['vel']:.2f} m/s, "
        f"{current_ego['acc_x']:.2f} m/s^2, "
        f"{current_ego['acc_y']:.2f} m/s^2, "
        f"{current_ego['steer']:.2f})"
    )
    return (
        "Historical ego (last 2 seconds), "
        "Format: (Velocity, Acceleration_x, Acceleration_y, yaw_rate): ["
        + ", ".join(parts) + "]\n"
    )


def build_image_frames_content(image_frame_history: deque):
    frames: List[CachedFrame] = list(image_frame_history)
    while len(frames) < IMAGE_HISTORY_FRAMES:
        frames.insert(0, frames[0])

    total_duration = (IMAGE_HISTORY_FRAMES - 1) * IMAGE_HISTORY_INTERVAL
    time_labels = [f"{-total_duration + i * IMAGE_HISTORY_INTERVAL:.1f}s"
                   for i in range(IMAGE_HISTORY_FRAMES)]
    content = "Here are drone images over the last 2 seconds of flight:\n"
    for label in time_labels:
        content += (
            f"[T={label}] 'CAM_FRONT': <image>, 'CAM_LEFT': <image>, "
            f"'CAM_RIGHT': <image>, 'CAM_DOWN': <image>\n"
        )
    return content, frames


def build_user_content(instruction: str, image_frame_history: deque,
                       history: deque, ego_history: deque,
                       current: DroneState, mission_goal: str = "FORWARD"):
    current_ego = compute_ego_state(current)

    image_content, ordered_frames = build_image_frames_content(image_frame_history)
    content = image_content
    content += "Here's some information you'll need:\n"
    if instruction:
        content += f"Instruction: {instruction}\n"
    content += build_history_ego_text(ego_history, current_ego)
    content += f"Mission Goal: {mission_goal}\n"
    content += "Flight Rules: Avoid collision with obstacles and terrain.\n"
    content += "- Maintain safe altitude above ground.\n"
    content += "- Stay within designated flight area.\n"
    content += "Based on the provided particulars, please output the plan waypoints (0.5s intervals) for the next 3 seconds.\n"
    return content, ordered_frames


# ==================== 模型推理 ====================

def build_model_inputs_cached(model, processor, tokenizer,
                              cached_frames: List[CachedFrame],
                              user_content: str, device: str = "cuda:0") -> dict:
    use_cache = cached_frames[0].image_embeds is not None

    if use_cache:
        all_grid_thw = torch.cat([f.grid_thw for f in cached_frames], dim=0)
    else:
        pixel_values_list, grid_thw_list = [], []
        for frame in cached_frames:
            for cam in _CAM_ORDER:
                img = frame.images[cam]
                img_proc = copy.deepcopy(processor.image_processor)
                img_proc.size["longest_edge"] = img_proc.max_pixels
                img_proc.size["shortest_edge"] = img_proc.min_pixels
                out = img_proc.preprocess(img, return_tensors="pt")
                t = out["pixel_values"]
                if isinstance(t, list):
                    t = t[0]
                pixel_values_list.append(t)
                grid_thw_list.append(out["image_grid_thw"][0])
        all_pixel_values = torch.cat(pixel_values_list, dim=0).to(device, torch.bfloat16)
        all_grid_thw = torch.cat([g.unsqueeze(0) for g in grid_thw_list], dim=0)

    merge_size = processor.image_processor.merge_size
    grid_thw_merged = [int(g.prod().item() // (merge_size ** 2)) for g in all_grid_thw]

    content_processed = user_content
    vis_idx = 0
    parts = content_processed.split("<image>")
    new_parts = []
    for i in range(len(parts) - 1):
        new_parts.append(parts[i])
        new_parts.append(
            "<|vision_start|>"
            + "<|image_pad|>" * grid_thw_merged[vis_idx]
            + "<|vision_end|>"
        )
        vis_idx += 1
    new_parts.append(parts[-1])
    content_processed = "".join(new_parts)

    tokenizer.chat_template = (
        "{% for message in messages %}"
        "{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}"
        "{% endfor %}"
        "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content_processed},
    ]
    input_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False
    )
    input_ids += tokenizer.apply_chat_template(
        [{"role": "assistant", "content": ""}],
        add_generation_prompt=True, tokenize=True,
    )
    input_ids_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    attn_mask = torch.ones(1, len(input_ids), dtype=torch.long, device=device)

    if use_cache:
        all_image_embeds = torch.cat(
            [f.image_embeds for f in cached_frames], dim=0
        ).to(device)

        with torch.no_grad():
            inputs_embeds = model.model.embed_tokens(input_ids_tensor)
        image_mask = (input_ids_tensor == model.config.image_token_id) \
                        .unsqueeze(-1).expand_as(inputs_embeds)
        inputs_embeds = inputs_embeds.masked_scatter(
            image_mask, all_image_embeds.to(inputs_embeds.dtype)
        )
        return {
            "input_ids": input_ids_tensor,
            "inputs_embeds": inputs_embeds,
            "attention_mask": attn_mask,
            "image_grid_thw": all_grid_thw.to(device),
        }
    else:
        return {
            "input_ids": input_ids_tensor,
            "attention_mask": attn_mask,
            "pixel_values": all_pixel_values,
            "image_grid_thw": all_grid_thw.to(device),
        }


def run_inference(model, processor, tokenizer, cached_frames: List[CachedFrame],
                  user_content: str, device: str = "cuda:0") -> List[Tuple[float, float, float]]:
    inputs = build_model_inputs_cached(
        model, processor, tokenizer, cached_frames, user_content, device
    )

    forward_kwargs = {
        "input_ids":       inputs["input_ids"],
        "attention_mask":  inputs["attention_mask"],
        "image_grid_thw":  inputs["image_grid_thw"],
    }
    if "inputs_embeds" in inputs:
        forward_kwargs["inputs_embeds"] = inputs["inputs_embeds"]
    else:
        forward_kwargs["pixel_values"] = inputs["pixel_values"]

    with torch.no_grad():
        outputs = model.forward(**forward_kwargs)

    waypoints = outputs.waypoints[0].cpu().float().tolist()
    return [(x, y, z) for x, y, z in waypoints]


def determine_mission_goal(instruction: str) -> str:
    inst_lower = instruction.lower()
    if "clockwise" in inst_lower and "counter" not in inst_lower:
        return "RIGHT"
    elif "counter-clockwise" in inst_lower or "counter clockwise" in inst_lower:
        return "LEFT"
    return "FORWARD"


# ==================== 主函数 ====================

def main():
    parser = argparse.ArgumentParser(description="AirSim VLA 部署推理 - LearnableQ 速度控制版本")
    parser.add_argument("--model_path", type=str, required=True,
                        help="LearnableQ checkpoint 路径")
    parser.add_argument("--use_lm_head", action="store_true",
                        help="使用 CE 模式（训练时 use_lm_head=True）；默认回归模式")
    parser.add_argument("--instruction", type=str,
                        default="Fly a circle clockwise with radius 50m at 200m altitude with 8m/s speed")
    parser.add_argument("--takeoff_altitude", type=float, default=200)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--max_speed", type=float, default=15.0,
                        help="速度向量最大限幅 (m/s)")
    parser.add_argument("--replan_interval", type=float, default=6.0,
                        help="重新推理最大间隔(秒)；实际上 waypoints 执行完即触发重推理")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--save_log", type=str, default=None)
    parser.add_argument("--save_image_dir", type=str, default=None)
    parser.add_argument("--use_int8", action="store_true",
                        help="启用 INT4(NF4) 量化以降低显存占用")
    parser.add_argument("--no_visual_cache", action="store_true",
                        help="禁用视觉编码器缓存：推理时实时运行 model.visual")
    args = parser.parse_args()

    model, processor, tokenizer = load_model(
        args.model_path, args.device,
        use_lm_head=args.use_lm_head,
        use_int8=args.use_int8,
    )

    print("连接 AirSim...")
    client = airsim.MultirotorClient()
    client.confirmConnection()

    try:
        vehicles = client.listVehicles()
        if vehicles:
            global VEHICLE_NAME
            VEHICLE_NAME = vehicles[0]
            print(f"使用无人机: {VEHICLE_NAME}")
    except Exception:
        pass

    client.enableApiControl(True, vehicle_name=VEHICLE_NAME)
    client.armDisarm(True, vehicle_name=VEHICLE_NAME)

    altitude = -args.takeoff_altitude
    print(f"起飞到 {args.takeoff_altitude}m...")
    client.takeoffAsync(vehicle_name=VEHICLE_NAME).join()
    client.moveToPositionAsync(100, 200, altitude, 5.0, timeout_sec=60,
                               vehicle_name=VEHICLE_NAME).join()
    print("起飞完成")
    time.sleep(2.0)

    traj_history = deque(maxlen=HISTORY_LEN)
    ego_history = deque(maxlen=4)
    image_frame_history: deque = deque(maxlen=IMAGE_HISTORY_FRAMES)
    last_image_capture = 0.0
    _frame_id_counter = 0

    # 速度控制关键变量
    current_waypoints_world: List[Tuple[float, float, float]] = []
    waypoint_velocities: List[Tuple[float, float, float]] = []
    current_wp_idx = 0          # 下一个待发送的速度段索引
    wp_start_time = 0.0         # 当前速度段开始时间
    path_following = False
    path_start_time = 0.0

    mission_goal = determine_mission_goal(args.instruction)
    print(f"Instruction: {args.instruction}")
    print(f"Mission Goal: {mission_goal}")
    print(f"重新推理间隔: {args.replan_interval}s")
    print(f"视觉编码缓存: {'禁用' if args.no_visual_cache else '启用'}")
    print(f"开始自主飞行 (最多 {args.max_steps} 步)...")

    flight_log = []
    last_traj_update = 0.0
    last_infer_time = 0.0
    infer_count = 0
    step = 0

    try:
        while step < args.max_steps:
            loop_start = time.time()

            # 1. 获取当前状态
            current = get_drone_state(client)

            if time.time() - last_traj_update >= HISTORY_INTERVAL:
                traj_history.append(current)
                ego_history.append(compute_ego_state(current))
                last_traj_update = time.time()

            if time.time() - last_image_capture >= IMAGE_HISTORY_INTERVAL:
                raw_frame = capture_images(client)
                if len(raw_frame) == 4:
                    if not args.no_visual_cache:
                        t_enc = time.time()
                        embeds_cpu, grid_cpu = encode_frame_images(
                            model, processor, raw_frame, args.device
                        )
                        enc_ms = (time.time() - t_enc) * 1000
                        cached = CachedFrame(
                            frame_id=_frame_id_counter,
                            images=raw_frame,
                            image_embeds=embeds_cpu,
                            grid_thw=grid_cpu,
                        )
                        if step % 20 == 0:
                            print(f"  [帧缓存] frame_id={cached.frame_id} "
                                  f"enc={enc_ms:.0f}ms "
                                  f"buf={len(image_frame_history)+1}/{IMAGE_HISTORY_FRAMES}")
                    else:
                        cached = CachedFrame(frame_id=_frame_id_counter, images=raw_frame)
                        if step % 20 == 0:
                            print(f"  [帧采集 no_cache] frame_id={cached.frame_id} "
                                  f"buf={len(image_frame_history)+1}/{IMAGE_HISTORY_FRAMES}")
                    image_frame_history.append(cached)
                    _frame_id_counter += 1
                last_image_capture = time.time()

            # 2. 判断是否需要重新推理
            need_replan = (
                not path_following
                or time.time() - last_infer_time >= args.replan_interval
            )

            if need_replan:
                print(f"\n{'='*60}")
                print(f"[推理 #{infer_count + 1}] 获取新速度序列...")

                if path_following:
                    client.cancelLastTask(vehicle_name=VEHICLE_NAME)
                    print("取消上一速度指令")

                if len(image_frame_history) == 0:
                    raw_frame = capture_images(client)
                    if len(raw_frame) == 4:
                        if not args.no_visual_cache:
                            embeds_cpu, grid_cpu = encode_frame_images(
                                model, processor, raw_frame, args.device
                            )
                            image_frame_history.append(CachedFrame(
                                frame_id=_frame_id_counter,
                                images=raw_frame,
                                image_embeds=embeds_cpu,
                                grid_thw=grid_cpu,
                            ))
                        else:
                            image_frame_history.append(CachedFrame(
                                frame_id=_frame_id_counter, images=raw_frame,
                            ))
                        _frame_id_counter += 1
                        last_image_capture = time.time()

                if len(image_frame_history) == 0:
                    print("图像采集失败，跳过本次推理")
                    time.sleep(0.1)
                    continue

                print(f"  图像帧历史: {len(image_frame_history)}/{IMAGE_HISTORY_FRAMES} 帧")

                ref_position = (current.pos_x, current.pos_y, current.pos_z)
                ref_yaw = current.yaw

                user_content, ordered_frames = build_user_content(
                    args.instruction, image_frame_history,
                    traj_history, ego_history, current, mission_goal
                )

                t0 = time.time()
                waypoints_ego = run_inference(
                    model, processor, tokenizer, ordered_frames, user_content, args.device
                )
                infer_time = time.time() - t0

                current_post = get_drone_state(client)
                actual_position = (current_post.pos_x, current_post.pos_y, current_post.pos_z)
                drift = math.sqrt(
                    (actual_position[0] - ref_position[0]) ** 2 +
                    (actual_position[1] - ref_position[1]) ** 2
                )

                if len(waypoints_ego) >= NUM_WAYPOINTS:
                    print(f"\n[调试] Mission Goal: {mission_goal}")
                    print(f"[调试] 参考位置:   ({ref_position[0]:.1f}, {ref_position[1]:.1f}), 朝向: {ref_yaw:.1f}°")
                    print(f"[调试] 推理后位置: ({actual_position[0]:.1f}, {actual_position[1]:.1f})  漂移: {drift:.1f}m / {infer_time:.2f}s")
                    print(f"[调试] 模型输出的自车坐标 waypoints (前 3 个):")
                    for i, wp in enumerate(waypoints_ego[:3]):
                        print(f"  WP[{i}]: ({wp[0]:.2f}, {wp[1]:.2f}, {wp[2]:.2f})")

                    waypoints_world = waypoints_ego_to_world(
                        waypoints_ego[:NUM_WAYPOINTS], ref_position, ref_yaw
                    )

                    # 计算速度向量序列（起点用推理后实际位置，与 waypoints_world 坐标系一致）
                    waypoint_velocities = compute_waypoint_velocities(
                        actual_position, waypoints_world, max_speed=args.max_speed
                    )
                    current_waypoints_world = waypoints_world
                    current_wp_idx = 0

                    print(f"[调试] 速度序列 (前 3 段):")
                    for i, (vx, vy, vz) in enumerate(waypoint_velocities[:3]):
                        spd = math.sqrt(vx**2 + vy**2 + vz**2)
                        print(f"  V[{i}]: ({vx:.2f}, {vy:.2f}, {vz:.2f}) m/s  |v|={spd:.2f} m/s")
                    print(f"Waypoints: {waypoints_world}")

                    # 发出第一段速度指令
                    vx, vy, vz = waypoint_velocities[0]
                    client.moveByVelocityAsync(
                        vx, vy, vz,
                        duration=WAYPOINT_INTERVAL,
                        drivetrain=airsim.DrivetrainType.ForwardOnly,
                        yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=0),
                        vehicle_name=VEHICLE_NAME,
                    )
                    current_wp_idx = 1
                    wp_start_time = time.time()
                    path_following = True
                    path_start_time = time.time()
                    last_infer_time = time.time()
                    infer_count += 1

                    flight_log.append({
                        "step": step,
                        "infer_count": infer_count,
                        "pos": list(actual_position),
                        "ref_position": ref_position,
                        "ref_yaw": ref_yaw,
                        "waypoints_ego": waypoints_ego[:NUM_WAYPOINTS],
                        "waypoints_world": waypoints_world,
                        "velocities": waypoint_velocities,
                        "infer_time": infer_time,
                    })

                    print(f"推理耗时: {infer_time:.2f}s")
                    print(f"{'='*60}")

                else:
                    print(f"waypoints 数量不足 ({len(waypoints_ego)}个)")

            # 3. 速度控制：按 waypoint 间隔逐步发送速度指令
            if path_following:
                elapsed = time.time() - path_start_time
                wp_elapsed = time.time() - wp_start_time

                # 当前速度段时间到，切换到下一段
                if wp_elapsed >= WAYPOINT_INTERVAL and current_wp_idx < len(waypoint_velocities):
                    vx, vy, vz = waypoint_velocities[current_wp_idx]
                    client.moveByVelocityAsync(
                        vx, vy, vz,
                        duration=WAYPOINT_INTERVAL,
                        drivetrain=airsim.DrivetrainType.ForwardOnly,
                        yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=0),
                        vehicle_name=VEHICLE_NAME,
                    )
                    current_wp_idx += 1
                    wp_start_time = time.time()

                pos = client.getMultirotorState(vehicle_name=VEHICLE_NAME).kinematics_estimated.position
                current_pos = [pos.x_val, pos.y_val, pos.z_val]

                if step % 5 == 0:
                    target_idx = min(current_wp_idx, len(current_waypoints_world) - 1)
                    target_pos = list(current_waypoints_world[target_idx]) if current_waypoints_world else current_pos
                    flight_log.append({
                        "step": step,
                        "pos": current_pos,
                        "target": target_pos,
                        "infer_count": infer_count,
                        "path_elapsed": elapsed,
                        "wp_idx": current_wp_idx,
                    })

                if step % 20 == 0:
                    print(f"  [Step {step}] 位置: ({pos.x_val:.1f}, {pos.y_val:.1f}, {-pos.z_val:.1f}), "
                          f"路径耗时: {elapsed:.1f}s, "
                          f"waypoint: {current_wp_idx}/{len(waypoint_velocities)}")

                # 所有速度段执行完毕或超过 replan_interval 时触发重新推理
                if current_wp_idx >= len(waypoint_velocities) or elapsed >= args.replan_interval:
                    path_following = False
                    print(f"  速度序列执行完成 "
                          f"(耗时: {elapsed:.1f}s, "
                          f"段: {current_wp_idx}/{len(waypoint_velocities)})")

            # 4. 可视化
            if args.visualize and step % 5 == 0 and image_frame_history:
                latest_frame = image_frame_history[-1].images
                if "CAM_FRONT" in latest_frame and args.save_image_dir:
                    front_cv = cv2.cvtColor(np.array(latest_frame["CAM_FRONT"]), cv2.COLOR_RGB2BGR)
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
