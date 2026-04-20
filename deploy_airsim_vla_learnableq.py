#!/usr/bin/env python3
"""
AirSim VLA 部署推理脚本 - LearnableQ 版本
==========================================
使用 Qwen2_5_VLForLearnableQ 预测 waypoints，支持两种输出模式：
  - 回归模式 (use_lm_head=False，默认)：trajectory_head + Smooth L1
  - CE 模式   (use_lm_head=True)         ：lm_head + 交叉熵

与 deploy_airsim_vla_regression.py 的区别：
  - 加载 Qwen2_5_VLForLearnableQ 模型类
  - 推理后从 outputs.waypoints 直接取 tensor（无文本 parse）
  - 必须将 tokenizer 挂载到模型（CE 和回归模式均需要 decode labels）
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
REPLAN_INTERVAL = 6.0    # 重新推理间隔 (秒)，与路径飞行时间匹配
NUM_WAYPOINTS = 6        # VLA 输出的 waypoints 数量
WAYPOINT_INTERVAL = 0.5  # waypoints 时间间隔 (秒)

# 历史轨迹（位置/ego状态）
HISTORY_LEN = 5
HISTORY_INTERVAL = 0.4

# 历史图像帧：5帧 × 4相机 = 20张，与训练数据 history5f 对齐
IMAGE_HISTORY_FRAMES = 5       # T=-2.0, -1.5, -1.0, -0.5, 0.0
IMAGE_HISTORY_INTERVAL = 0.5   # 每 0.5s 采集一帧，视觉编码在采集时完成（推理时复用）

SYSTEM_PROMPT = (
    "You're an autonomous drone's brain. "
    "Coordinates: X-axis is perpendicular, Y-axis is parallel to the direction you're facing, "
    "and Z-axis points upward. You're at point (0,0,0). Units: meters. "
    "Based on the provided particulars, please output the 3D plan waypoints (0.5s intervals) for the next 3 seconds."
)


# ==================== 数据类 ====================

@dataclass
class CachedFrame:
    """单帧（4路相机）数据，支持两种模式：

    visual_cache=True（默认）：
        image_embeds / grid_thw 在采集时预计算并存 CPU，
        推理时跳过 model.visual，只跑 LLM backbone。

    visual_cache=False（--no_visual_cache）：
        image_embeds / grid_thw 均为 None，
        推理时从 images 重新 preprocess + 运行 model.visual（原始路径）。
    """
    frame_id: int
    images: dict                             # {CAM_FRONT/LEFT/RIGHT/DOWN: PIL}
    image_embeds: Optional[torch.Tensor] = field(default=None)  # CPU tensor or None
    grid_thw: Optional[torch.Tensor] = field(default=None)      # (4,3) CPU or None


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
    """加载训练好的 LearnableQ VLA 模型"""
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
        # waypoint_mean/std 已保存在 checkpoint buffer 中（回归模式），from_pretrained 自动恢复
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

    # LearnableQ 模型在 CE 和回归模式下都需要 tokenizer（decode label 文本）
    model.tokenizer = tokenizer

    mode_str = "(INT4-NF4 量化)" if use_int8 else "(BF16)"
    print(f"模型加载完成 {mode_str}")
    if not use_lm_head and hasattr(model, "waypoint_mean"):
        print(f"  waypoint_mean: {model.waypoint_mean.tolist()}")
        print(f"  waypoint_std:  {model.waypoint_std.tolist()}")

    # try:
    #     if hasattr(torch, "compile") and not use_int8:
    #         print("启用 torch.compile 优化...")
    #         model = torch.compile(model, mode="reduce-overhead")
    #         print("torch.compile 启用成功")
    # except Exception as e:
    #     print(f"torch.compile 启用失败: {e}")

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
    """采集 4 路相机图像，返回 {CAM_FRONT/LEFT/RIGHT/DOWN: PIL.Image}"""
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
    """预处理4路相机图像并运行视觉编码器，返回 (image_embeds_cpu, grid_thw_cpu)。

    在每次图像采集时（0.5s 间隔）调用，将结果缓存到 CachedFrame。
    推理时不再调用 model.visual，只复用缓存，大幅降低推理延迟。

    返回:
        image_embeds : (n_vis_tokens_4cams, hidden_size)  CPU tensor
        grid_thw     : (4, 3)                              CPU tensor
    """
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
        )   # (n_vis_tokens_4cams, hidden_size)

    return image_embeds.cpu(), grid_thw.cpu()


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
    """自车坐标 → 世界坐标 (NED: X=北, Y=东, Z=下)

    当前匹配旧训练数据（XY bug 版本）：world_to_ego 用 cos(-yaw)、sin(-yaw)，
    ego_to_world 用其逆矩阵 R^T，即 cos(yaw)、sin(yaw)。

    TODO: 训练数据修复并重训后，换用正确公式：
        dx = -local_x * sin_yaw + local_y * cos_yaw
        dy =  local_x * cos_yaw + local_y * sin_yaw
    """
    cos_yaw = math.cos(yaw_rad)
    sin_yaw = math.sin(yaw_rad)
    # dx = local_x * cos_yaw - local_y * sin_yaw
    # dy = local_x * sin_yaw + local_y * cos_yaw
    dx = -local_x * sin_yaw + local_y * cos_yaw
    dy =  local_x * cos_yaw + local_y * sin_yaw
    world_x = ref_x + dx
    world_y = ref_y + dy
    world_z = ref_z - local_z
    return world_x, world_y, world_z


def waypoints_ego_to_world(waypoints_ego: List[Tuple[float, float, float]],
                            ref_pos: Tuple[float, float, float],
                            ref_yaw_deg: float) -> List[Tuple[float, float, float]]:
    """将自车坐标 waypoints 批量转换为世界坐标"""
    ref_x, ref_y, ref_z = ref_pos
    yaw_rad = math.radians(ref_yaw_deg)
    return [
        ego_to_world(lx, ly, lz, ref_x, ref_y, ref_z, yaw_rad)
        for lx, ly, lz in waypoints_ego
    ]


def calculate_path_length(path: List[airsim.Vector3r]) -> float:
    """计算路径总长度"""
    total = 0.0
    for i in range(1, len(path)):
        dx = path[i].x_val - path[i-1].x_val
        dy = path[i].y_val - path[i-1].y_val
        dz = path[i].z_val - path[i-1].z_val
        total += math.sqrt(dx*dx + dy*dy + dz*dz)
    return total


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
    """构建多帧图像 prompt 头部，与训练数据 history5f 格式完全一致。

    image_frame_history: deque[CachedFrame]，最老在前，最新在后。
    不足5帧时用最老帧向前填充（CachedFrame 对象直接重用）。

    返回:
        content_str      : 含 20 个 <image> 占位符的 prompt 头部
        ordered_frames   : List[CachedFrame]，长度=5，时间顺序
    """
    frames: List[CachedFrame] = list(image_frame_history)
    while len(frames) < IMAGE_HISTORY_FRAMES:
        frames.insert(0, frames[0])

    # time_labels 根据 IMAGE_HISTORY_FRAMES 动态生成，保证与 ordered_frames 长度一致
    total_duration = (IMAGE_HISTORY_FRAMES - 1) * IMAGE_HISTORY_INTERVAL  # 秒
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
    """返回 (user_content_str, ordered_cached_frames: List[CachedFrame])。"""
    current_ego = compute_ego_state(current)

    image_content, ordered_frames = build_image_frames_content(image_frame_history)
    content = image_content
    content += "Here's some information you'll need:\n"
    if instruction:
        content += f"Instruction: {instruction}\n"
    content += build_history_traj_text(history, current)
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
    """构建模型输入，自动根据 CachedFrame 是否含预计算 embeds 选择路径。

    visual_cache=True（image_embeds 不为 None）：
        跳过 model.visual，用缓存 embeds 在此完成 embed_tokens + masked_scatter，
        forward 不传 pixel_values。

    visual_cache=False（image_embeds 为 None）：
        从 CachedFrame.images 重新 preprocess，传 pixel_values 给 forward，
        model.forward 内部完整运行 model.visual（原始路径）。
    """
    use_cache = cached_frames[0].image_embeds is not None

    # ── 公共部分：收集 grid_thw，用于 <|image_pad|> token 数和 RoPE ──
    if use_cache:
        all_grid_thw = torch.cat([f.grid_thw for f in cached_frames], dim=0)  # (N_imgs, 3)
    else:
        # 无缓存时：重新 preprocess 获取 grid_thw 和 pixel_values
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

    # ── 构建 input_ids（含正确数量的 <|image_pad|>）────────────────
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
        # ── 缓存路径：embed_tokens + 手动 masked_scatter，不传 pixel_values ──
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
            "inputs_embeds": inputs_embeds,   # 预合并，forward 跳过 visual
            "attention_mask": attn_mask,
            "image_grid_thw": all_grid_thw.to(device),
        }
    else:
        # ── 无缓存路径：传 pixel_values，model.forward 内部运行 model.visual ──
        return {
            "input_ids": input_ids_tensor,
            "attention_mask": attn_mask,
            "pixel_values": all_pixel_values,
            "image_grid_thw": all_grid_thw.to(device),
        }


def run_inference(model, processor, tokenizer, cached_frames: List[CachedFrame],
                  user_content: str, device: str = "cuda:0") -> List[Tuple[float, float, float]]:
    """运行 LearnableQ 模型推理，直接返回 ego 坐标 waypoints。

    cached_frames 含预缓存 embeds（visual_cache=True）：
        model.forward 仅执行：text embed + LLM backbone + trajectory_head。
    cached_frames 无缓存（visual_cache=False，--no_visual_cache）：
        model.forward 完整执行：model.visual + LLM backbone + trajectory_head。
    """
    inputs = build_model_inputs_cached(
        model, processor, tokenizer, cached_frames, user_content, device
    )

    forward_kwargs = {
        "input_ids":       inputs["input_ids"],
        "attention_mask":  inputs["attention_mask"],
        "image_grid_thw":  inputs["image_grid_thw"],
    }
    # 缓存路径传 inputs_embeds（已含图像特征），无缓存路径传 pixel_values
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
    parser = argparse.ArgumentParser(description="AirSim VLA 部署推理 - LearnableQ 版本")
    parser.add_argument("--model_path", type=str, required=True,
                        help="LearnableQ checkpoint 路径")
    parser.add_argument("--use_lm_head", action="store_true",
                        help="使用 CE 模式（训练时 use_lm_head=True）；默认回归模式")
    parser.add_argument("--instruction", type=str,
                        default="Fly a circle clockwise with radius 50m at 200m altitude with 8m/s speed")
    parser.add_argument("--takeoff_altitude", type=float, default=200)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--base_speed", type=float, default=6.0,
                        help="基础速度，实际速度会根据路径长度调整")
    parser.add_argument("--replan_interval", type=float, default=6.0,
                        help="重新推理间隔(秒)")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--save_log", type=str, default=None)
    parser.add_argument("--save_image_dir", type=str, default=None)
    parser.add_argument("--use_int8", action="store_true",
                        help="启用 INT4(NF4) 量化以降低显存占用")
    parser.add_argument("--no_visual_cache", action="store_true",
                        help="禁用视觉编码器缓存：推理时实时运行 model.visual（原始路径，延迟更高）")
    args = parser.parse_args()

    # 加载模型
    model, processor, tokenizer = load_model(
        args.model_path, args.device,
        use_lm_head=args.use_lm_head,
        use_int8=args.use_int8,
    )

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
    except Exception:
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
    traj_history = deque(maxlen=HISTORY_LEN)
    ego_history = deque(maxlen=4)
    # 图像帧历史：5帧 CachedFrame，视觉编码在采集时完成
    image_frame_history: deque = deque(maxlen=IMAGE_HISTORY_FRAMES)
    last_image_capture = 0.0
    _frame_id_counter = 0

    # Path-based 控制关键变量
    current_waypoints_world: List[Tuple[float, float, float]] = []
    path_start_time = 0.0
    path_following = False

    mission_goal = determine_mission_goal(args.instruction)
    print(f"Instruction: {args.instruction}")
    print(f"Mission Goal: {mission_goal}")
    print(f"重新推理间隔: {args.replan_interval}s")
    print(f"视觉编码缓存: {'禁用（--no_visual_cache，推理时实时运行 ViT）' if args.no_visual_cache else '启用（采集时预编码，推理时跳过 ViT）'}")
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

            # 更新历史轨迹（位置/ego状态）
            if time.time() - last_traj_update >= HISTORY_INTERVAL:
                traj_history.append(current)
                ego_history.append(compute_ego_state(current))
                last_traj_update = time.time()

            # 更新图像帧历史（0.5s 采集一帧）
            # visual_cache=True（默认）：采集后立即运行视觉编码器并缓存，推理时跳过 ViT
            # visual_cache=False（--no_visual_cache）：只存 PIL 图像，推理时实时运行 ViT
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
                        cached = CachedFrame(
                            frame_id=_frame_id_counter,
                            images=raw_frame,
                        )
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
                print(f"[推理 #{infer_count + 1}] 获取新路径...")

                if path_following:
                    client.cancelLastTask(vehicle_name=VEHICLE_NAME)
                    print("取消上一路径")

                # 确保图像历史有至少 1 帧（首次推理前可能还没采到）
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
                                frame_id=_frame_id_counter,
                                images=raw_frame,
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

                # 推理期间（~1s）无人机继续飞行，重新查询实际位置
                # 若仍用 ref_position 作为路径起点，AirSim 会让无人机折返到推理前旧位置
                current_post = get_drone_state(client)
                actual_position = (current_post.pos_x, current_post.pos_y, current_post.pos_z)
                drift = math.sqrt(
                    (actual_position[0] - ref_position[0]) ** 2 +
                    (actual_position[1] - ref_position[1]) ** 2
                )

                if len(waypoints_ego) >= NUM_WAYPOINTS:
                    # ========== 调试信息 ==========
                    print(f"\n[调试] Mission Goal: {mission_goal}")
                    print(f"[调试] 参考位置:   ({ref_position[0]:.1f}, {ref_position[1]:.1f}), 朝向: {ref_yaw:.1f}°")
                    print(f"[调试] 推理后位置: ({actual_position[0]:.1f}, {actual_position[1]:.1f})  漂移: {drift:.1f}m / {infer_time:.2f}s")
                    print(f"[调试] 模型输出的自车坐标 waypoints (前 3 个):")
                    for i, wp in enumerate(waypoints_ego[:3]):
                        print(f"  WP[{i}]: ({wp[0]:.2f}, {wp[1]:.2f}, {wp[2]:.2f})")

                    avg_x = sum(wp[0] for wp in waypoints_ego[:NUM_WAYPOINTS]) / NUM_WAYPOINTS
                    print(f"[调试] Waypoints X 坐标平均值: {avg_x:.2f}")
                    if mission_goal == "RIGHT" and avg_x < 0:
                        print("  警告: Mission Goal=RIGHT 但 waypoints X < 0 (应该是右/正)")
                    elif mission_goal == "LEFT" and avg_x > 0:
                        print("  警告: Mission Goal=LEFT 但 waypoints X > 0 (应该是左/负)")
                    # ==============================

                    # 转换到世界坐标（基准仍为推理时的 ref_position/ref_yaw，与模型输入对齐）
                    waypoints_world = waypoints_ego_to_world(
                        waypoints_ego[:NUM_WAYPOINTS], ref_position, ref_yaw
                    )

                    print(f"[调试] 转换后的世界坐标 waypoints (前 3 个):")
                    for i, wp in enumerate(waypoints_world[:3]):
                        print(f"  WP[{i}]: ({wp[0]:.1f}, {wp[1]:.1f}, {wp[2]:.1f})")

                    # 路径起点用推理后实际位置，避免无人机折返到推理前旧位置
                    path = [airsim.Vector3r(actual_position[0], actual_position[1], actual_position[2])]
                    for wp in waypoints_world:
                        path.append(airsim.Vector3r(wp[0], wp[1], wp[2]))

                    # 在路径末尾追加 2 个外推点，防止 AirSim 在最后一个 waypoint 减速刹停。
                    # replan 的 cancelLastTask 发出时无人机仍在向更远的点加速，轨迹连贯。
                    # 外推方向 = 最后两点的切线方向，步长 = 最后段间距（保持曲率一致）
                    if len(waypoints_world) >= 2:
                        p2 = waypoints_world[-1]
                        p1 = waypoints_world[-2]
                        dx, dy, dz = p2[0]-p1[0], p2[1]-p1[1], p2[2]-p1[2]
                        for k in range(1, 3):
                            path.append(airsim.Vector3r(p2[0]+k*dx, p2[1]+k*dy, p2[2]+k*dz))

                    path_length = calculate_path_length(path)

                    if path_length > 0.1:
                        target_speed = max(2.0, min(path_length / args.replan_interval, 15.0))
                    else:
                        target_speed = args.base_speed

                    print(f"生成路径: {len(path)} 个点, 长度: {path_length:.2f}m")
                    print(f"目标速度: {target_speed:.2f}m/s (飞行时间≈{args.replan_interval}s)")
                    print(f"Waypoints: {waypoints_world}")

                    client.moveOnPathAsync(
                        path,
                        velocity=target_speed,
                        timeout_sec=args.replan_interval + 5,  # 主循环会主动 cancel，此 timeout 不应触发
                        drivetrain=airsim.DrivetrainType.ForwardOnly,
                        yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=0),
                        vehicle_name=VEHICLE_NAME,
                    )

                    current_waypoints_world = waypoints_world
                    path_following = True
                    path_start_time = time.time()
                    last_infer_time = time.time()
                    infer_count += 1

                    flight_log.append({
                        "step": step,
                        "infer_count": infer_count,
                        "pos": list(actual_position),
                        "target": list(waypoints_world[0]),
                        "ref_position": ref_position,
                        "ref_yaw": ref_yaw,
                        "waypoints_ego": waypoints_ego[:NUM_WAYPOINTS],
                        "waypoints_world": waypoints_world,
                        "path_length": path_length,
                        "target_speed": target_speed,
                        "infer_time": infer_time,
                    })

                    print(f"推理耗时: {infer_time:.2f}s")
                    print(f"{'='*60}")

                else:
                    print(f"waypoints 数量不足 ({len(waypoints_ego)}个)")

            # 3. 检查路径状态并记录
            if path_following:
                elapsed = time.time() - path_start_time
                pos = client.getMultirotorState(vehicle_name=VEHICLE_NAME).kinematics_estimated.position
                current_pos = [pos.x_val, pos.y_val, pos.z_val]

                if step % 5 == 0 or elapsed >= args.replan_interval:
                    progress = min(1.0, elapsed / args.replan_interval)
                    target_idx = min(
                        int(progress * len(current_waypoints_world)),
                        len(current_waypoints_world) - 1,
                    )
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

            # 4. 可视化（直接用最新 CachedFrame，避免重复采集）
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
