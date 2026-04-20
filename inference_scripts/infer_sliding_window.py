"""
infer_sliding_window.py — 滑窗 KV Cache 推理脚本
=================================================
演示如何用 SlidingWindowInference 替代逐帧全量推理。

与普通推理的计算量对比（window_size=5, n_cams=4）：
  普通:   每帧都对 5×4=20 张图跑完整 ViT+backbone
  滑窗:   只对 1×4=4 张新图跑 ViT，backbone 只处理新帧 token

用法：
    python -m inference_scripts.infer_sliding_window \
        --model_path ./result/evodrivevla/nuscenes/xxx_learnableq \
        --data_path  ../data/xxx_nuscenes/Drive_KD_val_his_ego_history5f.json \
        --img_dir    ../data/xxx_nuscenes \
        --output     ./result_sliding.json
"""

import os
import sys
import json
import torch
import argparse
import pathlib
from PIL import Image
from tqdm import tqdm
from transformers import AutoTokenizer, AutoProcessor

project_root = pathlib.Path(__file__).parent.parent
sys.path.append(str(project_root))

from model.language_models.modeling_qwen2_5_vl_learnableq import Qwen2_5_VLForLearnableQ
from model.language_models.sliding_window_inference import SlidingWindowInference

CAM_TYPES = ["CAM_FRONT", "CAM_LEFT", "CAM_RIGHT", "CAM_DOWN"]
N_CAMS = len(CAM_TYPES)
WINDOW_SIZE = 5   # 与训练时历史帧数一致


# ─── 工具 ──────────────────────────────────────────────────────────────────
def load_model(model_path: str, device: str = "cuda"):
    model = Qwen2_5_VLForLearnableQ.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    processor = AutoProcessor.from_pretrained(model_path, use_fast=True)
    model.tokenizer = tokenizer
    return model, tokenizer, processor


def process_images(image_paths: list[str], processor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    把一组图像路径处理成 (pixel_values, image_grid_thw)。
    pixel_values: (N, C, H, W)
    image_grid_thw: (N, 3)
    """
    all_pv, all_thw = [], []
    for path in image_paths:
        img = Image.open(path).convert("RGB")
        out = processor.image_processor.preprocess(img, return_tensors="pt")
        pv = out["pixel_values"]
        if isinstance(pv, list):
            pv = pv[0]
        thw = out["image_grid_thw"][0]
        all_pv.append(pv)
        all_thw.append(thw.unsqueeze(0))
    pixel_values = torch.cat(all_pv, dim=0)           # (N, C, H, W)
    image_grid_thw = torch.cat(all_thw, dim=0)        # (N, 3)
    return pixel_values, image_grid_thw


def build_full_input(sample: dict, img_dir: str, tokenizer, processor, device: str):
    """
    把 JSON 样本（含 5 帧 × 4 摄像头 = 20 张图像）处理成完整 input_ids / pixel_values。
    """
    # 图像路径
    image_paths = [os.path.join(img_dir, p) for p in sample["images"]]
    pixel_values, image_grid_thw = process_images(image_paths, processor)

    # 文字序列（用 apply_chat_template）
    messages = [
        {"role": "system", "content": sample["system"]},
        {"role": "user",   "content": sample["messages"][0]["content"]},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    # 把 <image> 替换为正确数量的 image_pad token（复用 data_qwen 的逻辑）
    # 简化：直接 tokenize，Qwen2.5-VL processor 会处理
    from qwenvl.data.data_qwen import preprocess_qwen_2_visual  # 内部工具
    # 此处直接用 processor 的完整流程
    inputs = processor(text=[text], images=pixel_values, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    pixel_values = inputs["pixel_values"].to(device)
    image_grid_thw = inputs["image_grid_thw"].to(device)
    return input_ids, attention_mask, pixel_values, image_grid_thw


def make_new_frame_input_ids(n_img_tokens: int, model, device: str) -> tuple:
    """
    构造仅包含图像占位 token 的 input_ids，用于给 get_rope_index() 计算位置。
    结构: <|vision_start|> <|image_pad|>×n <|vision_end|>  ×n_cams
    """
    vision_start = model.config.vision_start_token_id
    vision_end   = model.config.vision_end_token_id
    image_pad    = model.config.image_token_id

    tokens_per_cam = n_img_tokens // N_CAMS
    seq = []
    for _ in range(N_CAMS):
        seq += [vision_start] + [image_pad] * tokens_per_cam + [vision_end]
    input_ids = torch.tensor([seq], dtype=torch.long, device=device)
    attn_mask = torch.ones_like(input_ids)
    return input_ids, attn_mask


# ─── 主推理循环 ────────────────────────────────────────────────────────────
def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, processor = load_model(args.model_path, device)

    data = json.load(open(args.data_path))
    infer = SlidingWindowInference(model, window_size=WINDOW_SIZE)
    results = []

    for i, sample in enumerate(tqdm(data)):
        # 每个样本都是独立序列（不跨样本复用 cache）
        infer.reset()

        # ── 冷启动：处理完整 5 帧 ────────────────────────────────────────
        input_ids, attention_mask, pixel_values, image_grid_thw = build_full_input(
            sample, args.img_dir, tokenizer, processor, device
        )

        waypoints = infer.cold_start(
            input_ids, attention_mask,
            pixel_values, image_grid_thw,
            n_cams_per_frame=N_CAMS,
        )

        # ── 模拟滑动：用样本里已有的当前帧再滑一次 ──────────────────────
        # 实际部署时此处改为读取实时摄像头图像
        current_frame_paths = [
            os.path.join(args.img_dir, p)
            for p in sample["images"][-N_CAMS:]  # 最后4张 = 当前帧
        ]
        new_pv, new_thw = process_images(current_frame_paths, processor)
        new_pv = new_pv.to(device)
        new_thw = new_thw.to(device)

        # 计算新帧图像 token 数
        s = model.config.vision_config.spatial_merge_size
        n_img_tokens = int(sum(
            thw[0] * thw[1] * thw[2] / (s * s) for thw in new_thw
        ))
        new_input_ids, new_attn = make_new_frame_input_ids(n_img_tokens, model, device)

        waypoints_slide = infer.step(new_pv, new_thw, new_input_ids, new_attn)

        # 记录结果
        wp_list = waypoints_slide[0].cpu().float().tolist()
        results.append({
            "id": sample["id"],
            "waypoints": wp_list,
            "gt": sample["messages"][1]["content"],
        })

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"已保存 {len(results)} 条结果到 {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_path",  required=True)
    parser.add_argument("--img_dir",    required=True)
    parser.add_argument("--output",     default="./result_sliding.json")
    run(parser.parse_args())
