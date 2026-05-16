"""DDP inference for Qwen2_5_VLForActionToken on drone search-and-track data.

For each val sample we:
  1. Build inputs through LazySupervisedDataset (with inference=False so labels
     remain attached; the model's forward strips the visible answer tokens via
     _strip_supervised_answer_tokens and appends learned action slots).
  2. Run model.forward(do_sample=False) -> argmax over the per-position
     action_logits -> outputs.waypoints in original units (denormalized).
  3. Persist {id, predict_vec, action_token_ids} per sample. Metrics are
     computed separately by eval_planning/evaluation/eval_drone_action.py.
"""

import json
import os
from dataclasses import dataclass, field

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import transformers
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor

from model.language_models.modeling_qwen2_5_vl_action_token import (
    Qwen2_5_VLForActionToken,
)
from qwenvl.data.data_qwen import make_supervised_data_module
from qwenvl.train.argument import (
    DataArguments,
    EvalArguments,
    ModelArguments,
    TrainingArguments,
)


@dataclass
class ActionTokenArguments:
    num_action_steps: int = field(default=1)
    action_dim: int = field(default=6)
    action_bins: int = field(default=256)


def _add_action_tokens(tokenizer, action_bins):
    toks = [f"<action_{i}>" for i in range(int(action_bins))]
    existing = set(tokenizer.additional_special_tokens)
    to_add = [t for t in toks if t not in existing]
    if to_add:
        tokenizer.add_special_tokens({"additional_special_tokens": to_add})


def run_worker(rank, world_size, model_args, data_args, eval_args,
               training_args, action_args, attn_implementation="sdpa"):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    # action tokens may or may not already be in the checkpoint tokenizer;
    # add any missing ones idempotently.
    _add_action_tokens(tokenizer, action_args.action_bins)

    model = Qwen2_5_VLForActionToken.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_implementation,
        device_map={"": rank},
        num_action_steps=action_args.num_action_steps,
        action_dim=action_args.action_dim,
        action_bins=action_args.action_bins,
    ).eval()
    model.resize_token_embeddings(len(tokenizer))
    model.set_action_tokenizer(tokenizer)
    model.config.use_cache = False
    model.to(device).bfloat16()

    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
        use_fast=True,
        min_pixels=data_args.min_pixels,
        max_pixels=data_args.max_pixels,
    )
    data_args.image_processor = processor.image_processor
    data_args.model_type = "qwen2.5vl"
    # Keep labels attached: the action-token model relies on `labels` to mask
    # the visible answer span and inject action slots. GT for metrics is parsed
    # separately from the val JSON in eval_drone_action.py.
    data_args.inference = False
    data_args.random = False

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    dataset = data_module["train_dataset"]
    collate = data_module["data_collator"]

    total = len(dataset)
    per_rank = (total + world_size - 1) // world_size
    start = rank * per_rank
    end = min(start + per_rank, total)
    indices = list(range(start, end))
    subset = torch.utils.data.Subset(dataset, indices)
    loader = DataLoader(
        subset,
        batch_size=1,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate,
        shuffle=False,
    )
    if rank == 0:
        print(f"Total val samples: {total} | per-GPU: {len(subset)}")

    out_rows = []
    for batch in tqdm(loader, disable=(rank != 0)):
        model_inputs = {}
        for k, v in batch.items():
            if k in (
                "input_ids", "attention_mask", "labels", "pixel_values",
                "image_grid_thw", "position_ids",
            ):
                if isinstance(v, torch.Tensor):
                    if v.is_floating_point():
                        v = v.to(device=device, dtype=torch.bfloat16)
                    else:
                        v = v.to(device)
                model_inputs[k] = v

        with torch.no_grad():
            outputs = model.forward(
                **model_inputs,
                do_sample=False,
                return_dict=True,
            )
        # waypoints shape: [B, num_action_steps, action_dim]
        wp = outputs.waypoints.float().cpu()
        ids = outputs.action_token_ids.cpu()
        for bi, sid in enumerate(batch["id"]):
            out_rows.append({
                "id": sid,
                "predict_vec": wp[bi].reshape(-1).tolist(),  # length = steps*dim
                "action_token_ids": ids[bi].tolist(),
            })

    tmp_path = eval_args.eval_save_path + f".rank{rank}.json"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(out_rows, f, ensure_ascii=False, indent=2)

    dist.barrier()
    if rank == 0:
        merged, seen = [], set()
        for r in range(world_size):
            p = eval_args.eval_save_path + f".rank{r}.json"
            for item in json.load(open(p, "r", encoding="utf-8")):
                if item["id"] in seen:
                    continue
                merged.append(item)
                seen.add(item["id"])
            os.remove(p)
        merged.sort(key=lambda x: str(x["id"]))
        with open(eval_args.eval_save_path, "w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(merged)} predictions -> {eval_args.eval_save_path}")

    dist.barrier()
    dist.destroy_process_group()


def main():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, EvalArguments, TrainingArguments,
         ActionTokenArguments)
    )
    (model_args, data_args, eval_args, training_args,
     action_args) = parser.parse_args_into_dataclasses()

    world_size = torch.cuda.device_count()
    if world_size == 0:
        raise RuntimeError("No CUDA devices visible for inference.")
    print(f"Launching DDP inference on {world_size} GPUs ...")
    mp.spawn(
        run_worker,
        args=(world_size, model_args, data_args, eval_args, training_args,
              action_args),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    main()
