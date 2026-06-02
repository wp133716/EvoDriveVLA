"""Minimal MLP baseline for state-only action prediction.

Goal: establish a floor for how well a tiny model can predict the 5 continuous
action dims from purely numeric state. If this baseline already matches the
GT step-to-step variance on pitch/yaw/zoom, the VLM head is overkill.

Architecture:
    flatten[history_len * (STATE_DIM + mode_embed_dim)]
    -> Linear(hidden) -> SiLU -> Linear(hidden) -> SiLU
    -> Linear(horizon * LABEL_DIM)
    -> reshape to (horizon, LABEL_DIM)

Training:
    - Standardize state and labels using train-split stats
    - MSE on standardized labels
    - Reports val MAE in original units, per-dim and per coarse phase
    - Reports identity baseline (repeat current state for the 3 slow dims,
      zero for the 2 delta dims) for comparison

Usage:
    python -m state_policy.train_mlp \
        --root /home/user/remote_disk/0522-Tianjin_Datasets/20260515_182514__20260516_092922_merged \
        --val_ratio 0.5 --split_seed 0 --epochs 5
"""

from __future__ import annotations

import argparse
import bisect
import datetime
import json
import time
from collections import Counter, defaultdict
from pathlib import Path


_RUN_TS = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm.auto import tqdm

from state_policy.dataset import (
    FLAT_DT_COL,
    HORIZON_MAX,
    HISTORY_LEN_DEFAULT,
    LABEL_DIM,
    LABEL_NAMES,
    PHASES,
    PHASE_TO_COARSE,
    STATE_DIM,
    STATE_NAMES,
    StateOnlyDataset,
    split_episodes,
    GRID_TOTAL,
)
from state_policy.models import N_PHASES, build_model


# ----------------------------- collate -----------------------------

def make_collate():
    def collate(batch):
        state = torch.from_numpy(np.stack([b["state"] for b in batch]))
        mode = torch.from_numpy(np.stack([b["mode"] for b in batch]))
        label = torch.from_numpy(np.stack([b["label"] for b in batch]))
        label_grid = torch.from_numpy(np.stack([b["label_grid"] for b in batch]))
        mask = torch.from_numpy(np.stack([b["label_mask"] for b in batch]))
        return state, mode, label, label_grid, mask
    return collate


# ----------------------------- stats -----------------------------

def _per_sample_phase(ds: StateOnlyDataset) -> list:
    """Return coarse-phase label per sample in ds.index order."""
    out = []
    for ei, fi in ds.index:
        raw = ds.episodes[ei].frames[fi].get("mode")
        out.append(PHASE_TO_COARSE.get(raw, "UNKNOWN"))
    return out


def _per_sample_has_grid_transition(ds: StateOnlyDataset) -> np.ndarray:
    """Bool array: True if horizon contains any change in grid_idx.

    Uses ep.step_ts + flat_8x9.dt_ms[i] bisect, identical math to
    `_extract_label` so the answer matches what the model is trained on.
    """
    has = np.zeros(len(ds), dtype=bool)
    horizon = ds.horizon
    for i, (ei, fi) in enumerate(ds.index):
        ep = ds.episodes[ei]
        if ep.step_ts.size == 0:
            continue
        cur_ts = int(ep.frame_ts[fi])
        vals = ep.frames[fi].get("output", {}).get("flat_8x9", {}).get("values")
        if not vals:
            continue
        step_ts_list = ep.step_ts.tolist()
        seen = set()
        for s in range(horizon):
            abs_t = cur_ts + int(vals[s][FLAT_DT_COL])
            k = max(0, bisect.bisect_right(step_ts_list, abs_t) - 1)
            k = min(k, ep.step_grid_idx.size - 1)
            seen.add(int(ep.step_grid_idx[k]))
            if len(seen) > 1:
                has[i] = True
                break
    return has


def _per_sample_near_dwell(ds: StateOnlyDataset, window_s: float = 0.3) -> np.ndarray:
    """Bool: True if time_in_current_grid is within `window_s` of dwell_limit
    (i.e. about to switch)."""
    near = np.zeros(len(ds), dtype=bool)
    for i, (ei, fi) in enumerate(ds.index):
        ep = ds.episodes[ei]
        if ep.dwell_limit_s <= 0 or ep.frame_time_in_grid_s.size == 0:
            continue
        t = float(ep.frame_time_in_grid_s[fi])
        if abs(t - ep.dwell_limit_s) < window_s:
            near[i] = True
    return near


def compute_sample_weights(
    ds: StateOnlyDataset,
    mode: str = "full",
    transition_mult: float = 5.0,
    near_dwell_mult: float = 2.0,
    near_dwell_window_s: float = 0.3,
) -> np.ndarray:
    """Per-sample weights for WeightedRandomSampler.

    mode:
      'phase'        -> inverse coarse-phase frequency only
      'full'         -> phase + horizon-transition boost + near-dwell boost
    """
    phases = _per_sample_phase(ds)
    counts = Counter(phases)
    inv = {ph: 1.0 / max(c, 1) for ph, c in counts.items()}
    w = np.asarray([inv[ph] for ph in phases], dtype=np.float64)

    if mode == "full":
        has_trans = _per_sample_has_grid_transition(ds)
        near = _per_sample_near_dwell(ds, window_s=near_dwell_window_s)
        w = w * np.where(has_trans, transition_mult, 1.0)
        w = w * np.where(near, near_dwell_mult, 1.0)
    # Renormalize to a friendly scale (mean=1); WeightedRandomSampler only
    # needs proportionality but mean=1 makes the printed diagnostics readable.
    w = w * (len(w) / w.sum())
    return w


def _print_sampling_diagnostics(ds: StateOnlyDataset, weights: np.ndarray) -> None:
    """Show how phase distribution shifts under the weights."""
    phases = _per_sample_phase(ds)
    raw = Counter(phases)
    weighted = defaultdict(float)
    for ph, w in zip(phases, weights):
        weighted[ph] += w
    total_w = sum(weighted.values())
    print(f"\n[balance] phase distribution shift (raw % -> sampled %):")
    for ph in sorted(raw.keys()):
        raw_pct = 100.0 * raw[ph] / len(phases)
        w_pct = 100.0 * weighted[ph] / total_w
        print(f"  {ph:<10}  raw={raw_pct:>5.2f}%   sampled={w_pct:>5.2f}%")


def compute_train_stats(ds: StateOnlyDataset, n: int = 5000) -> dict:
    stats = ds.compute_stats(n_samples=n)
    # Clamp std so binary fields (0/1 grid bitmasks) and near-constant physical
    # dims (uav_z/roll/pitch) don't blow up when out-of-distribution at eval.
    state_std = np.maximum(np.asarray(stats["state_std"], dtype=np.float32), 1.0)
    label_std = np.maximum(np.asarray(stats["label_std"], dtype=np.float32), 1.0)
    return {
        "state_mean": np.asarray(stats["state_mean"], dtype=np.float32),
        "state_std": state_std,
        "label_mean": np.asarray(stats["label_mean"], dtype=np.float32),
        "label_std": label_std,
    }


# ----------------------------- eval -----------------------------

@torch.no_grad()
def eval_mae(model, loader, stats, device, ds_for_phase: StateOnlyDataset):
    """Return dict of MAE / grid-accuracy summaries."""
    model.eval()
    state_mean = torch.from_numpy(stats["state_mean"]).to(device)
    state_std = torch.from_numpy(stats["state_std"]).to(device)
    label_mean = torch.from_numpy(stats["label_mean"]).to(device)
    label_std = torch.from_numpy(stats["label_std"]).to(device)

    gp_idx = STATE_NAMES.index("gim_pitch")
    gy_idx = STATE_NAMES.index("gim_yaw")

    sum_abs_model = torch.zeros(LABEL_DIM, device=device)
    sum_abs_ident = torch.zeros(LABEL_DIM, device=device)
    grid_correct = 0
    grid_total = 0
    cont_count = 0
    phase_sums = defaultdict(lambda: [torch.zeros(LABEL_DIM, device=device), 0, 0, 0])
    # phase_sums[ph] = [sum_abs (5,), n_valid_cont, grid_correct, grid_total]

    for state, mode, label, label_grid, mask in tqdm(
        loader, desc="eval", leave=False, dynamic_ncols=True
    ):
        state = state.to(device)
        mode = mode.to(device)
        label = label.to(device)
        label_grid = label_grid.to(device)
        state_norm = (state - state_mean) / state_std
        # Model forward
        pred_norm, grid_logits = model(state_norm, mode)
        pred = pred_norm * label_std + label_mean
        grid_pred = grid_logits.argmax(dim=-1)                            # (B, T)
        # Identity baseline (continuous only)
        cur_pitch = state[:, -1, gp_idx]
        cur_yaw = state[:, -1, gy_idx]
        ident = torch.zeros_like(label)
        ident[..., 2] = cur_pitch.unsqueeze(-1)
        ident[..., 3] = cur_yaw.unsqueeze(-1)
        ident[..., 4] = label_mean[4]
        m = mask.to(device).float().unsqueeze(-1)                         # (B, T, 1)
        abs_model = ((pred - label).abs() * m).sum(dim=(0, 1))
        abs_ident = ((ident - label).abs() * m).sum(dim=(0, 1))
        n_valid = int(m.sum().item() / LABEL_DIM)
        sum_abs_model += abs_model
        sum_abs_ident += abs_ident
        cont_count += n_valid
        # Grid accuracy (mask in 2D)
        mask2d = mask.to(device)                                          # (B, T) bool
        grid_correct += int(((grid_pred == label_grid) & mask2d).sum().item())
        grid_total += int(mask2d.sum().item())
        # Per-phase
        cur_mode = mode[:, -1]
        for ph_id in cur_mode.unique().tolist():
            sel = cur_mode == ph_id
            ph_name = PHASES[ph_id] if 0 <= ph_id < N_PHASES else "unknown"
            coarse = PHASE_TO_COARSE.get(ph_name, ph_name.upper())
            m_sel = m[sel]
            n_sel = int(m_sel.sum().item() / LABEL_DIM)
            if n_sel == 0:
                continue
            ph_sum_abs = ((pred[sel] - label[sel]).abs() * m_sel).sum(dim=(0, 1))
            phase_sums[coarse][0] += ph_sum_abs
            phase_sums[coarse][1] += n_sel
            mask2d_sel = mask2d[sel]
            phase_sums[coarse][2] += int(
                ((grid_pred[sel] == label_grid[sel]) & mask2d_sel).sum().item())
            phase_sums[coarse][3] += int(mask2d_sel.sum().item())

    return {
        "model_mae": (sum_abs_model / max(cont_count, 1)).cpu().numpy(),
        "ident_mae": (sum_abs_ident / max(cont_count, 1)).cpu().numpy(),
        "grid_acc": grid_correct / max(grid_total, 1),
        "n_valid_cont": cont_count,
        "n_valid_grid": grid_total,
        "by_phase": {
            ph: {
                "mae": (s[0].cpu().numpy() / max(s[1], 1)).tolist(),
                "grid_acc": s[2] / max(s[3], 1),
                "n_cont": s[1],
                "n_grid": s[3],
            }
            for ph, s in phase_sums.items()
        },
    }


def _build_scheduler(opt, args):
    """Cosine + optional linear warmup, stepped per-epoch. Returns None for 'none'."""
    if args.lr_schedule == "none":
        return None
    if args.lr_schedule != "cosine":
        return None
    if args.warmup_epochs > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(
            opt, start_factor=0.01, end_factor=1.0,
            total_iters=args.warmup_epochs,
        )
        main_steps = max(1, args.epochs - args.warmup_epochs)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=main_steps)
        return torch.optim.lr_scheduler.SequentialLR(
            opt, [warmup, cosine], milestones=[args.warmup_epochs],
        )
    return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs))


def _save_ckpt(path, model, opt, epoch, args, stats, splits, ev, metric_value):
    """Save self-contained ckpt for inference + reproducibility."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": opt.state_dict(),
        "args": vars(args),
        "stats": {k: v.tolist() for k, v in stats.items()},
        "splits": {k: sorted(list(v)) for k, v in splits.items()},
        "val_summary": {
            "grid_acc": ev.get("grid_acc"),
            "model_mae": ev.get("model_mae").tolist() if ev.get("model_mae") is not None else None,
        } if ev else None,
        "best_metric_val": metric_value,
    }, path)


def _print_dim_table(title, mae, ident=None):
    print(f"\n{title}")
    print(f"  {'dim':<14} {'MAE':>10}" + (f"  {'identity':>10}  {'gain':>10}" if ident is not None else ""))
    for k, name in enumerate(LABEL_NAMES):
        if ident is not None:
            gain = ident[k] - mae[k]
            sign = "+" if gain > 0 else ""
            print(f"  {name:<14} {mae[k]:>10.4f}  {ident[k]:>10.4f}  {sign}{gain:>9.4f}")
        else:
            print(f"  {name:<14} {mae[k]:>10.4f}")


# ----------------------------- main -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root",
                    default="/home/user/remote_disk/0522-Tianjin_Datasets/"
                            "20260515_182514__20260516_092922_merged")
    ap.add_argument("--history_len", type=int, default=HISTORY_LEN_DEFAULT)
    ap.add_argument("--horizon", type=int, default=HORIZON_MAX)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--test_ratio", type=float, default=0.1)
    ap.add_argument("--split_seed", type=int, default=0)
    ap.add_argument("--episode_list", default="")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--arch", choices=["mlp", "transformer"], default="transformer")
    ap.add_argument("--hidden", type=int, default=256,
                    help="MLP hidden width (ignored by transformer).")
    ap.add_argument("--mode_embed", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.0,
                    help="Dropout for either arch.")
    # transformer-only
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--n_layers", type=int, default=3)
    ap.add_argument("--ffn_dim", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--save_metrics", default="state_mlp_metrics.json")
    # ----- ckpt / logging / schedule -----
    ap.add_argument("--tb_logdir", default=f"runs/{_RUN_TS}",
                    help="TensorBoard log dir. Empty = no tb.")
    ap.add_argument("--wandb_project", default="",
                    help="wandb project name. Empty = no wandb.")
    ap.add_argument("--wandb_run_name", default="",
                    help="wandb run name (default: auto-generated by wandb).")
    ap.add_argument("--wandb_entity", default="",
                    help="wandb entity / team. Empty = default user.")
    ap.add_argument("--wandb_upload_ckpt", action="store_true",
                    help="Upload best & last ckpts to wandb at end of run.")
    ap.add_argument("--save_best_ckpt", default=f"ckpts/{_RUN_TS}_best.pt",
                    help="Path to save best val ckpt (by --best_metric). Empty = no save.")
    ap.add_argument("--save_last_ckpt", default=f"ckpts/{_RUN_TS}_last.pt",
                    help="Path to save final-epoch ckpt. Empty = no save.")
    ap.add_argument("--best_metric", default="grid_acc",
                    choices=["grid_acc", "neg_cont_mae"],
                    help="Metric tracked for --save_best_ckpt. "
                         "grid_acc: maximize; neg_cont_mae: -mean(model_mae).")
    ap.add_argument("--eval_test_at_end", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="Final eval on test split using best ckpt. "
                         "Disable with --no-eval_test_at_end. "
                         "Silently skipped if test split is empty.")
    ap.add_argument("--lr_schedule", choices=["none", "cosine"], default="cosine")
    ap.add_argument("--warmup_epochs", type=int, default=2,
                    help="Linear warmup epochs prepended to --lr_schedule.")
    ap.add_argument("--grid_loss_weight", type=float, default=1.0,
                    help="lambda * CE(grid_logits) added to MSE(continuous).")
    ap.add_argument("--balance", choices=["none", "phase", "full"], default="full",
                    help="Train sampler balancing. 'phase' = inverse coarse-phase "
                         "freq; 'full' = phase + horizon-transition boost + near-dwell boost.")
    ap.add_argument("--transition_mult", type=float, default=5.0,
                    help="Weight multiplier when horizon contains a grid_idx change.")
    ap.add_argument("--near_dwell_mult", type=float, default=2.0,
                    help="Weight multiplier when time_in_grid is within --near_dwell_window of dwell_limit.")
    ap.add_argument("--near_dwell_window_s", type=float, default=0.3)
    args = ap.parse_args()

    explicit = json.loads(Path(args.episode_list).read_text()) if args.episode_list else None
    splits = split_episodes(args.root, val_ratio=args.val_ratio,
                            test_ratio=args.test_ratio,
                            seed=args.split_seed, explicit=explicit)
    print(f"[split] " + ", ".join(f"{k}={len(v)}" for k, v in splits.items()))

    train_ds = StateOnlyDataset(args.root, history_len=args.history_len,
                                horizon=args.horizon, episode_ids=splits["train"])
    val_ds = StateOnlyDataset(args.root, history_len=args.history_len,
                              horizon=args.horizon, episode_ids=splits["val"])
    print(f"[ds] train samples={len(train_ds)}  val samples={len(val_ds)}")
    if len(train_ds) == 0 or len(val_ds) == 0:
        print("[err] empty split; tweak --val_ratio or supply --episode_list.")
        return

    stats = compute_train_stats(train_ds, n=5000)

    train_sampler = None
    train_weights = None
    if args.balance != "none":
        train_weights = compute_sample_weights(
            train_ds,
            mode=args.balance,
            transition_mult=args.transition_mult,
            near_dwell_mult=args.near_dwell_mult,
            near_dwell_window_s=args.near_dwell_window_s,
        )
        _print_sampling_diagnostics(train_ds, train_weights)
        train_sampler = WeightedRandomSampler(
            torch.from_numpy(train_weights).double(),
            num_samples=len(train_ds),
            replacement=True,
        )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers, collate_fn=make_collate(),
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=make_collate())

    device = torch.device(args.device)
    model = build_model(
        args.arch,
        history_len=args.history_len, horizon=args.horizon,
        hidden=args.hidden, mode_embed=args.mode_embed, dropout=args.dropout,
        d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers,
        ffn_dim=args.ffn_dim,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] {model.__class__.__name__}  "
          f"total={n_params:,}  trainable={n_trainable:,}")
    for name, mod in model.named_children():
        n_mod = sum(p.numel() for p in mod.parameters())
        if n_mod == 0:
            continue
        pct = 100.0 * n_mod / max(n_params, 1)
        print(f"          {name:<14} {n_mod:>10,}  ({pct:>5.1f}%)")

    # ---- LR schedule ----
    scheduler = _build_scheduler(opt, args)

    # ---- TensorBoard ----
    writer = None
    if args.tb_logdir:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(args.tb_logdir)
        print(f"[tb] writing to {args.tb_logdir}")

    # ---- wandb ----
    wandb_run = None
    if args.wandb_project:
        import wandb
        init_kwargs = {"project": args.wandb_project, "config": vars(args)}
        if args.wandb_run_name:
            init_kwargs["name"] = args.wandb_run_name
        if args.wandb_entity:
            init_kwargs["entity"] = args.wandb_entity
        wandb_run = wandb.init(**init_kwargs)
        print(f"[wandb] project={args.wandb_project}  run={wandb_run.name}")

    # ---- Optional test loader ----
    test_loader = None
    test_ds = None
    if args.eval_test_at_end and splits.get("test"):
        test_ds = StateOnlyDataset(args.root, history_len=args.history_len,
                                   horizon=args.horizon,
                                   episode_ids=splits["test"])
        if len(test_ds) > 0:
            test_loader = DataLoader(
                test_ds, batch_size=args.batch_size, shuffle=False,
                num_workers=args.num_workers, collate_fn=make_collate())
            print(f"[test] reserved {len(test_ds)} samples for final eval")

    state_mean_t = torch.from_numpy(stats["state_mean"]).to(device)
    state_std_t = torch.from_numpy(stats["state_std"]).to(device)
    label_mean_t = torch.from_numpy(stats["label_mean"]).to(device)
    label_std_t = torch.from_numpy(stats["label_std"]).to(device)

    metrics_log = []
    best_metric_val = -float("inf")
    best_epoch = -1
    global_step = 0
    ev = None
    for ep in range(args.epochs):
        model.train()
        t0 = time.time()
        running_total = 0.0
        running_cont = 0.0
        running_grid = 0.0
        n_batch = 0
        pbar = tqdm(train_loader, desc=f"epoch {ep+1}/{args.epochs}",
                    leave=False, dynamic_ncols=True)
        for state, mode, label, label_grid, mask in pbar:
            state = state.to(device)
            mode = mode.to(device)
            label = label.to(device)
            label_grid = label_grid.to(device)
            mask = mask.to(device)
            state_n = (state - state_mean_t) / state_std_t
            label_n = (label - label_mean_t) / label_std_t
            pred_n, grid_logits = model(state_n, mode)
            m = mask.float().unsqueeze(-1)
            cont_loss = ((pred_n - label_n).pow(2) * m).sum() / (m.sum() * LABEL_DIM + 1e-8)
            # CE per valid step then mean
            B, T = label_grid.shape
            ce_flat = F.cross_entropy(
                grid_logits.reshape(B * T, GRID_TOTAL),
                label_grid.reshape(B * T),
                reduction="none",
            ).reshape(B, T)
            grid_loss = (ce_flat * mask.float()).sum() / (mask.float().sum() + 1e-8)
            loss = cont_loss + args.grid_loss_weight * grid_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            running_total += loss.item()
            running_cont += cont_loss.item()
            running_grid += grid_loss.item()
            n_batch += 1
            if writer is not None:
                writer.add_scalar("train/loss_total", loss.item(), global_step)
                writer.add_scalar("train/loss_cont", cont_loss.item(), global_step)
                writer.add_scalar("train/loss_grid", grid_loss.item(), global_step)
                writer.add_scalar("train/lr", opt.param_groups[0]["lr"], global_step)
            if wandb_run is not None:
                wandb_run.log({
                    "train/loss_total": loss.item(),
                    "train/loss_cont": cont_loss.item(),
                    "train/loss_grid": grid_loss.item(),
                    "train/lr": opt.param_groups[0]["lr"],
                }, step=global_step)
            if n_batch % 10 == 0:
                pbar.set_postfix(
                    loss=f"{loss.item():.3f}",
                    cont=f"{cont_loss.item():.3f}",
                    grid=f"{grid_loss.item():.3f}",
                    lr=f"{opt.param_groups[0]['lr']:.2e}",
                )
            global_step += 1
        if scheduler is not None:
            scheduler.step()
        dt = time.time() - t0
        ev = eval_mae(model, val_loader, stats, device, val_ds)
        model_mae = ev["model_mae"]
        ident_mae = ev["ident_mae"]
        grid_acc = ev["grid_acc"]
        by_phase = ev["by_phase"]
        print(f"\n=== Epoch {ep+1}/{args.epochs}  "
              f"loss={running_total/max(n_batch,1):.4f}  "
              f"(cont={running_cont/max(n_batch,1):.4f}  grid={running_grid/max(n_batch,1):.4f})  "
              f"({dt:.1f}s)  val_n={ev['n_valid_cont']} ===")
        _print_dim_table("[val] overall MAE (model vs identity)",
                         model_mae, ident_mae)
        print(f"\n[val] grid_idx accuracy: {grid_acc:.4f}  "
              f"({grid_acc * ev['n_valid_grid']:.0f}/{ev['n_valid_grid']})")
        if by_phase:
            print("\n[val] per-phase MAE + grid_acc:")
            print(f"  {'phase':<10} " + " ".join(f"{n:>10}" for n in LABEL_NAMES)
                  + f"  {'grid_acc':>10}")
            for ph, d in sorted(by_phase.items()):
                row = " ".join(f"{m:>10.4f}" for m in d["mae"])
                print(f"  {ph:<10} {row}  {d['grid_acc']:>10.4f}")
        metrics_log.append({
            "epoch": ep + 1,
            "train_loss": running_total / max(n_batch, 1),
            "train_cont_loss": running_cont / max(n_batch, 1),
            "train_grid_loss": running_grid / max(n_batch, 1),
            "val_model_mae": model_mae.tolist(),
            "val_identity_mae": ident_mae.tolist(),
            "val_grid_acc": grid_acc,
            "val_by_phase": by_phase,
        })

        # ---- TB per-epoch logs ----
        if writer is not None:
            writer.add_scalar("val/grid_acc", grid_acc, ep + 1)
            for i, name in enumerate(LABEL_NAMES):
                writer.add_scalar(f"val/mae_{name}", float(model_mae[i]), ep + 1)
                writer.add_scalar(f"val/ident_mae_{name}", float(ident_mae[i]), ep + 1)
            for ph, d in by_phase.items():
                writer.add_scalar(f"val_phase/{ph}/grid_acc", d["grid_acc"], ep + 1)
        if wandb_run is not None:
            ep_log = {
                "epoch": ep + 1,
                "val/grid_acc": grid_acc,
                "train/epoch_loss_total": running_total / max(n_batch, 1),
                "train/epoch_loss_cont": running_cont / max(n_batch, 1),
                "train/epoch_loss_grid": running_grid / max(n_batch, 1),
            }
            for i, name in enumerate(LABEL_NAMES):
                ep_log[f"val/mae_{name}"] = float(model_mae[i])
                ep_log[f"val/ident_mae_{name}"] = float(ident_mae[i])
            for ph, d in by_phase.items():
                ep_log[f"val_phase/{ph}/grid_acc"] = d["grid_acc"]
            wandb_run.log(ep_log, step=global_step)

        # ---- Best ckpt tracking ----
        current_metric = (grid_acc if args.best_metric == "grid_acc"
                          else -float(model_mae.mean()))
        if current_metric > best_metric_val:
            best_metric_val = current_metric
            best_epoch = ep + 1
            if args.save_best_ckpt:
                _save_ckpt(args.save_best_ckpt, model, opt, ep + 1,
                           args, stats, splits, ev, current_metric)
                print(f"[ckpt] new best {args.best_metric}={current_metric:.4f}"
                      f" -> {args.save_best_ckpt}")

    if args.save_last_ckpt and ev is not None:
        _save_ckpt(args.save_last_ckpt, model, opt, args.epochs,
                   args, stats, splits, ev, None)
        print(f"[ckpt] last-epoch -> {args.save_last_ckpt}")

    # ---- Final test-split evaluation ----
    test_results = None
    if test_loader is not None:
        # Restore best ckpt for test if we have one
        if args.save_best_ckpt and Path(args.save_best_ckpt).exists():
            ck = torch.load(args.save_best_ckpt, map_location=device,
                            weights_only=False)
            model.load_state_dict(ck["model_state_dict"])
            print(f"\n[test] restored best ckpt from epoch {ck.get('epoch', '?')}"
                  f" ({args.best_metric}={ck.get('best_metric_val', '?')})")
        else:
            print(f"\n[test] using last-epoch weights")
        tev = eval_mae(model, test_loader, stats, device, test_ds)
        print(f"=== Test set eval  n={tev['n_valid_cont']} ===")
        _print_dim_table("[test] MAE (model vs identity)",
                         tev["model_mae"], tev["ident_mae"])
        print(f"\n[test] grid_idx accuracy: {tev['grid_acc']:.4f}")
        if tev["by_phase"]:
            print("\n[test] per-phase MAE + grid_acc:")
            print(f"  {'phase':<10} " + " ".join(f"{n:>10}" for n in LABEL_NAMES)
                  + f"  {'grid_acc':>10}")
            for ph, d in sorted(tev["by_phase"].items()):
                row = " ".join(f"{m:>10.4f}" for m in d["mae"])
                print(f"  {ph:<10} {row}  {d['grid_acc']:>10.4f}")
        test_results = {
            "n_valid_cont": tev["n_valid_cont"],
            "n_valid_grid": tev["n_valid_grid"],
            "grid_acc": tev["grid_acc"],
            "model_mae": tev["model_mae"].tolist(),
            "ident_mae": tev["ident_mae"].tolist(),
            "by_phase": tev["by_phase"],
        }
        if writer is not None:
            writer.add_scalar("test/grid_acc", tev["grid_acc"], 0)
            for i, name in enumerate(LABEL_NAMES):
                writer.add_scalar(f"test/mae_{name}", float(tev["model_mae"][i]), 0)
        if wandb_run is not None:
            test_log = {"test/grid_acc": tev["grid_acc"]}
            for i, name in enumerate(LABEL_NAMES):
                test_log[f"test/mae_{name}"] = float(tev["model_mae"][i])
            for ph, d in tev["by_phase"].items():
                test_log[f"test_phase/{ph}/grid_acc"] = d["grid_acc"]
            wandb_run.log(test_log, step=global_step)
            wandb_run.summary.update(test_log)
            wandb_run.summary["best_epoch"] = best_epoch
            wandb_run.summary["best_metric_val"] = best_metric_val

    if writer is not None:
        writer.close()

    if wandb_run is not None:
        if args.wandb_upload_ckpt:
            for p in (args.save_best_ckpt, args.save_last_ckpt):
                if p and Path(p).exists():
                    wandb_run.save(p, policy="now")
                    print(f"[wandb] uploaded {p}")
        wandb_run.finish()

    print(f"\n[summary] best {args.best_metric}={best_metric_val:.4f} "
          f"at epoch {best_epoch}/{args.epochs}")

    Path(args.save_metrics).write_text(json.dumps({
        "args": vars(args),
        "stats": {k: v.tolist() for k, v in stats.items()},
        "splits": {k: sorted(list(v)) for k, v in splits.items()},
        "log": metrics_log,
        "best": {
            "epoch": best_epoch,
            "metric_name": args.best_metric,
            "metric_value": best_metric_val,
        },
        "test_results": test_results,
    }, indent=2))
    print(f"\n[done] metrics saved -> {args.save_metrics}")


if __name__ == "__main__":
    main()
