# state_policy

State-only drone action prediction. Skips the VLM entirely: reads raw episode
data, builds a fixed-size numeric state vector per frame, predicts the next
`horizon` steps of 5 continuous action dims plus a categorical `grid_idx`.

Designed for the same data format produced by the AirSim collector
(`frames_vla.jsonl` + `steps.jsonl` + `meta.json` per `ep_*/`).

## Files

| File           | Role                                                                              |
| -------------- | --------------------------------------------------------------------------------- |
| `dataset.py`   | `StateOnlyDataset`, episode walker, grid context loader, train/val split helper   |
| `models.py`    | `StateMLP`, `StateTransformer`, `build_model(arch=...)` factory                   |
| `train_mlp.py` | Training loop (despite the name, drives both MLP and Transformer via `--arch`)    |
| `README.md`    | This file                                                                         |

## Data shape

| Field        | Shape                          | Notes                                                                |
| ------------ | ------------------------------ | -------------------------------------------------------------------- |
| `state`      | `(history_len, 117)` float32   | 14 base + 5 grid scalar + 49 target bitmask + 49 searched bitmask    |
| `mode`       | `(history_len,)` int64         | phase id; -1 maps to N_PHASES in embedding                           |
| `label`      | `(horizon, 5)` float32         | pixel_dx, pixel_dy, gim_pitch, gim_yaw, zoom                         |
| `label_grid` | `(horizon,)` int64             | grid_idx 0..48                                                       |
| `label_mask` | `(horizon,)` bool              | per-step validity                                                    |
| `sample_id`  | str                            | `<episode_id>_<frame_id>`                                            |

`STATE_NAMES`, `LABEL_NAMES`, `GRID_ROWS/COLS/TOTAL`, `HORIZON_MAX`,
`HISTORY_LEN_DEFAULT` live at the top of `dataset.py`.

## 1. Inspect dataset

```bash
python -m state_policy.dataset \
  --root /home/user/remote_disk/0522-Tianjin_Datasets/20260515_182514__20260516_092922_merged \
  --history_len 5 --horizon 8
```

Prints episode count, sample count, phase distribution, and per-dim
mean/std over a 5k-sample stat draw.

### Train/val split

```bash
# 90 / 10 split by episode, deterministic
python -m state_policy.dataset --root ... \
  --split train --val_ratio 0.1 --split_seed 42

# Explicit override
cat > /tmp/eps.json <<'EOF'
{"train": ["ep_000005", "ep_000007"], "val": ["ep_000259"]}
EOF
python -m state_policy.dataset --episode_list /tmp/eps.json --split val
```

Splitting is by **episode**, not by frame, to avoid temporal leakage.

## 2. Train

```bash
# MLP baseline
python -m state_policy.train_mlp \
  --root /path/to/dataset \
  --arch mlp --hidden 256 \
  --val_ratio 0.1 --split_seed 42 \
  --epochs 20 --batch_size 1024 \
  --device cuda

# Transformer baseline
python -m state_policy.train_mlp \
  --root /path/to/dataset \
  --arch transformer --d_model 128 --n_heads 4 --n_layers 3 --ffn_dim 256 \
  --val_ratio 0.1 --split_seed 42 \
  --epochs 20 --batch_size 1024 \
  --device cuda
```

Both archs output two heads:

- `cont_pred (B, T, 5)` -> MSE on standardized labels
- `grid_logits (B, T, 49)` -> Cross-entropy on `label_grid`

Total loss: `MSE + grid_loss_weight * CE`, default weight 1.0.

## 3. Sample balancing

The dataset is heavily imbalanced (TRACK + REPORT dominate, SELECT is rare,
within-phase grid transitions are even rarer). Three modes:

| `--balance`      | What it does                                                   |
| ---------------- | -------------------------------------------------------------- |
| `none` (default) | Plain shuffle                                                  |
| `phase`          | Inverse coarse-phase frequency only                            |
| `full`           | Phase + horizon-contains-transition x N + near-dwell-edge x N  |

```bash
python -m state_policy.train_mlp --balance full \
  --transition_mult 5.0 \
  --near_dwell_mult 2.0 \
  --near_dwell_window_s 0.3 \
  ...
```

When balancing is on, the trainer prints the raw vs sampled phase
distribution so you can confirm the shift.

## 4. Output

- Per-epoch console log: train_loss (cont / grid split), val MAE table
  (model vs identity baseline), grid_idx_accuracy overall + per phase
- `state_mlp_metrics.json` (or whatever `--save_metrics` points to):
  args + train stats + splits + per-epoch metrics log

## 5. Python API

```python
from state_policy import StateOnlyDataset, split_episodes
from state_policy.models import build_model

splits = split_episodes(root, val_ratio=0.1, seed=42)
train_ds = StateOnlyDataset(root, history_len=5, horizon=8,
                            episode_ids=splits["train"])
val_ds   = StateOnlyDataset(root, history_len=5, horizon=8,
                            episode_ids=splits["val"])

model = build_model("transformer", history_len=5, horizon=8,
                    d_model=128, n_heads=4, n_layers=3, ffn_dim=256)
# model(state, mode) -> (cont_pred (B,T,5), grid_logits (B,T,49))
```

## 6. Common flags reference

```text
Dataset                Default        Note
--root                 (required)     Episode root directory
--history_len          5              Past frames in state window
--horizon              8              Future steps to predict (<= 8)
--split                all            all / train / val / test
--val_ratio            0.1            By-episode ratio
--test_ratio           0.0
--split_seed           0
--episode_list         ""             JSON overriding the random split

Training
--arch                 mlp            mlp | transformer
--epochs               5
--batch_size           256
--lr                   3e-4
--device               cuda if avail
--save_metrics         state_mlp_metrics.json

Model: MLP
--hidden               256
--mode_embed           8
--dropout              0.0

Model: Transformer
--d_model              128
--n_heads              4
--n_layers             3
--ffn_dim              256
--dropout              0.1

Loss
--grid_loss_weight     1.0            lambda * CE added to MSE

Balancing
--balance              none           none | phase | full
--transition_mult      5.0
--near_dwell_mult      2.0
--near_dwell_window_s  0.3

Logging / ckpt / test / schedule
--tb_logdir            ""             TensorBoard log directory (empty = off)
--wandb_project        ""             wandb project name (empty = off)
--wandb_run_name       ""             wandb run name (default: auto)
--wandb_entity         ""             wandb entity/team
--wandb_upload_ckpt    off            Also upload best & last ckpts to wandb
--save_best_ckpt       ""             Save best-val ckpt to this path
--save_last_ckpt       ""             Save final-epoch ckpt to this path
--best_metric          grid_acc       grid_acc | neg_cont_mae
--eval_test_at_end     off            Final eval on test split using best ckpt
--lr_schedule          none           none | cosine
--warmup_epochs        0              Linear warmup epochs before cosine
```

### TensorBoard + wandb keys logged

| key | when |
| --- | ---- |
| `train/loss_total`, `train/loss_cont`, `train/loss_grid`, `train/lr` | per training step (`global_step`) |
| `train/epoch_loss_*` (wandb only) | per epoch |
| `val/grid_acc`, `val/mae_<dim>`, `val/ident_mae_<dim>` | per epoch |
| `val_phase/<PHASE>/grid_acc` | per epoch per coarse phase |
| `test/grid_acc`, `test/mae_<dim>`, `test_phase/<PHASE>/grid_acc` | once after training |
| wandb `summary.best_epoch`, `summary.best_metric_val` | once at end |

## 8. Ckpt format

Both `--save_best_ckpt` and `--save_last_ckpt` write a single .pt with:

```text
epoch                  int
model_state_dict       weights only, load with build_model(args.arch, ...)
optimizer_state_dict   Adam state (for resume)
args                   dict — full CLI args used for the run
stats                  state_mean/std + label_mean/std (for normalization)
splits                 train/val/test episode ids
val_summary            grid_acc + model_mae at this epoch
best_metric_val        scalar
```

Reload for inference:

```python
import torch
from state_policy.models import build_model

ckpt = torch.load("ckpts/state_tx_best.pt", weights_only=False)
a = ckpt["args"]
model = build_model(
    a["arch"], history_len=a["history_len"], horizon=a["horizon"],
    hidden=a["hidden"], mode_embed=a["mode_embed"], dropout=a["dropout"],
    d_model=a["d_model"], n_heads=a["n_heads"], n_layers=a["n_layers"],
    ffn_dim=a["ffn_dim"],
)
model.load_state_dict(ckpt["model_state_dict"])
state_mean = torch.tensor(ckpt["stats"]["state_mean"])
state_std  = torch.tensor(ckpt["stats"]["state_std"])
label_mean = torch.tensor(ckpt["stats"]["label_mean"])
label_std  = torch.tensor(ckpt["stats"]["label_std"])
```

## 7. Notes on what's been validated

- Dataset loader and grid context (target/searched bitmask + timer) verified
  on the local 2-episode merged dataset
- Overfit test (train = val = ep_000005) reaches grid_idx_accuracy ~ 0.91
  for both archs in 5 epochs
- `--balance full` shifts SELECT from 3% to ~25% of sampled batches and
  pushes SELECT grid_idx_accuracy from 0.66 to 0.89 in the smoke test
- Transformer beats MLP by ~30% on gim_pitch MAE and ~15% on gim_yaw MAE in
  the overfit test, at 3x CPU time and 26% more params
- Real metrics require multi-episode server data; the local 2-ep split is
  only useful for smoke testing
