"""State-only dataset for drone action prediction.

Reads raw episodes directly from the merged dataset directory; no VLM, no
images, no text prompts. Each sample is a fixed-size numeric state window
(history_len frames) plus the next horizon-step action label, both already
present in frames_vla.jsonl's `output.flat_8x9`.

Dataset layout (each ep_NNNNNN/ under root):
    frames_vla.jsonl         per-frame state + output future actions
    steps.jsonl              grid transitions (used to attach current_grid)
    meta.json                episode-level metadata
    images/...               IGNORED here

Usage:
    python -m state_policy.dataset \
        --root /home/user/remote_disk/.../20260515_182514__20260516_092922_merged

Prints schema + phase distribution + per-dim mean/std over a 5k-sample stat
draw. That output is what you check before training anything.
"""

from __future__ import annotations

import argparse
import bisect
import json
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


PHASES = ("prepare", "search", "select", "first_track", "second_track", "report")
PHASE_TO_ID = {p: i for i, p in enumerate(PHASES)}
PHASE_TO_COARSE = {  # roll first_track + second_track into TRACK to match val JSON
    "prepare": "PREPARE",
    "search": "SEARCH",
    "select": "SELECT",
    "first_track": "TRACK",
    "second_track": "TRACK",
    "report": "REPORT",
}

_BASE_STATE_NAMES = (
    "uav_x", "uav_y", "uav_z",
    "uav_roll", "uav_pitch", "uav_yaw",
    "gim_pitch", "gim_yaw",
    "view_x_512", "view_y_512", "fov_area",
    "atr_count", "atr_x", "atr_y",
)

GRID_ROWS = 7
GRID_COLS = 7
GRID_TOTAL = GRID_ROWS * GRID_COLS  # 49

_GRID_CTX_NAMES = (
    "cur_grid_row_norm", "cur_grid_col_norm",
    "time_in_current_grid_s",
    "dwell_limit_s",
    "lkh_progress",
) + tuple(f"target_g{i:02d}" for i in range(GRID_TOTAL)) \
  + tuple(f"searched_g{i:02d}" for i in range(GRID_TOTAL))

STATE_NAMES = _BASE_STATE_NAMES + _GRID_CTX_NAMES
STATE_DIM = len(STATE_NAMES)  # 14 + 5 + 49 + 49 = 117

LABEL_NAMES = ("pixel_dx", "pixel_dy", "gim_pitch", "gim_yaw", "zoom")
LABEL_DIM = len(LABEL_NAMES)
LABEL_COLS_IN_FLAT_8X9 = (2, 3, 4, 5, 6)  # delta_px_x/y, gimbal_pitch/yaw, zoom
FLAT_DT_COL = 7  # dt_ms column in flat_8x9 (absolute ms from current frame)

HISTORY_LEN_DEFAULT = 50
HORIZON_MAX = 8  # flat_8x9 always has 8 rows; allow truncation to fewer steps.


def _parse_grid_id(gid):
    """Parse 'row-col' to (row, col, idx) or (None, None, None)."""
    if not isinstance(gid, str) or "-" not in gid:
        return None, None, None
    parts = gid.split("-")
    if len(parts) != 2:
        return None, None, None
    try:
        row, col = int(parts[0]), int(parts[1])
    except ValueError:
        return None, None, None
    if not (0 <= row < GRID_ROWS and 0 <= col < GRID_COLS):
        return None, None, None
    return row, col, row * GRID_COLS + col


def _extract_base_state(frame: dict) -> np.ndarray:
    pos = frame["uav_position_m"]
    att = frame["uav_attitude_deg"]
    gim = frame["gimbal_attitude_deg"]
    vc = frame["current_view_center"]
    prm = frame.get("prompt") or {}
    atr_list = prm.get("atr_pixel_coords") or []
    # Raw data: list of [x, y] pairs. gen_data adds w/h/score/type but the
    # geometric signal is just the first candidate's pixel center.
    if atr_list:
        first = atr_list[0]
        if isinstance(first, dict):
            atr_x = float(first.get("cx", 0.0))
            atr_y = float(first.get("cy", 0.0))
        else:
            atr_x = float(first[0]) if len(first) > 0 else 0.0
            atr_y = float(first[1]) if len(first) > 1 else 0.0
    else:
        atr_x = 0.0
        atr_y = 0.0
    return np.asarray([
        pos["x"], pos["y"], pos["z"],
        att["roll"], att["pitch"], att["yaw"],
        gim["pitch"], gim["yaw"],
        vc["grid_512"]["x"], vc["grid_512"]["y"], vc["fov_area_m2"],
        float(prm.get("atr_count", 0)),
        atr_x, atr_y,
    ], dtype=np.float32)


def _extract_state(frame: dict, ep: "_EpisodeFrames", fi: int) -> np.ndarray:
    """14 base dims + grid context (cur grid one-hotish, timer, dwell, progress,
    target & searched bitmasks)."""
    base = _extract_base_state(frame)
    cur_idx = int(ep.frame_grid_idx[fi]) if ep.frame_grid_idx.size else 0
    cur_row = cur_idx // GRID_COLS
    cur_col = cur_idx % GRID_COLS
    ctx_scalar = np.asarray([
        cur_row / max(GRID_ROWS - 1, 1),
        cur_col / max(GRID_COLS - 1, 1),
        float(ep.frame_time_in_grid_s[fi]) if ep.frame_time_in_grid_s.size else 0.0,
        float(ep.dwell_limit_s),
        float(ep.frame_lkh_progress[fi]) if ep.frame_lkh_progress.size else 0.0,
    ], dtype=np.float32)
    target_mask = ep.target_mask.astype(np.float32)
    searched_mask = (ep.frame_searched_mask[fi].astype(np.float32)
                     if ep.frame_searched_mask.size else np.zeros(GRID_TOTAL, dtype=np.float32))
    return np.concatenate([base, ctx_scalar, target_mask, searched_mask])


def _extract_label(frame: dict, ep: "_EpisodeFrames", horizon: int = HORIZON_MAX):
    """Return (label_cont[h, 5], label_grid[h] int64, mask[h])."""
    out = frame.get("output") or {}
    flat = out.get("flat_8x9") or {}
    vals = flat.get("values")
    mask = flat.get("valid_mask")
    if not vals or not mask:
        return None, None, None
    arr = np.asarray(vals, dtype=np.float32)  # (8, 9)
    if arr.shape != (HORIZON_MAX, 9):
        return None, None, None
    h = min(int(horizon), HORIZON_MAX)
    cont = arr[:h, list(LABEL_COLS_IN_FLAT_8X9)]

    # Future grid_idx per step: look up grid at (current_ts + dt_ms[i]) in
    # the episode step table. If steps aren't loaded, fall back to current.
    cur_ts = int(frame.get("timestamp", 0))
    grid_label = np.empty(h, dtype=np.int64)
    if ep.step_ts.size > 0:
        step_ts_list = ep.step_ts.tolist()
        for i in range(h):
            abs_t = cur_ts + int(arr[i, FLAT_DT_COL])
            k = bisect.bisect_right(step_ts_list, abs_t) - 1
            if k < 0:
                k = 0
            if k >= ep.step_grid_idx.size:
                k = ep.step_grid_idx.size - 1
            grid_label[i] = int(ep.step_grid_idx[k])
    else:
        grid_label[:] = 0
    return cont, grid_label, np.asarray(mask[:h], dtype=bool)


@dataclass
class _EpisodeFrames:
    ep_dir: Path
    frames: list
    mode_ids: np.ndarray                  # (N,) phase id per frame
    frame_ts: np.ndarray                  # (N,) frame timestamps, int64 ms
    # ---- grid context (filled by _attach_grid_ctx) ----
    target_mask: np.ndarray = field(default_factory=lambda: np.zeros(GRID_TOTAL, dtype=bool))
    dwell_limit_s: float = 0.0
    step_ts: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    step_grid_idx: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int32))
    step_searched: np.ndarray = field(default_factory=lambda: np.empty((0, GRID_TOTAL), dtype=bool))
    frame_grid_idx: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int32))
    frame_searched_mask: np.ndarray = field(default_factory=lambda: np.empty((0, GRID_TOTAL), dtype=bool))
    frame_time_in_grid_s: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))
    frame_lkh_progress: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))


def _attach_grid_ctx(ep: _EpisodeFrames) -> None:
    """Populate target_mask, dwell_limit_s, and per-frame grid arrays using
    steps.jsonl + meta.json. Robust to either file missing."""
    meta_path = ep.ep_dir / "meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            meta = {}
    else:
        meta = {}
    ep.dwell_limit_s = float(meta.get("dwell_time_s", 0.0))
    lkh_path = meta.get("lkh_path") or []
    tmask = np.zeros(GRID_TOTAL, dtype=bool)
    for gid in lkh_path:
        _, _, idx = _parse_grid_id(gid)
        if idx is not None:
            tmask[idx] = True
    ep.target_mask = tmask
    n_targets = max(int(tmask.sum()), 1)

    steps_path = ep.ep_dir / "steps.jsonl"
    if not steps_path.exists():
        # No steps -> leave per-frame arrays at zero defaults
        n = len(ep.frames)
        ep.frame_grid_idx = np.zeros(n, dtype=np.int32)
        ep.frame_searched_mask = np.zeros((n, GRID_TOTAL), dtype=bool)
        ep.frame_time_in_grid_s = np.zeros(n, dtype=np.float32)
        ep.frame_lkh_progress = np.zeros(n, dtype=np.float32)
        return

    steps = []
    with steps_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            steps.append(json.loads(line))
    steps.sort(key=lambda s: s.get("timestamp", 0))
    step_ts = np.asarray([int(s["timestamp"]) for s in steps], dtype=np.int64)
    step_grids = [_parse_grid_id(s.get("current_grid_id")) for s in steps]
    step_grid_idx = np.asarray(
        [g[2] if g[2] is not None else 0 for g in step_grids], dtype=np.int32)
    # Cumulative searched mask per step entry.
    step_searched = np.zeros((len(steps), GRID_TOTAL), dtype=bool)
    cur = np.zeros(GRID_TOTAL, dtype=bool)
    for k, idx in enumerate(step_grid_idx.tolist()):
        cur[idx] = True
        step_searched[k] = cur
    ep.step_ts = step_ts
    ep.step_grid_idx = step_grid_idx
    ep.step_searched = step_searched

    # Per-frame: latest step whose timestamp <= frame timestamp.
    n = len(ep.frames)
    frame_grid = np.zeros(n, dtype=np.int32)
    frame_searched = np.zeros((n, GRID_TOTAL), dtype=bool)
    frame_time_in_grid = np.zeros(n, dtype=np.float32)
    frame_lkh_progress = np.zeros(n, dtype=np.float32)
    step_ts_list = step_ts.tolist()
    for fi in range(n):
        t = int(ep.frame_ts[fi])
        k = bisect.bisect_right(step_ts_list, t) - 1
        if k < 0:
            k = 0
        frame_grid[fi] = step_grid_idx[k]
        frame_searched[fi] = step_searched[k]
        frame_time_in_grid[fi] = max(0.0, (t - step_ts[k]) / 1000.0)
        frame_lkh_progress[fi] = step_searched[k].sum() / n_targets
    ep.frame_grid_idx = frame_grid
    ep.frame_searched_mask = frame_searched
    ep.frame_time_in_grid_s = frame_time_in_grid
    ep.frame_lkh_progress = frame_lkh_progress


def _load_episode(ep_dir: Path) -> Optional[_EpisodeFrames]:
    fpath = ep_dir / "frames_vla.jsonl"
    if not fpath.exists():
        return None
    frames = []
    with fpath.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            frames.append(json.loads(line))
    if not frames:
        return None
    modes = np.asarray(
        [PHASE_TO_ID.get(fr.get("mode"), -1) for fr in frames],
        dtype=np.int64,
    )
    frame_ts = np.asarray(
        [int(fr.get("timestamp", 0)) for fr in frames], dtype=np.int64,
    )
    ep = _EpisodeFrames(ep_dir=ep_dir, frames=frames, mode_ids=modes, frame_ts=frame_ts)
    _attach_grid_ctx(ep)
    return ep


class StateOnlyDataset:
    """Walks ep_* dirs under `root`, yields (state, mode, label, mask) tuples.

    A sample at index i corresponds to one "current frame" with:
      - state: (history_len, STATE_DIM) float32, padded with current frame
        when history is shorter than history_len at the start of the episode
      - mode: (history_len,) int64, phase id per history step
      - label: (HORIZON, LABEL_DIM) float32, future actions
      - mask: (HORIZON,) bool, validity per future step
      - sample_id: str, "<ep>_<frame_id>" for joining with eval outputs

    Frames where the label flat_8x9 is missing or partially invalid are
    silently skipped.
    """

    def __init__(
        self,
        root: str | Path,
        history_len: int = HISTORY_LEN_DEFAULT,
        horizon: int = HORIZON_MAX,
        phase_filter: Optional[set] = None,
        episode_ids: Optional[set] = None,
        max_episodes: int = 0,
        require_full_horizon: bool = True,
    ):
        self.root = Path(root)
        self.history_len = int(history_len)
        self.horizon = min(int(horizon), HORIZON_MAX)
        self.phase_filter = phase_filter
        self.require_full_horizon = require_full_horizon

        self.episodes: list[_EpisodeFrames] = []
        ep_dirs = sorted(p for p in self.root.glob("ep_*") if p.is_dir())
        for ed in ep_dirs:
            if episode_ids is not None and ed.name not in episode_ids:
                continue
            ep = _load_episode(ed)
            if ep is None:
                continue
            self.episodes.append(ep)
            if max_episodes and len(self.episodes) >= max_episodes:
                break

        # Flat index of (episode_idx, frame_idx) for valid samples.
        self.index: list[tuple[int, int]] = []
        for ei, ep in enumerate(self.episodes):
            for fi, frame in enumerate(ep.frames):
                if phase_filter and frame.get("mode") not in phase_filter:
                    continue
                out = frame.get("output") or {}
                flat = out.get("flat_8x9") or {}
                mask = flat.get("valid_mask")
                if not mask:
                    continue
                if require_full_horizon and not all(mask[:self.horizon]):
                    continue
                self.index.append((ei, fi))

    # --- Dataset API ----------------------------------------------------

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> dict:
        ei, fi = self.index[i]
        ep = self.episodes[ei]
        history_len = self.history_len

        states = np.empty((history_len, STATE_DIM), dtype=np.float32)
        modes = np.empty((history_len,), dtype=np.int64)
        for k in range(history_len):
            # k = 0 is the oldest, k = history_len - 1 is the current frame.
            offset = history_len - 1 - k
            j = max(0, fi - offset)
            states[k] = _extract_state(ep.frames[j], ep, j)
            modes[k] = ep.mode_ids[j]

        frame = ep.frames[fi]
        label, label_grid, mask = _extract_label(frame, ep, horizon=self.horizon)
        return {
            "state": states,
            "mode": modes,
            "label": label,
            "label_grid": label_grid,
            "label_mask": mask,
            "sample_id": f"{int(frame['episode_id']):06d}_{int(frame['frame_id']):06d}",
        }

    # --- Stats helpers --------------------------------------------------

    def compute_stats(self, n_samples: int = 5000, seed: int = 0) -> dict:
        if len(self) == 0:
            return {}
        rng = random.Random(seed)
        idx = rng.sample(range(len(self)), min(n_samples, len(self)))
        states = []
        labels = []
        for k in idx:
            s = self[k]
            # Use the most recent state row (current frame) for state stats.
            states.append(s["state"][-1])
            labels.append(s["label"])
        states = np.stack(states, axis=0)              # (n, STATE_DIM)
        labels = np.concatenate(labels, axis=0)        # (n*HORIZON, LABEL_DIM)
        return {
            "n_samples_used": len(idx),
            "state_mean": states.mean(axis=0).tolist(),
            "state_std": (states.std(axis=0) + 1e-6).tolist(),
            "label_mean": labels.mean(axis=0).tolist(),
            "label_std": (labels.std(axis=0) + 1e-6).tolist(),
        }

    def phase_counts(self) -> Counter:
        ct = Counter()
        for ei, fi in self.index:
            ct[self.episodes[ei].frames[fi].get("mode", "?")] += 1
        return ct


def split_episodes(
    root: str | Path,
    val_ratio: float = 0.1,
    test_ratio: float = 0.0,
    seed: int = 0,
    explicit: Optional[dict] = None,
) -> dict:
    """Partition ep_* directory names under root into train/val/test sets.

    If `explicit` is provided (a dict with keys like {"train": [...], "val": [...]}),
    those lists override any ratio-based logic. Names should be the directory
    basenames (e.g. "ep_000005").

    Ratio-based split is deterministic given `seed` and uses a sorted episode
    list so adding new episodes only appends them (doesn't reshuffle).
    """
    root = Path(root)
    all_eps = sorted(
        p.name for p in root.glob("ep_*") if p.is_dir() and (p / "frames_vla.jsonl").exists()
    )
    if explicit:
        used = set()
        result = {}
        for split, names in explicit.items():
            sset = set(names) & set(all_eps)
            result[split] = sset
            used |= sset
        remaining = [e for e in all_eps if e not in used]
        result.setdefault("train", set()).update(remaining)
        return result

    rng = random.Random(int(seed))
    shuffled = list(all_eps)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_test = int(round(n * float(test_ratio)))
    n_val = int(round(n * float(val_ratio)))
    # Guarantee at least 1 episode per requested split when there is room.
    # Val has priority over test (more important for monitoring); otherwise
    # small datasets (n < 10) silently get empty val/test.
    if float(val_ratio) > 0 and n_val == 0 and n - n_test >= 2:
        n_val = 1
    if float(test_ratio) > 0 and n_test == 0 and n - n_val >= 2:
        n_test = 1
    test = set(shuffled[:n_test])
    val = set(shuffled[n_test:n_test + n_val])
    train = set(shuffled[n_test + n_val:])
    return {"train": train, "val": val, "test": test}


def _print_stats_table(names, mean, std):
    for name, m, s in zip(names, mean, std):
        print(f"  {name:<14} mean={m:>+12.3f}   std={s:>10.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        default="/home/user/remote_disk/0522-Tianjin_Datasets/"
                "20260515_182514__20260516_092922_merged",
    )
    ap.add_argument("--history_len", type=int, default=HISTORY_LEN_DEFAULT)
    ap.add_argument("--horizon", type=int, default=HORIZON_MAX,
                    help=f"Future steps to predict (1..{HORIZON_MAX}, clamped).")
    ap.add_argument("--phase", default="",
                    help="Comma-separated raw mode filter "
                         "(prepare/search/select/first_track/second_track/report). "
                         "Empty = all.")
    ap.add_argument("--max_episodes", type=int, default=0)
    ap.add_argument("--allow_partial_horizon", action="store_true",
                    help="Keep frames whose label mask isn't fully True over horizon.")
    ap.add_argument("--split", choices=["all", "train", "val", "test"], default="all",
                    help="Which split to load. Use 'all' to see the whole set.")
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--test_ratio", type=float, default=0.0)
    ap.add_argument("--split_seed", type=int, default=0)
    ap.add_argument("--episode_list", default="",
                    help="Optional JSON file overriding the random split. "
                         'Format: {"train": ["ep_000005", ...], "val": [...]}')
    args = ap.parse_args()

    phase_filter = ({p.strip() for p in args.phase.split(",") if p.strip()}
                    if args.phase else None)

    explicit = None
    if args.episode_list:
        explicit = json.loads(Path(args.episode_list).read_text())
    splits = split_episodes(
        root=args.root,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.split_seed,
        explicit=explicit,
    )
    print(f"[ds] split sizes (episode count): "
          + ", ".join(f"{k}={len(v)}" for k, v in splits.items()))

    if args.split == "all":
        episode_ids = None  # load everything
    else:
        episode_ids = splits.get(args.split, set())
        if not episode_ids:
            print(f"[ds] split '{args.split}' is empty; nothing to load.")
            return

    print(f"[ds] scanning {args.root}  (split={args.split})")
    ds = StateOnlyDataset(
        root=args.root,
        history_len=args.history_len,
        horizon=args.horizon,
        phase_filter=phase_filter,
        episode_ids=episode_ids,
        max_episodes=args.max_episodes,
        require_full_horizon=not args.allow_partial_horizon,
    )
    print(f"[ds] episodes: {len(ds.episodes)}")
    print(f"[ds] samples : {len(ds)}")
    if len(ds) == 0:
        return

    print("\n[ds] phase distribution (raw mode):")
    for k, v in sorted(ds.phase_counts().items()):
        coarse = PHASE_TO_COARSE.get(k, "?")
        print(f"  {k:<15} {v:>8}   -> coarse={coarse}")

    s0 = ds[0]
    print(f"\n[ds] sample 0:")
    print(f"  sample_id    = {s0['sample_id']}")
    print(f"  state shape  = {s0['state'].shape}    (history × {STATE_DIM} dims)")
    print(f"  mode shape   = {s0['mode'].shape}")
    print(f"  label shape  = {s0['label'].shape}    (horizon × {LABEL_DIM} dims)")
    print(f"  mask         = {s0['label_mask'].tolist()}")

    print("\n[ds] computing normalization stats (sampled current-frame state and 8-step labels)...")
    stats = ds.compute_stats(n_samples=5000)
    print(f"[ds] sampled n = {stats['n_samples_used']}")
    print("\nstate dims:")
    _print_stats_table(STATE_NAMES, stats["state_mean"], stats["state_std"])
    print("\nlabel dims (across all 8 future steps):")
    _print_stats_table(LABEL_NAMES, stats["label_mean"], stats["label_std"])


if __name__ == "__main__":
    main()
