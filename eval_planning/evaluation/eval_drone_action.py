"""Evaluate drone search/track action predictions.

Inputs
  --result_file : JSON from inference_scripts.infer_drone_action_token
                  -> list of {id, predict_vec, action_token_ids}
  --gt_file     : the val JSON used as dataset_use (parses assistant text)
  --save_file   : where the metrics summary is written

Metrics
  per-dim MAE / RMSE for the 6 dims
                  (pixel_dx, pixel_dy, gimbal_pitch, gimbal_yaw, zoom, grid_idx)
  grid_idx exact-match accuracy (after rounding to nearest int)
  row / col absolute error  (using --grid_cols to decompose grid_idx)

Also separates metrics by phase (SEARCH / TRACK / SELECT) if "Current Phase:" is
present in the user prompt.
"""

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path


DIM_NAMES = ("pixel_dx", "pixel_dy", "gimbal_pitch", "gimbal_yaw", "zoom", "grid_idx")
GRID_IDX_DIM = DIM_NAMES.index("grid_idx")


def parse_action_text(text):
    """Return list of 6 floats from the assistant text, or None on failure."""
    m = re.search(r"\(([^()]*)\)", text)
    if not m:
        return None
    parts = [p.strip() for p in m.group(1).split(",")]
    if len(parts) != len(DIM_NAMES):
        return None
    try:
        return [float(p) for p in parts]
    except ValueError:
        return None


def parse_phase(user_text):
    m = re.search(r"Current Phase:\s*([A-Z_]+)", user_text or "")
    return m.group(1) if m else "UNKNOWN"


def aggregate(rows, grid_cols):
    n = len(rows)
    if n == 0:
        return {"n": 0}
    sum_abs = [0.0] * len(DIM_NAMES)
    sum_sq = [0.0] * len(DIM_NAMES)
    grid_correct = 0
    row_abs = 0.0
    col_abs = 0.0
    for gt, pr in rows:
        for k in range(len(DIM_NAMES)):
            d = pr[k] - gt[k]
            sum_abs[k] += abs(d)
            sum_sq[k] += d * d
        gt_g = int(round(gt[GRID_IDX_DIM]))
        pr_g = int(round(pr[GRID_IDX_DIM]))
        if gt_g == pr_g:
            grid_correct += 1
        row_abs += abs(gt_g // grid_cols - pr_g // grid_cols)
        col_abs += abs(gt_g % grid_cols - pr_g % grid_cols)
    return {
        "n": n,
        "per_dim_mae": {DIM_NAMES[k]: round(sum_abs[k] / n, 4) for k in range(len(DIM_NAMES))},
        "per_dim_rmse": {DIM_NAMES[k]: round(math.sqrt(sum_sq[k] / n), 4) for k in range(len(DIM_NAMES))},
        "grid_idx_accuracy": round(grid_correct / n, 4),
        "grid_row_mae": round(row_abs / n, 4),
        "grid_col_mae": round(col_abs / n, 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result_file", required=True)
    ap.add_argument("--gt_file", required=True)
    ap.add_argument("--save_file", required=True)
    ap.add_argument("--grid_cols", type=int, default=7,
                    help="Used to decompose grid_idx into (row, col).")
    args = ap.parse_args()

    preds = {p["id"]: p for p in json.load(open(args.result_file))}
    gt_samples = json.load(open(args.gt_file))

    all_rows = []
    by_phase = defaultdict(list)
    skipped = {"no_pred": 0, "bad_gt": 0, "bad_pred": 0}

    for s in gt_samples:
        sid = s["id"]
        p = preds.get(sid)
        if p is None:
            skipped["no_pred"] += 1
            continue
        gt_vec = parse_action_text(s["messages"][1]["content"])
        if gt_vec is None:
            skipped["bad_gt"] += 1
            continue
        pred_vec = p.get("predict_vec")
        if not pred_vec or len(pred_vec) != len(DIM_NAMES):
            skipped["bad_pred"] += 1
            continue
        all_rows.append((gt_vec, pred_vec))
        phase = parse_phase(s["messages"][0]["content"])
        by_phase[phase].append((gt_vec, pred_vec))

    metrics = {
        "overall": aggregate(all_rows, args.grid_cols),
        "by_phase": {ph: aggregate(rows, args.grid_cols) for ph, rows in by_phase.items()},
        "skipped": skipped,
        "grid_cols": args.grid_cols,
    }

    Path(args.save_file).parent.mkdir(parents=True, exist_ok=True)
    Path(args.save_file).write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    print(f"Wrote metrics -> {args.save_file}")


if __name__ == "__main__":
    main()
