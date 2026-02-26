#!/usr/bin/env python3
"""
Analyze temporal macro dynamics of smile onset trajectories in feature space.

Input can be provided in either mode:
1) --manifest-csv with columns: pt,label
2) --pts ... --labels ...

Each pt file should be output from extract_base_to_peak.py and contain:
  - delta: Tensor[L,D], where delta(t)=f(t)-baseline

Main idea:
  - Resample each segment to normalized time grid K (0..100%)
  - Compute macro temporal signals:
      amp(t)       = ||delta(t)||
      vel(t)       = ||delta(t+1)-delta(t)||
      progress(t)  = projection on endpoint direction
      deviation(t) = orthogonal magnitude to endpoint direction
      align(t)     = cosine(step_dir, endpoint_dir)
      curv(t)      = ||second difference||
  - Aggregate by label and compare between labels.
"""

from __future__ import annotations

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


def load_delta(pt_path: Path) -> torch.Tensor:
    obj = torch.load(pt_path, map_location="cpu")
    if "delta" not in obj:
        raise KeyError(f"{pt_path} missing 'delta'. keys={list(obj.keys())}")
    delta = obj["delta"]
    if not isinstance(delta, torch.Tensor) or delta.ndim != 2:
        raise ValueError(
            f"{pt_path} delta must be Tensor[L,D], got {type(delta)} shape={getattr(delta, 'shape', None)}"
        )
    if delta.shape[0] < 2:
        raise ValueError(f"{pt_path} has too few frames in delta: L={delta.shape[0]}")
    return delta.float()


def resample_time_linear(seq: torch.Tensor, k: int) -> torch.Tensor:
    # seq: [L,D] -> [K,D]
    l, _ = seq.shape
    t = torch.linspace(0, l - 1, steps=k)
    t0 = torch.floor(t).long()
    t1 = torch.clamp(t0 + 1, max=l - 1)
    w = (t - t0.float()).unsqueeze(1)
    return (1.0 - w) * seq[t0] + w * seq[t1]


def compute_curves(delta_k: torch.Tensor, eps: float = 1e-8) -> Dict[str, np.ndarray]:
    # delta_k: [K,D]
    amp = torch.linalg.norm(delta_k, dim=1)  # [K]

    step = delta_k[1:] - delta_k[:-1]  # [K-1,D]
    vel = torch.linalg.norm(step, dim=1)  # [K-1]

    end = delta_k[-1]
    u = end / (torch.linalg.norm(end) + eps)  # endpoint direction [D]

    progress = (delta_k * u).sum(dim=1)  # [K]
    proj = progress.unsqueeze(1) * u.unsqueeze(0)
    deviation = torch.linalg.norm(delta_k - proj, dim=1)  # [K]

    step_n = step / (torch.linalg.norm(step, dim=1, keepdim=True) + eps)
    align = (step_n * u.unsqueeze(0)).sum(dim=1).clamp(-1.0, 1.0)  # [K-1]

    # second-order finite difference as curvature proxy in feature space
    second = delta_k[2:] - 2.0 * delta_k[1:-1] + delta_k[:-2]  # [K-2,D]
    curv = torch.linalg.norm(second, dim=1)  # [K-2]

    return {
        "amp": amp.numpy(),
        "vel": vel.numpy(),
        "progress": progress.numpy(),
        "deviation": deviation.numpy(),
        "align": align.numpy(),
        "curv": curv.numpy(),
        "delta_k": delta_k.numpy(),
    }


def read_manifest_csv(path: Path) -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"pt", "label"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise RuntimeError(f"Manifest missing fields: {sorted(missing)}")
        for r in reader:
            rows.append((r["pt"], r["label"]))
    if not rows:
        raise RuntimeError(f"No rows found in manifest: {path}")
    return rows


def cohen_d(x: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    nx = len(x)
    ny = len(y)
    if nx < 2 or ny < 2:
        return float("nan")
    vx = np.var(x, ddof=1)
    vy = np.var(y, ddof=1)
    pooled = ((nx - 1) * vx + (ny - 1) * vy) / max(nx + ny - 2, 1)
    return float((np.mean(x) - np.mean(y)) / np.sqrt(pooled + eps))


def summarize_segment_features(curves: Dict[str, np.ndarray], k: int) -> Dict[str, float]:
    amp = curves["amp"]
    vel = curves["vel"]
    deviation = curves["deviation"]
    align = curves["align"]
    curv = curves["curv"]

    peak_idx = int(np.argmax(amp))
    peak_pct = 100.0 * peak_idx / max(k - 1, 1)

    q20 = max(1, int(0.2 * k))
    q50 = max(1, int(0.5 * k))
    early_gain = float(amp[q20 - 1] - amp[0])
    mid_gain = float(amp[q50 - 1] - amp[q20 - 1])

    return {
        "auc_amp": float(np.mean(amp)),
        "peak_amp": float(np.max(amp)),
        "peak_time_pct": peak_pct,
        "mean_vel": float(np.mean(vel)),
        "std_vel": float(np.std(vel)),
        "mean_deviation": float(np.mean(deviation)),
        "max_deviation": float(np.max(deviation)),
        "mean_align": float(np.mean(align)),
        "mean_curv": float(np.mean(curv)) if len(curv) > 0 else 0.0,
        "early_gain_0_20pct": early_gain,
        "mid_gain_20_50pct": mid_gain,
    }


def save_curve_plot(
    curves_by_label: Dict[str, Dict[str, np.ndarray]],
    curve_name: str,
    x: np.ndarray,
    out_path: Path,
    ylabel: str,
) -> None:
    fig = plt.figure(figsize=(8, 5), dpi=160)
    ax = fig.add_subplot(111)
    for label, bundle in sorted(curves_by_label.items(), key=lambda z: z[0]):
        mean = bundle[f"{curve_name}_mean"]
        std = bundle[f"{curve_name}_std"]
        ax.plot(x, mean, label=label)
        ax.fill_between(x, mean - std, mean + std, alpha=0.18)
    ax.set_xlabel("time (% of start->peak)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{curve_name} over normalized time")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare temporal macro dynamics between smile labels from base_to_peak delta trajectories."
    )
    parser.add_argument("--manifest-csv", type=Path, default=None, help="CSV with columns: pt,label")
    parser.add_argument("--pts", nargs="+", default=None, help="List of pt files")
    parser.add_argument("--labels", nargs="+", default=None, help="List of labels, same length as --pts")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--K", type=int, default=101, help="Resample points on normalized time")
    parser.add_argument("--distance-metric", choices=["cos", "l2"], default="cos")
    args = parser.parse_args()

    if args.manifest_csv is not None:
        items = read_manifest_csv(args.manifest_csv)
    else:
        if args.pts is None or args.labels is None:
            raise ValueError("Use either --manifest-csv or both --pts and --labels.")
        if len(args.pts) != len(args.labels):
            raise ValueError("--pts and --labels length mismatch.")
        items = list(zip(args.pts, args.labels))

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    per_label_delta: Dict[str, List[np.ndarray]] = {}
    per_label_curves: Dict[str, Dict[str, List[np.ndarray]]] = {}
    segment_rows: List[dict] = []

    for i, (pt, label) in enumerate(items):
        pt_path = Path(pt)
        delta = load_delta(pt_path)
        delta_k = resample_time_linear(delta, args.K)
        curves = compute_curves(delta_k)

        per_label_delta.setdefault(label, []).append(curves["delta_k"])
        per_label_curves.setdefault(
            label,
            {"amp": [], "vel": [], "progress": [], "deviation": [], "align": [], "curv": []},
        )
        for key in ["amp", "vel", "progress", "deviation", "align", "curv"]:
            per_label_curves[label][key].append(curves[key])

        feat = summarize_segment_features(curves, args.K)
        feat.update({"segment_index": i, "label": label, "pt": str(pt_path)})
        segment_rows.append(feat)

    seg_df = pd.DataFrame(segment_rows)
    seg_df.to_csv(out_dir / "per_segment_macro_features.csv", index=False, encoding="utf-8")

    curve_stats_by_label: Dict[str, Dict[str, np.ndarray]] = {}
    curve_rows: List[dict] = []
    x_full = np.linspace(0, 100, args.K)
    x_km1 = np.linspace(0, 100, args.K - 1)
    x_km2 = np.linspace(0, 100, args.K - 2)

    for label, curves in per_label_curves.items():
        bundle: Dict[str, np.ndarray] = {}
        for name in ["amp", "progress", "deviation"]:
            arr = np.stack(curves[name], axis=0)  # [N,K]
            bundle[f"{name}_mean"] = arr.mean(axis=0)
            bundle[f"{name}_std"] = arr.std(axis=0)
        for name in ["vel", "align"]:
            arr = np.stack(curves[name], axis=0)  # [N,K-1]
            bundle[f"{name}_mean"] = arr.mean(axis=0)
            bundle[f"{name}_std"] = arr.std(axis=0)
        curv_arr = np.stack(curves["curv"], axis=0)  # [N,K-2]
        bundle["curv_mean"] = curv_arr.mean(axis=0)
        bundle["curv_std"] = curv_arr.std(axis=0)
        curve_stats_by_label[label] = bundle

        for j, pct in enumerate(x_full):
            curve_rows.append(
                {
                    "label": label,
                    "percent": pct,
                    "amp_mean": bundle["amp_mean"][j],
                    "amp_std": bundle["amp_std"][j],
                    "progress_mean": bundle["progress_mean"][j],
                    "progress_std": bundle["progress_std"][j],
                    "deviation_mean": bundle["deviation_mean"][j],
                    "deviation_std": bundle["deviation_std"][j],
                }
            )

    pd.DataFrame(curve_rows).to_csv(out_dir / "macro_curves_mean.csv", index=False, encoding="utf-8")

    save_curve_plot(curve_stats_by_label, "amp", x_full, out_dir / "amp_curve_by_label.png", "||delta||")
    save_curve_plot(curve_stats_by_label, "vel", x_km1, out_dir / "vel_curve_by_label.png", "||delta(t+1)-delta(t)||")
    save_curve_plot(curve_stats_by_label, "deviation", x_full, out_dir / "deviation_curve_by_label.png", "orthogonal magnitude")
    save_curve_plot(curve_stats_by_label, "align", x_km1, out_dir / "align_curve_by_label.png", "cos(step, endpoint_dir)")
    save_curve_plot(curve_stats_by_label, "curv", x_km2, out_dir / "curv_curve_by_label.png", "||second diff||")

    # Pairwise distance between mean delta trajectories
    pair_rows: List[dict] = []
    fig = plt.figure(figsize=(8, 5), dpi=160)
    ax = fig.add_subplot(111)
    labels_sorted = sorted(per_label_delta.keys())
    for a, b in combinations(labels_sorted, 2):
        da = np.stack(per_label_delta[a], axis=0).mean(axis=0)  # [K,D]
        db = np.stack(per_label_delta[b], axis=0).mean(axis=0)  # [K,D]

        if args.distance_metric == "cos":
            na = da / (np.linalg.norm(da, axis=1, keepdims=True) + 1e-8)
            nb = db / (np.linalg.norm(db, axis=1, keepdims=True) + 1e-8)
            dist = 1.0 - np.sum(na * nb, axis=1)
            ylabel = "cosine distance (1-cos)"
        else:
            dist = np.linalg.norm(da - db, axis=1)
            ylabel = "L2 distance"

        ax.plot(x_full, dist, label=f"{a} vs {b}")
        for j, pct in enumerate(x_full):
            pair_rows.append(
                {
                    "pair": f"{a}_vs_{b}",
                    "percent": pct,
                    "distance": float(dist[j]),
                    "metric": args.distance_metric,
                }
            )

    ax.set_xlabel("time (% of start->peak)")
    ax.set_ylabel(ylabel)
    ax.set_title("Distance between label mean trajectories")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"mean_delta_distance_{args.distance_metric}.png")
    plt.close(fig)
    pd.DataFrame(pair_rows).to_csv(out_dir / "mean_delta_pairwise_distance.csv", index=False, encoding="utf-8")

    group_summary = (
        seg_df.groupby("label", dropna=False)
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    group_summary.columns = ["_".join([c for c in col if c]) for col in group_summary.columns.values]
    group_summary.to_csv(out_dir / "group_macro_summary.csv", index=False, encoding="utf-8")

    effect_sizes: Dict[str, float] = {}
    if len(labels_sorted) == 2:
        l0, l1 = labels_sorted
        for col in [
            "auc_amp",
            "peak_amp",
            "peak_time_pct",
            "mean_vel",
            "std_vel",
            "mean_deviation",
            "max_deviation",
            "mean_align",
            "mean_curv",
            "early_gain_0_20pct",
            "mid_gain_20_50pct",
        ]:
            x = seg_df.loc[seg_df["label"] == l0, col].to_numpy(dtype=float)
            y = seg_df.loc[seg_df["label"] == l1, col].to_numpy(dtype=float)
            effect_sizes[col] = cohen_d(x, y)

    summary = {
        "num_segments": len(seg_df),
        "labels": labels_sorted,
        "num_segments_by_label": {k: int(len(v)) for k, v in per_label_delta.items()},
        "K": int(args.K),
        "distance_metric": args.distance_metric,
        "effect_size_cohen_d_if_two_labels": effect_sizes,
        "outputs": {
            "per_segment": str(out_dir / "per_segment_macro_features.csv"),
            "group_summary": str(out_dir / "group_macro_summary.csv"),
            "curve_mean_csv": str(out_dir / "macro_curves_mean.csv"),
            "pair_distance_csv": str(out_dir / "mean_delta_pairwise_distance.csv"),
        },
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[OK] done")
    print(f"segments={len(seg_df)}, labels={labels_sorted}, K={args.K}")
    print(f"saved to: {out_dir}")


if __name__ == "__main__":
    main()
