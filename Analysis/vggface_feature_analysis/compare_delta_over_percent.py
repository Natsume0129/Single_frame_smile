#!/usr/bin/env python3
# compare_delta_over_percent.py
#
# Compare delta(x%) across labels after time normalization.
# Input: output .pt(s) from extract_base_to_peak.py
# Each pt must contain: delta [L,D], meta, etc.
#
# For each segment:
#   delta_seq(t) = delta[t]  (already f(t)-b)
#   resample to K points along time -> delta_pct[k], k=0..K-1 (0..100%)
#
# For each label:
#   mean_delta_pct[label][k] = average over segments of that label
#
# Then for each pair of labels (A,B):
#   dist[k] = cosine_distance(mean_delta_pct[A][k], mean_delta_pct[B][k])
#   or l2_distance if chosen
#
# Outputs:
#   - pairwise_delta_distance.png
#   - pairwise_delta_distance.csv

'''
Docstring for Analysis.vggface_feature_analysis.compare_delta_over_percent
python compare_delta_over_percent.py 
  --pts "E:\Matsuda_data\vgg-face_analysis\classic_segments\basetopeak\ambiguous_2318-2324_rvm_base_to_peak.pt" "E:\Matsuda_data\vgg-face_analysis\classic_segments\basetopeak\polite_0912-0921_rvm_base_to_peak.pt" "E:\Matsuda_data\vgg-face_analysis\classic_segments\basetopeak\true_1544-1551_rvm_base_to_peak.pt"
  --labels ambiguous polite true 
  --out-dir "E:\Matsuda_data\vgg-face_analysis\classic_segments\gap512_rvm_not_normalized"
  --K 101 
  --metric cos

'''

import argparse
import csv
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


def load_delta(pt_path: Path) -> torch.Tensor:
    obj = torch.load(pt_path, map_location="cpu")
    if "delta" not in obj:
        raise KeyError(f"{pt_path} missing 'delta'. keys={list(obj.keys())}")
    delta = obj["delta"]
    if not isinstance(delta, torch.Tensor) or delta.ndim != 2:
        raise ValueError(f"{pt_path} delta must be Tensor[L,D]. got {type(delta)} shape={getattr(delta,'shape',None)}")
    return delta.float()  # [L,D]


def resample_time_linear(seq: torch.Tensor, K: int) -> torch.Tensor:
    """
    seq: [L,D] -> [K,D] by linear interpolation along time axis.
    """
    L, D = seq.shape
    if L < 2:
        raise ValueError(f"Need L>=2 for interpolation, got L={L}")
    # positions in original index space
    t = torch.linspace(0, L - 1, steps=K)
    t0 = torch.floor(t).long()
    t1 = torch.clamp(t0 + 1, max=L - 1)
    w = (t - t0.float()).unsqueeze(1)  # [K,1]
    out = (1 - w) * seq[t0] + w * seq[t1]
    return out  # [K,D]


def cosine_distance(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    a,b: [K,D] -> [K] cosine distance (1 - cos)
    """
    a_n = a / (a.norm(dim=1, keepdim=True) + eps)
    b_n = b / (b.norm(dim=1, keepdim=True) + eps)
    cos = (a_n * b_n).sum(dim=1).clamp(-1.0, 1.0)
    return 1.0 - cos


def l2_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    a,b: [K,D] -> [K] L2 distance
    """
    return (a - b).norm(dim=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pts", nargs="+", required=True, help="pt files (outputs of extract_base_to_peak.py)")
    ap.add_argument("--labels", nargs="+", required=True, help="label for each pt, same length as --pts")
    ap.add_argument("--out-dir", required=True, help="output directory")
    ap.add_argument("--K", type=int, default=101, help="number of percent points (default 101 => 0..100%)")
    ap.add_argument("--metric", choices=["cos", "l2"], default="cos", help="distance metric on delta(x%)")
    ap.add_argument("--normalize-delta", action="store_true",
                    help="normalize delta vectors per time point before averaging (direction-only)")
    ap.add_argument("--no-mean", action="store_true",
                    help="do NOT average segments by label (treat each pt as its own label instance; mainly for debugging)")
    args = ap.parse_args()

    if len(args.pts) != len(args.labels):
        raise ValueError(f"--labels length must match --pts. got {len(args.labels)} vs {len(args.pts)}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # group resampled delta by label
    grouped: Dict[str, List[torch.Tensor]] = {}
    for p, lab in zip(args.pts, args.labels):
        delta = load_delta(Path(p))           # [L,D]
        delta_k = resample_time_linear(delta, args.K)  # [K,D]
        if args.normalize_delta:
            delta_k = F.normalize(delta_k, dim=1)
        grouped.setdefault(lab, []).append(delta_k)

    # build representative delta trajectory per label
    reps: Dict[str, torch.Tensor] = {}
    if args.no_mean:
        # treat each file as unique key: label#i
        for lab, seqs in grouped.items():
            for i, s in enumerate(seqs):
                reps[f"{lab}#{i+1}"] = s
    else:
        for lab, seqs in grouped.items():
            reps[lab] = torch.stack(seqs, dim=0).mean(dim=0)  # [K,D]

    rep_labels = list(reps.keys())
    if len(rep_labels) < 2:
        raise RuntimeError("Need at least 2 labels/instances to compare.")

    # compute pairwise distance curves
    percent = torch.linspace(0, 100, steps=args.K).tolist()
    pair_curves: List[Tuple[str, str, List[float]]] = []

    for a, b in combinations(rep_labels, 2):
        A = reps[a]  # [K,D]
        B = reps[b]  # [K,D]
        if args.metric == "cos":
            d = cosine_distance(A, B)
            metric_name = "cosine_distance(1-cos)"
        else:
            d = l2_distance(A, B)
            metric_name = "l2_distance"
        pair_curves.append((a, b, d.tolist()))

    # plot
    plt.figure()
    for a, b, d in pair_curves:
        plt.plot(percent, d, label=f"{a} vs {b}")
    plt.xlabel("time (% of start->peak)")
    plt.ylabel(metric_name)
    plt.title("Delta difference over normalized time")
    plt.legend()
    plt.tight_layout()
    out_png = out_dir / "pairwise_delta_distance.png"
    plt.savefig(out_png, dpi=200)
    plt.close()

    # save CSV: columns = percent + each pair
    out_csv = out_dir / "pairwise_delta_distance.csv"
    fieldnames = ["percent"] + [f"{a}_vs_{b}" for a, b, _ in pair_curves]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for i, pct in enumerate(percent):
            row = {"percent": pct}
            for (a, b, d) in pair_curves:
                row[f"{a}_vs_{b}"] = d[i]
            w.writerow(row)

    print("[OK] saved:")
    print(" ", out_png)
    print(" ", out_csv)


if __name__ == "__main__":
    main()
