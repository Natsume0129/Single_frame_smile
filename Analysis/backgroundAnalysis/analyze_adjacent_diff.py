# analyze_adjacent_diff.py
# Purpose:
#   Adjacent-frame difference analysis within a single clip:
#   - BG:  cos_dist(u_bg[t], u_bg[t+1])
#   - RVM: cos_dist(u_rvm[t], u_rvm[t+1])
#   - Cross-domain (optional): cos_dist(u_bg[t], u_rvm[t])  (same-frame)
#
# Usage:
#   python analyze_adjacent_diff.py --bg_pt path/to/bg.pt --rvm_pt path/to/rvm.pt --out_dir out_adj
#
# Output:
#   out_dir/
#     report_adj.json
#     raw_adj_metrics.pt

import argparse
import os
import json
from typing import Any, Tuple

import torch

COMMON_KEYS = ["feat", "feats", "fc7", "features", "embedding", "embeddings", "x"]


def _to_td_tensor(obj: Any) -> torch.Tensor:
    if torch.is_tensor(obj):
        x = obj
    elif isinstance(obj, dict):
        found = None
        for k in COMMON_KEYS:
            if k in obj and torch.is_tensor(obj[k]):
                found = obj[k]
                break
        if found is None:
            for v in obj.values():
                if torch.is_tensor(v):
                    found = v
                    break
        if found is None:
            raise ValueError(f"Dict contains no tensor under keys {COMMON_KEYS} and no tensor values.")
        x = found
    elif isinstance(obj, (list, tuple)):
        if len(obj) == 0:
            raise ValueError("Empty list/tuple in pt file.")
        if all(torch.is_tensor(v) for v in obj):
            x = torch.stack([v.squeeze() for v in obj], dim=0)
        else:
            raise ValueError("List/tuple contains non-tensor elements; unsupported.")
    else:
        raise ValueError(f"Unsupported pt content type: {type(obj)}")

    if x.dim() == 1:
        x = x.unsqueeze(0)
    elif x.dim() == 2:
        pass
    elif x.dim() == 3 and x.size(1) == 1:
        x = x.squeeze(1)
    else:
        raise ValueError(f"Unsupported tensor shape: {tuple(x.shape)} (expect [T,D] or similar)")

    return x.detach().to(torch.float32).cpu()


def l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(dim=1, keepdim=True) + eps)


def cosine_distance_pairwise(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    # u,v must be L2-normalized with same shape [N,D]
    return 1.0 - (u * v).sum(dim=1)


def summarize(name: str, x: torch.Tensor) -> dict:
    x = x.cpu()
    return {
        f"{name}_mean": float(x.mean().item()),
        f"{name}_median": float(x.median().item()),
        f"{name}_p95": float(x.quantile(0.95).item()),
        f"{name}_p99": float(x.quantile(0.99).item()),
        f"{name}_min": float(x.min().item()),
        f"{name}_max": float(x.max().item()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bg_pt", required=True)
    ap.add_argument("--rvm_pt", required=True)
    ap.add_argument("--out_dir", default="out_adj")
    ap.add_argument("--also_same_frame", action="store_true",
                    help="also compute same-frame bg-vs-rvm cosine distance for reference")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    bg = _to_td_tensor(torch.load(args.bg_pt, map_location="cpu"))
    rvm = _to_td_tensor(torch.load(args.rvm_pt, map_location="cpu"))

    if bg.shape != rvm.shape:
        raise ValueError(f"Shape mismatch: bg {tuple(bg.shape)} vs rvm {tuple(rvm.shape)}")

    T, D = bg.shape
    if T < 2:
        raise ValueError(f"Need at least 2 frames for adjacent analysis, got T={T}")

    ubg = l2_normalize(bg)
    urvm = l2_normalize(rvm)

    # Adjacent diffs within each domain
    d_bg_adj = cosine_distance_pairwise(ubg[:-1], ubg[1:])       # length T-1
    d_rvm_adj = cosine_distance_pairwise(urvm[:-1], urvm[1:])    # length T-1

    # Optional: same-frame cross-domain reference
    d_same = None
    if args.also_same_frame:
        d_same = cosine_distance_pairwise(ubg, urvm)             # length T

    report = {
        "meta": {
            "T": int(T),
            "D": int(D),
            "bg_pt": os.path.abspath(args.bg_pt),
            "rvm_pt": os.path.abspath(args.rvm_pt),
        },
        "adjacent_bg": summarize("bg_adj", d_bg_adj),
        "adjacent_rvm": summarize("rvm_adj", d_rvm_adj),
    }

    if d_same is not None:
        report["same_frame_bg_vs_rvm"] = summarize("same", d_same)

    torch.save(
        {
            "bg_adj": d_bg_adj,
            "rvm_adj": d_rvm_adj,
            "same_frame": d_same,
        },
        os.path.join(args.out_dir, "raw_adj_metrics.pt"),
    )

    with open(os.path.join(args.out_dir, "report_adj.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("[DONE] Wrote:")
    print(" -", os.path.join(args.out_dir, "report_adj.json"))
    print(" -", os.path.join(args.out_dir, "raw_adj_metrics.pt"))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
