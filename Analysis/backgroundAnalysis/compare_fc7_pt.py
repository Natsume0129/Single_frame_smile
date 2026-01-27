# compare_fc7_pt.py
# Usage:
#   python compare_fc7_pt.py --bg_pt path/to/bg.pt --rvm_pt path/to/rvm.pt --out_dir out --k 20
#
# Expected .pt formats (any of these):
#   1) Tensor [T, D]
#   2) Dict with a tensor under common keys: 'feat','feats','fc7','features'
#   3) List of tensors length T (each [D] or [1,D])

import argparse
import os
import json
from typing import Tuple, Union, Any

import torch


COMMON_KEYS = ["feat", "feats", "fc7", "features", "embedding", "embeddings", "x"]


def _to_td_tensor(obj: Any) -> torch.Tensor:
    """
    Convert various saved formats into a float32 Tensor of shape [T, D].
    """
    if torch.is_tensor(obj):
        x = obj
    elif isinstance(obj, dict):
        found = None
        for k in COMMON_KEYS:
            if k in obj and torch.is_tensor(obj[k]):
                found = obj[k]
                break
        if found is None:
            # try first tensor value
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
        x = x.unsqueeze(0)  # [1, D]
    elif x.dim() == 2:
        pass  # [T, D]
    elif x.dim() == 3 and x.size(1) == 1:
        x = x.squeeze(1)  # [T, D]
    else:
        raise ValueError(f"Unsupported tensor shape: {tuple(x.shape)} (expect [T,D] or similar)")

    return x.detach().to(torch.float32).cpu()


def l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(dim=1, keepdim=True) + eps)


def cosine_distance(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    # u,v are L2-normalized: [T,D]
    return 1.0 - (u * v).sum(dim=1)


def relative_change(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    # x,y: [T,D] (not normalized)
    return (x - y).norm(dim=1) / (x.norm(dim=1) + eps)


def neighbor_overlap(u: torch.Tensor, v: torch.Tensor, k: int = 20) -> torch.Tensor:
    """
    For each t, compute kNN indices within the same clip (excluding self) in u-space and v-space,
    then compute overlap ratio |N_u ∩ N_v| / k.

    Complexity: O(T^2) memory/time for cdist; suitable for T up to a few thousands.
    """
    T = u.size(0)
    k_eff = min(k, max(T - 1, 1))

    du = torch.cdist(u, u)  # [T,T]
    dv = torch.cdist(v, v)

    # exclude self by setting diagonal large
    inf = torch.tensor(float("inf"))
    du.fill_diagonal_(inf)
    dv.fill_diagonal_(inf)

    iu = torch.topk(du, k=k_eff, largest=False).indices  # [T,k]
    iv = torch.topk(dv, k=k_eff, largest=False).indices

    # compute overlap per row
    # Use set intersection via broadcasting equality (T,k,k) -> count matches
    overlap = (iu.unsqueeze(2) == iv.unsqueeze(1)).any(dim=2).float().sum(dim=1) / float(k_eff)
    return overlap


def temporal_smoothness(u: torch.Tensor) -> torch.Tensor:
    """
    Adjacent-frame distance: ||u(t)-u(t+1)||_2 for t=0..T-2
    """
    return (u[1:] - u[:-1]).norm(dim=1)


def pca_top_explained(x: torch.Tensor) -> Tuple[float, float]:
    """
    Return explained variance ratio of PC1 and cumulative PC1-2 on x [T,D].
    Uses SVD on centered data.
    """
    x = x - x.mean(dim=0, keepdim=True)
    # SVD on [T,D] with T usually smaller than D; full_matrices=False
    # singular values s relate to variance: var = s^2 / (T-1)
    U, S, Vh = torch.linalg.svd(x, full_matrices=False)
    if S.numel() == 0:
        return 0.0, 0.0
    var = (S ** 2) / max(x.size(0) - 1, 1)
    total = var.sum().item() if var.numel() > 0 else 0.0
    if total <= 0:
        return 0.0, 0.0
    pc1 = (var[0].item() / total)
    pc12 = (var[:2].sum().item() / total) if var.numel() >= 2 else pc1
    return pc1, pc12


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
    ap.add_argument("--bg_pt", required=True, help="pt file for original background features")
    ap.add_argument("--rvm_pt", required=True, help="pt file for RVM (green background) features")
    ap.add_argument("--out_dir", default="out_compare", help="output directory")
    ap.add_argument("--k", type=int, default=20, help="k for kNN overlap (within-clip)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    bg_obj = torch.load(args.bg_pt, map_location="cpu")
    rvm_obj = torch.load(args.rvm_pt, map_location="cpu")

    bg = _to_td_tensor(bg_obj)
    rvm = _to_td_tensor(rvm_obj)

    if bg.shape != rvm.shape:
        raise ValueError(f"Shape mismatch: bg {tuple(bg.shape)} vs rvm {tuple(rvm.shape)}")

    T, D = bg.shape
    print(f"[INFO] Loaded: T={T}, D={D}")

    ubg = l2_normalize(bg)
    urvm = l2_normalize(rvm)

    # 1) per-frame cosine distance
    dcos = cosine_distance(ubg, urvm)

    # 2) relative change (amplitude-sensitive)
    rchg = relative_change(bg, rvm)

    # 3) delta shift stats
    delta = ubg - urvm
    mean_delta_norm = float(delta.mean(dim=0).norm().item())
    pc1, pc12 = pca_top_explained(delta)

    # 4) kNN overlap within clip
    try:
        ov = neighbor_overlap(ubg, urvm, k=args.k)
        ov_summary = summarize("knn_overlap", ov)
    except RuntimeError as e:
        # if T too large and memory hits, skip
        print(f"[WARN] kNN overlap skipped due to: {e}")
        ov = None
        ov_summary = {}

    # 5) temporal smoothness
    sm_bg = temporal_smoothness(ubg)
    sm_rvm = temporal_smoothness(urvm)

    # Summaries
    report = {
        "meta": {
            "T": int(T),
            "D": int(D),
            "k": int(args.k),
            "bg_pt": os.path.abspath(args.bg_pt),
            "rvm_pt": os.path.abspath(args.rvm_pt),
        },
        "cosine_distance": summarize("dcos", dcos),
        "relative_change": summarize("rchg", rchg),
        "delta_shift": {
            "mean_delta_norm": mean_delta_norm,
            "delta_pca_pc1_explained": float(pc1),
            "delta_pca_pc1_2_explained": float(pc12),
        },
        "temporal_smoothness_bg": summarize("smooth_bg", sm_bg),
        "temporal_smoothness_rvm": summarize("smooth_rvm", sm_rvm),
        **ov_summary,
    }

    # Save raw vectors for plotting elsewhere
    torch.save(
        {
            "dcos": dcos,
            "rchg": rchg,
            "smooth_bg": sm_bg,
            "smooth_rvm": sm_rvm,
            "mean_delta_norm": mean_delta_norm,
            "delta_pca_pc1_explained": pc1,
            "delta_pca_pc1_2_explained": pc12,
            "knn_overlap": ov,
        },
        os.path.join(args.out_dir, "raw_metrics.pt"),
    )

    with open(os.path.join(args.out_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("[DONE] Wrote:")
    print(" -", os.path.join(args.out_dir, "report.json"))
    print(" -", os.path.join(args.out_dir, "raw_metrics.pt"))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
