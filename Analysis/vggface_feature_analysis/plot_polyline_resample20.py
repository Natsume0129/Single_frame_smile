#!/usr/bin/env python3
# plot_polyline_resample20.py
#
# For each segment output-pt (from extract_base_to_peak.py):
#   delta: [L,D]  (already f(t)-b)
# Resample delta to K nodes (default 20) -> delta_k: [K,D]
# Build an interpretable 2D coordinate system using endpoint direction:
#   u = normalize(delta_k[-1])
#   x_k = delta_k · u
#   y_k = || delta_k - x_k u ||
# Plot polyline through (x_k, y_k) with K nodes.
# Optional: draw arrows for step vectors between nodes.
'''
Docstring for Analysis.vggface_feature_analysis.plot_polyline_resample20

python plot_polyline_resample20.py 
  --pts "E:\Matsuda_data\vgg-face_analysis\classic_segments\basetopeak\ambiguous_2318-2324_rvm_base_to_peak.pt" "E:\Matsuda_data\vgg-face_analysis\classic_segments\basetopeak\polite_0912-0921_rvm_base_to_peak.pt" "E:\Matsuda_data\vgg-face_analysis\classic_segments\basetopeak\true_1544-1551_rvm_base_to_peak.pt"
  --labels ambiguous polite true 
  --K 50 
  --out "E:\Matsuda_data\vgg-face_analysis\classic_segments\gap512_rvm_not_normalized"

'''

import argparse
from pathlib import Path
import torch
import matplotlib.pyplot as plt


def load_delta(pt_path: Path) -> torch.Tensor:
    obj = torch.load(pt_path, map_location="cpu")
    if "delta" not in obj:
        raise KeyError(f"{pt_path} missing 'delta'. keys={list(obj.keys())}")
    delta = obj["delta"]
    if not isinstance(delta, torch.Tensor) or delta.ndim != 2:
        raise ValueError(f"{pt_path} delta must be Tensor[L,D]. got {type(delta)} shape={getattr(delta,'shape',None)}")
    return delta.float()


def resample_time_linear(seq: torch.Tensor, K: int) -> torch.Tensor:
    # seq: [L,D] -> [K,D]
    L, D = seq.shape
    if L < 2:
        raise ValueError(f"Need L>=2 for interpolation, got L={L}")
    t = torch.linspace(0, L - 1, steps=K)
    t0 = torch.floor(t).long()
    t1 = torch.clamp(t0 + 1, max=L - 1)
    w = (t - t0.float()).unsqueeze(1)  # [K,1]
    return (1 - w) * seq[t0] + w * seq[t1]


def progress_deviation(delta_k: torch.Tensor, eps: float = 1e-8):
    # delta_k: [K,D]
    end = delta_k[-1]
    u = end / (end.norm() + eps)               # [D]
    x = (delta_k * u).sum(dim=1)               # [K]
    proj = x.unsqueeze(1) * u.unsqueeze(0)     # [K,D]
    y = (delta_k - proj).norm(dim=1)           # [K]
    return x, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pts", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--K", type=int, default=20, help="number of resampled nodes (default 20)")
    ap.add_argument("--arrows", action="store_true", help="draw step arrows between nodes")
    args = ap.parse_args()

    if len(args.pts) != len(args.labels):
        raise ValueError("--labels length must match --pts")

    plt.figure()

    for p, lab in zip(args.pts, args.labels):
        delta = load_delta(Path(p))              # [L,D]
        delta_k = resample_time_linear(delta, args.K)  # [K,D]
        x, y = progress_deviation(delta_k)       # [K], [K]

        # plot polyline with markers
        plt.plot(x.numpy(), y.numpy(), marker="o", label=lab)

        # optional arrows: step vectors in this 2D space
        if args.arrows and args.K >= 2:
            dx = (x[1:] - x[:-1]).numpy()
            dy = (y[1:] - y[:-1]).numpy()
            plt.quiver(
                x[:-1].numpy(), y[:-1].numpy(),
                dx, dy,
                angles="xy", scale_units="xy", scale=1.0, width=0.003
            )

        # mark start and peak nodes
        plt.scatter([x[0].item()], [y[0].item()])
        plt.scatter([x[-1].item()], [y[-1].item()])

    plt.xlabel("progress along start→peak direction")
    plt.ylabel("deviation from that direction")
    plt.title(f"Resampled polyline pattern (K={args.K})")
    plt.legend()
    plt.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()

    print("[OK] saved:", out_path)


if __name__ == "__main__":
    main()
