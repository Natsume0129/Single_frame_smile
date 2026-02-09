# tsne_plot_from_npz.py
# Input: features .npz (paths, feats) + dataset .dat (rel_path label)
# Output: t-SNE scatter plot (blue=0, red=1). Works for any feature dimension D.
#
# Notes:
# - t-SNE is slow on high-D; this script does PCA->50D first (standard practice).
# - Set PERPLEXITY based on N (rule of thumb: 5~50; must be < N).
# - For reproducibility, RANDOM_STATE is fixed.

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# =========================
# Config (edit here only)
# =========================
@dataclass
class CFG:
    # feature npz produced by your extractor
    FEATURES_NPZ: str = r"E:\Single_frame_smile\data\Isornot\features\resnet50_matched_features.npz"

    # dat file: "images/dataset/xxx.png 0/1"
    DAT_FILE: str = r"E:\Single_frame_smile\data\Isornot\matched.dat"

    # output dir + figure name
    OUT_DIR: str = r"E:\Single_frame_smile\data\Isornot\features\tsne_output"
    FIG_NAME: str = "tsne_resnet50_matched.png"  # <- change this name
    # pre-reduction (recommended before t-SNE)
    PRE_PCA_DIM: int = 50

    # t-SNE params
    PERPLEXITY: float = 30.0
    LEARNING_RATE: float = 200.0
    N_ITER: int = 1500
    INIT: str = "pca"          # "pca" or "random"
    METRIC: str = "euclidean"  # keep default

    RANDOM_STATE: int = 42

    # plot
    POINT_SIZE: float = 10.0
    ALPHA: float = 0.6

cfg = CFG()


# =========================
# I/O helpers
# =========================
def read_dat_labels(dat_file: str) -> dict:
    """
    Returns: dict[rel_path] = int(label)
    """
    m = {}
    with open(dat_file, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            rel_path, y = ln.split()
            m[rel_path] = int(y)
    return m


def load_features(npz_path: str):
    z = np.load(npz_path, allow_pickle=True)
    if "paths" not in z or "feats" not in z:
        raise KeyError("NPZ must contain arrays: 'paths' and 'feats'")
    paths = z["paths"]
    feats = z["feats"]
    if feats.ndim != 2:
        raise ValueError(f"Expected feats shape (N, D), got {feats.shape}")
    return paths, feats


# =========================
# Main
# =========================
def main():
    Path(cfg.OUT_DIR).mkdir(parents=True, exist_ok=True)

    paths, feats = load_features(cfg.FEATURES_NPZ)
    label_map = read_dat_labels(cfg.DAT_FILE)

    # Align labels to the order of NPZ paths
    labels = []
    missing = []
    for p in paths:
        p_str = str(p)
        if p_str not in label_map:
            missing.append(p_str)
            labels.append(-1)
        else:
            labels.append(label_map[p_str])

    labels = np.array(labels, dtype=np.int32)

    # Drop missing (fail-safe)
    keep = labels >= 0
    if not np.all(keep):
        print(f"[WARN] {np.sum(~keep)} paths in NPZ not found in DAT. Dropping them.")
        for ex in missing[:5]:
            print("  missing:", ex)

    feats = feats[keep]
    labels = labels[keep]

    n, d = feats.shape
    print(f"N={n}, D={d}")

    # Perplexity must be < N
    if cfg.PERPLEXITY >= n:
        raise ValueError(f"PERPLEXITY ({cfg.PERPLEXITY}) must be < N ({n}).")

    # Pre-PCA to speed up and denoise (standard)
    pre_dim = min(cfg.PRE_PCA_DIM, d)
    if pre_dim < d:
        pca = PCA(n_components=pre_dim, random_state=cfg.RANDOM_STATE)
        feats_low = pca.fit_transform(feats)
        print(f"Pre-PCA: {d} -> {pre_dim}")
    else:
        feats_low = feats

    tsne = TSNE(
        n_components=2,
        perplexity=cfg.PERPLEXITY,
        learning_rate=cfg.LEARNING_RATE,
        max_iter=cfg.N_ITER, 
        init=cfg.INIT,
        metric=cfg.METRIC,
        random_state=cfg.RANDOM_STATE,
        verbose=1,
    )
    xy = tsne.fit_transform(feats_low)

    mask0 = labels == 0
    mask1 = labels == 1

    plt.figure()
    plt.scatter(xy[mask0, 0], xy[mask0, 1], s=cfg.POINT_SIZE, alpha=cfg.ALPHA, label="0")
    plt.scatter(xy[mask1, 0], xy[mask1, 1], s=cfg.POINT_SIZE, alpha=cfg.ALPHA, label="1")
    plt.title(f"t-SNE (D={d}, prePCA={pre_dim}, perplexity={cfg.PERPLEXITY})")
    plt.xlabel("tSNE-1")
    plt.ylabel("tSNE-2")
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(cfg.OUT_DIR, cfg.FIG_NAME)
    plt.savefig(out_path, dpi=300)
    print("Saved figure:", out_path)


if __name__ == "__main__":
    main()
