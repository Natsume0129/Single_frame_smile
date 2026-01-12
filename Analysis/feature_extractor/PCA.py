# pca_plot_from_npz.py
# Input: features .npz (paths, feats) + dataset .dat (rel_path label)
# Output: PCA scatter plot (blue=0, red=1). Works for any feature dimension D.

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# =========================
# Config (edit here only)
# =========================
@dataclass
class CFG:
    # feature npz produced by your extractor
    FEATURES_NPZ: str = r"E:\Single_frame_smile\data\Isornot\features\vgg16_matched_features.npz"

    # dat file: "images/dataset/xxx.png 0/1"
    DAT_FILE: str = r"E:\Single_frame_smile\data\Isornot\matched.dat"

    # output dir + figure name
    OUT_DIR: str = r"E:\Single_frame_smile\data\Isornot\features\pca_output"
    FIG_NAME: str = "pca_vgg16_matched.png"  # <- change this name

    # PCA params
    N_COMPONENTS: int = 2
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

    # If some paths are missing in dat, drop them (fail-safe)
    keep = labels >= 0
    if not np.all(keep):
        print(f"[WARN] {np.sum(~keep)} paths in NPZ not found in DAT. They will be dropped.")
        # show a few examples
        for ex in missing[:5]:
            print("  missing:", ex)

    feats = feats[keep]
    labels = labels[keep]
    paths = paths[keep]

    # PCA (dimension-agnostic)
    pca = PCA(n_components=cfg.N_COMPONENTS, random_state=cfg.RANDOM_STATE)
    xy = pca.fit_transform(feats)

    # Colors: blue for 0, red for 1
    mask0 = labels == 0
    mask1 = labels == 1

    plt.figure()
    plt.scatter(xy[mask0, 0], xy[mask0, 1], s=cfg.POINT_SIZE, alpha=cfg.ALPHA, label="0 (not smile)")
    plt.scatter(xy[mask1, 0], xy[mask1, 1], s=cfg.POINT_SIZE, alpha=cfg.ALPHA, label="1 (smile)")

    # Title includes explained variance (useful in slides)
    evr = pca.explained_variance_ratio_
    title = f"PCA (D={feats.shape[1]})  PC1 {evr[0]*100:.2f}%  PC2 {evr[1]*100:.2f}%"
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(cfg.OUT_DIR, cfg.FIG_NAME)
    plt.savefig(out_path, dpi=300)
    print("Saved figure:", out_path)
    print("N =", feats.shape[0], "D =", feats.shape[1])


if __name__ == "__main__":
    main()
