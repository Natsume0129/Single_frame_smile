from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(r"E:\Single_frame_smile\Analysis\Heatmap\output")
OUT_DIR = ROOT / "still_images"
CLASSES = ("polite", "truesmile", "ambiguous")
COLORS = {"polite": "#1f77b4", "truesmile": "#2ca02c", "ambiguous": "#ff7f0e"}
THRESHOLD = 0.5


def extract_last_int(path: Path) -> int:
    m = re.search(r"(\d+)(?!.*\d)", path.stem)
    if not m:
        return 0
    return int(m.group(1))


def list_heatmaps(class_dir: Path) -> list[Path]:
    files = [p for p in class_dir.iterdir() if p.is_file() and p.name.endswith("_heatmap.npy")]
    return sorted(files, key=extract_last_int)


def compute_metrics(heatmap_paths: list[Path]) -> tuple[list[int], list[int], list[float]]:
    x = []
    area = []
    total = []
    for idx, path in enumerate(heatmap_paths):
        arr = np.load(path, allow_pickle=False)
        mask = arr > THRESHOLD
        x.append(idx)
        area.append(int(mask.sum()))
        total.append(float(arr[mask].sum()))
    return x, area, total


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    per_class = {}
    for class_name in CLASSES:
        heatmaps = list_heatmaps(ROOT / class_name)
        per_class[class_name] = compute_metrics(heatmaps)

    fig, ax = plt.subplots(figsize=(9, 5), dpi=160)
    for class_name in CLASSES:
        x, area, _ = per_class[class_name]
        ax.plot(x, area, linewidth=2.0, color=COLORS[class_name], label=class_name)
    ax.set_title(f"Area of heatmap > {THRESHOLD} over time")
    ax.set_xlabel("Frame index")
    ax.set_ylabel("Area (pixel count)")
    ax.legend(loc="best")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "heatmap_area_over_threshold_over_time.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5), dpi=160)
    for class_name in CLASSES:
        x, _, total = per_class[class_name]
        ax.plot(x, total, linewidth=2.0, color=COLORS[class_name], label=class_name)
    ax.set_title(f"Sum of heatmap values > {THRESHOLD} over time")
    ax.set_xlabel("Frame index")
    ax.set_ylabel("Sum of heat values")
    ax.legend(loc="best")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "heatmap_sum_over_threshold_over_time.png")
    plt.close(fig)

    print(f"[THRESHOLD_METRICS] Saved plots to: {OUT_DIR}")


if __name__ == "__main__":
    main()
