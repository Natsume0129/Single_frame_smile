from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUTPUT_ROOT = Path(r"E:\Matsuda_data\DTW_analysis")
CSV_DIR = OUTPUT_ROOT / "csv"
PLOTS_DIR = OUTPUT_ROOT / "plots"

BRANCHES = [
    "magnitude_dtw",
    "magnitude_dtw_band",
    "velocity_dtw",
    "velocity_dtw_band",
    "pca10_dtw",
    "pca10_dtw_band",
    "pca20_dtw",
    "pca20_dtw_band",
    "pca30_dtw",
    "pca30_dtw_band",
]

PAIRS = [
    "polite_vs_polite",
    "ambiguous_vs_ambiguous",
    "truesmile_vs_truesmile",
    "ambiguous_vs_polite",
    "polite_vs_truesmile",
    "ambiguous_vs_truesmile",
]

PAIR_LABELS = {
    "polite_vs_polite": "polite-polite",
    "ambiguous_vs_ambiguous": "ambiguous-ambiguous",
    "truesmile_vs_truesmile": "truesmile-truesmile",
    "ambiguous_vs_polite": "ambiguous-polite",
    "polite_vs_truesmile": "polite-truesmile",
    "ambiguous_vs_truesmile": "ambiguous-truesmile",
}


def load_branch_stats(branch: str) -> dict[str, float]:
    path = CSV_DIR / f"dtw_statistics_{branch}.csv"
    out: dict[str, float] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            out[row["pair"]] = float(row["median"])
    return out


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    raw = np.zeros((len(PAIRS), len(BRANCHES)), dtype=np.float64)
    normalized = np.zeros_like(raw)
    best_inter_vs_intra = np.zeros((len(BRANCHES),), dtype=np.float64)

    for j, branch in enumerate(BRANCHES):
        stats = load_branch_stats(branch)
        vals = np.asarray([stats[pair] for pair in PAIRS], dtype=np.float64)
        raw[:, j] = vals
        vmin = float(np.min(vals))
        vmax = float(np.max(vals))
        if vmax - vmin <= 1e-12:
            normalized[:, j] = 0.0
        else:
            normalized[:, j] = (vals - vmin) / (vmax - vmin)

        intra_mean = np.mean([stats["polite_vs_polite"], stats["ambiguous_vs_ambiguous"], stats["truesmile_vs_truesmile"]])
        best_inter = min(stats["ambiguous_vs_polite"], stats["polite_vs_truesmile"], stats["ambiguous_vs_truesmile"])
        best_inter_vs_intra[j] = best_inter / intra_mean

    fig = plt.figure(figsize=(14, 9), dpi=160)
    gs = fig.add_gridspec(2, 1, height_ratios=[3.2, 1.2], hspace=0.28)

    ax1 = fig.add_subplot(gs[0, 0])
    im = ax1.imshow(normalized, aspect="auto", cmap="YlGnBu", origin="upper")
    ax1.set_xticks(range(len(BRANCHES)))
    ax1.set_xticklabels(BRANCHES, rotation=30, ha="right")
    ax1.set_yticks(range(len(PAIRS)))
    ax1.set_yticklabels([PAIR_LABELS[p] for p in PAIRS])
    ax1.set_title("Branch-wise normalized median DTW distance\n(lower = more similar within each branch)")
    cbar = fig.colorbar(im, ax=ax1, fraction=0.028, pad=0.02)
    cbar.set_label("Normalized median DTW")

    for i in range(len(PAIRS)):
        for j in range(len(BRANCHES)):
            value = raw[i, j]
            ax1.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="black",
            )

    ax2 = fig.add_subplot(gs[1, 0])
    x = np.arange(len(BRANCHES))
    ax2.plot(x, best_inter_vs_intra, marker="o", linewidth=2.0, color="#d62728")
    ax2.axhline(1.0, linestyle="--", color="gray", linewidth=1.2)
    ax2.set_xticks(x)
    ax2.set_xticklabels(BRANCHES, rotation=30, ha="right")
    ax2.set_ylabel("best inter / mean intra")
    ax2.set_title("How close the best inter-class relation is to the average intra-class compactness")
    ax2.grid(alpha=0.25)

    for j, v in enumerate(best_inter_vs_intra):
        ax2.text(j, v + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle(
        "DTW Summary Figure\n"
        "Across methods, ambiguous-polite is the closest inter-class pair;\n"
        "intra-class compactness is not extremely tight and sometimes overlaps with inter-class closeness.",
        fontsize=14,
        y=0.98,
    )
    fig.subplots_adjust(left=0.09, right=0.96, bottom=0.10, top=0.90, hspace=0.34)
    fig.savefig(PLOTS_DIR / "dtw_conclusion_summary.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
