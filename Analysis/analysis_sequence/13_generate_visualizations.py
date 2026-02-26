from __future__ import annotations

import csv
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from common.base import CLASS_NAMES, PipelineConfig, SequenceTaskBase


COLORS = {"polite": "#1f77b4", "truesmile": "#2ca02c", "ambiguous": "#ff7f0e"}


class VisualizationTask(SequenceTaskBase):
    def _load_class_curves(self, category: str, file_name: str) -> dict[str, np.ndarray]:
        grouped: dict[str, list[np.ndarray]] = defaultdict(list)
        for seq in self.discover_sequences():
            arr = self.load_npy(self.metrics_seq_dir(category, seq) / file_name)
            grouped[seq.class_name].append(arr.astype(np.float32))
        out: dict[str, np.ndarray] = {}
        for cls, values in grouped.items():
            out[cls] = np.stack(values, axis=0).mean(axis=0)
        return out

    def _save_curve_csv(self, path, curves: dict[str, np.ndarray]) -> None:
        rows = []
        n = self.cfg.norm_len
        for cls, curve in curves.items():
            for t in range(n):
                rows.append({"class": cls, "t_index": t, "value": float(curve[t])})
        self.write_csv(path, rows, ["class", "t_index", "value"])

    def run(self) -> None:
        plots = self.cfg.output_root / "plots"
        plots.mkdir(parents=True, exist_ok=True)

        d_curves = self._load_class_curves("normalized", "distance_norm.npy")
        v_curves = self._load_class_curves("normalized", "velocity_norm.npy")

        self._save_curve_csv(self.cfg.output_root / "csv" / "mean_magnitude_curve.csv", d_curves)
        self._save_curve_csv(self.cfg.output_root / "csv" / "mean_velocity_curve.csv", v_curves)

        t = np.arange(self.cfg.norm_len)
        fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
        for cls in CLASS_NAMES:
            if cls in d_curves:
                ax.plot(t, d_curves[cls], linewidth=2.0, color=COLORS[cls], label=cls)
        ax.set_title("Mean Magnitude Curve by Class")
        ax.set_xlabel("Normalized Time Index")
        ax.set_ylabel("Magnitude")
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(plots / "mean_magnitude_curve.png")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
        for cls in CLASS_NAMES:
            if cls in v_curves:
                ax.plot(t, v_curves[cls], linewidth=2.0, color=COLORS[cls], label=cls)
        ax.set_title("Mean Velocity Curve by Class")
        ax.set_xlabel("Normalized Time Index")
        ax.set_ylabel("Velocity")
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(plots / "mean_velocity_curve.png")
        plt.close(fig)

        curve_csv = self.cfg.output_root / "csv" / "class_distance_curve.csv"
        rows = []
        with curve_csv.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
        grouped: dict[str, list[tuple[int, float]]] = defaultdict(list)
        for r in rows:
            grouped[r["pair"]].append((int(r["t_index"]), float(r["diff_norm"])))
        fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
        for pair, items in grouped.items():
            items = sorted(items, key=lambda x: x[0])
            ax.plot([i[0] for i in items], [i[1] for i in items], linewidth=2.0, label=pair)
        ax.set_title("Class Distance Over Time")
        ax.set_xlabel("Normalized Time Index")
        ax.set_ylabel("Distance Norm")
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(plots / "class_distance_over_time.png")
        plt.close(fig)

        dur_rows = []
        with (self.cfg.output_root / "csv" / "duration_per_sequence.csv").open(
            "r", encoding="utf-8", newline=""
        ) as f:
            reader = csv.DictReader(f)
            for row in reader:
                dur_rows.append(row)
        dur_grouped: dict[str, list[float]] = defaultdict(list)
        for r in dur_rows:
            dur_grouped[r["class"]].append(float(r["duration_seconds"]))
        fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
        for cls in CLASS_NAMES:
            if cls in dur_grouped:
                ax.hist(dur_grouped[cls], bins=15, alpha=0.45, label=cls, color=COLORS[cls])
        ax.set_title("Duration Distribution by Class")
        ax.set_xlabel("Duration (seconds)")
        ax.set_ylabel("Count")
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(plots / "duration_distribution.png")
        plt.close(fig)

        print("[STEP13] Saved required visualization outputs.")


def main() -> None:
    parser = SequenceTaskBase.build_common_arg_parser("Step 13: generate required visualizations.")
    args = parser.parse_args()
    task = VisualizationTask(PipelineConfig.from_args(args))
    task.run()


if __name__ == "__main__":
    main()

