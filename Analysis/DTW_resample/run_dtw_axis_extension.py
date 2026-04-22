from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import importlib.util
import matplotlib.pyplot as plt
import numpy as np
import sys


ANALYSIS_PROJECTION_COMMON = Path(__file__).resolve().parent.parent / "analysis_projection" / "common.py"
_proj_spec = importlib.util.spec_from_file_location("analysis_projection_common_axis_extension", ANALYSIS_PROJECTION_COMMON)
if _proj_spec is None or _proj_spec.loader is None:
    raise RuntimeError(f"Cannot load analysis_projection common module from {ANALYSIS_PROJECTION_COMMON}")
_proj_module = importlib.util.module_from_spec(_proj_spec)
sys.modules["analysis_projection_common_axis_extension"] = _proj_module
_proj_spec.loader.exec_module(_proj_module)
CLASS_NAMES = _proj_module.CLASS_NAMES
compute_axis_metrics = _proj_module.compute_axis_metrics
compute_summary_stats = _proj_module.compute_summary_stats


COLORS = {"polite": "#1f77b4", "truesmile": "#2ca02c", "ambiguous": "#ff7f0e"}


@dataclass
class AxisExtensionConfig:
    dtw_resample_root: Path = Path(r"E:\Matsuda_data\DTW_resample_output")
    output_root: Path = Path(r"E:\Matsuda_data\DTW_resample_output\axis_extension")
    norm_len: int = 20

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "AxisExtensionConfig":
        return cls(
            dtw_resample_root=Path(args.dtw_resample_root),
            output_root=Path(args.output_root),
            norm_len=int(args.norm_len),
        )


class AxisExtensionPipeline:
    def __init__(self, config: AxisExtensionConfig):
        self.cfg = config
        for sub in ("csv", "plots", "report"):
            (self.cfg.output_root / sub).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def write_csv(path: Path, rows: Iterable[dict], fieldnames: list[str]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    @staticmethod
    def read_csv(path: Path) -> list[dict[str, str]]:
        with path.open("r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))

    @staticmethod
    def load_npy(path: Path) -> np.ndarray:
        return np.load(path, allow_pickle=False)

    def representative_sequences(self) -> dict[str, str]:
        rows = self.read_csv(self.cfg.dtw_resample_root / "csv" / "representative_sequences.csv")
        return {row["class"]: row["representative_sequence_id"] for row in rows}

    def prototype_path(self, class_name: str, seq_id: str) -> Path:
        return self.cfg.dtw_resample_root / "metrics" / "resampled20_aligned" / class_name / seq_id / "aligned_resampled20.npy"

    def all_sequences(self) -> list[tuple[str, str, Path]]:
        items = []
        root = self.cfg.dtw_resample_root / "metrics" / "resampled20_aligned"
        for class_name in CLASS_NAMES:
            class_dir = root / class_name
            if not class_dir.is_dir():
                continue
            for seq_dir in sorted(class_dir.iterdir(), key=lambda p: p.name):
                path = seq_dir / "aligned_resampled20.npy"
                if path.exists():
                    items.append((class_name, seq_dir.name, path))
        return items

    def compute_for_axis(self, axis_class: str) -> tuple[list[dict], list[dict], list[dict]]:
        reps = self.representative_sequences()
        protos = {
            class_name: self.load_npy(self.prototype_path(class_name, reps[class_name])).astype(np.float32)
            for class_name in CLASS_NAMES
        }
        axis_proto = protos[axis_class]
        axis = axis_proto[-1] - axis_proto[0]
        axis_norm = float(np.linalg.norm(axis))

        prototype_rows: list[dict] = []
        per_sequence_rows: list[dict] = []

        for class_name, proto in protos.items():
            projection_length, projection_ratio, off_axis_distance, off_axis_ratio = compute_axis_metrics(proto, axis)
            for t in range(self.cfg.norm_len):
                prototype_rows.append(
                    {
                        "axis_class": axis_class,
                        "class": class_name,
                        "representative_sequence_id": reps[class_name],
                        "time_index": t,
                        "projection_length": float(projection_length[t]),
                        "projection_ratio": float(projection_ratio[t]),
                        "off_axis_distance": float(off_axis_distance[t]),
                        "off_axis_ratio": float(off_axis_ratio[t]),
                    }
                )

        for class_name, seq_id, path in self.all_sequences():
            arr = self.load_npy(path).astype(np.float32)
            projection_length, projection_ratio, off_axis_distance, off_axis_ratio = compute_axis_metrics(arr, axis)
            for t in range(self.cfg.norm_len):
                per_sequence_rows.append(
                    {
                        "axis_class": axis_class,
                        "class": class_name,
                        "sequence_id": seq_id,
                        "time_index": t,
                        "projection_length": float(projection_length[t]),
                        "projection_ratio": float(projection_ratio[t]),
                        "off_axis_distance": float(off_axis_distance[t]),
                        "off_axis_ratio": float(off_axis_ratio[t]),
                    }
                )

        stats_grouped: dict[tuple[str, str, int], list[float]] = defaultdict(list)
        for row in per_sequence_rows:
            t = int(row["time_index"])
            stats_grouped[("projection_length", row["class"], t)].append(float(row["projection_length"]))
            stats_grouped[("off_axis_distance", row["class"], t)].append(float(row["off_axis_distance"]))

        stats_rows: list[dict] = []
        for (metric_type, class_name, t), values in sorted(stats_grouped.items()):
            stats = compute_summary_stats(values)
            stats_rows.append(
                {
                    "axis_class": axis_class,
                    "metric_type": metric_type,
                    "class": class_name,
                    "time_index": t,
                    **stats,
                }
            )

        return prototype_rows, per_sequence_rows, stats_rows, axis_norm

    def plot_t_s(self, prototype_rows: list[dict], axis_class: str) -> None:
        grouped: dict[str, list[tuple[int, float]]] = defaultdict(list)
        for row in prototype_rows:
            grouped[row["class"]].append((int(row["time_index"]), float(row["projection_length"])))
        for key in grouped:
            grouped[key].sort(key=lambda x: x[0])

        fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
        for class_name in CLASS_NAMES:
            items = grouped[class_name]
            ax.plot([t for t, _ in items], [v for _, v in items], linewidth=2.0, color=COLORS[class_name], label=class_name)
        ax.set_title(f"t-s plot (base axis = {axis_class})")
        ax.set_xlabel("Time Index")
        ax.set_ylabel("Projection Length")
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(self.cfg.output_root / "plots" / f"t_s_axis_{axis_class}.png")
        plt.close(fig)

    def plot_t_d(self, prototype_rows: list[dict], axis_class: str) -> None:
        grouped: dict[str, list[tuple[int, float]]] = defaultdict(list)
        for row in prototype_rows:
            grouped[row["class"]].append((int(row["time_index"]), float(row["off_axis_distance"])))
        for key in grouped:
            grouped[key].sort(key=lambda x: x[0])

        fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
        for class_name in CLASS_NAMES:
            items = grouped[class_name]
            ax.plot([t for t, _ in items], [v for _, v in items], linewidth=2.0, color=COLORS[class_name], label=class_name)
        ax.set_title(f"t-d plot (base axis = {axis_class})")
        ax.set_xlabel("Time Index")
        ax.set_ylabel("Off-axis Distance")
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(self.cfg.output_root / "plots" / f"t_d_axis_{axis_class}.png")
        plt.close(fig)

    def plot_s_d(self, prototype_rows: list[dict], axis_class: str) -> None:
        grouped_s: dict[str, list[float]] = defaultdict(list)
        grouped_d: dict[str, list[float]] = defaultdict(list)
        for row in prototype_rows:
            grouped_s[row["class"]].append(float(row["projection_length"]))
            grouped_d[row["class"]].append(float(row["off_axis_distance"]))

        fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
        for class_name in CLASS_NAMES:
            x = grouped_s[class_name]
            y = grouped_d[class_name]
            ax.plot(x, y, linewidth=2.0, color=COLORS[class_name], label=class_name)
            ax.scatter(x[0], y[0], color=COLORS[class_name], s=20)
            ax.scatter(x[-1], y[-1], color=COLORS[class_name], s=30, marker="x")
        ax.set_title(f"s-d plot (base axis = {axis_class})")
        ax.set_xlabel("Projection Length")
        ax.set_ylabel("Off-axis Distance")
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(self.cfg.output_root / "plots" / f"s_d_axis_{axis_class}.png")
        plt.close(fig)

    def run(self) -> None:
        summary_lines = ["# DTW-resampled axis extension", ""]
        for axis_class in ("truesmile", "polite"):
            prototype_rows, per_sequence_rows, stats_rows, axis_norm = self.compute_for_axis(axis_class)

            self.write_csv(
                self.cfg.output_root / "csv" / f"prototype_metrics_axis_{axis_class}.csv",
                prototype_rows,
                [
                    "axis_class",
                    "class",
                    "representative_sequence_id",
                    "time_index",
                    "projection_length",
                    "projection_ratio",
                    "off_axis_distance",
                    "off_axis_ratio",
                ],
            )
            self.write_csv(
                self.cfg.output_root / "csv" / f"per_sequence_metrics_axis_{axis_class}.csv",
                per_sequence_rows,
                [
                    "axis_class",
                    "class",
                    "sequence_id",
                    "time_index",
                    "projection_length",
                    "projection_ratio",
                    "off_axis_distance",
                    "off_axis_ratio",
                ],
            )
            self.write_csv(
                self.cfg.output_root / "csv" / f"statistics_axis_{axis_class}.csv",
                stats_rows,
                ["axis_class", "metric_type", "class", "time_index", "mean", "std", "median", "q1", "q3"],
            )

            self.plot_t_s(prototype_rows, axis_class)
            self.plot_t_d(prototype_rows, axis_class)
            self.plot_s_d(prototype_rows, axis_class)

            summary_lines.append(f"## Base axis = {axis_class}")
            summary_lines.append(f"- axis_norm = {axis_norm:.4f}")
            grouped_proto: dict[str, list[dict]] = defaultdict(list)
            for row in prototype_rows:
                grouped_proto[row["class"]].append(row)
            for class_name in CLASS_NAMES:
                items = sorted(grouped_proto[class_name], key=lambda r: int(r["time_index"]))
                along_end = float(items[-1]["projection_length"])
                along_peak = max(float(r["projection_length"]) for r in items)
                d_end = float(items[-1]["off_axis_distance"])
                d_peak = max(float(r["off_axis_distance"]) for r in items)
                summary_lines.append(
                    f"- {class_name}: s_end={along_end:.4f}, s_peak={along_peak:.4f}, d_end={d_end:.4f}, d_peak={d_peak:.4f}"
                )
            summary_lines.append("")

        report_path = self.cfg.output_root / "report" / "axis_extension_summary.md"
        report_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
        print(f"[DTW_AXIS_EXTENSION] Finished. Report saved to: {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate t-s, t-d, s-d plots for true/polite base axes on DTW-resampled data.")
    parser.add_argument("--dtw_resample_root", default=r"E:\Matsuda_data\DTW_resample_output")
    parser.add_argument("--output_root", default=r"E:\Matsuda_data\DTW_resample_output\axis_extension")
    parser.add_argument("--norm_len", type=int, default=20)
    args = parser.parse_args()
    pipeline = AxisExtensionPipeline(AxisExtensionConfig.from_args(args))
    pipeline.run()


if __name__ == "__main__":
    main()
