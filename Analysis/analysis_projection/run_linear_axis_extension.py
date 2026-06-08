from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

from common import CLASS_NAMES, ProjectionConfig, ProjectionTaskBase, SequenceInfo, compute_axis_metrics, compute_summary_stats


COLORS = {"polite": "#1f77b4", "truesmile": "#2ca02c", "ambiguous": "#ff7f0e"}


@dataclass
class LinearAxisExtensionConfig:
    analysis_input_root: Path = Path(r"E:\Matsuda_data\2-27meeting")
    projection_output_root: Path = Path(r"E:\Matsuda_data\3-10meeting")
    output_root: Path = Path(r"E:\Matsuda_data\3-10meeting\linear_axis_extension")
    method: str = "methodB"
    norm_len: int = 20
    nearest_count: int = 6

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "LinearAxisExtensionConfig":
        return cls(
            analysis_input_root=Path(args.analysis_input_root),
            projection_output_root=Path(args.projection_output_root),
            output_root=Path(args.output_root),
            method=args.method,
            norm_len=int(args.norm_len),
            nearest_count=int(args.nearest_count),
        )


class LinearAxisExtensionPipeline(ProjectionTaskBase):
    def __init__(self, config: LinearAxisExtensionConfig):
        projection_cfg = ProjectionConfig(
            analysis_input_root=config.analysis_input_root,
            output_root=config.projection_output_root,
            norm_len=config.norm_len,
        )
        super().__init__(projection_cfg)
        self.ext_cfg = config
        if self.ext_cfg.method not in {"methodA", "methodB"}:
            raise ValueError("--method must be methodA or methodB")
        for sub in ("csv", "plots", "report"):
            (self.ext_cfg.output_root / sub).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def sequence_sort_key(seq_id: str) -> tuple[int, int | str]:
        try:
            return (0, int(seq_id))
        except ValueError:
            return (1, seq_id)

    @staticmethod
    def write_rows(path: Path, rows: Iterable[dict], fieldnames: list[str]) -> None:
        ProjectionTaskBase.write_csv(path, rows, fieldnames)

    def prototype_suffix(self) -> str:
        return self.ext_cfg.method

    def prototype_path(self, class_name: str) -> Path:
        return self.method_proto(
            self.ext_cfg.method,
            f"prototype_{class_name}_{self.prototype_suffix()}.npy",
        )

    def load_prototypes(self) -> dict[str, np.ndarray]:
        return {
            class_name: self.load_npy(self.prototype_path(class_name)).astype(np.float32)
            for class_name in CLASS_NAMES
        }

    def representative_sequence_ids(self) -> dict[str, str]:
        if self.ext_cfg.method == "methodA":
            return {class_name: "median" for class_name in CLASS_NAMES}
        meta = self.load_json(self.method_proto("methodB", "projection_meta_methodB.json"))
        return {class_name: str(meta[class_name]["sequence_id"]) for class_name in CLASS_NAMES}

    def all_sequences(self) -> list[SequenceInfo]:
        return self.discover_sequences()

    def nearest_sequences_to_prototype(self, count: int, protos: dict[str, np.ndarray]) -> tuple[dict[str, set[str]], list[dict]]:
        selected: dict[str, set[str]] = {}
        selected_rows: list[dict] = []
        reps = self.representative_sequence_ids()

        for class_name in CLASS_NAMES:
            distances: list[tuple[str, float]] = []
            proto = protos[class_name]
            for seq in self.sequences_for_class(class_name):
                arr = self.load_npy(self.normalized_seq_path(seq)).astype(np.float32)
                distance = float(np.linalg.norm(arr - proto))
                distances.append((seq.sequence_id, distance))

            ranked = sorted(distances, key=lambda item: (item[1], self.sequence_sort_key(item[0])))
            chosen = [seq_id for seq_id, _ in ranked[: min(count, len(ranked))]]
            selected[class_name] = set(chosen)
            for rank, (seq_id, distance) in enumerate(ranked[: min(count, len(ranked))], start=1):
                selected_rows.append(
                    {
                        "method": self.ext_cfg.method,
                        "class": class_name,
                        "rank": rank,
                        "sequence_id": seq_id,
                        "prototype_sequence_id": reps[class_name],
                        "euclidean_distance_to_prototype": distance,
                    }
                )

        return selected, selected_rows

    def compute_for_axis(self, axis_class: str, protos: dict[str, np.ndarray]) -> tuple[list[dict], list[dict], list[dict], float]:
        reps = self.representative_sequence_ids()
        axis_proto = protos[axis_class]
        axis = axis_proto[-1] - axis_proto[0]
        axis_norm = float(np.linalg.norm(axis))

        prototype_rows: list[dict] = []
        per_sequence_rows: list[dict] = []

        for class_name, proto in protos.items():
            projection_length, projection_ratio, off_axis_distance, off_axis_ratio = compute_axis_metrics(proto, axis)
            for t in range(self.ext_cfg.norm_len):
                prototype_rows.append(
                    {
                        "method": self.ext_cfg.method,
                        "axis_class": axis_class,
                        "class": class_name,
                        "prototype_sequence_id": reps[class_name],
                        "time_index": t,
                        "projection_length": float(projection_length[t]),
                        "projection_ratio": float(projection_ratio[t]),
                        "off_axis_distance": float(off_axis_distance[t]),
                        "off_axis_ratio": float(off_axis_ratio[t]),
                    }
                )

        for seq in self.all_sequences():
            arr = self.load_npy(self.normalized_seq_path(seq)).astype(np.float32)
            projection_length, projection_ratio, off_axis_distance, off_axis_ratio = compute_axis_metrics(arr, axis)
            for t in range(self.ext_cfg.norm_len):
                per_sequence_rows.append(
                    {
                        "method": self.ext_cfg.method,
                        "axis_class": axis_class,
                        "class": seq.class_name,
                        "sequence_id": seq.sequence_id,
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
            stats_rows.append(
                {
                    "method": self.ext_cfg.method,
                    "axis_class": axis_class,
                    "metric_type": metric_type,
                    "class": class_name,
                    "time_index": t,
                    **compute_summary_stats(values),
                }
            )

        return prototype_rows, per_sequence_rows, stats_rows, axis_norm

    def plot_s_d(self, prototype_rows: list[dict], axis_class: str) -> None:
        grouped: dict[str, list[dict]] = defaultdict(list)
        for row in prototype_rows:
            grouped[row["class"]].append(row)

        fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
        for class_name in CLASS_NAMES:
            items = sorted(grouped[class_name], key=lambda r: int(r["time_index"]))
            x = [float(r["projection_length"]) for r in items]
            y = [float(r["off_axis_distance"]) for r in items]
            ax.plot(x, y, linewidth=2.0, color=COLORS[class_name], label=class_name)
            ax.scatter(x[0], y[0], color=COLORS[class_name], s=20)
            ax.scatter(x[-1], y[-1], color=COLORS[class_name], s=30, marker="x")
        ax.set_title(f"linear-aligned s-d plot ({self.ext_cfg.method}, base axis = {axis_class})")
        ax.set_xlabel("Projection Length")
        ax.set_ylabel("Off-axis Distance")
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(self.ext_cfg.output_root / "plots" / f"s_d_axis_{axis_class}_{self.ext_cfg.method}.png")
        plt.close(fig)

    def plot_s_d_all_sequences(self, per_sequence_rows: list[dict], prototype_rows: list[dict], axis_class: str) -> None:
        grouped_seq: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for row in per_sequence_rows:
            grouped_seq[(row["class"], row["sequence_id"])].append(row)

        grouped_proto: dict[str, list[dict]] = defaultdict(list)
        for row in prototype_rows:
            grouped_proto[row["class"]].append(row)

        fig, ax = plt.subplots(figsize=(7.5, 6.5), dpi=150)
        labeled_classes: set[str] = set()
        for class_name in CLASS_NAMES:
            for (seq_class, _), rows in grouped_seq.items():
                if seq_class != class_name:
                    continue
                items = sorted(rows, key=lambda r: int(r["time_index"]))
                x = [float(r["projection_length"]) for r in items]
                y = [float(r["off_axis_distance"]) for r in items]
                label = f"{class_name} sequences" if class_name not in labeled_classes else None
                ax.plot(x, y, linewidth=0.8, alpha=0.18, color=COLORS[class_name], label=label)
                labeled_classes.add(class_name)

        for class_name in CLASS_NAMES:
            items = sorted(grouped_proto[class_name], key=lambda r: int(r["time_index"]))
            x = [float(r["projection_length"]) for r in items]
            y = [float(r["off_axis_distance"]) for r in items]
            ax.plot(x, y, linewidth=2.6, color=COLORS[class_name], label=f"{class_name} prototype")
            ax.scatter(x[0], y[0], color=COLORS[class_name], s=24)
            ax.scatter(x[-1], y[-1], color=COLORS[class_name], s=42, marker="x")

        ax.set_title(f"linear-aligned all-sequence s-d plot ({self.ext_cfg.method}, base axis = {axis_class})")
        ax.set_xlabel("Projection Length")
        ax.set_ylabel("Off-axis Distance")
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(self.ext_cfg.output_root / "plots" / f"s_d_all_sequences_axis_{axis_class}_{self.ext_cfg.method}.png")
        plt.close(fig)

    def plot_s_d_nearest_sequences(
        self,
        per_sequence_rows: list[dict],
        prototype_rows: list[dict],
        selected_ids: dict[str, set[str]],
        axis_class: str,
    ) -> None:
        grouped_seq: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for row in per_sequence_rows:
            class_name = row["class"]
            seq_id = row["sequence_id"]
            if seq_id in selected_ids.get(class_name, set()):
                grouped_seq[(class_name, seq_id)].append(row)

        grouped_proto: dict[str, list[dict]] = defaultdict(list)
        for row in prototype_rows:
            grouped_proto[row["class"]].append(row)

        fig, ax = plt.subplots(figsize=(7.5, 6.5), dpi=150)
        for class_name in CLASS_NAMES:
            class_items = [
                ((seq_class, seq_id), rows)
                for (seq_class, seq_id), rows in grouped_seq.items()
                if seq_class == class_name
            ]
            class_items.sort(key=lambda item: self.sequence_sort_key(item[0][1]))
            for idx, ((_, _), rows) in enumerate(class_items):
                items = sorted(rows, key=lambda r: int(r["time_index"]))
                x = [float(r["projection_length"]) for r in items]
                y = [float(r["off_axis_distance"]) for r in items]
                label = f"{class_name} nearest {self.ext_cfg.nearest_count}" if idx == 0 else None
                ax.plot(x, y, linewidth=1.1, alpha=0.45, color=COLORS[class_name], label=label)

        for class_name in CLASS_NAMES:
            items = sorted(grouped_proto[class_name], key=lambda r: int(r["time_index"]))
            x = [float(r["projection_length"]) for r in items]
            y = [float(r["off_axis_distance"]) for r in items]
            ax.plot(x, y, linewidth=2.8, color=COLORS[class_name], label=f"{class_name} prototype")
            ax.scatter(x[0], y[0], color=COLORS[class_name], s=24)
            ax.scatter(x[-1], y[-1], color=COLORS[class_name], s=42, marker="x")

        ax.set_title(f"linear-aligned nearest-{self.ext_cfg.nearest_count} s-d plot ({self.ext_cfg.method}, base axis = {axis_class})")
        ax.set_xlabel("Projection Length")
        ax.set_ylabel("Off-axis Distance")
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(
            self.ext_cfg.output_root
            / "plots"
            / f"s_d_nearest{self.ext_cfg.nearest_count}_axis_{axis_class}_{self.ext_cfg.method}.png"
        )
        plt.close(fig)

    def run(self) -> None:
        protos = self.load_prototypes()
        selected_ids, selected_rows = self.nearest_sequences_to_prototype(self.ext_cfg.nearest_count, protos)
        self.write_rows(
            self.ext_cfg.output_root / "csv" / f"nearest{self.ext_cfg.nearest_count}_to_prototype_sequences_{self.ext_cfg.method}.csv",
            selected_rows,
            [
                "method",
                "class",
                "rank",
                "sequence_id",
                "prototype_sequence_id",
                "euclidean_distance_to_prototype",
            ],
        )

        summary_lines = [
            "# Linear-aligned axis extension",
            "",
            f"- method = {self.ext_cfg.method}",
            f"- input = {self.ext_cfg.analysis_input_root}",
            f"- output = {self.ext_cfg.output_root}",
            f"- nearest_count = {self.ext_cfg.nearest_count}",
            "",
        ]

        for axis_class in ("truesmile", "polite"):
            prototype_rows, per_sequence_rows, stats_rows, axis_norm = self.compute_for_axis(axis_class, protos)
            self.write_rows(
                self.ext_cfg.output_root / "csv" / f"prototype_metrics_axis_{axis_class}_{self.ext_cfg.method}.csv",
                prototype_rows,
                [
                    "method",
                    "axis_class",
                    "class",
                    "prototype_sequence_id",
                    "time_index",
                    "projection_length",
                    "projection_ratio",
                    "off_axis_distance",
                    "off_axis_ratio",
                ],
            )
            self.write_rows(
                self.ext_cfg.output_root / "csv" / f"per_sequence_metrics_axis_{axis_class}_{self.ext_cfg.method}.csv",
                per_sequence_rows,
                [
                    "method",
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
            self.write_rows(
                self.ext_cfg.output_root / "csv" / f"statistics_axis_{axis_class}_{self.ext_cfg.method}.csv",
                stats_rows,
                ["method", "axis_class", "metric_type", "class", "time_index", "mean", "std", "median", "q1", "q3"],
            )

            self.plot_s_d(prototype_rows, axis_class)
            self.plot_s_d_all_sequences(per_sequence_rows, prototype_rows, axis_class)
            self.plot_s_d_nearest_sequences(per_sequence_rows, prototype_rows, selected_ids, axis_class)

            summary_lines.append(f"## Base axis = {axis_class}")
            summary_lines.append(f"- axis_norm = {axis_norm:.4f}")
            grouped_proto: dict[str, list[dict]] = defaultdict(list)
            for row in prototype_rows:
                grouped_proto[row["class"]].append(row)
            for class_name in CLASS_NAMES:
                items = sorted(grouped_proto[class_name], key=lambda r: int(r["time_index"]))
                s_end = float(items[-1]["projection_length"])
                s_peak = max(float(r["projection_length"]) for r in items)
                d_end = float(items[-1]["off_axis_distance"])
                d_peak = max(float(r["off_axis_distance"]) for r in items)
                summary_lines.append(
                    f"- {class_name}: s_end={s_end:.4f}, s_peak={s_peak:.4f}, d_end={d_end:.4f}, d_peak={d_peak:.4f}"
                )
            summary_lines.append("")

        report_path = self.ext_cfg.output_root / "report" / f"linear_axis_extension_summary_{self.ext_cfg.method}.md"
        report_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
        print(f"[LINEAR_AXIS_EXTENSION] Finished. Report saved to: {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate s-d plots on linearly normalized, non-DTW trajectories.")
    parser.add_argument("--analysis_input_root", default=r"E:\Matsuda_data\2-27meeting")
    parser.add_argument("--projection_output_root", default=r"E:\Matsuda_data\3-10meeting")
    parser.add_argument("--output_root", default=r"E:\Matsuda_data\3-10meeting\linear_axis_extension")
    parser.add_argument("--method", choices=("methodA", "methodB"), default="methodB")
    parser.add_argument("--norm_len", type=int, default=20)
    parser.add_argument("--nearest_count", type=int, default=6)
    args = parser.parse_args()
    pipeline = LinearAxisExtensionPipeline(LinearAxisExtensionConfig.from_args(args))
    pipeline.run()


if __name__ == "__main__":
    main()
