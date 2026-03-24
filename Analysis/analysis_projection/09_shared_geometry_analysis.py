from __future__ import annotations

import argparse
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from common import CLASS_NAMES, ProjectionConfig, ProjectionTaskBase, SequenceInfo, compute_summary_stats


COLORS = {"polite": "#1f77b4", "truesmile": "#2ca02c", "ambiguous": "#ff7f0e"}


@dataclass
class GeometryConfig:
    analysis_input_root: Path
    output_root: Path
    source_input_root: Path
    progress_bins: int

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "GeometryConfig":
        return cls(
            analysis_input_root=Path(args.analysis_input_root),
            output_root=Path(args.output_root),
            source_input_root=Path(args.source_input_root),
            progress_bins=int(args.progress_bins),
        )


class SharedGeometryTask(ProjectionTaskBase):
    def __init__(self, config: GeometryConfig):
        super().__init__(
            ProjectionConfig(
                analysis_input_root=config.analysis_input_root,
                output_root=config.output_root,
                norm_len=config.progress_bins,
            )
        )
        self.source_input_root = config.source_input_root
        self.progress_bins = config.progress_bins
        self.shared_csv_dir = self.cfg.output_root / "shared" / "csv"
        self.shared_plot_dir = self.cfg.output_root / "shared" / "plots"
        self.shared_report_dir = self.cfg.output_root / "shared" / "report"
        for p in (self.shared_csv_dir, self.shared_plot_dir, self.shared_report_dir):
            p.mkdir(parents=True, exist_ok=True)

    def original_rel_path(self, seq: SequenceInfo) -> Path:
        return (
            self.cfg.analysis_input_root
            / "metrics"
            / "sequence_features_rel"
            / seq.class_name
            / seq.sequence_id
            / "sequence_features_rel.npy"
        )

    def frame_names_path(self, seq: SequenceInfo) -> Path:
        return (
            self.cfg.analysis_input_root
            / "metrics"
            / "sequence_features"
            / seq.class_name
            / seq.sequence_id
            / "frame_names.json"
        )

    def source_frame_path(self, seq: SequenceInfo, frame_name: str) -> Path:
        return self.source_input_root / seq.class_name / seq.sequence_id / frame_name

    @staticmethod
    def pairwise_distance_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a2 = np.sum(a * a, axis=1, keepdims=True)
        b2 = np.sum(b * b, axis=1, keepdims=True).T
        dist2 = np.maximum(a2 + b2 - 2.0 * (a @ b.T), 0.0)
        return np.sqrt(dist2)

    @staticmethod
    def tangent_vector(curve: np.ndarray, idx: int) -> np.ndarray:
        if curve.shape[0] == 1:
            return np.zeros((curve.shape[1],), dtype=np.float32)
        if idx == 0:
            return curve[1] - curve[0]
        if idx == curve.shape[0] - 1:
            return curve[-1] - curve[-2]
        return curve[idx + 1] - curve[idx - 1]

    def progress_bin(self, idx: int, length: int) -> int:
        if length <= 1:
            return 0
        ratio = idx / float(length - 1)
        return min(self.progress_bins - 1, int(round(ratio * (self.progress_bins - 1))))

    def load_all_sequences(self) -> dict[tuple[str, str], dict]:
        cache: dict[tuple[str, str], dict] = {}
        for seq in self.discover_sequences():
            arr = self.load_npy(self.original_rel_path(seq)).astype(np.float32)
            frame_names = self.load_json(self.frame_names_path(seq))
            assert isinstance(frame_names, list)
            cache[(seq.class_name, seq.sequence_id)] = {
                "seq": seq,
                "arr": arr,
                "frame_names": frame_names,
            }
        return cache

    def compute(self) -> tuple[list[dict], list[dict]]:
        cache = self.load_all_sequences()
        seqs = [v["seq"] for v in cache.values()]
        min_rows: list[dict] = []
        tangent_rows: list[dict] = []

        for i, seq1 in enumerate(seqs):
            data1 = cache[(seq1.class_name, seq1.sequence_id)]
            arr1 = data1["arr"]
            frames1 = data1["frame_names"]
            for j in range(i + 1, len(seqs)):
                seq2 = seqs[j]
                if seq1.class_name == seq2.class_name:
                    continue
                data2 = cache[(seq2.class_name, seq2.sequence_id)]
                arr2 = data2["arr"]
                frames2 = data2["frame_names"]

                dist_mat = self.pairwise_distance_matrix(arr1, arr2)
                flat_idx = int(np.argmin(dist_mat))
                t1_min, t2_min = np.unravel_index(flat_idx, dist_mat.shape)
                min_rows.append(
                    {
                        "curve1_class": seq1.class_name,
                        "curve1_sequence_id": seq1.sequence_id,
                        "curve1_time_index": int(t1_min),
                        "curve1_frame_name": frames1[t1_min],
                        "curve2_class": seq2.class_name,
                        "curve2_sequence_id": seq2.sequence_id,
                        "curve2_time_index": int(t2_min),
                        "curve2_frame_name": frames2[t2_min],
                        "min_distance": float(dist_mat[t1_min, t2_min]),
                    }
                )

                for base_seq, other_seq, base_arr, other_arr, base_frames, other_frames, direction in (
                    (seq1, seq2, arr1, arr2, frames1, frames2, "forward"),
                    (seq2, seq1, arr2, arr1, frames2, frames1, "reverse"),
                ):
                    dist_mat_dir = dist_mat if direction == "forward" else dist_mat.T
                    for t in range(base_arr.shape[0]):
                        nearest_idx = int(np.argmin(dist_mat_dir[t]))
                        d_vec = other_arr[nearest_idx] - base_arr[t]
                        tau = self.tangent_vector(base_arr, t).astype(np.float64)
                        tau_norm = float(np.linalg.norm(tau))
                        if tau_norm <= 1e-12:
                            u_tan = np.zeros_like(tau)
                            signed_parallel = 0.0
                            dist_parallel = 0.0
                            d_parallel = np.zeros_like(tau)
                        else:
                            u_tan = tau / tau_norm
                            signed_parallel = float(np.dot(d_vec, u_tan))
                            d_parallel = signed_parallel * u_tan
                            dist_parallel = float(np.linalg.norm(d_parallel))
                        d_normal = d_vec - d_parallel
                        tangent_rows.append(
                            {
                                "curve1_class": base_seq.class_name,
                                "curve1_sequence_id": base_seq.sequence_id,
                                "curve1_time_index": int(t),
                                "curve1_frame_name": base_frames[t],
                                "curve2_class": other_seq.class_name,
                                "curve2_sequence_id": other_seq.sequence_id,
                                "nearest_time_index_on_curve2": int(nearest_idx),
                                "nearest_frame_name_on_curve2": other_frames[nearest_idx],
                                "progress_bin": int(self.progress_bin(t, base_arr.shape[0])),
                                "dist_total": float(np.linalg.norm(d_vec)),
                                "signed_parallel": signed_parallel,
                                "dist_parallel": dist_parallel,
                                "dist_normal": float(np.linalg.norm(d_normal)),
                            }
                        )
        return min_rows, tangent_rows

    def write_outputs(self, min_rows: list[dict], tangent_rows: list[dict]) -> None:
        self.write_csv(
            self.shared_csv_dir / "curve_min_distance.csv",
            min_rows,
            [
                "curve1_class",
                "curve1_sequence_id",
                "curve1_time_index",
                "curve1_frame_name",
                "curve2_class",
                "curve2_sequence_id",
                "curve2_time_index",
                "curve2_frame_name",
                "min_distance",
            ],
        )
        self.write_csv(
            self.shared_csv_dir / "tangent_relative_distance.csv",
            tangent_rows,
            [
                "curve1_class",
                "curve1_sequence_id",
                "curve1_time_index",
                "curve1_frame_name",
                "curve2_class",
                "curve2_sequence_id",
                "nearest_time_index_on_curve2",
                "nearest_frame_name_on_curve2",
                "progress_bin",
                "dist_total",
                "signed_parallel",
                "dist_parallel",
                "dist_normal",
            ],
        )

    def write_min_distance_statistics(self, min_rows: list[dict]) -> list[dict]:
        grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
        for row in min_rows:
            pair = tuple(sorted((row["curve1_class"], row["curve2_class"])))
            grouped[pair].append(float(row["min_distance"]))

        stats_rows: list[dict] = []
        for pair, values in sorted(grouped.items()):
            stats = compute_summary_stats(values)
            stats_rows.append(
                {
                    "pair": f"{pair[0]}_vs_{pair[1]}",
                    "class_a": pair[0],
                    "class_b": pair[1],
                    "count": len(values),
                    **stats,
                }
            )

        self.write_csv(
            self.shared_csv_dir / "curve_min_distance_statistics.csv",
            stats_rows,
            ["pair", "class_a", "class_b", "count", "mean", "std", "median", "q1", "q3"],
        )
        return stats_rows

    def write_statistics(self, tangent_rows: list[dict]) -> list[dict]:
        grouped: dict[tuple[str, str, int, str], list[float]] = defaultdict(list)
        for row in tangent_rows:
            for metric in ("dist_total", "dist_parallel", "dist_normal"):
                grouped[
                    (row["curve1_class"], row["curve2_class"], int(row["progress_bin"]), metric)
                ].append(float(row[metric]))

        stats_rows: list[dict] = []
        for (curve1_class, curve2_class, progress_bin, metric), values in sorted(grouped.items()):
            stats = compute_summary_stats(values)
            stats_rows.append(
                {
                    "curve1_class": curve1_class,
                    "curve2_class": curve2_class,
                    "progress_bin": progress_bin,
                    "metric": metric,
                    **stats,
                }
            )

        self.write_csv(
            self.shared_csv_dir / "tangent_relative_statistics.csv",
            stats_rows,
            ["curve1_class", "curve2_class", "progress_bin", "metric", "mean", "std", "median", "q1", "q3"],
        )
        return stats_rows

    def plot_min_distance_distribution(self, min_rows: list[dict]) -> None:
        grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
        for row in min_rows:
            pair = tuple(sorted((row["curve1_class"], row["curve2_class"])))
            grouped[pair].append(float(row["min_distance"]))

        ordered_pairs = [
            ("polite", "truesmile"),
            ("ambiguous", "truesmile"),
            ("ambiguous", "polite"),
        ]
        labels = [f"{a} vs {b}" for a, b in ordered_pairs]
        data = [grouped[(a, b)] for a, b in ordered_pairs]

        fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
        box = ax.boxplot(data, patch_artist=True, tick_labels=labels, widths=0.55)
        for patch, (a, b) in zip(box["boxes"], ordered_pairs):
            patch.set_facecolor(COLORS[a])
            patch.set_alpha(0.25)
        ax.set_title("Minimum Distance Distribution by Class Pair")
        ax.set_ylabel("Minimum Distance")
        fig.tight_layout()
        fig.savefig(self.shared_plot_dir / "curve_min_distance_distribution.png")
        plt.close(fig)

    def plot_min_distance_examples_by_pair(self, min_rows: list[dict], top_k: int = 6) -> None:
        grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for row in min_rows:
            pair = tuple(sorted((row["curve1_class"], row["curve2_class"])))
            grouped[pair].append(row)

        for pair, rows in grouped.items():
            top_rows = sorted(rows, key=lambda r: float(r["min_distance"]))[:top_k]
            if not top_rows:
                continue
            fig, axes = plt.subplots(len(top_rows), 2, figsize=(8, 3 * len(top_rows)), dpi=140)
            axes_arr = np.atleast_2d(axes)
            for row_idx, row in enumerate(top_rows):
                left_ax = axes_arr[row_idx, 0]
                right_ax = axes_arr[row_idx, 1]
                for ax in (left_ax, right_ax):
                    ax.axis("off")
                seq1 = SequenceInfo(class_name=row["curve1_class"], sequence_id=row["curve1_sequence_id"])
                seq2 = SequenceInfo(class_name=row["curve2_class"], sequence_id=row["curve2_sequence_id"])
                img1 = self.source_frame_path(seq1, row["curve1_frame_name"])
                img2 = self.source_frame_path(seq2, row["curve2_frame_name"])
                left_ax.imshow(plt.imread(img1))
                right_ax.imshow(plt.imread(img2))
                left_ax.set_title(
                    f"{row['curve1_class']} / {row['curve1_sequence_id']} / t={row['curve1_time_index']}",
                    fontsize=8,
                )
                right_ax.set_title(
                    f"{row['curve2_class']} / {row['curve2_sequence_id']} / t={row['curve2_time_index']}",
                    fontsize=8,
                )
            fig.suptitle(f"Top Minimum-Distance Examples: {pair[0]} vs {pair[1]}", fontsize=12)
            fig.tight_layout()
            fig.savefig(self.shared_plot_dir / f"curve_min_distance_examples_{pair[0]}_vs_{pair[1]}.png")
            plt.close(fig)

    def plot_tangent_metric(self, stats_rows: list[dict], metric: str) -> None:
        grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for row in stats_rows:
            if row["metric"] == metric:
                grouped[(row["curve1_class"], row["curve2_class"])].append(row)

        for anchor_class in CLASS_NAMES:
            fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
            for target_class in CLASS_NAMES:
                if target_class == anchor_class:
                    continue
                items = sorted(grouped[(anchor_class, target_class)], key=lambda r: int(r["progress_bin"]))
                if not items:
                    continue
                x = [int(r["progress_bin"]) for r in items]
                mean = np.asarray([float(r["mean"]) for r in items], dtype=np.float64)
                q1 = np.asarray([float(r["q1"]) for r in items], dtype=np.float64)
                q3 = np.asarray([float(r["q3"]) for r in items], dtype=np.float64)
                ax.fill_between(x, q1, q3, color=COLORS[target_class], alpha=0.15)
                ax.plot(x, mean, color=COLORS[target_class], linewidth=2.0, label=f"{target_class} vs {anchor_class}")
            ax.set_title(f"{metric} curves (anchor={anchor_class})")
            ax.set_xlabel("Relative Progress Bin")
            ax.set_ylabel("Distance")
            ax.legend(loc="best")
            fig.tight_layout()
            fig.savefig(self.shared_plot_dir / f"{metric}_anchor_{anchor_class}.png")
            plt.close(fig)

    def write_report(self, min_rows: list[dict], min_stats_rows: list[dict], stats_rows: list[dict]) -> None:
        lines = ["# Shared Geometry Summary", ""]
        lines.append("## Curve-to-Curve Minimum Distance")
        pair_grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for row in min_rows:
            pair_grouped[tuple(sorted((row["curve1_class"], row["curve2_class"])))].append(row)
        min_stats_map = {
            (row["class_a"], row["class_b"]): row for row in min_stats_rows
        }
        for pair, rows in sorted(pair_grouped.items()):
            best = min(rows, key=lambda r: float(r["min_distance"]))
            stat = min_stats_map[pair]
            lines.append(
                f"- {pair[0]} vs {pair[1]}: best={float(best['min_distance']):.4f}, median={float(stat['median']):.4f}, "
                f"q1-q3=({float(stat['q1']):.4f}, {float(stat['q3']):.4f}), "
                f"best_frames=({best['curve1_sequence_id']}:{best['curve1_time_index']}, {best['curve2_sequence_id']}:{best['curve2_time_index']})"
            )
        lines.append("")
        lines.append("## Tangent-Relative Statistics")
        metric_grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
        for row in stats_rows:
            metric_grouped[(row["curve1_class"], row["curve2_class"], row["metric"])].append(row)
        for metric in ("dist_total", "dist_parallel", "dist_normal"):
            lines.append(f"### {metric}")
            for curve1_class in CLASS_NAMES:
                for curve2_class in CLASS_NAMES:
                    if curve1_class == curve2_class:
                        continue
                    items = sorted(metric_grouped[(curve1_class, curve2_class, metric)], key=lambda r: int(r["progress_bin"]))
                    if not items:
                        continue
                    end = items[-1]
                    peak = max(items, key=lambda r: float(r["mean"]))
                    lines.append(
                        f"- {curve2_class} relative to {curve1_class}: mean_end={float(end['mean']):.4f}, "
                        f"mean_peak={float(peak['mean']):.4f} at bin={peak['progress_bin']}"
                    )
        (self.shared_report_dir / "curve_geometry_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def run(self) -> None:
        min_rows, tangent_rows = self.compute()
        self.write_outputs(min_rows, tangent_rows)
        min_stats_rows = self.write_min_distance_statistics(min_rows)
        stats_rows = self.write_statistics(tangent_rows)
        self.plot_min_distance_distribution(min_rows)
        self.plot_min_distance_examples_by_pair(min_rows)
        for metric in ("dist_total", "dist_parallel", "dist_normal"):
            self.plot_tangent_metric(stats_rows, metric)
        self.write_report(min_rows, min_stats_rows, stats_rows)
        print("[09] Saved shared geometry analysis outputs.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Shared geometry analysis on original pre-resampling sequences.")
    parser.add_argument("--analysis_input_root", default=r"E:\Matsuda_data\2-27meeting")
    parser.add_argument("--output_root", default=r"E:\Matsuda_data\3-10meeting")
    parser.add_argument("--source_input_root", default=r"E:\Matsuda_data\2-18meeting")
    parser.add_argument("--progress_bins", type=int, default=20)
    args = parser.parse_args()
    task = SharedGeometryTask(GeometryConfig.from_args(args))
    task.run()


if __name__ == "__main__":
    main()
