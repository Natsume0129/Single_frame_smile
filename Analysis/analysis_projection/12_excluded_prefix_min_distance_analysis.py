from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from common import CLASS_NAMES, ProjectionConfig, ProjectionTaskBase, SequenceInfo, compute_summary_stats


COLORS = {"polite": "#1f77b4", "truesmile": "#2ca02c", "ambiguous": "#ff7f0e"}


@dataclass
class ExcludedPrefixConfig:
    analysis_input_root: Path
    output_root: Path
    progress_bins: int
    exclude_values: list[int]

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "ExcludedPrefixConfig":
        exclude_values = [int(v) for v in str(args.exclude_values).split(",") if v.strip()]
        return cls(
            analysis_input_root=Path(args.analysis_input_root),
            output_root=Path(args.output_root),
            progress_bins=int(args.progress_bins),
            exclude_values=exclude_values,
        )


class ExcludedPrefixMinDistanceTask(ProjectionTaskBase):
    def __init__(self, config: ExcludedPrefixConfig):
        super().__init__(
            ProjectionConfig(
                analysis_input_root=config.analysis_input_root,
                output_root=config.output_root,
                norm_len=config.progress_bins,
            )
        )
        self.progress_bins = config.progress_bins
        self.exclude_values = config.exclude_values
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

    @staticmethod
    def pairwise_distance_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a2 = np.sum(a * a, axis=1, keepdims=True)
        b2 = np.sum(b * b, axis=1, keepdims=True).T
        dist2 = np.maximum(a2 + b2 - 2.0 * (a @ b.T), 0.0)
        return np.sqrt(dist2)

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
                "length": arr.shape[0],
            }
        return cache

    def progress_bin(self, idx: int, length: int, exclude_first: int) -> int:
        remaining = length - exclude_first
        if remaining <= 1:
            return 0
        ratio = (idx - exclude_first) / float(remaining - 1)
        ratio = max(0.0, min(1.0, ratio))
        return min(self.progress_bins - 1, int(round(ratio * (self.progress_bins - 1))))

    def compute_all_pairs(self, cache: dict[tuple[str, str], dict], exclude_first: int) -> list[dict]:
        seqs = [v["seq"] for v in cache.values()]
        rows: list[dict] = []
        skipped_pairs = 0

        for i, seq1 in enumerate(seqs):
            data1 = cache[(seq1.class_name, seq1.sequence_id)]
            arr1_full = data1["arr"]
            frames1 = data1["frame_names"]
            len1 = data1["length"]
            if len1 <= exclude_first:
                continue
            arr1 = arr1_full[exclude_first:]
            for j in range(i + 1, len(seqs)):
                seq2 = seqs[j]
                data2 = cache[(seq2.class_name, seq2.sequence_id)]
                arr2_full = data2["arr"]
                frames2 = data2["frame_names"]
                len2 = data2["length"]
                if len2 <= exclude_first:
                    skipped_pairs += 1
                    continue
                arr2 = arr2_full[exclude_first:]
                if arr1.shape[0] == 0 or arr2.shape[0] == 0:
                    skipped_pairs += 1
                    continue

                dist_mat = self.pairwise_distance_matrix(arr1, arr2)
                flat_idx = int(np.argmin(dist_mat))
                local_t1, local_t2 = np.unravel_index(flat_idx, dist_mat.shape)
                t1_min = local_t1 + exclude_first
                t2_min = local_t2 + exclude_first
                relation = "intra_class" if seq1.class_name == seq2.class_name else "inter_class"

                rows.append(
                    {
                        "exclude_first": exclude_first,
                        "relation_type": relation,
                        "sequence1_class": seq1.class_name,
                        "sequence1_id": seq1.sequence_id,
                        "sequence1_length": len1,
                        "sequence1_time_index": int(t1_min),
                        "sequence1_frame_name": frames1[t1_min],
                        "sequence2_class": seq2.class_name,
                        "sequence2_id": seq2.sequence_id,
                        "sequence2_length": len2,
                        "sequence2_time_index": int(t2_min),
                        "sequence2_frame_name": frames2[t2_min],
                        "min_distance": float(dist_mat[local_t1, local_t2]),
                    }
                )

        summary_lines = [
            f"# Excluded Prefix Summary (exclude_first={exclude_first})",
            "",
            f"- total_pairs_kept: {len(rows)}",
            f"- skipped_pairs: {skipped_pairs}",
        ]
        (self.shared_report_dir / f"curve_min_distance_exclude_first_{exclude_first}_summary.md").write_text(
            "\n".join(summary_lines) + "\n",
            encoding="utf-8",
        )
        return rows

    def write_all_pairs_csv(self, rows: list[dict], exclude_first: int) -> None:
        self.write_csv(
            self.shared_csv_dir / f"curve_min_distance_all_pairs_exclude_first_{exclude_first}.csv",
            rows,
            [
                "exclude_first",
                "relation_type",
                "sequence1_class",
                "sequence1_id",
                "sequence1_length",
                "sequence1_time_index",
                "sequence1_frame_name",
                "sequence2_class",
                "sequence2_id",
                "sequence2_length",
                "sequence2_time_index",
                "sequence2_frame_name",
                "min_distance",
            ],
        )

    def write_pair_stats(self, rows: list[dict], exclude_first: int) -> list[dict]:
        grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
        grouped_relation: dict[tuple[str, str], list[float]] = defaultdict(list)
        for row in rows:
            pair = tuple(sorted((row["sequence1_class"], row["sequence2_class"])))
            grouped[pair].append(float(row["min_distance"]))
            grouped_relation[(row["relation_type"], "_".join(pair))].append(float(row["min_distance"]))

        stats_rows: list[dict] = []
        for pair, values in sorted(grouped.items()):
            stats = compute_summary_stats(values)
            relation = "intra_class" if pair[0] == pair[1] else "inter_class"
            stats_rows.append(
                {
                    "exclude_first": exclude_first,
                    "relation_type": relation,
                    "pair": f"{pair[0]}_vs_{pair[1]}",
                    "class_a": pair[0],
                    "class_b": pair[1],
                    "count": len(values),
                    **stats,
                }
            )

        self.write_csv(
            self.shared_csv_dir / f"curve_min_distance_statistics_exclude_first_{exclude_first}.csv",
            stats_rows,
            ["exclude_first", "relation_type", "pair", "class_a", "class_b", "count", "mean", "std", "median", "q1", "q3"],
        )
        return stats_rows

    def write_position_distribution(self, rows: list[dict], exclude_first: int) -> None:
        raw_counts_1: Counter[int] = Counter()
        raw_counts_2: Counter[int] = Counter()
        raw_counts_combined: Counter[int] = Counter()
        progress_counts_combined: Counter[int] = Counter()

        for row in rows:
            t1 = int(row["sequence1_time_index"])
            t2 = int(row["sequence2_time_index"])
            len1 = int(row["sequence1_length"])
            len2 = int(row["sequence2_length"])
            raw_counts_1[t1] += 1
            raw_counts_2[t2] += 1
            raw_counts_combined[t1] += 1
            raw_counts_combined[t2] += 1
            progress_counts_combined[self.progress_bin(t1, len1, exclude_first)] += 1
            progress_counts_combined[self.progress_bin(t2, len2, exclude_first)] += 1

        raw_rows = [
            {
                "time_index": idx,
                "count_sequence1": raw_counts_1[idx],
                "count_sequence2": raw_counts_2[idx],
                "count_combined": raw_counts_combined[idx],
            }
            for idx in sorted(raw_counts_combined)
        ]
        self.write_csv(
            self.shared_csv_dir / f"curve_min_distance_position_distribution_raw_exclude_first_{exclude_first}.csv",
            raw_rows,
            ["time_index", "count_sequence1", "count_sequence2", "count_combined"],
        )

        progress_rows = [
            {
                "progress_bin": idx,
                "count_combined": progress_counts_combined[idx],
            }
            for idx in range(self.progress_bins)
        ]
        self.write_csv(
            self.shared_csv_dir / f"curve_min_distance_position_distribution_progress_exclude_first_{exclude_first}.csv",
            progress_rows,
            ["progress_bin", "count_combined"],
        )

        fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
        x = [row["time_index"] for row in raw_rows]
        y = [row["count_combined"] for row in raw_rows]
        ax.bar(x, y, color="#4c78a8")
        ax.set_title(f"Minimum-Distance Position Distribution (Raw Index, exclude first {exclude_first})")
        ax.set_xlabel("Raw Time Index")
        ax.set_ylabel("Count")
        fig.tight_layout()
        fig.savefig(self.shared_plot_dir / f"curve_min_distance_position_distribution_raw_exclude_first_{exclude_first}.png")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
        x = list(range(self.progress_bins))
        y = [progress_counts_combined[i] for i in x]
        ax.bar(x, y, color="#f58518")
        ax.set_title(f"Minimum-Distance Position Distribution (Progress, exclude first {exclude_first})")
        ax.set_xlabel("Progress Bin")
        ax.set_ylabel("Count")
        fig.tight_layout()
        fig.savefig(self.shared_plot_dir / f"curve_min_distance_position_distribution_progress_exclude_first_{exclude_first}.png")
        plt.close(fig)

    def plot_distance_distribution(self, stats_rows: list[dict], exclude_first: int) -> None:
        ordered_pairs = [
            ("polite", "truesmile"),
            ("ambiguous", "truesmile"),
            ("ambiguous", "polite"),
            ("polite", "polite"),
            ("truesmile", "truesmile"),
            ("ambiguous", "ambiguous"),
        ]
        value_map = { (row["class_a"], row["class_b"]): row for row in stats_rows }
        # Use raw all-pair rows to draw boxplots
        rows = self.read_csv(self.shared_csv_dir / f"curve_min_distance_all_pairs_exclude_first_{exclude_first}.csv")
        grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
        for row in rows:
            pair = tuple(sorted((row["sequence1_class"], row["sequence2_class"])))
            grouped[pair].append(float(row["min_distance"]))

        labels = []
        data = []
        for a, b in ordered_pairs:
            pair = tuple(sorted((a, b)))
            if pair in grouped:
                labels.append(f"{pair[0]} vs {pair[1]}")
                data.append(grouped[pair])

        fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
        box = ax.boxplot(data, patch_artist=True, tick_labels=labels, widths=0.55)
        for patch, label in zip(box["boxes"], labels):
            if "truesmile" in label:
                patch.set_facecolor(COLORS["truesmile"])
            elif "ambiguous" in label and "polite" in label:
                patch.set_facecolor(COLORS["ambiguous"])
            elif "polite" in label:
                patch.set_facecolor(COLORS["polite"])
            else:
                patch.set_facecolor("#999999")
            patch.set_alpha(0.25)
        ax.set_title(f"Minimum Distance Distribution by Pair (exclude first {exclude_first})")
        ax.set_ylabel("Minimum Distance")
        ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        fig.savefig(self.shared_plot_dir / f"curve_min_distance_distribution_exclude_first_{exclude_first}.png")
        plt.close(fig)

    def write_summary(self, stats_rows: list[dict], exclude_first: int) -> None:
        lines = [f"# Minimum Distance Summary After Excluding First {exclude_first} Frames", ""]
        lines.append("## Pair Statistics")
        for row in stats_rows:
            lines.append(
                f"- {row['pair']} ({row['relation_type']}): count={row['count']}, mean={float(row['mean']):.4f}, "
                f"median={float(row['median']):.4f}, q1-q3=({float(row['q1']):.4f}, {float(row['q3']):.4f})"
            )
        (self.shared_report_dir / f"curve_min_distance_statistics_exclude_first_{exclude_first}.md").write_text(
            "\n".join(lines) + "\n",
            encoding="utf-8",
        )

    def run(self) -> None:
        cache = self.load_all_sequences()
        for exclude_first in self.exclude_values:
            rows = self.compute_all_pairs(cache, exclude_first)
            self.write_all_pairs_csv(rows, exclude_first)
            stats_rows = self.write_pair_stats(rows, exclude_first)
            self.write_position_distribution(rows, exclude_first)
            self.plot_distance_distribution(stats_rows, exclude_first)
            self.write_summary(stats_rows, exclude_first)
        print("[12] Saved excluded-prefix minimum-distance analyses.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute minimum distances after excluding the first N frames.")
    parser.add_argument("--analysis_input_root", default=r"E:\Matsuda_data\2-27meeting")
    parser.add_argument("--output_root", default=r"E:\Matsuda_data\3-10meeting")
    parser.add_argument("--progress_bins", type=int, default=20)
    parser.add_argument("--exclude_values", default="5,10")
    args = parser.parse_args()
    task = ExcludedPrefixMinDistanceTask(ExcludedPrefixConfig.from_args(args))
    task.run()


if __name__ == "__main__":
    main()
