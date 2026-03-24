from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from common import ProjectionConfig, ProjectionTaskBase, SequenceInfo


@dataclass
class PositionConfig:
    analysis_input_root: Path
    output_root: Path
    progress_bins: int

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "PositionConfig":
        return cls(
            analysis_input_root=Path(args.analysis_input_root),
            output_root=Path(args.output_root),
            progress_bins=int(args.progress_bins),
        )


class MinDistancePositionTask(ProjectionTaskBase):
    def __init__(self, config: PositionConfig):
        super().__init__(
            ProjectionConfig(
                analysis_input_root=config.analysis_input_root,
                output_root=config.output_root,
                norm_len=config.progress_bins,
            )
        )
        self.progress_bins = config.progress_bins
        self.shared_csv_dir = self.cfg.output_root / "shared" / "csv"
        self.shared_plot_dir = self.cfg.output_root / "shared" / "plots"
        self.shared_report_dir = self.cfg.output_root / "shared" / "report"
        for p in (self.shared_csv_dir, self.shared_plot_dir, self.shared_report_dir):
            p.mkdir(parents=True, exist_ok=True)

    def frame_names_path(self, seq: SequenceInfo) -> Path:
        return (
            self.cfg.analysis_input_root
            / "metrics"
            / "sequence_features"
            / seq.class_name
            / seq.sequence_id
            / "frame_names.json"
        )

    def sequence_length_map(self) -> dict[tuple[str, str], int]:
        out: dict[tuple[str, str], int] = {}
        for seq in self.discover_sequences():
            names = self.load_json(self.frame_names_path(seq))
            assert isinstance(names, list)
            out[(seq.class_name, seq.sequence_id)] = len(names)
        return out

    def progress_bin(self, idx: int, length: int) -> int:
        if length <= 1:
            return 0
        ratio = idx / float(length - 1)
        return min(self.progress_bins - 1, int(round(ratio * (self.progress_bins - 1))))

    def run(self) -> None:
        rows = self.read_csv(self.shared_csv_dir / "curve_min_distance_all_pairs.csv")
        length_map = self.sequence_length_map()

        raw_counts_1: Counter[int] = Counter()
        raw_counts_2: Counter[int] = Counter()
        raw_counts_combined: Counter[int] = Counter()

        progress_counts_1: Counter[int] = Counter()
        progress_counts_2: Counter[int] = Counter()
        progress_counts_combined: Counter[int] = Counter()

        heatmap_progress = np.zeros((self.progress_bins, self.progress_bins), dtype=np.int32)
        relation_progress: dict[str, Counter[int]] = defaultdict(Counter)

        enriched_rows: list[dict] = []

        for row in rows:
            t1 = int(row["sequence1_time_index"])
            t2 = int(row["sequence2_time_index"])
            key1 = (row["sequence1_class"], row["sequence1_id"])
            key2 = (row["sequence2_class"], row["sequence2_id"])
            len1 = length_map[key1]
            len2 = length_map[key2]
            p1 = self.progress_bin(t1, len1)
            p2 = self.progress_bin(t2, len2)

            raw_counts_1[t1] += 1
            raw_counts_2[t2] += 1
            raw_counts_combined[t1] += 1
            raw_counts_combined[t2] += 1

            progress_counts_1[p1] += 1
            progress_counts_2[p2] += 1
            progress_counts_combined[p1] += 1
            progress_counts_combined[p2] += 1

            heatmap_progress[p1, p2] += 1

            relation = row["relation_type"]
            relation_progress[relation][p1] += 1
            relation_progress[relation][p2] += 1

            enriched = dict(row)
            enriched["sequence1_length"] = len1
            enriched["sequence2_length"] = len2
            enriched["sequence1_progress_bin"] = p1
            enriched["sequence2_progress_bin"] = p2
            enriched_rows.append(enriched)

        self.write_csv(
            self.shared_csv_dir / "curve_min_distance_all_pairs_enriched.csv",
            enriched_rows,
            [
                "relation_type",
                "sequence1_class",
                "sequence1_id",
                "sequence1_time_index",
                "sequence1_frame_name",
                "sequence1_length",
                "sequence1_progress_bin",
                "sequence2_class",
                "sequence2_id",
                "sequence2_time_index",
                "sequence2_frame_name",
                "sequence2_length",
                "sequence2_progress_bin",
                "min_distance",
            ],
        )

        raw_rows = []
        for idx in sorted(raw_counts_combined):
            raw_rows.append(
                {
                    "time_index": idx,
                    "count_sequence1": raw_counts_1[idx],
                    "count_sequence2": raw_counts_2[idx],
                    "count_combined": raw_counts_combined[idx],
                }
            )
        self.write_csv(
            self.shared_csv_dir / "curve_min_distance_position_distribution_raw.csv",
            raw_rows,
            ["time_index", "count_sequence1", "count_sequence2", "count_combined"],
        )

        progress_rows = []
        for idx in range(self.progress_bins):
            progress_rows.append(
                {
                    "progress_bin": idx,
                    "count_sequence1": progress_counts_1[idx],
                    "count_sequence2": progress_counts_2[idx],
                    "count_combined": progress_counts_combined[idx],
                    "count_inter_class": relation_progress["inter_class"][idx],
                    "count_intra_class": relation_progress["intra_class"][idx],
                }
            )
        self.write_csv(
            self.shared_csv_dir / "curve_min_distance_position_distribution_progress.csv",
            progress_rows,
            [
                "progress_bin",
                "count_sequence1",
                "count_sequence2",
                "count_combined",
                "count_inter_class",
                "count_intra_class",
            ],
        )

        fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
        x = [row["time_index"] for row in raw_rows]
        y = [row["count_combined"] for row in raw_rows]
        ax.bar(x, y, color="#4c78a8")
        ax.set_title("Minimum-Distance Position Distribution (Raw Time Index)")
        ax.set_xlabel("Raw Time Index")
        ax.set_ylabel("Count")
        fig.tight_layout()
        fig.savefig(self.shared_plot_dir / "curve_min_distance_position_distribution_raw.png")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
        x = list(range(self.progress_bins))
        y = [progress_counts_combined[i] for i in x]
        ax.bar(x, y, color="#f58518")
        ax.set_title("Minimum-Distance Position Distribution (Relative Progress)")
        ax.set_xlabel("Progress Bin")
        ax.set_ylabel("Count")
        fig.tight_layout()
        fig.savefig(self.shared_plot_dir / "curve_min_distance_position_distribution_progress.png")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
        im = ax.imshow(heatmap_progress, origin="lower", aspect="auto", cmap="YlGnBu")
        ax.set_title("Minimum-Distance Position Joint Distribution")
        ax.set_xlabel("Sequence 2 Progress Bin")
        ax.set_ylabel("Sequence 1 Progress Bin")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(self.shared_plot_dir / "curve_min_distance_position_heatmap_progress.png")
        plt.close(fig)

        total_combined = sum(raw_counts_combined.values())
        first3_raw = sum(raw_counts_combined[i] for i in range(3))
        first5_raw = sum(raw_counts_combined[i] for i in range(5))
        first3_progress = sum(progress_counts_combined[i] for i in range(3))
        first5_progress = sum(progress_counts_combined[i] for i in range(5))

        summary_lines = [
            "# Minimum-Distance Position Distribution Summary",
            "",
            f"- total_sequence_positions_counted: {total_combined}",
            f"- first_3_raw_indices_count: {first3_raw} ({first3_raw / total_combined:.2%})",
            f"- first_5_raw_indices_count: {first5_raw} ({first5_raw / total_combined:.2%})",
            f"- first_3_progress_bins_count: {first3_progress} ({first3_progress / total_combined:.2%})",
            f"- first_5_progress_bins_count: {first5_progress} ({first5_progress / total_combined:.2%})",
            "",
            "## Top Raw Time Indices",
        ]
        for idx, count in raw_counts_combined.most_common(10):
            summary_lines.append(f"- raw_time_index={idx}: count={count}")
        summary_lines.append("")
        summary_lines.append("## Top Progress Bins")
        for idx, count in progress_counts_combined.most_common(10):
            summary_lines.append(f"- progress_bin={idx}: count={count}")

        (self.shared_report_dir / "curve_min_distance_position_distribution_summary.md").write_text(
            "\n".join(summary_lines) + "\n",
            encoding="utf-8",
        )

        print("[11] Saved minimum-distance position distribution analysis.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze where minimum-distance points tend to occur.")
    parser.add_argument("--analysis_input_root", default=r"E:\Matsuda_data\2-27meeting")
    parser.add_argument("--output_root", default=r"E:\Matsuda_data\3-10meeting")
    parser.add_argument("--progress_bins", type=int, default=20)
    args = parser.parse_args()
    task = MinDistancePositionTask(PositionConfig.from_args(args))
    task.run()


if __name__ == "__main__":
    main()
