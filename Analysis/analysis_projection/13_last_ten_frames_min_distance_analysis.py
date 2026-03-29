from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from common import ProjectionConfig, ProjectionTaskBase, SequenceInfo, compute_summary_stats


@dataclass
class LastTenConfig:
    analysis_input_root: Path
    output_root: Path
    tail_length: int

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "LastTenConfig":
        return cls(
            analysis_input_root=Path(args.analysis_input_root),
            output_root=Path(args.output_root),
            tail_length=int(args.tail_length),
        )


class LastTenMinDistanceTask(ProjectionTaskBase):
    def __init__(self, config: LastTenConfig):
        super().__init__(
            ProjectionConfig(
                analysis_input_root=config.analysis_input_root,
                output_root=config.output_root,
                norm_len=20,
            )
        )
        self.tail_length = config.tail_length
        self.shared_csv_dir = self.cfg.output_root / "shared" / "csv"
        self.shared_report_dir = self.cfg.output_root / "shared" / "report"
        self.shared_csv_dir.mkdir(parents=True, exist_ok=True)
        self.shared_report_dir.mkdir(parents=True, exist_ok=True)

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

    def run(self) -> None:
        cache = self.load_all_sequences()
        seqs = [v["seq"] for v in cache.values()]
        rows: list[dict] = []

        for i, seq1 in enumerate(seqs):
            data1 = cache[(seq1.class_name, seq1.sequence_id)]
            arr1_full = data1["arr"]
            frames1 = data1["frame_names"]
            len1 = data1["length"]
            start1 = max(0, len1 - self.tail_length)
            arr1 = arr1_full[start1:]

            for j in range(i + 1, len(seqs)):
                seq2 = seqs[j]
                data2 = cache[(seq2.class_name, seq2.sequence_id)]
                arr2_full = data2["arr"]
                frames2 = data2["frame_names"]
                len2 = data2["length"]
                start2 = max(0, len2 - self.tail_length)
                arr2 = arr2_full[start2:]

                dist_mat = self.pairwise_distance_matrix(arr1, arr2)
                flat_idx = int(np.argmin(dist_mat))
                local_t1, local_t2 = np.unravel_index(flat_idx, dist_mat.shape)
                t1_min = local_t1 + start1
                t2_min = local_t2 + start2
                relation = "intra_class" if seq1.class_name == seq2.class_name else "inter_class"

                rows.append(
                    {
                        "tail_length": self.tail_length,
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

        rows.sort(
            key=lambda r: (
                r["relation_type"],
                r["sequence1_class"],
                r["sequence2_class"],
                r["sequence1_id"],
                r["sequence2_id"],
            )
        )

        all_pairs_csv = self.shared_csv_dir / f"curve_min_distance_all_pairs_last_{self.tail_length}_frames.csv"
        self.write_csv(
            all_pairs_csv,
            rows,
            [
                "tail_length",
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

        grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
        for row in rows:
            pair = tuple(sorted((row["sequence1_class"], row["sequence2_class"])))
            grouped[pair].append(float(row["min_distance"]))

        stats_rows: list[dict] = []
        for pair, values in sorted(grouped.items()):
            stats = compute_summary_stats(values)
            relation = "intra_class" if pair[0] == pair[1] else "inter_class"
            stats_rows.append(
                {
                    "tail_length": self.tail_length,
                    "relation_type": relation,
                    "pair": f"{pair[0]}_vs_{pair[1]}",
                    "class_a": pair[0],
                    "class_b": pair[1],
                    "count": len(values),
                    **stats,
                }
            )

        stats_csv = self.shared_csv_dir / f"curve_min_distance_statistics_last_{self.tail_length}_frames.csv"
        self.write_csv(
            stats_csv,
            stats_rows,
            ["tail_length", "relation_type", "pair", "class_a", "class_b", "count", "mean", "std", "median", "q1", "q3"],
        )

        lines = [f"# Last {self.tail_length} Frames Minimum Distance Summary", ""]
        lines.append("## Intra-class")
        for row in stats_rows:
            if row["relation_type"] != "intra_class":
                continue
            lines.append(
                f"- {row['pair']}: count={row['count']}, mean={float(row['mean']):.4f}, median={float(row['median']):.4f}, "
                f"q1-q3=({float(row['q1']):.4f}, {float(row['q3']):.4f})"
            )
        lines.append("")
        lines.append("## Inter-class")
        for row in stats_rows:
            if row["relation_type"] != "inter_class":
                continue
            lines.append(
                f"- {row['pair']}: count={row['count']}, mean={float(row['mean']):.4f}, median={float(row['median']):.4f}, "
                f"q1-q3=({float(row['q1']):.4f}, {float(row['q3']):.4f})"
            )
        lines.append("")
        lines.append(f"- all_pairs_csv: {all_pairs_csv}")
        lines.append(f"- stats_csv: {stats_csv}")
        (self.shared_report_dir / f"curve_min_distance_statistics_last_{self.tail_length}_frames.md").write_text(
            "\n".join(lines) + "\n",
            encoding="utf-8",
        )
        print(f"[13] Saved last-{self.tail_length}-frames minimum-distance analysis.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute minimum distance using only the last N frames of each sequence.")
    parser.add_argument("--analysis_input_root", default=r"E:\Matsuda_data\2-27meeting")
    parser.add_argument("--output_root", default=r"E:\Matsuda_data\3-10meeting")
    parser.add_argument("--tail_length", type=int, default=10)
    args = parser.parse_args()
    task = LastTenMinDistanceTask(LastTenConfig.from_args(args))
    task.run()


if __name__ == "__main__":
    main()
