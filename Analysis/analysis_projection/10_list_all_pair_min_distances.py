from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from common import ProjectionConfig, ProjectionTaskBase, SequenceInfo


@dataclass
class AllPairConfig:
    analysis_input_root: Path
    output_root: Path

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "AllPairConfig":
        return cls(
            analysis_input_root=Path(args.analysis_input_root),
            output_root=Path(args.output_root),
        )


class AllPairMinDistanceTask(ProjectionTaskBase):
    def __init__(self, config: AllPairConfig):
        super().__init__(
            ProjectionConfig(
                analysis_input_root=config.analysis_input_root,
                output_root=config.output_root,
                norm_len=20,
            )
        )
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
            }
        return cache

    def run(self) -> None:
        cache = self.load_all_sequences()
        seqs = [v["seq"] for v in cache.values()]
        rows: list[dict] = []

        for i, seq1 in enumerate(seqs):
            data1 = cache[(seq1.class_name, seq1.sequence_id)]
            arr1 = data1["arr"]
            frames1 = data1["frame_names"]
            for j in range(i + 1, len(seqs)):
                seq2 = seqs[j]
                data2 = cache[(seq2.class_name, seq2.sequence_id)]
                arr2 = data2["arr"]
                frames2 = data2["frame_names"]

                dist_mat = self.pairwise_distance_matrix(arr1, arr2)
                flat_idx = int(np.argmin(dist_mat))
                t1_min, t2_min = np.unravel_index(flat_idx, dist_mat.shape)
                relation = "intra_class" if seq1.class_name == seq2.class_name else "inter_class"

                rows.append(
                    {
                        "relation_type": relation,
                        "sequence1_class": seq1.class_name,
                        "sequence1_id": seq1.sequence_id,
                        "sequence1_time_index": int(t1_min),
                        "sequence1_frame_name": frames1[t1_min],
                        "sequence2_class": seq2.class_name,
                        "sequence2_id": seq2.sequence_id,
                        "sequence2_time_index": int(t2_min),
                        "sequence2_frame_name": frames2[t2_min],
                        "min_distance": float(dist_mat[t1_min, t2_min]),
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

        out_csv = self.shared_csv_dir / "curve_min_distance_all_pairs.csv"
        self.write_csv(
            out_csv,
            rows,
            [
                "relation_type",
                "sequence1_class",
                "sequence1_id",
                "sequence1_time_index",
                "sequence1_frame_name",
                "sequence2_class",
                "sequence2_id",
                "sequence2_time_index",
                "sequence2_frame_name",
                "min_distance",
            ],
        )

        inter_count = sum(1 for r in rows if r["relation_type"] == "inter_class")
        intra_count = sum(1 for r in rows if r["relation_type"] == "intra_class")
        summary = [
            "# All Pair Minimum Distance Summary",
            "",
            f"- total_pairs: {len(rows)}",
            f"- inter_class_pairs: {inter_count}",
            f"- intra_class_pairs: {intra_count}",
            "",
            f"- output_csv: {out_csv}",
        ]
        (self.shared_report_dir / "curve_min_distance_all_pairs_summary.md").write_text(
            "\n".join(summary) + "\n",
            encoding="utf-8",
        )
        print(f"[10] Saved all-pair minimum-distance list: {out_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(description="List minimum distances for all sequence pairs, including intra-class pairs.")
    parser.add_argument("--analysis_input_root", default=r"E:\Matsuda_data\2-27meeting")
    parser.add_argument("--output_root", default=r"E:\Matsuda_data\3-10meeting")
    args = parser.parse_args()
    task = AllPairMinDistanceTask(AllPairConfig.from_args(args))
    task.run()


if __name__ == "__main__":
    main()
