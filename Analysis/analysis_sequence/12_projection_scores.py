from __future__ import annotations

import numpy as np

from common.base import PipelineConfig, SequenceTaskBase


class ProjectionScoreTask(SequenceTaskBase):
    def run(self) -> None:
        diff_dict = np.load(
            self.cfg.output_root / "prototypes" / "class_difference_vectors.npy",
            allow_pickle=True,
        ).item()

        rows: list[dict] = []
        for seq in self.discover_sequences():
            f_norm = self.load_npy(self.metrics_seq_dir("normalized", seq) / "normalized_sequence.npy")
            for pair_name, delta in diff_dict.items():
                score = np.sum(f_norm * delta, axis=1)
                for t, val in enumerate(score.tolist()):
                    rows.append(
                        {
                            "class": seq.class_name,
                            "sequence_id": seq.sequence_id,
                            "pair": pair_name,
                            "t_index": t,
                            "score": float(val),
                        }
                    )

        self.write_csv(
            self.cfg.output_root / "csv" / "projection_scores.csv",
            rows,
            ["class", "sequence_id", "pair", "t_index", "score"],
        )
        print(f"[STEP12] Saved projection scores rows={len(rows)}")


def main() -> None:
    parser = SequenceTaskBase.build_common_arg_parser("Step 12: projection scores on class pair axes.")
    args = parser.parse_args()
    task = ProjectionScoreTask(PipelineConfig.from_args(args))
    task.run()


if __name__ == "__main__":
    main()

