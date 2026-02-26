from __future__ import annotations

import numpy as np

from common.base import PipelineConfig, SequenceTaskBase


class MagnitudeTask(SequenceTaskBase):
    def run(self) -> None:
        for seq in self.discover_sequences():
            feat_rel = self.load_npy(
                self.metrics_seq_dir("sequence_features_rel", seq) / "sequence_features_rel.npy"
            )
            distance = np.linalg.norm(feat_rel, axis=1).astype(np.float32)

            out_dir = self.metrics_seq_dir("distance", seq)
            self.save_npy(out_dir / "distance_curve.npy", distance)
            self.save_json(
                out_dir / "metrics.json",
                {
                    "peak_magnitude": float(distance.max()),
                    "mean_magnitude": float(distance.mean()),
                    "std_magnitude": float(distance.std()),
                },
            )
            print(f"[STEP3] {seq.class_name}/{seq.sequence_id}: T={distance.shape[0]}")


def main() -> None:
    parser = SequenceTaskBase.build_common_arg_parser("Step 3: compute magnitude curve and metrics.")
    args = parser.parse_args()
    task = MagnitudeTask(PipelineConfig.from_args(args))
    task.run()


if __name__ == "__main__":
    main()

