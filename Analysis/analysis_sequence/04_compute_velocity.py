from __future__ import annotations

import numpy as np

from common.base import PipelineConfig, SequenceTaskBase


class VelocityTask(SequenceTaskBase):
    def run(self) -> None:
        for seq in self.discover_sequences():
            feat_rel = self.load_npy(
                self.metrics_seq_dir("sequence_features_rel", seq) / "sequence_features_rel.npy"
            )
            velocity = np.zeros((feat_rel.shape[0],), dtype=np.float32)
            if feat_rel.shape[0] > 1:
                delta = feat_rel[1:] - feat_rel[:-1]
                velocity[1:] = np.linalg.norm(delta, axis=1)

            out_dir = self.metrics_seq_dir("velocity", seq)
            self.save_npy(out_dir / "velocity_curve.npy", velocity)
            self.save_json(
                out_dir / "metrics.json",
                {
                    "mean_velocity": float(velocity.mean()),
                    "peak_velocity": float(velocity.max()),
                    "total_motion_energy": float(velocity.sum()),
                },
            )
            print(f"[STEP4] {seq.class_name}/{seq.sequence_id}: T={velocity.shape[0]}")


def main() -> None:
    parser = SequenceTaskBase.build_common_arg_parser("Step 4: compute velocity curve and metrics.")
    args = parser.parse_args()
    task = VelocityTask(PipelineConfig.from_args(args))
    task.run()


if __name__ == "__main__":
    main()

