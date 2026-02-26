from __future__ import annotations

import numpy as np

from common.base import PipelineConfig, SequenceTaskBase


class BaselineAlignTask(SequenceTaskBase):
    def run(self) -> None:
        for seq in self.discover_sequences():
            in_dir = self.metrics_seq_dir("sequence_features", seq)
            feat = self.load_npy(in_dir / "sequence_features.npy")
            f0 = feat[:5].mean(axis=0, keepdims=True)
            feat_rel = feat - f0

            out_dir = self.metrics_seq_dir("sequence_features_rel", seq)
            self.save_npy(out_dir / "sequence_features_rel.npy", feat_rel.astype(np.float32))
            self.save_npy(out_dir / "baseline_f0.npy", f0.astype(np.float32))
            print(f"[STEP2] {seq.class_name}/{seq.sequence_id}: shape={tuple(feat_rel.shape)}")


def main() -> None:
    parser = SequenceTaskBase.build_common_arg_parser("Step 2: baseline alignment with first 5 frames.")
    args = parser.parse_args()
    task = BaselineAlignTask(PipelineConfig.from_args(args))
    task.run()


if __name__ == "__main__":
    main()

