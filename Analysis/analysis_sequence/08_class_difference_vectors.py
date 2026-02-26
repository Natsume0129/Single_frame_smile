from __future__ import annotations

import numpy as np

from common.base import PipelineConfig, SequenceTaskBase


class ClassDifferenceTask(SequenceTaskBase):
    def run(self) -> None:
        proto_dir = self.cfg.output_root / "prototypes"
        p_polite = np.load(proto_dir / "prototype_polite.npy")
        p_true = np.load(proto_dir / "prototype_truesmile.npy")
        p_amb = np.load(proto_dir / "prototype_ambiguous.npy")

        diff_dict = {
            "polite_vs_truesmile": (p_polite - p_true).astype(np.float32),
            "polite_vs_ambiguous": (p_polite - p_amb).astype(np.float32),
            "truesmile_vs_ambiguous": (p_true - p_amb).astype(np.float32),
        }
        np.save(self.cfg.output_root / "prototypes" / "class_difference_vectors.npy", diff_dict, allow_pickle=True)
        print("[STEP8] Saved class difference dict with 3 keys.")


def main() -> None:
    parser = SequenceTaskBase.build_common_arg_parser("Step 8: class difference vectors.")
    args = parser.parse_args()
    task = ClassDifferenceTask(PipelineConfig.from_args(args))
    task.run()


if __name__ == "__main__":
    main()

