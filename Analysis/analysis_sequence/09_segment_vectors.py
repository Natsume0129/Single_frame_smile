from __future__ import annotations

import numpy as np

from common.base import CLASS_NAMES, PipelineConfig, SequenceTaskBase


class SegmentVectorTask(SequenceTaskBase):
    def run(self) -> None:
        proto_dir = self.cfg.output_root / "prototypes"
        seg_dict: dict[str, np.ndarray] = {}
        for class_name in CLASS_NAMES:
            proto = np.load(proto_dir / f"prototype_{class_name}.npy")
            seg = proto[1:] - proto[:-1]
            seg_dict[class_name] = seg.astype(np.float32)
            print(f"[STEP9] {class_name}: segments={seg.shape[0]}")
        np.save(proto_dir / "segment_vectors.npy", seg_dict, allow_pickle=True)


def main() -> None:
    parser = SequenceTaskBase.build_common_arg_parser("Step 9: segment vectors for prototypes.")
    args = parser.parse_args()
    task = SegmentVectorTask(PipelineConfig.from_args(args))
    task.run()


if __name__ == "__main__":
    main()

