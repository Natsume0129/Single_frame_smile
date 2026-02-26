from __future__ import annotations

import numpy as np

from common.base import CLASS_NAMES, PipelineConfig, SequenceTaskBase


def medoid_index_by_frobenius(seqs: np.ndarray) -> int:
    n = seqs.shape[0]
    costs = np.zeros((n,), dtype=np.float64)
    for i in range(n):
        diff = seqs[i][None, :, :] - seqs
        d = np.sqrt(np.sum(diff * diff, axis=(1, 2)))
        costs[i] = d.sum()
    return int(np.argmin(costs))


class PrototypeTask(SequenceTaskBase):
    def run(self) -> None:
        out_dir = self.cfg.output_root / "prototypes"
        out_dir.mkdir(parents=True, exist_ok=True)
        summary: dict[str, dict] = {}

        for class_name in CLASS_NAMES:
            seq_infos = [s for s in self.discover_sequences() if s.class_name == class_name]
            if not seq_infos:
                continue

            arrs = []
            sequence_ids = []
            for seq in seq_infos:
                arr = self.load_npy(
                    self.metrics_seq_dir("normalized", seq) / "normalized_sequence.npy"
                ).astype(np.float32)
                arrs.append(arr)
                sequence_ids.append(seq.sequence_id)
            stacked = np.stack(arrs, axis=0)  # [N, T, D]

            median_proto = np.median(stacked, axis=0).astype(np.float32)
            medoid_idx = medoid_index_by_frobenius(stacked)
            medoid_proto = stacked[medoid_idx]
            medoid_seq_id = sequence_ids[medoid_idx]

            np.save(out_dir / f"prototype_{class_name}.npy", median_proto)
            np.save(out_dir / f"prototype_{class_name}_medoid.npy", medoid_proto)
            summary[class_name] = {
                "num_sequences": int(stacked.shape[0]),
                "shape": [int(stacked.shape[1]), int(stacked.shape[2])],
                "medoid_sequence_id": medoid_seq_id,
            }
            print(f"[STEP7] {class_name}: N={stacked.shape[0]}, medoid={medoid_seq_id}")

        self.save_json(out_dir / "prototype_meta.json", summary)


def main() -> None:
    parser = SequenceTaskBase.build_common_arg_parser("Step 7: build class prototypes.")
    args = parser.parse_args()
    task = PrototypeTask(PipelineConfig.from_args(args))
    task.run()


if __name__ == "__main__":
    main()

