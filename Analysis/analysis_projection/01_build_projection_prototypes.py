from __future__ import annotations

import numpy as np

from common import CLASS_NAMES, ProjectionConfig, ProjectionTaskBase, medoid_index_by_frobenius


class BuildProjectionPrototypesTask(ProjectionTaskBase):
    def run(self) -> None:
        meta_a: dict[str, dict] = {}
        meta_b: dict[str, dict] = {}

        for class_name in CLASS_NAMES:
            seq_infos = self.sequences_for_class(class_name)
            if not seq_infos:
                continue

            stacked = np.stack(
                [self.load_npy(self.normalized_seq_path(seq)).astype(np.float32) for seq in seq_infos],
                axis=0,
            )
            median_proto = np.median(stacked, axis=0).astype(np.float32)
            medoid_idx = medoid_index_by_frobenius(stacked)
            medoid_proto = stacked[medoid_idx].astype(np.float32)
            medoid_seq = seq_infos[medoid_idx]

            self.save_npy(self.method_proto("methodA", f"prototype_{class_name}_methodA.npy"), median_proto)
            self.save_npy(self.method_proto("methodB", f"prototype_{class_name}_methodB.npy"), medoid_proto)

            meta_a[class_name] = {
                "num_sequences": int(stacked.shape[0]),
                "shape": [int(stacked.shape[1]), int(stacked.shape[2])],
            }
            meta_b[class_name] = {
                "num_sequences": int(stacked.shape[0]),
                "shape": [int(stacked.shape[1]), int(stacked.shape[2])],
                "sequence_id": medoid_seq.sequence_id,
                "normalized_frames_dir": str(self.normalized_frames_dir(medoid_seq)),
                "sampled_frames_json": str(self.sampled_frames_path(medoid_seq)),
            }
            print(f"[01] {class_name}: N={stacked.shape[0]}, medoid={medoid_seq.sequence_id}")

        self.save_json(self.method_proto("methodA", "projection_meta_methodA.json"), meta_a)
        self.save_json(self.method_proto("methodB", "projection_meta_methodB.json"), meta_b)


def main() -> None:
    parser = ProjectionTaskBase.build_common_arg_parser("Build projection prototypes.")
    args = parser.parse_args()
    task = BuildProjectionPrototypesTask(ProjectionConfig.from_args(args))
    task.run()


if __name__ == "__main__":
    main()
