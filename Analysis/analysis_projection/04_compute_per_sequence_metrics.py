from __future__ import annotations

import numpy as np

from common import CLASS_NAMES, ProjectionConfig, ProjectionTaskBase, compute_axis_metrics


class PerSequenceMetricsTask(ProjectionTaskBase):
    def load_prototypes(self, method: str) -> dict[str, np.ndarray]:
        suffix = "methodA" if method == "methodA" else "methodB"
        return {
            class_name: self.load_npy(self.method_proto(method, f"prototype_{class_name}_{suffix}.npy")).astype(np.float32)
            for class_name in CLASS_NAMES
        }

    def run_for_method(self, method: str) -> None:
        protos = self.load_prototypes(method)
        axis = protos["truesmile"][-1] - protos["truesmile"][0]

        projection_rows: list[dict] = []
        direct_rows: list[dict] = []

        for seq in self.discover_sequences():
            arr = self.load_npy(self.normalized_seq_path(seq)).astype(np.float32)
            _, projection_ratio, _, off_axis_ratio = compute_axis_metrics(arr, axis)
            for t in range(self.cfg.norm_len):
                projection_rows.append(
                    {
                        "method": method,
                        "class": seq.class_name,
                        "sequence_id": seq.sequence_id,
                        "time_index": t,
                        "projection_ratio": float(projection_ratio[t]),
                        "off_axis_ratio": float(off_axis_ratio[t]),
                    }
                )

            for anchor in CLASS_NAMES:
                diff = np.linalg.norm(arr - protos[anchor], axis=1)
                for t, value in enumerate(diff):
                    direct_rows.append(
                        {
                            "method": method,
                            "anchor_class": anchor,
                            "target_class": seq.class_name,
                            "sequence_id": seq.sequence_id,
                            "time_index": t,
                            "difference_norm": float(value),
                        }
                    )

        self.write_csv(
            self.method_csv(method, f"projection_per_sequence_{method}.csv"),
            projection_rows,
            ["method", "class", "sequence_id", "time_index", "projection_ratio", "off_axis_ratio"],
        )
        self.write_csv(
            self.method_csv(method, f"per_sequence_direct_distance_{method}.csv"),
            direct_rows,
            ["method", "anchor_class", "target_class", "sequence_id", "time_index", "difference_norm"],
        )
        print(f"[04] Saved per-sequence metrics for {method}.")

    def run(self) -> None:
        for method in ("methodA", "methodB"):
            self.run_for_method(method)


def main() -> None:
    parser = ProjectionTaskBase.build_common_arg_parser("Compute per-sequence projection metrics.")
    args = parser.parse_args()
    task = PerSequenceMetricsTask(ProjectionConfig.from_args(args))
    task.run()


if __name__ == "__main__":
    main()
