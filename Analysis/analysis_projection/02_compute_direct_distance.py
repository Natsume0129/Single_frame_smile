from __future__ import annotations

import numpy as np

from common import CLASS_NAMES, ProjectionConfig, ProjectionTaskBase


class DirectDistanceTask(ProjectionTaskBase):
    def load_prototypes(self, method: str) -> dict[str, np.ndarray]:
        suffix = "methodA" if method == "methodA" else "methodB"
        return {
            class_name: self.load_npy(self.method_proto(method, f"prototype_{class_name}_{suffix}.npy")).astype(np.float32)
            for class_name in CLASS_NAMES
        }

    def run_for_method(self, method: str) -> None:
        protos = self.load_prototypes(method)
        rows: list[dict] = []
        for anchor in CLASS_NAMES:
            for target in CLASS_NAMES:
                if target == anchor:
                    continue
                diff = np.linalg.norm(protos[anchor] - protos[target], axis=1)
                for t, value in enumerate(diff):
                    rows.append(
                        {
                            "method": method,
                            "anchor_class": anchor,
                            "target_class": target,
                            "time_index": t,
                            "difference_norm": float(value),
                        }
                    )
        self.write_csv(
            self.method_csv(method, f"direct_distance_{method}.csv"),
            rows,
            ["method", "anchor_class", "target_class", "time_index", "difference_norm"],
        )
        print(f"[02] Saved direct-distance CSV for {method}.")

    def run(self) -> None:
        for method in ("methodA", "methodB"):
            self.run_for_method(method)


def main() -> None:
    parser = ProjectionTaskBase.build_common_arg_parser("Compute prototype direct distances.")
    args = parser.parse_args()
    task = DirectDistanceTask(ProjectionConfig.from_args(args))
    task.run()


if __name__ == "__main__":
    main()
