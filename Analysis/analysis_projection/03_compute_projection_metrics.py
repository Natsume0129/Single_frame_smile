from __future__ import annotations

import numpy as np

from common import CLASS_NAMES, ProjectionConfig, ProjectionTaskBase, compute_axis_metrics


class ProjectionMetricsTask(ProjectionTaskBase):
    def load_prototypes(self, method: str) -> dict[str, np.ndarray]:
        suffix = "methodA" if method == "methodA" else "methodB"
        return {
            class_name: self.load_npy(self.method_proto(method, f"prototype_{class_name}_{suffix}.npy")).astype(np.float32)
            for class_name in CLASS_NAMES
        }

    def run_for_method(self, method: str) -> None:
        protos = self.load_prototypes(method)
        true_proto = protos["truesmile"]
        axis = true_proto[-1] - true_proto[0]
        axis_norm = float(np.linalg.norm(axis))

        along_rows: list[dict] = []
        off_rows: list[dict] = []
        for class_name, proto in protos.items():
            projection_length, projection_ratio, off_axis_distance, off_axis_ratio = compute_axis_metrics(proto, axis)
            for t in range(self.cfg.norm_len):
                along_rows.append(
                    {
                        "method": method,
                        "class": class_name,
                        "time_index": t,
                        "projection_length": float(projection_length[t]),
                        "projection_ratio": float(projection_ratio[t]),
                    }
                )
                off_rows.append(
                    {
                        "method": method,
                        "class": class_name,
                        "time_index": t,
                        "off_axis_distance": float(off_axis_distance[t]),
                        "off_axis_ratio": float(off_axis_ratio[t]),
                    }
                )

        self.write_csv(
            self.method_csv(method, f"projection_along_{method}.csv"),
            along_rows,
            ["method", "class", "time_index", "projection_length", "projection_ratio"],
        )
        self.write_csv(
            self.method_csv(method, f"projection_off_{method}.csv"),
            off_rows,
            ["method", "class", "time_index", "off_axis_distance", "off_axis_ratio"],
        )
        self.save_json(
            self.method_proto(method, f"axis_meta_{method}.json"),
            {
                "method": method,
                "norm_len": self.cfg.norm_len,
                "axis_norm": axis_norm,
                "prototype_start_index": 0,
                "prototype_end_index": self.cfg.norm_len - 1,
            },
        )
        print(f"[03] Saved projection metrics for {method}.")

    def run(self) -> None:
        for method in ("methodA", "methodB"):
            self.run_for_method(method)


def main() -> None:
    parser = ProjectionTaskBase.build_common_arg_parser("Compute projection metrics.")
    args = parser.parse_args()
    task = ProjectionMetricsTask(ProjectionConfig.from_args(args))
    task.run()


if __name__ == "__main__":
    main()
