from __future__ import annotations

from collections import defaultdict

from common import ProjectionConfig, ProjectionTaskBase, compute_summary_stats


class StatisticsTask(ProjectionTaskBase):
    def run_for_method(self, method: str) -> None:
        projection_rows = self.read_csv(self.method_csv(method, f"projection_per_sequence_{method}.csv"))
        direct_rows = self.read_csv(self.method_csv(method, f"per_sequence_direct_distance_{method}.csv"))

        grouped_projection: dict[tuple[str, str, int], list[float]] = defaultdict(list)
        for row in projection_rows:
            t = int(row["time_index"])
            grouped_projection[("projection_ratio", row["class"], t)].append(float(row["projection_ratio"]))
            grouped_projection[("off_axis_ratio", row["class"], t)].append(float(row["off_axis_ratio"]))

        projection_stats_rows: list[dict] = []
        for (metric_type, class_name, t), values in sorted(grouped_projection.items()):
            stats = compute_summary_stats(values)
            projection_stats_rows.append(
                {
                    "method": method,
                    "metric_type": metric_type,
                    "class": class_name,
                    "anchor_class": "",
                    "time_index": t,
                    **stats,
                }
            )

        grouped_direct: dict[tuple[str, str, int], list[float]] = defaultdict(list)
        for row in direct_rows:
            t = int(row["time_index"])
            grouped_direct[(row["anchor_class"], row["target_class"], t)].append(float(row["difference_norm"]))

        direct_stats_rows: list[dict] = []
        for (anchor_class, target_class, t), values in sorted(grouped_direct.items()):
            stats = compute_summary_stats(values)
            direct_stats_rows.append(
                {
                    "method": method,
                    "metric_type": "difference_norm",
                    "class": target_class,
                    "anchor_class": anchor_class,
                    "time_index": t,
                    **stats,
                }
            )

        fieldnames = ["method", "metric_type", "class", "anchor_class", "time_index", "mean", "std", "median", "q1", "q3"]
        self.write_csv(self.method_csv(method, f"projection_statistics_{method}.csv"), projection_stats_rows, fieldnames)
        self.write_csv(self.method_csv(method, f"direct_distance_statistics_{method}.csv"), direct_stats_rows, fieldnames)
        print(f"[05] Saved statistics for {method}.")

    def run(self) -> None:
        for method in ("methodA", "methodB"):
            self.run_for_method(method)


def main() -> None:
    parser = ProjectionTaskBase.build_common_arg_parser("Compute statistical summaries.")
    args = parser.parse_args()
    task = StatisticsTask(ProjectionConfig.from_args(args))
    task.run()


if __name__ == "__main__":
    main()
