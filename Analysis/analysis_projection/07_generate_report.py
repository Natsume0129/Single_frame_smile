from __future__ import annotations

from collections import defaultdict

from common import CLASS_NAMES, ProjectionConfig, ProjectionTaskBase


class ReportTask(ProjectionTaskBase):
    def summarize_direct_distance(self, method: str) -> list[str]:
        rows = self.read_csv(self.method_csv(method, f"direct_distance_{method}.csv"))
        grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            grouped[(row["anchor_class"], row["target_class"])].append(row)
        lines = ["## Direct Distance"]
        for anchor in CLASS_NAMES:
            lines.append(f"### Anchor: {anchor}")
            for target in CLASS_NAMES:
                if target == anchor:
                    continue
                items = sorted(grouped[(anchor, target)], key=lambda r: int(r["time_index"]))
                values = [float(item["difference_norm"]) for item in items]
                max_idx = max(range(len(values)), key=lambda i: values[i])
                lines.append(f"- {target}: start={values[0]:.4f}, peak={values[max_idx]:.4f} at t={max_idx}, end={values[-1]:.4f}")
        return lines

    def summarize_projection(self, method: str) -> list[str]:
        along_rows = self.read_csv(self.method_csv(method, f"projection_along_{method}.csv"))
        off_rows = self.read_csv(self.method_csv(method, f"projection_off_{method}.csv"))
        grouped_along: dict[str, list[dict[str, str]]] = defaultdict(list)
        grouped_off: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in along_rows:
            grouped_along[row["class"]].append(row)
        for row in off_rows:
            grouped_off[row["class"]].append(row)
        lines = ["## Axis Metrics"]
        for class_name in CLASS_NAMES:
            along_items = sorted(grouped_along[class_name], key=lambda r: int(r["time_index"]))
            off_items = sorted(grouped_off[class_name], key=lambda r: int(r["time_index"]))
            along_vals = [float(item["projection_ratio"]) for item in along_items]
            off_vals = [float(item["off_axis_ratio"]) for item in off_items]
            lines.append(f"- {class_name}: along_end={along_vals[-1]:.4f}, along_peak={max(along_vals):.4f}, off_end={off_vals[-1]:.4f}, off_peak={max(off_vals):.4f}")
        return lines

    def summarize_statistics(self, method: str) -> list[str]:
        rows = self.read_csv(self.method_csv(method, f"projection_statistics_{method}.csv"))
        grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            grouped[(row["metric_type"], row["class"])].append(row)
        lines = ["## Per-Sequence Statistics"]
        for metric_type in ("projection_ratio", "off_axis_ratio"):
            lines.append(f"### {metric_type}")
            for class_name in CLASS_NAMES:
                items = sorted(grouped[(metric_type, class_name)], key=lambda r: int(r["time_index"]))
                end = items[-1]
                peak = max(items, key=lambda r: float(r["mean"]))
                lines.append(
                    f"- {class_name}: mean_end={float(end['mean']):.4f}, mean_peak={float(peak['mean']):.4f} at t={peak['time_index']}, iqr_end=({float(end['q1']):.4f}, {float(end['q3']):.4f})"
                )
        return lines

    def run_for_method(self, method: str) -> None:
        lines = [f"# Projection Summary ({method})", ""]
        lines.extend(self.summarize_direct_distance(method))
        lines.append("")
        lines.extend(self.summarize_projection(method))
        lines.append("")
        lines.extend(self.summarize_statistics(method))
        lines.append("")
        lines.append("## Notes")
        lines.append("- Method A uses median prototype trajectories.")
        lines.append("- Method B uses medoid prototype trajectories and preserves real sequence IDs.")
        self.method_report(method, f"projection_summary_{method}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"[07] Saved report for {method}.")

    def run(self) -> None:
        for method in ("methodA", "methodB"):
            self.run_for_method(method)


def main() -> None:
    parser = ProjectionTaskBase.build_common_arg_parser("Generate projection-analysis reports.")
    args = parser.parse_args()
    task = ReportTask(ProjectionConfig.from_args(args))
    task.run()


if __name__ == "__main__":
    main()
