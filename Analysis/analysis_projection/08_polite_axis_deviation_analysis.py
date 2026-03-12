from __future__ import annotations

from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from common import (
    CLASS_NAMES,
    ProjectionConfig,
    ProjectionTaskBase,
    compute_axis_metrics,
    compute_summary_stats,
)


COLORS = {"polite": "#1f77b4", "truesmile": "#2ca02c", "ambiguous": "#ff7f0e"}


class PoliteAxisDeviationTask(ProjectionTaskBase):
    def load_prototypes(self, method: str) -> dict[str, np.ndarray]:
        suffix = "methodA" if method == "methodA" else "methodB"
        return {
            class_name: self.load_npy(
                self.method_proto(method, f"prototype_{class_name}_{suffix}.npy")
            ).astype(np.float32)
            for class_name in CLASS_NAMES
        }

    def compute_for_method(self, method: str) -> None:
        protos = self.load_prototypes(method)
        polite_proto = protos["polite"]
        axis = polite_proto[-1] - polite_proto[0]
        axis_norm = float(np.linalg.norm(axis))

        proto_rows: list[dict] = []
        per_sequence_rows: list[dict] = []

        for class_name, proto in protos.items():
            _, _, off_axis_distance, off_axis_ratio = compute_axis_metrics(proto, axis)
            for t in range(self.cfg.norm_len):
                proto_rows.append(
                    {
                        "method": method,
                        "axis_class": "polite",
                        "class": class_name,
                        "time_index": t,
                        "off_axis_distance": float(off_axis_distance[t]),
                        "off_axis_ratio": float(off_axis_ratio[t]),
                    }
                )

        for seq in self.discover_sequences():
            arr = self.load_npy(self.normalized_seq_path(seq)).astype(np.float32)
            _, _, off_axis_distance, off_axis_ratio = compute_axis_metrics(arr, axis)
            for t in range(self.cfg.norm_len):
                per_sequence_rows.append(
                    {
                        "method": method,
                        "axis_class": "polite",
                        "class": seq.class_name,
                        "sequence_id": seq.sequence_id,
                        "time_index": t,
                        "off_axis_distance": float(off_axis_distance[t]),
                        "off_axis_ratio": float(off_axis_ratio[t]),
                    }
                )

        self.write_csv(
            self.method_csv(method, f"polite_axis_off_{method}.csv"),
            proto_rows,
            ["method", "axis_class", "class", "time_index", "off_axis_distance", "off_axis_ratio"],
        )
        self.write_csv(
            self.method_csv(method, f"polite_axis_off_per_sequence_{method}.csv"),
            per_sequence_rows,
            [
                "method",
                "axis_class",
                "class",
                "sequence_id",
                "time_index",
                "off_axis_distance",
                "off_axis_ratio",
            ],
        )
        self.save_json(
            self.method_proto(method, f"polite_axis_meta_{method}.json"),
            {
                "method": method,
                "axis_class": "polite",
                "axis_norm": axis_norm,
                "prototype_start_index": 0,
                "prototype_end_index": self.cfg.norm_len - 1,
            },
        )
        print(f"[08] Saved polite-axis deviation CSVs for {method}.")

    def compute_statistics_for_method(self, method: str) -> None:
        rows = self.read_csv(self.method_csv(method, f"polite_axis_off_per_sequence_{method}.csv"))
        grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
        for row in rows:
            grouped[(row["class"], int(row["time_index"]))].append(float(row["off_axis_ratio"]))

        stats_rows: list[dict] = []
        for (class_name, time_index), values in sorted(grouped.items()):
            stats = compute_summary_stats(values)
            stats_rows.append(
                {
                    "method": method,
                    "axis_class": "polite",
                    "metric_type": "off_axis_ratio",
                    "class": class_name,
                    "time_index": time_index,
                    **stats,
                }
            )

        self.write_csv(
            self.method_csv(method, f"polite_axis_off_statistics_{method}.csv"),
            stats_rows,
            ["method", "axis_class", "metric_type", "class", "time_index", "mean", "std", "median", "q1", "q3"],
        )
        print(f"[08] Saved polite-axis statistics for {method}.")

    def plot_for_method(self, method: str) -> None:
        proto_rows = self.read_csv(self.method_csv(method, f"polite_axis_off_{method}.csv"))
        stats_rows = self.read_csv(self.method_csv(method, f"polite_axis_off_statistics_{method}.csv"))

        grouped_proto: dict[str, list[tuple[int, float]]] = defaultdict(list)
        grouped_stats: dict[str, list[dict[str, str]]] = defaultdict(list)

        for row in proto_rows:
            grouped_proto[row["class"]].append((int(row["time_index"]), float(row["off_axis_ratio"])))
        for key in grouped_proto:
            grouped_proto[key].sort(key=lambda item: item[0])

        for row in stats_rows:
            grouped_stats[row["class"]].append(row)
        for key in grouped_stats:
            grouped_stats[key].sort(key=lambda item: int(item["time_index"]))

        fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
        for class_name in CLASS_NAMES:
            items = grouped_proto[class_name]
            ax.plot(
                [x[0] for x in items],
                [x[1] for x in items],
                linewidth=2.0,
                color=COLORS[class_name],
                label=class_name,
            )
        ax.set_title(f"Deviation from Polite-Smile Axis ({method})")
        ax.set_xlabel("Time Index")
        ax.set_ylabel("Ratio")
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(self.method_plot(method, f"polite_axis_off_{method}.png"))
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
        for class_name in CLASS_NAMES:
            proto = np.asarray([v for _, v in grouped_proto[class_name]], dtype=np.float64)
            items = grouped_stats[class_name]
            t = [int(r["time_index"]) for r in items]
            mean = np.asarray([float(r["mean"]) for r in items], dtype=np.float64)
            q1 = np.asarray([float(r["q1"]) for r in items], dtype=np.float64)
            q3 = np.asarray([float(r["q3"]) for r in items], dtype=np.float64)
            ax.fill_between(t, q1, q3, color=COLORS[class_name], alpha=0.15)
            ax.plot(t, mean, color=COLORS[class_name], linestyle="--", linewidth=1.5, label=f"{class_name} mean")
            ax.plot(t, proto, color=COLORS[class_name], linewidth=2.0, label=f"{class_name} proto")
        ax.set_title(f"Per-Sequence Band Plot (polite axis off ratio, {method})")
        ax.set_xlabel("Time Index")
        ax.set_ylabel("Ratio")
        ax.legend(loc="best", ncol=2, fontsize=8)
        fig.tight_layout()
        fig.savefig(self.method_plot(method, f"polite_axis_off_band_{method}.png"))
        plt.close(fig)
        print(f"[08] Saved polite-axis plots for {method}.")

    def write_report_for_method(self, method: str) -> None:
        proto_rows = self.read_csv(self.method_csv(method, f"polite_axis_off_{method}.csv"))
        stats_rows = self.read_csv(self.method_csv(method, f"polite_axis_off_statistics_{method}.csv"))

        grouped_proto: dict[str, list[dict[str, str]]] = defaultdict(list)
        grouped_stats: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in proto_rows:
            grouped_proto[row["class"]].append(row)
        for row in stats_rows:
            grouped_stats[row["class"]].append(row)

        lines = [f"# Polite-Axis Deviation Summary ({method})", ""]
        lines.append("## Prototype Curves")
        for class_name in CLASS_NAMES:
            proto_items = sorted(grouped_proto[class_name], key=lambda r: int(r["time_index"]))
            values = [float(item["off_axis_ratio"]) for item in proto_items]
            peak_idx = max(range(len(values)), key=lambda i: values[i])
            lines.append(
                f"- {class_name}: start={values[0]:.4f}, peak={values[peak_idx]:.4f} at t={peak_idx}, end={values[-1]:.4f}"
            )
        lines.append("")
        lines.append("## Per-Sequence Means")
        for class_name in CLASS_NAMES:
            stat_items = sorted(grouped_stats[class_name], key=lambda r: int(r["time_index"]))
            end = stat_items[-1]
            peak = max(stat_items, key=lambda r: float(r["mean"]))
            lines.append(
                f"- {class_name}: mean_end={float(end['mean']):.4f}, mean_peak={float(peak['mean']):.4f} at t={peak['time_index']}, "
                f"iqr_end=({float(end['q1']):.4f}, {float(end['q3']):.4f})"
            )
        lines.append("")
        lines.append("## Notes")
        lines.append("- This analysis uses the polite-smile prototype vector as the base axis.")
        lines.append("- The metric is the off-axis deviation ratio relative to the polite-smile axis.")
        self.method_report(method, f"polite_axis_summary_{method}.md").write_text(
            "\n".join(lines) + "\n",
            encoding="utf-8",
        )
        print(f"[08] Saved polite-axis summary for {method}.")

    def run(self) -> None:
        for method in ("methodA", "methodB"):
            self.compute_for_method(method)
            self.compute_statistics_for_method(method)
            self.plot_for_method(method)
            self.write_report_for_method(method)


def main() -> None:
    parser = ProjectionTaskBase.build_common_arg_parser(
        "Compute extra polite-axis deviation analysis."
    )
    args = parser.parse_args()
    task = PoliteAxisDeviationTask(ProjectionConfig.from_args(args))
    task.run()


if __name__ == "__main__":
    main()
