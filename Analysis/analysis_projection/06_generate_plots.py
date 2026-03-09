from __future__ import annotations

from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from common import CLASS_NAMES, ProjectionConfig, ProjectionTaskBase, SequenceInfo


COLORS = {"polite": "#1f77b4", "truesmile": "#2ca02c", "ambiguous": "#ff7f0e"}


class PlotTask(ProjectionTaskBase):
    def load_rows_grouped(self, path, key_fields: tuple[str, ...], value_field: str) -> dict[tuple[str, ...], list[tuple[int, float]]]:
        rows = self.read_csv(path)
        grouped: dict[tuple[str, ...], list[tuple[int, float]]] = defaultdict(list)
        for row in rows:
            key = tuple(row[field] for field in key_fields)
            grouped[key].append((int(row["time_index"]), float(row[value_field])))
        for key in grouped:
            grouped[key].sort(key=lambda item: item[0])
        return grouped

    def plot_direct_distance(self, method: str) -> None:
        grouped = self.load_rows_grouped(self.method_csv(method, f"direct_distance_{method}.csv"), ("anchor_class", "target_class"), "difference_norm")
        for anchor in CLASS_NAMES:
            fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
            for target in CLASS_NAMES:
                if target == anchor:
                    continue
                items = grouped[(anchor, target)]
                ax.plot([x[0] for x in items], [x[1] for x in items], linewidth=2.0, color=COLORS[target], label=f"{target} vs {anchor}")
            ax.set_title(f"Direct Distance Curves (anchor={anchor}, {method})")
            ax.set_xlabel("Time Index")
            ax.set_ylabel("Distance")
            ax.legend(loc="best")
            fig.tight_layout()
            fig.savefig(self.method_plot(method, f"direct_distance_anchor_{anchor}_{method}.png"))
            plt.close(fig)

    def plot_projection_curve(self, method: str, metric: str) -> None:
        filename = f"projection_along_{method}.csv" if metric == "projection_ratio" else f"projection_off_{method}.csv"
        grouped = self.load_rows_grouped(self.method_csv(method, filename), ("class",), metric)
        fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
        for class_name in CLASS_NAMES:
            items = grouped[(class_name,)]
            ax.plot([x[0] for x in items], [x[1] for x in items], linewidth=2.0, color=COLORS[class_name], label=class_name)
        ax.set_title("Projection Along True-Smile Axis" if metric == "projection_ratio" else "Deviation from True-Smile Axis")
        ax.set_xlabel("Time Index")
        ax.set_ylabel("Ratio")
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(self.method_plot(method, f"{'projection_along' if metric == 'projection_ratio' else 'projection_off'}_{method}.png"))
        plt.close(fig)

    def plot_phase(self, method: str) -> None:
        along = self.load_rows_grouped(self.method_csv(method, f"projection_along_{method}.csv"), ("class",), "projection_ratio")
        off = self.load_rows_grouped(self.method_csv(method, f"projection_off_{method}.csv"), ("class",), "off_axis_ratio")
        fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
        for class_name in CLASS_NAMES:
            x = [v for _, v in along[(class_name,)]]
            y = [v for _, v in off[(class_name,)]]
            ax.plot(x, y, linewidth=2.0, color=COLORS[class_name], label=class_name)
            ax.scatter(x[0], y[0], color=COLORS[class_name], s=20)
            ax.scatter(x[-1], y[-1], color=COLORS[class_name], s=30, marker="x")
        ax.set_title(f"Along vs Off Phase Plot ({method})")
        ax.set_xlabel("Projection Ratio")
        ax.set_ylabel("Off-Axis Ratio")
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(self.method_plot(method, f"projection_phase_{method}.png"))
        plt.close(fig)

    def plot_band_metrics(self, method: str) -> None:
        stats_rows = self.read_csv(self.method_csv(method, f"projection_statistics_{method}.csv"))
        proto_along = self.load_rows_grouped(self.method_csv(method, f"projection_along_{method}.csv"), ("class",), "projection_ratio")
        proto_off = self.load_rows_grouped(self.method_csv(method, f"projection_off_{method}.csv"), ("class",), "off_axis_ratio")
        grouped_stats: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
        for row in stats_rows:
            grouped_stats[(row["metric_type"], row["class"])].append(row)

        for metric_type, out_name, proto_source in (
            ("projection_ratio", f"projection_along_band_{method}.png", proto_along),
            ("off_axis_ratio", f"projection_off_band_{method}.png", proto_off),
        ):
            fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
            for class_name in CLASS_NAMES:
                items = sorted(grouped_stats[(metric_type, class_name)], key=lambda r: int(r["time_index"]))
                t = [int(r["time_index"]) for r in items]
                mean = np.asarray([float(r["mean"]) for r in items], dtype=np.float64)
                q1 = np.asarray([float(r["q1"]) for r in items], dtype=np.float64)
                q3 = np.asarray([float(r["q3"]) for r in items], dtype=np.float64)
                proto = np.asarray([v for _, v in proto_source[(class_name,)]], dtype=np.float64)
                ax.fill_between(t, q1, q3, color=COLORS[class_name], alpha=0.15)
                ax.plot(t, mean, color=COLORS[class_name], linestyle="--", linewidth=1.5, label=f"{class_name} mean")
                ax.plot(t, proto, color=COLORS[class_name], linewidth=2.0, label=f"{class_name} proto")
            ax.set_title(f"Per-Sequence Band Plot ({metric_type}, {method})")
            ax.set_xlabel("Time Index")
            ax.set_ylabel("Ratio")
            ax.legend(loc="best", ncol=2, fontsize=8)
            fig.tight_layout()
            fig.savefig(self.method_plot(method, out_name))
            plt.close(fig)

    def plot_direct_distance_bands(self, method: str) -> None:
        rows = self.read_csv(self.method_csv(method, f"direct_distance_statistics_{method}.csv"))
        grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            grouped[(row["anchor_class"], row["class"])].append(row)
        for anchor in CLASS_NAMES:
            fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
            for target in CLASS_NAMES:
                if target == anchor:
                    continue
                items = sorted(grouped[(anchor, target)], key=lambda r: int(r["time_index"]))
                t = [int(r["time_index"]) for r in items]
                mean = np.asarray([float(r["mean"]) for r in items], dtype=np.float64)
                q1 = np.asarray([float(r["q1"]) for r in items], dtype=np.float64)
                q3 = np.asarray([float(r["q3"]) for r in items], dtype=np.float64)
                ax.fill_between(t, q1, q3, color=COLORS[target], alpha=0.15)
                ax.plot(t, mean, color=COLORS[target], linewidth=2.0, label=target)
            ax.set_title(f"Per-Sequence Direct Distance Bands (anchor={anchor}, {method})")
            ax.set_xlabel("Time Index")
            ax.set_ylabel("Distance")
            ax.legend(loc="best")
            fig.tight_layout()
            fig.savefig(self.method_plot(method, f"direct_distance_band_anchor_{anchor}_{method}.png"))
            plt.close(fig)

    def plot_method_b_montages(self) -> None:
        meta = self.load_json(self.method_proto("methodB", "projection_meta_methodB.json"))
        assert isinstance(meta, dict)
        for class_name in CLASS_NAMES:
            info = meta.get(class_name)
            if not isinstance(info, dict):
                continue
            seq = SequenceInfo(class_name=class_name, sequence_id=str(info["sequence_id"]))
            frames = sorted(self.normalized_frames_dir(seq).glob("*.png"))
            if not frames:
                continue
            cols = 5
            rows = int(np.ceil(len(frames) / cols))
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.0), dpi=120)
            axes_arr = np.atleast_2d(axes)
            for ax in axes_arr.ravel():
                ax.axis("off")
            for idx, frame_path in enumerate(frames):
                ax = axes_arr.ravel()[idx]
                ax.imshow(plt.imread(frame_path))
                ax.set_title(frame_path.stem, fontsize=8)
                ax.axis("off")
            fig.suptitle(f"Method B Prototype Frames: {class_name} / seq {seq.sequence_id}", fontsize=12)
            fig.tight_layout()
            fig.savefig(self.method_plot("methodB", f"prototype_frames_methodB_{class_name}.png"))
            plt.close(fig)

    def run(self) -> None:
        for method in ("methodA", "methodB"):
            self.plot_direct_distance(method)
            self.plot_projection_curve(method, "projection_ratio")
            self.plot_projection_curve(method, "off_axis_ratio")
            self.plot_phase(method)
            self.plot_band_metrics(method)
            self.plot_direct_distance_bands(method)
        self.plot_method_b_montages()
        print("[06] Saved all projection-analysis plots.")


def main() -> None:
    parser = ProjectionTaskBase.build_common_arg_parser("Generate projection-analysis plots.")
    args = parser.parse_args()
    task = PlotTask(ProjectionConfig.from_args(args))
    task.run()


if __name__ == "__main__":
    main()
