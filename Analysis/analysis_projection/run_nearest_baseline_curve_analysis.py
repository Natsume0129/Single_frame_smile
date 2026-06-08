from __future__ import annotations

import argparse
import csv
import html
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import CLASS_NAMES


METHODS = ("methodA", "methodB")
BASELINE_CLASSES = ("truesmile", "polite")
COLORS = {"polite": "#1f77b4", "truesmile": "#2ca02c", "ambiguous": "#ff7f0e"}

CSV_FIELDS = [
    "method",
    "baseline_class",
    "target_class",
    "source_type",
    "sequence_id",
    "rank",
    "target_stage_percent",
    "nearest_baseline_progress_percent",
    "nearest_distance",
    "nearest_sample_index",
    "target_fixed_axis_s_percent",
    "target_fixed_axis_d",
    "nearest_fixed_axis_s_percent",
    "nearest_fixed_axis_d",
]


@dataclass
class NearestBaselineConfig:
    analysis_input_root: Path = Path(r"E:\Matsuda_data\2-27meeting")
    projection_output_root: Path = Path(r"E:\Matsuda_data\3-10meeting")
    output_root: Path = Path(r"E:\Matsuda_data\3-10meeting\nearest_baseline_curve")
    target_stages: tuple[int, ...] = tuple(range(5, 101, 5))
    search_step_percent: float = 1.0
    baseline_classes: tuple[str, ...] = BASELINE_CLASSES

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "NearestBaselineConfig":
        stages = tuple(int(x.strip()) for x in args.target_stages.split(",") if x.strip())
        baseline_classes = tuple(x.strip() for x in args.baseline_classes.split(",") if x.strip())
        return cls(
            analysis_input_root=Path(args.analysis_input_root),
            projection_output_root=Path(args.projection_output_root),
            output_root=Path(args.output_root),
            target_stages=stages,
            search_step_percent=float(args.search_step_percent),
            baseline_classes=baseline_classes,
        )


class NearestBaselineCurveAnalysis:
    def __init__(self, cfg: NearestBaselineConfig):
        self.cfg = cfg
        self.csv_dir = cfg.output_root / "csv"
        self.plot_dir = cfg.output_root / "plots"
        self.report_dir = cfg.output_root / "report"
        for path in (self.csv_dir, self.plot_dir, self.report_dir):
            path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def write_csv(path: Path, rows: Iterable[dict], fieldnames: list[str]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    @staticmethod
    def point_at_percent(curve: np.ndarray, percent: float) -> np.ndarray:
        if percent < 0.0 or percent > 100.0:
            raise ValueError(f"percent must be in [0, 100], got {percent}")
        pos = (percent / 100.0) * (curve.shape[0] - 1)
        lo = int(np.floor(pos))
        hi = int(np.ceil(pos))
        if lo == hi:
            return curve[lo]
        alpha = pos - lo
        return (1.0 - alpha) * curve[lo] + alpha * curve[hi]

    @staticmethod
    def fixed_axis_sd(point: np.ndarray, base_curve: np.ndarray) -> tuple[float, float]:
        axis = base_curve[-1] - base_curve[0]
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm <= 1e-12:
            return float("nan"), float("nan")
        axis_unit = axis / axis_norm
        delta = point - base_curve[0]
        s_percent = float(np.dot(delta, axis_unit) / axis_norm * 100.0)
        projected = np.dot(delta, axis_unit) * axis_unit
        d = float(np.linalg.norm(delta - projected))
        return s_percent, d

    def search_percents(self) -> np.ndarray:
        step = self.cfg.search_step_percent
        if step <= 0.0:
            raise ValueError("--search_step_percent must be positive")
        values = list(np.arange(0.0, 100.0 + step * 0.5, step, dtype=np.float64))
        if values[-1] < 100.0:
            values.append(100.0)
        clipped = sorted({round(float(v), 8) for v in values if 0.0 <= v <= 100.0})
        if clipped[-1] != 100.0:
            clipped.append(100.0)
        return np.asarray(clipped, dtype=np.float64)

    def prototype_path(self, method: str, class_name: str) -> Path:
        return (
            self.cfg.projection_output_root
            / method
            / "prototypes"
            / f"prototype_{class_name}_{method}.npy"
        )

    def load_prototypes(self, method: str) -> dict[str, np.ndarray]:
        return {
            class_name: np.load(self.prototype_path(method, class_name), allow_pickle=False).astype(np.float64)
            for class_name in CLASS_NAMES
        }

    def normalized_sequence_path(self, class_name: str, sequence_id: str) -> Path:
        return (
            self.cfg.analysis_input_root
            / "metrics"
            / "normalized"
            / class_name
            / str(sequence_id)
            / "normalized_sequence.npy"
        )

    def nearest6_csv_path(self, method: str) -> Path:
        return (
            self.cfg.projection_output_root
            / "linear_axis_extension"
            / "csv"
            / f"nearest6_to_prototype_sequences_{method}.csv"
        )

    def load_nearest6_curves(self, method: str) -> dict[str, list[tuple[str, str, np.ndarray]]]:
        selected: dict[str, list[tuple[str, str, np.ndarray]]] = {class_name: [] for class_name in CLASS_NAMES}
        with self.nearest6_csv_path(method).open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                class_name = row["class"]
                sequence_id = row["sequence_id"]
                rank = row["rank"]
                arr = np.load(
                    self.normalized_sequence_path(class_name, sequence_id),
                    allow_pickle=False,
                ).astype(np.float64)
                selected[class_name].append((rank, sequence_id, arr))
        return selected

    def compute_rows_for_curve(
        self,
        method: str,
        baseline_class: str,
        target_class: str,
        source_type: str,
        sequence_id: str,
        rank: str,
        baseline_curve: np.ndarray,
        target_curve: np.ndarray,
    ) -> list[dict]:
        search_percents = self.search_percents()
        baseline_samples = np.vstack([self.point_at_percent(baseline_curve, p) for p in search_percents])
        rows: list[dict] = []

        for stage in self.cfg.target_stages:
            target_point = self.point_at_percent(target_curve, float(stage))
            diff = baseline_samples - target_point
            squared_dist = np.einsum("ij,ij->i", diff, diff)
            nearest_idx = int(np.argmin(squared_dist))
            nearest_point = baseline_samples[nearest_idx]
            nearest_distance = float(np.sqrt(squared_dist[nearest_idx]))
            target_s, target_d = self.fixed_axis_sd(target_point, baseline_curve)
            nearest_s, nearest_d = self.fixed_axis_sd(nearest_point, baseline_curve)

            rows.append(
                {
                    "method": method,
                    "baseline_class": baseline_class,
                    "target_class": target_class,
                    "source_type": source_type,
                    "sequence_id": sequence_id,
                    "rank": rank,
                    "target_stage_percent": stage,
                    "nearest_baseline_progress_percent": float(search_percents[nearest_idx]),
                    "nearest_distance": nearest_distance,
                    "nearest_sample_index": nearest_idx,
                    "target_fixed_axis_s_percent": target_s,
                    "target_fixed_axis_d": target_d,
                    "nearest_fixed_axis_s_percent": nearest_s,
                    "nearest_fixed_axis_d": nearest_d,
                }
            )

        return rows

    def compute_method_rows(self, method: str) -> tuple[list[dict], list[dict]]:
        prototypes = self.load_prototypes(method)
        nearest_curves = self.load_nearest6_curves(method)
        prototype_rows: list[dict] = []
        nearest6_rows: list[dict] = []

        for baseline_class in self.cfg.baseline_classes:
            baseline_curve = prototypes[baseline_class]

            for target_class in CLASS_NAMES:
                prototype_rows.extend(
                    self.compute_rows_for_curve(
                        method=method,
                        baseline_class=baseline_class,
                        target_class=target_class,
                        source_type="prototype",
                        sequence_id="prototype",
                        rank="",
                        baseline_curve=baseline_curve,
                        target_curve=prototypes[target_class],
                    )
                )

            for target_class in CLASS_NAMES:
                for rank, sequence_id, target_curve in nearest_curves[target_class]:
                    nearest6_rows.extend(
                        self.compute_rows_for_curve(
                            method=method,
                            baseline_class=baseline_class,
                            target_class=target_class,
                            source_type="nearest6",
                            sequence_id=sequence_id,
                            rank=rank,
                            baseline_curve=baseline_curve,
                            target_curve=target_curve,
                        )
                    )

        return prototype_rows, nearest6_rows

    @staticmethod
    def rows_for(rows: list[dict], method: str, baseline_class: str) -> list[dict]:
        return [r for r in rows if r["method"] == method and r["baseline_class"] == baseline_class]

    @staticmethod
    def class_rows(rows: list[dict], class_name: str) -> list[dict]:
        return [r for r in rows if r["target_class"] == class_name]

    @staticmethod
    def grouped_by_sequence(rows: list[dict]) -> dict[str, list[dict]]:
        grouped: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            grouped[str(row["sequence_id"])].append(row)
        return grouped

    @staticmethod
    def sorted_by_stage(rows: list[dict]) -> list[dict]:
        return sorted(rows, key=lambda r: float(r["target_stage_percent"]))

    def plot_demo(self) -> Path:
        t = np.linspace(0.0, 1.0, 101)
        baseline = np.column_stack([t, 0.22 * np.sin(np.pi * t)])
        target = np.column_stack([t, 0.34 * np.sin(np.pi * (t + 0.08)) + 0.10])

        fig, ax = plt.subplots(figsize=(7.0, 4.4))
        ax.plot(baseline[:, 0], baseline[:, 1], color="#2ca02c", linewidth=2.4, label="C_baseline")
        ax.plot(target[:, 0], target[:, 1], color="#1f77b4", linewidth=2.4, label="C_2")

        for percent in (25, 50, 75):
            target_point = self.point_at_percent(target, percent)
            diff = baseline - target_point
            nearest_idx = int(np.argmin(np.einsum("ij,ij->i", diff, diff)))
            nearest_point = baseline[nearest_idx]
            ax.plot([target_point[0], nearest_point[0]], [target_point[1], nearest_point[1]], color="#444444", linestyle="--")
            ax.scatter([target_point[0]], [target_point[1]], color="#1f77b4", s=36, zorder=4)
            ax.scatter([nearest_point[0]], [nearest_point[1]], color="#2ca02c", s=36, zorder=4)

        ax.set_title("Nearest-baseline coordinate definition")
        ax.set_xlabel("illustrative 2D feature coordinate 1")
        ax.set_ylabel("illustrative 2D feature coordinate 2")
        ax.legend(frameon=False, loc="best")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        path = self.plot_dir / "demo_nearest_baseline_definition.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        return path

    def plot_prototype_curve(self, rows: list[dict], method: str, baseline_class: str) -> Path:
        fig, ax = plt.subplots(figsize=(7.2, 5.2))
        for target_class in CLASS_NAMES:
            selected = self.sorted_by_stage(self.class_rows(rows, target_class))
            if not selected:
                continue
            x = [r["nearest_baseline_progress_percent"] for r in selected]
            y = [r["nearest_distance"] for r in selected]
            ax.plot(
                x,
                y,
                marker="o",
                markersize=3.4,
                linewidth=2.0,
                color=COLORS.get(target_class),
                label=target_class,
            )
            for stage in (5, 50, 100):
                stage_row = next((r for r in selected if int(r["target_stage_percent"]) == stage), None)
                if stage_row is not None:
                    ax.annotate(
                        f"{stage}%",
                        (stage_row["nearest_baseline_progress_percent"], stage_row["nearest_distance"]),
                        textcoords="offset points",
                        xytext=(4, 4),
                        fontsize=7,
                    )
        ax.set_title(f"Prototype new curve | baseline={baseline_class} | {method}")
        ax.set_xlabel("nearest baseline progress (%)")
        ax.set_ylabel("nearest vector length (L2)")
        ax.set_xlim(-2, 102)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        path = self.plot_dir / f"prototype_new_curve_baseline_{baseline_class}_{method}.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        return path

    def plot_prototype_stage(self, rows: list[dict], method: str, baseline_class: str, value_key: str, ylabel: str, suffix: str) -> Path:
        fig, ax = plt.subplots(figsize=(7.2, 5.0))
        for target_class in CLASS_NAMES:
            selected = self.sorted_by_stage(self.class_rows(rows, target_class))
            if not selected:
                continue
            x = [r["target_stage_percent"] for r in selected]
            y = [r[value_key] for r in selected]
            ax.plot(x, y, marker="o", markersize=3.2, linewidth=2.0, color=COLORS.get(target_class), label=target_class)
        if value_key == "nearest_baseline_progress_percent":
            ax.plot([0, 100], [0, 100], color="#777777", linestyle=":", linewidth=1.5, label="stage=nearest progress")
            ax.set_xlim(-2, 102)
            ax.set_ylim(-2, 102)
        else:
            ax.set_xlim(-2, 102)
            ax.set_ylim(bottom=0)
        ax.set_title(f"Prototype {suffix} | baseline={baseline_class} | {method}")
        ax.set_xlabel("target stage on C_2 (%)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        path = self.plot_dir / f"prototype_{suffix}_baseline_{baseline_class}_{method}.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        return path

    def plot_nearest6_curve(self, rows: list[dict], proto_rows: list[dict], method: str, baseline_class: str) -> Path:
        fig, ax = plt.subplots(figsize=(7.6, 5.4))
        for target_class in CLASS_NAMES:
            class_selected = self.class_rows(rows, target_class)
            sequence_groups = self.grouped_by_sequence(class_selected)
            label_used = False
            for sequence_id, seq_rows in sorted(sequence_groups.items(), key=lambda item: int(item[0]) if item[0].isdigit() else item[0]):
                selected = self.sorted_by_stage(seq_rows)
                x = [r["nearest_baseline_progress_percent"] for r in selected]
                y = [r["nearest_distance"] for r in selected]
                ax.plot(
                    x,
                    y,
                    marker="o",
                    markersize=2.3,
                    linewidth=1.0,
                    alpha=0.35,
                    color=COLORS.get(target_class),
                    label=f"{target_class} nearest6" if not label_used else None,
                )
                label_used = True

            proto_selected = self.sorted_by_stage(self.class_rows(proto_rows, target_class))
            if proto_selected:
                ax.plot(
                    [r["nearest_baseline_progress_percent"] for r in proto_selected],
                    [r["nearest_distance"] for r in proto_selected],
                    color=COLORS.get(target_class),
                    linewidth=2.4,
                    linestyle="--",
                    label=f"{target_class} prototype",
                )

        ax.set_title(f"Nearest6 new curves | baseline={baseline_class} | {method}")
        ax.set_xlabel("nearest baseline progress (%)")
        ax.set_ylabel("nearest vector length (L2)")
        ax.set_xlim(-2, 102)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8, ncols=2)
        fig.tight_layout()
        path = self.plot_dir / f"nearest6_new_curve_baseline_{baseline_class}_{method}.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        return path

    def plot_nearest6_band(
        self,
        rows: list[dict],
        proto_rows: list[dict],
        method: str,
        baseline_class: str,
        value_key: str,
        ylabel: str,
        suffix: str,
    ) -> Path:
        fig, ax = plt.subplots(figsize=(7.6, 5.2))
        stages = np.asarray(self.cfg.target_stages, dtype=np.float64)

        for target_class in CLASS_NAMES:
            means: list[float] = []
            q1s: list[float] = []
            q3s: list[float] = []
            for stage in self.cfg.target_stages:
                values = [
                    float(r[value_key])
                    for r in rows
                    if r["target_class"] == target_class and int(r["target_stage_percent"]) == stage
                ]
                arr = np.asarray(values, dtype=np.float64)
                means.append(float(np.nanmean(arr)) if arr.size else np.nan)
                q1s.append(float(np.nanquantile(arr, 0.25)) if arr.size else np.nan)
                q3s.append(float(np.nanquantile(arr, 0.75)) if arr.size else np.nan)

            color = COLORS.get(target_class)
            ax.plot(stages, means, color=color, linewidth=2.1, label=f"{target_class} nearest6 mean")
            ax.fill_between(stages, q1s, q3s, color=color, alpha=0.15)

            proto_selected = self.sorted_by_stage(self.class_rows(proto_rows, target_class))
            if proto_selected:
                ax.plot(
                    [r["target_stage_percent"] for r in proto_selected],
                    [r[value_key] for r in proto_selected],
                    color=color,
                    linestyle="--",
                    linewidth=1.5,
                    alpha=0.85,
                    label=f"{target_class} prototype",
                )

        if value_key == "nearest_baseline_progress_percent":
            ax.plot([0, 100], [0, 100], color="#777777", linestyle=":", linewidth=1.5, label="stage=nearest progress")
            ax.set_ylim(-2, 102)
        else:
            ax.set_ylim(bottom=0)
        ax.set_xlim(-2, 102)
        ax.set_title(f"Nearest6 {suffix} band | baseline={baseline_class} | {method}")
        ax.set_xlabel("target stage on C_2 (%)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8, ncols=2)
        fig.tight_layout()
        path = self.plot_dir / f"nearest6_{suffix}_band_baseline_{baseline_class}_{method}.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        return path

    def make_plots(self, prototype_rows: list[dict], nearest6_rows: list[dict]) -> list[Path]:
        plot_paths = [self.plot_demo()]
        for method in METHODS:
            for baseline_class in self.cfg.baseline_classes:
                proto_selected = self.rows_for(prototype_rows, method, baseline_class)
                nearest_selected = self.rows_for(nearest6_rows, method, baseline_class)
                plot_paths.extend(
                    [
                        self.plot_prototype_curve(proto_selected, method, baseline_class),
                        self.plot_prototype_stage(
                            proto_selected,
                            method,
                            baseline_class,
                            "nearest_baseline_progress_percent",
                            "nearest baseline progress (%)",
                            "nearest_progress",
                        ),
                        self.plot_prototype_stage(
                            proto_selected,
                            method,
                            baseline_class,
                            "nearest_distance",
                            "nearest vector length (L2)",
                            "nearest_distance",
                        ),
                        self.plot_nearest6_curve(nearest_selected, proto_selected, method, baseline_class),
                        self.plot_nearest6_band(
                            nearest_selected,
                            proto_selected,
                            method,
                            baseline_class,
                            "nearest_baseline_progress_percent",
                            "nearest baseline progress (%)",
                            "nearest_progress",
                        ),
                        self.plot_nearest6_band(
                            nearest_selected,
                            proto_selected,
                            method,
                            baseline_class,
                            "nearest_distance",
                            "nearest vector length (L2)",
                            "nearest_distance",
                        ),
                    ]
                )
        return plot_paths

    @staticmethod
    def fmt(value: object, digits: int = 3) -> str:
        if isinstance(value, (float, np.floating)):
            if np.isnan(value):
                return "nan"
            return f"{float(value):.{digits}f}"
        return str(value)

    def summary_rows(self, prototype_rows: list[dict], nearest6_rows: list[dict]) -> list[dict]:
        rows: list[dict] = []
        for method in METHODS:
            for baseline_class in self.cfg.baseline_classes:
                for target_class in CLASS_NAMES:
                    proto = [
                        r
                        for r in prototype_rows
                        if r["method"] == method
                        and r["baseline_class"] == baseline_class
                        and r["target_class"] == target_class
                        and int(r["target_stage_percent"]) == 100
                    ]
                    if proto:
                        r = proto[0]
                        rows.append(
                            {
                                "method": method,
                                "baseline": baseline_class,
                                "target": target_class,
                                "source": "prototype",
                                "progress_100": r["nearest_baseline_progress_percent"],
                                "distance_100": r["nearest_distance"],
                            }
                        )

                    near = [
                        r
                        for r in nearest6_rows
                        if r["method"] == method
                        and r["baseline_class"] == baseline_class
                        and r["target_class"] == target_class
                        and int(r["target_stage_percent"]) == 100
                    ]
                    if near:
                        progress = np.asarray([r["nearest_baseline_progress_percent"] for r in near], dtype=np.float64)
                        distance = np.asarray([r["nearest_distance"] for r in near], dtype=np.float64)
                        rows.append(
                            {
                                "method": method,
                                "baseline": baseline_class,
                                "target": target_class,
                                "source": "nearest6 mean",
                                "progress_100": float(np.mean(progress)),
                                "distance_100": float(np.mean(distance)),
                            }
                        )
        return rows

    def relative_img(self, path: Path) -> str:
        return os.path.relpath(path, self.report_dir).replace("\\", "/")

    def write_html_report(self, prototype_rows: list[dict], nearest6_rows: list[dict], plot_paths: list[Path]) -> Path:
        path_by_name = {p.name: p for p in plot_paths}
        summary = self.summary_rows(prototype_rows, nearest6_rows)
        html_lines = [
            "<!doctype html>",
            "<html>",
            "<head>",
            '<meta charset="utf-8">',
            "<title>Nearest-baseline curve analysis</title>",
            "<style>",
            "body{font-family:Arial,sans-serif;line-height:1.55;margin:28px;color:#222;max-width:1180px}",
            "h1{font-size:28px;margin-bottom:6px} h2{font-size:22px;margin-top:28px}",
            "h3{font-size:17px;margin-top:22px}",
            ".formula{background:#f6f6f6;border-left:4px solid #777;padding:12px 14px;margin:12px 0;font-family:Consolas,monospace}",
            ".grid{display:grid;grid-template-columns:1fr 1fr;gap:18px;align-items:start}",
            "figure{margin:0 0 18px 0;border:1px solid #ddd;padding:10px;background:#fff}",
            "figure.wide{grid-column:1/-1}",
            "img{max-width:100%;height:auto;display:block}",
            "figcaption{font-size:13px;color:#555;margin-top:8px}",
            "table{border-collapse:collapse;font-size:13px;margin:12px 0 24px 0}",
            "th,td{border:1px solid #ddd;padding:6px 8px;text-align:right}",
            "th:first-child,td:first-child,th:nth-child(2),td:nth-child(2),th:nth-child(3),td:nth-child(3),th:nth-child(4),td:nth-child(4){text-align:left}",
            "th{background:#f4f4f4}",
            "</style>",
            "</head>",
            "<body>",
            "<h1>Nearest-baseline curve analysis</h1>",
            "<p>This report uses the linear-normalized 20-point fc7 trajectories. Baseline curves are methodA/methodB prototypes for truesmile and polite; target curves are class prototypes and the previous nearest-6 real sequences.</p>",
            "<h2>Coordinate Definition</h2>",
            "<p>For a target stage t on C<sub>2</sub>, the closest point is searched on the whole baseline curve C<sub>baseline</sub>. The new plotted coordinate is not the old fixed-axis s-d coordinate. Its x value is the nearest baseline progress, and its y value is the length of the vector linking C<sub>2</sub>(t) to that nearest baseline point.</p>",
            "<div class=\"formula\">",
            "C_baseline(tau), C_2(t), t in {5%, 10%, ..., 100%}<br>",
            "tau*(t) = argmin_tau || C_2(t) - C_baseline(tau) ||_2<br>",
            "x_new(t) = 100 * tau*(t)<br>",
            "y_new(t) = || C_2(t) - C_baseline(tau*(t)) ||_2",
            "</div>",
            "<p>Interpretation: x_new tells which stage of the baseline curve is most similar to the current target stage. y_new tells how far the target point still is from that nearest baseline stage.</p>",
        ]

        demo = path_by_name.get("demo_nearest_baseline_definition.png")
        if demo is not None:
            html_lines.extend(
                [
                    '<figure class="wide">',
                    f'<img src="{html.escape(self.relative_img(demo))}" alt="Nearest-baseline coordinate definition">',
                    "<figcaption>Definition demo: each dashed segment is the nearest vector from a target stage to the baseline curve.</figcaption>",
                    "</figure>",
                ]
            )

        html_lines.extend(
            [
                "<h2>Endpoint Summary</h2>",
                "<table>",
                "<tr><th>method</th><th>baseline</th><th>target</th><th>source</th><th>nearest progress at 100%</th><th>distance at 100%</th></tr>",
            ]
        )
        for row in summary:
            html_lines.append(
                "<tr>"
                f"<td>{html.escape(str(row['method']))}</td>"
                f"<td>{html.escape(str(row['baseline']))}</td>"
                f"<td>{html.escape(str(row['target']))}</td>"
                f"<td>{html.escape(str(row['source']))}</td>"
                f"<td>{self.fmt(row['progress_100'])}</td>"
                f"<td>{self.fmt(row['distance_100'])}</td>"
                "</tr>"
            )
        html_lines.extend(["</table>", "<h2>Figures</h2>"])

        for method in METHODS:
            for baseline_class in self.cfg.baseline_classes:
                html_lines.append(f"<h3>{html.escape(method)} / baseline={html.escape(baseline_class)}</h3>")
                html_lines.append('<div class="grid">')
                figure_specs = [
                    (
                        f"prototype_new_curve_baseline_{baseline_class}_{method}.png",
                        "Prototype new curve: connect each target stage in stage order after remapping it to nearest baseline progress and nearest-vector length.",
                        "wide",
                    ),
                    (
                        f"prototype_nearest_progress_baseline_{baseline_class}_{method}.png",
                        "Prototype progress map: x is target stage; y is nearest baseline progress. The diagonal means stage-to-stage matching.",
                        "",
                    ),
                    (
                        f"prototype_nearest_distance_baseline_{baseline_class}_{method}.png",
                        "Prototype distance map: x is target stage; y is nearest-vector length.",
                        "",
                    ),
                    (
                        f"nearest6_new_curve_baseline_{baseline_class}_{method}.png",
                        "Nearest-6 new curves: thin lines are the six closest real sequences per class; dashed lines are prototypes.",
                        "wide",
                    ),
                    (
                        f"nearest6_nearest_progress_band_baseline_{baseline_class}_{method}.png",
                        "Nearest-6 progress band: line is mean across nearest-6 sequences; shaded area is the interquartile range.",
                        "",
                    ),
                    (
                        f"nearest6_nearest_distance_band_baseline_{baseline_class}_{method}.png",
                        "Nearest-6 distance band: line is mean nearest-vector length; shaded area is the interquartile range.",
                        "",
                    ),
                ]
                for filename, caption, klass in figure_specs:
                    plot_path = path_by_name.get(filename)
                    if plot_path is None:
                        continue
                    class_attr = f' class="{klass}"' if klass else ""
                    html_lines.extend(
                        [
                            f"<figure{class_attr}>",
                            f'<img src="{html.escape(self.relative_img(plot_path))}" alt="{html.escape(filename)}">',
                            f"<figcaption>{html.escape(caption)}</figcaption>",
                            "</figure>",
                        ]
                    )
                html_lines.append("</div>")

        html_lines.extend(["</body>", "</html>"])
        report_path = self.report_dir / "nearest_baseline_curve_report.html"
        report_path.write_text("\n".join(html_lines), encoding="utf-8")
        return report_path

    def run(self) -> dict[str, Path | int]:
        prototype_rows: list[dict] = []
        nearest6_rows: list[dict] = []
        for method in METHODS:
            method_prototype_rows, method_nearest6_rows = self.compute_method_rows(method)
            prototype_rows.extend(method_prototype_rows)
            nearest6_rows.extend(method_nearest6_rows)

        prototype_csv = self.csv_dir / "prototype_nearest_baseline_curve_all.csv"
        nearest6_csv = self.csv_dir / "nearest6_nearest_baseline_curve_all.csv"
        self.write_csv(prototype_csv, prototype_rows, CSV_FIELDS)
        self.write_csv(nearest6_csv, nearest6_rows, CSV_FIELDS)

        for method in METHODS:
            self.write_csv(
                self.csv_dir / f"prototype_nearest_baseline_curve_{method}.csv",
                [r for r in prototype_rows if r["method"] == method],
                CSV_FIELDS,
            )
            self.write_csv(
                self.csv_dir / f"nearest6_nearest_baseline_curve_{method}.csv",
                [r for r in nearest6_rows if r["method"] == method],
                CSV_FIELDS,
            )

        summary_csv = self.csv_dir / "endpoint_100_summary.csv"
        summary_fields = ["method", "baseline", "target", "source", "progress_100", "distance_100"]
        self.write_csv(summary_csv, self.summary_rows(prototype_rows, nearest6_rows), summary_fields)

        plot_paths = self.make_plots(prototype_rows, nearest6_rows)
        report_path = self.write_html_report(prototype_rows, nearest6_rows, plot_paths)

        return {
            "prototype_rows": len(prototype_rows),
            "nearest6_rows": len(nearest6_rows),
            "plots": len(plot_paths),
            "prototype_csv": prototype_csv,
            "nearest6_csv": nearest6_csv,
            "summary_csv": summary_csv,
            "report": report_path,
        }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute nearest-baseline progress-distance curves.")
    parser.add_argument("--analysis_input_root", default=str(NearestBaselineConfig.analysis_input_root))
    parser.add_argument("--projection_output_root", default=str(NearestBaselineConfig.projection_output_root))
    parser.add_argument("--output_root", default=str(NearestBaselineConfig.output_root))
    parser.add_argument("--target_stages", default=",".join(str(x) for x in NearestBaselineConfig.target_stages))
    parser.add_argument("--search_step_percent", type=float, default=NearestBaselineConfig.search_step_percent)
    parser.add_argument("--baseline_classes", default=",".join(NearestBaselineConfig.baseline_classes))
    return parser


def main() -> None:
    cfg = NearestBaselineConfig.from_args(build_arg_parser().parse_args())
    result = NearestBaselineCurveAnalysis(cfg).run()
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
