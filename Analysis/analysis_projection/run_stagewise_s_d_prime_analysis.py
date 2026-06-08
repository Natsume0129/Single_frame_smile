from __future__ import annotations

import argparse
import csv
import html
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import CLASS_NAMES


COLORS = {"polite": "#1f77b4", "truesmile": "#2ca02c", "ambiguous": "#ff7f0e"}
METHODS = ("methodA", "methodB")
AXIS_CLASSES = ("truesmile", "polite")


@dataclass
class StagewiseConfig:
    analysis_input_root: Path = Path(r"E:\Matsuda_data\2-27meeting")
    projection_output_root: Path = Path(r"E:\Matsuda_data\3-10meeting")
    output_root: Path = Path(r"E:\Matsuda_data\3-10meeting\stagewise_s_d_prime")
    stages: tuple[int, ...] = tuple(range(10, 101, 10))

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "StagewiseConfig":
        stages = tuple(int(x.strip()) for x in args.stages.split(",") if x.strip())
        return cls(
            analysis_input_root=Path(args.analysis_input_root),
            projection_output_root=Path(args.projection_output_root),
            output_root=Path(args.output_root),
            stages=stages,
        )


class StagewiseSDPrimeAnalysis:
    def __init__(self, cfg: StagewiseConfig):
        self.cfg = cfg
        for sub in ("csv", "plots", "report"):
            (self.cfg.output_root / sub).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def write_csv(path: Path, rows: Iterable[dict], fieldnames: list[str]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    @staticmethod
    def point_at_fraction(curve: np.ndarray, fraction: float) -> np.ndarray:
        if fraction < 0.0 or fraction > 1.0:
            raise ValueError(f"fraction must be in [0, 1], got {fraction}")
        pos = fraction * (curve.shape[0] - 1)
        lo = int(np.floor(pos))
        hi = int(np.ceil(pos))
        if lo == hi:
            return curve[lo]
        alpha = pos - lo
        return (1.0 - alpha) * curve[lo] + alpha * curve[hi]

    def prototype_path(self, method: str, class_name: str) -> Path:
        return (
            self.cfg.projection_output_root
            / method
            / "prototypes"
            / f"prototype_{class_name}_{method}.npy"
        )

    def load_prototypes(self, method: str) -> dict[str, np.ndarray]:
        return {
            class_name: np.load(self.prototype_path(method, class_name), allow_pickle=False).astype(np.float32)
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

    def load_nearest6_curves(self, method: str) -> dict[str, list[tuple[str, np.ndarray]]]:
        path = self.nearest6_csv_path(method)
        selected: dict[str, list[tuple[str, np.ndarray]]] = {class_name: [] for class_name in CLASS_NAMES}
        with path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                class_name = row["class"]
                seq_id = row["sequence_id"]
                arr = np.load(self.normalized_sequence_path(class_name, seq_id), allow_pickle=False).astype(np.float32)
                selected[class_name].append((seq_id, arr))
        return selected

    def compute_rows_for_axis(
        self,
        method: str,
        axis_class: str,
        protos: dict[str, np.ndarray],
    ) -> list[dict]:
        base_curve = protos[axis_class]
        base_start = base_curve[0]
        rows: list[dict] = []

        for target_class in CLASS_NAMES:
            target_curve = protos[target_class]
            target_start = target_curve[0]
            for stage in self.cfg.stages:
                fraction = stage / 100.0
                v_base = self.point_at_fraction(base_curve, fraction) - base_start
                v_target = self.point_at_fraction(target_curve, fraction) - target_start
                base_norm = float(np.linalg.norm(v_base))
                target_norm = float(np.linalg.norm(v_target))

                if base_norm <= 1e-12:
                    s_prime = float("nan")
                    d_prime = float("nan")
                    angle_deg = float("nan")
                    total_difference = float("nan")
                    residual_norm = float("nan")
                    target_norm_ratio = float("nan")
                else:
                    s_prime = float(np.dot(v_base, v_target) / (base_norm * base_norm))
                    projected = s_prime * v_base
                    residual = v_target - projected
                    residual_norm = float(np.linalg.norm(residual))
                    d_prime = residual_norm / base_norm
                    total_difference = float(np.linalg.norm(v_target - v_base) / base_norm)
                    target_norm_ratio = target_norm / base_norm
                    if target_norm <= 1e-12:
                        angle_deg = float("nan")
                    else:
                        cosine = float(np.dot(v_base, v_target) / (base_norm * target_norm))
                        cosine = max(-1.0, min(1.0, cosine))
                        angle_deg = float(np.degrees(np.arccos(cosine)))

                rows.append(
                    {
                        "method": method,
                        "axis_class": axis_class,
                        "target_class": target_class,
                        "stage_percent": stage,
                        "s_prime": s_prime,
                        "d_prime": d_prime,
                        "base_norm": base_norm,
                        "target_norm": target_norm,
                        "target_norm_ratio": target_norm_ratio,
                        "residual_norm": residual_norm,
                        "total_difference_ratio": total_difference,
                        "angle_deg": angle_deg,
                    }
                )

        return rows

    def compute_nearest_rows_for_axis(
        self,
        method: str,
        axis_class: str,
        protos: dict[str, np.ndarray],
        nearest_curves: dict[str, list[tuple[str, np.ndarray]]],
    ) -> list[dict]:
        base_curve = protos[axis_class]
        base_start = base_curve[0]
        rows: list[dict] = []

        for target_class in CLASS_NAMES:
            for seq_id, target_curve in nearest_curves[target_class]:
                target_start = target_curve[0]
                for stage in self.cfg.stages:
                    fraction = stage / 100.0
                    v_base = self.point_at_fraction(base_curve, fraction) - base_start
                    v_target = self.point_at_fraction(target_curve, fraction) - target_start
                    base_norm = float(np.linalg.norm(v_base))
                    target_norm = float(np.linalg.norm(v_target))

                    if base_norm <= 1e-12:
                        s_prime = float("nan")
                        d_prime = float("nan")
                        angle_deg = float("nan")
                        total_difference = float("nan")
                        residual_norm = float("nan")
                        target_norm_ratio = float("nan")
                    else:
                        s_prime = float(np.dot(v_base, v_target) / (base_norm * base_norm))
                        projected = s_prime * v_base
                        residual = v_target - projected
                        residual_norm = float(np.linalg.norm(residual))
                        d_prime = residual_norm / base_norm
                        total_difference = float(np.linalg.norm(v_target - v_base) / base_norm)
                        target_norm_ratio = target_norm / base_norm
                        if target_norm <= 1e-12:
                            angle_deg = float("nan")
                        else:
                            cosine = float(np.dot(v_base, v_target) / (base_norm * target_norm))
                            cosine = max(-1.0, min(1.0, cosine))
                            angle_deg = float(np.degrees(np.arccos(cosine)))

                    rows.append(
                        {
                            "method": method,
                            "axis_class": axis_class,
                            "target_class": target_class,
                            "sequence_id": seq_id,
                            "stage_percent": stage,
                            "s_prime": s_prime,
                            "d_prime": d_prime,
                            "base_norm": base_norm,
                            "target_norm": target_norm,
                            "target_norm_ratio": target_norm_ratio,
                            "residual_norm": residual_norm,
                            "total_difference_ratio": total_difference,
                            "angle_deg": angle_deg,
                        }
                    )

        return rows

    @staticmethod
    def rows_by_class(rows: list[dict]) -> dict[str, list[dict]]:
        grouped: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            grouped[row["target_class"]].append(row)
        for items in grouped.values():
            items.sort(key=lambda r: int(r["stage_percent"]))
        return grouped

    def plot_s_prime(self, rows: list[dict], method: str, axis_class: str) -> Path:
        grouped = self.rows_by_class(rows)
        fig, ax = plt.subplots(figsize=(8.2, 5.0), dpi=150)
        for class_name in CLASS_NAMES:
            items = grouped[class_name]
            ax.plot(
                [r["stage_percent"] for r in items],
                [r["s_prime"] for r in items],
                marker="o",
                linewidth=1.8,
                color=COLORS[class_name],
                label=class_name,
            )
        ax.axhline(1.0, color="#222222", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.axhline(0.0, color="#222222", linewidth=0.8, alpha=0.35)
        ax.set_title(f"stage-wise s' ({method}, base = {axis_class})")
        ax.set_xlabel("Stage percent")
        ax.set_ylabel("s' = dot(v_base, v_target) / ||v_base||^2")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        out = self.cfg.output_root / "plots" / f"stagewise_s_prime_axis_{axis_class}_{method}.png"
        fig.savefig(out)
        plt.close(fig)
        return out

    def plot_d_prime(self, rows: list[dict], method: str, axis_class: str) -> Path:
        grouped = self.rows_by_class(rows)
        fig, ax = plt.subplots(figsize=(8.2, 5.0), dpi=150)
        for class_name in CLASS_NAMES:
            items = grouped[class_name]
            ax.plot(
                [r["stage_percent"] for r in items],
                [r["d_prime"] for r in items],
                marker="o",
                linewidth=1.8,
                color=COLORS[class_name],
                label=class_name,
            )
        ax.axhline(0.0, color="#222222", linewidth=0.8, alpha=0.35)
        ax.set_title(f"stage-wise d' ({method}, base = {axis_class})")
        ax.set_xlabel("Stage percent")
        ax.set_ylabel("d' = ||v_target - s' v_base|| / ||v_base||")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        out = self.cfg.output_root / "plots" / f"stagewise_d_prime_axis_{axis_class}_{method}.png"
        fig.savefig(out)
        plt.close(fig)
        return out

    def plot_s_d_points(self, rows: list[dict], method: str, axis_class: str) -> Path:
        grouped = self.rows_by_class(rows)
        fig, ax = plt.subplots(figsize=(7.2, 6.0), dpi=150)
        for class_name in CLASS_NAMES:
            items = grouped[class_name]
            x = [r["s_prime"] for r in items]
            y = [r["d_prime"] for r in items]
            ax.scatter(x, y, s=36, color=COLORS[class_name], label=class_name)
            for r in items:
                if int(r["stage_percent"]) in {10, 50, 100}:
                    ax.annotate(
                        f"{r['stage_percent']}%",
                        (r["s_prime"], r["d_prime"]),
                        textcoords="offset points",
                        xytext=(4, 3),
                        fontsize=7,
                        color=COLORS[class_name],
                    )
        ax.scatter([1.0], [0.0], marker="x", s=60, color="#111111", label="base target")
        ax.axhline(0.0, color="#222222", linewidth=0.8, alpha=0.35)
        ax.axvline(1.0, color="#222222", linewidth=0.8, linestyle="--", alpha=0.35)
        ax.set_title(f"stage-wise s'-d' points ({method}, base = {axis_class})")
        ax.set_xlabel("s'")
        ax.set_ylabel("d'")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        out = self.cfg.output_root / "plots" / f"stagewise_s_d_points_axis_{axis_class}_{method}.png"
        fig.savefig(out)
        plt.close(fig)
        return out

    def plot_stage_segments(self, rows: list[dict], method: str, axis_class: str) -> Path:
        grouped_stage: dict[int, list[dict]] = defaultdict(list)
        for row in rows:
            grouped_stage[int(row["stage_percent"])].append(row)

        finite_s = [float(r["s_prime"]) for r in rows if np.isfinite(float(r["s_prime"]))]
        finite_d = [float(r["d_prime"]) for r in rows if np.isfinite(float(r["d_prime"]))]
        xmin = min(-0.1, min(finite_s, default=0.0) - 0.15)
        xmax = max(1.15, max(finite_s, default=1.0) + 0.15)
        ymax = max(0.35, max(finite_d, default=0.0) + 0.15)

        fig, axes = plt.subplots(2, 5, figsize=(13.5, 6.0), dpi=150, sharex=True, sharey=True)
        axes_list = list(axes.ravel())
        for ax, stage in zip(axes_list, self.cfg.stages):
            ax.plot([0.0, 1.0], [0.0, 0.0], color="#111111", linewidth=1.6, label="base")
            ax.scatter([1.0], [0.0], marker="x", color="#111111", s=28)
            for row in sorted(grouped_stage[stage], key=lambda r: CLASS_NAMES.index(r["target_class"])):
                class_name = row["target_class"]
                x = float(row["s_prime"])
                y = float(row["d_prime"])
                if not np.isfinite(x) or not np.isfinite(y):
                    continue
                ax.plot([0.0, x], [0.0, y], color=COLORS[class_name], linewidth=1.4, alpha=0.85)
                ax.scatter([x], [y], color=COLORS[class_name], s=20)
            ax.set_title(f"{stage}%")
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(-0.05, ymax)
            ax.grid(True, alpha=0.22)

        for ax in axes_list[5:]:
            ax.set_xlabel("s'")
        for ax in (axes_list[0], axes_list[5]):
            ax.set_ylabel("d'")

        handles = [
            plt.Line2D([0], [0], color="#111111", lw=1.6, label=f"{axis_class} base"),
            *[
                plt.Line2D([0], [0], color=COLORS[class_name], lw=1.6, label=class_name)
                for class_name in CLASS_NAMES
            ],
        ]
        fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9, bbox_to_anchor=(0.5, 0.01))
        fig.suptitle(f"10 discontinuous stage-wise normalized segments ({method}, base = {axis_class})", y=0.99)
        fig.tight_layout(rect=(0, 0.07, 1, 0.93))
        out = self.cfg.output_root / "plots" / f"stagewise_segments_axis_{axis_class}_{method}.png"
        fig.savefig(out)
        plt.close(fig)
        return out

    @staticmethod
    def grouped_nearest_by_class_seq(rows: list[dict]) -> dict[tuple[str, str], list[dict]]:
        grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for row in rows:
            grouped[(row["target_class"], row["sequence_id"])].append(row)
        for items in grouped.values():
            items.sort(key=lambda r: int(r["stage_percent"]))
        return grouped

    @staticmethod
    def stage_stats(rows: list[dict], metric: str) -> dict[str, dict[int, dict[str, float]]]:
        grouped: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
        for row in rows:
            grouped[row["target_class"]][int(row["stage_percent"])].append(float(row[metric]))

        stats: dict[str, dict[int, dict[str, float]]] = {}
        for class_name, by_stage in grouped.items():
            stats[class_name] = {}
            for stage, values in by_stage.items():
                arr = np.asarray(values, dtype=np.float64)
                stats[class_name][stage] = {
                    "mean": float(np.mean(arr)),
                    "q1": float(np.quantile(arr, 0.25)),
                    "q3": float(np.quantile(arr, 0.75)),
                }
        return stats

    def plot_nearest_metric_band(
        self,
        nearest_rows: list[dict],
        prototype_rows: list[dict],
        method: str,
        axis_class: str,
        metric: str,
        ylabel: str,
    ) -> Path:
        stats = self.stage_stats(nearest_rows, metric)
        proto_grouped = self.rows_by_class(prototype_rows)

        fig, ax = plt.subplots(figsize=(8.4, 5.1), dpi=150)
        stages = list(self.cfg.stages)
        for class_name in CLASS_NAMES:
            class_stats = stats[class_name]
            means = [class_stats[stage]["mean"] for stage in stages]
            q1 = [class_stats[stage]["q1"] for stage in stages]
            q3 = [class_stats[stage]["q3"] for stage in stages]
            ax.plot(stages, means, color=COLORS[class_name], linewidth=1.9, label=f"{class_name} nearest6 mean")
            ax.fill_between(stages, q1, q3, color=COLORS[class_name], alpha=0.14)

            proto_items = proto_grouped[class_name]
            ax.plot(
                [r["stage_percent"] for r in proto_items],
                [r[metric] for r in proto_items],
                color=COLORS[class_name],
                linewidth=1.1,
                linestyle="--",
                alpha=0.85,
                label=f"{class_name} prototype",
            )

        if metric == "s_prime":
            ax.axhline(1.0, color="#222222", linewidth=0.8, linestyle="--", alpha=0.45)
        ax.axhline(0.0, color="#222222", linewidth=0.8, alpha=0.32)
        ax.set_title(f"nearest6 {metric} band ({method}, base = {axis_class})")
        ax.set_xlabel("Stage percent")
        ax.set_ylabel(ylabel)
        ax.legend(loc="best", fontsize=7)
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        out = self.cfg.output_root / "plots" / f"nearest6_{metric}_band_axis_{axis_class}_{method}.png"
        fig.savefig(out)
        plt.close(fig)
        return out

    def plot_nearest_s_d_points(self, nearest_rows: list[dict], prototype_rows: list[dict], method: str, axis_class: str) -> Path:
        grouped_seq = self.grouped_nearest_by_class_seq(nearest_rows)
        proto_grouped = self.rows_by_class(prototype_rows)

        fig, ax = plt.subplots(figsize=(7.4, 6.1), dpi=150)
        labeled_classes: set[str] = set()
        for (class_name, _), rows in grouped_seq.items():
            label = f"{class_name} nearest6" if class_name not in labeled_classes else None
            ax.scatter(
                [r["s_prime"] for r in rows],
                [r["d_prime"] for r in rows],
                s=18,
                alpha=0.28,
                color=COLORS[class_name],
                label=label,
            )
            labeled_classes.add(class_name)

        for class_name in CLASS_NAMES:
            items = proto_grouped[class_name]
            ax.scatter(
                [r["s_prime"] for r in items],
                [r["d_prime"] for r in items],
                s=44,
                marker="x",
                color=COLORS[class_name],
                label=f"{class_name} prototype",
            )

        ax.scatter([1.0], [0.0], marker="x", s=66, color="#111111", label="base target")
        ax.axhline(0.0, color="#222222", linewidth=0.8, alpha=0.35)
        ax.axvline(1.0, color="#222222", linewidth=0.8, linestyle="--", alpha=0.35)
        ax.set_title(f"nearest6 stage-wise s'-d' points ({method}, base = {axis_class})")
        ax.set_xlabel("s'")
        ax.set_ylabel("d'")
        ax.legend(loc="best", fontsize=7)
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        out = self.cfg.output_root / "plots" / f"nearest6_s_d_points_axis_{axis_class}_{method}.png"
        fig.savefig(out)
        plt.close(fig)
        return out

    def plot_nearest_stage_segments(
        self,
        nearest_rows: list[dict],
        prototype_rows: list[dict],
        method: str,
        axis_class: str,
    ) -> Path:
        grouped_stage: dict[int, list[dict]] = defaultdict(list)
        for row in nearest_rows:
            grouped_stage[int(row["stage_percent"])].append(row)

        proto_stage: dict[int, list[dict]] = defaultdict(list)
        for row in prototype_rows:
            proto_stage[int(row["stage_percent"])].append(row)

        finite_s = [float(r["s_prime"]) for r in nearest_rows + prototype_rows if np.isfinite(float(r["s_prime"]))]
        finite_d = [float(r["d_prime"]) for r in nearest_rows + prototype_rows if np.isfinite(float(r["d_prime"]))]
        xmin = min(-0.2, min(finite_s, default=0.0) - 0.15)
        xmax = max(1.15, max(finite_s, default=1.0) + 0.15)
        ymax = max(0.35, max(finite_d, default=0.0) + 0.15)

        fig, axes = plt.subplots(2, 5, figsize=(13.5, 6.0), dpi=150, sharex=True, sharey=True)
        axes_list = list(axes.ravel())
        for ax, stage in zip(axes_list, self.cfg.stages):
            ax.plot([0.0, 1.0], [0.0, 0.0], color="#111111", linewidth=1.6)
            ax.scatter([1.0], [0.0], marker="x", color="#111111", s=28)

            for row in grouped_stage[stage]:
                class_name = row["target_class"]
                x = float(row["s_prime"])
                y = float(row["d_prime"])
                if not np.isfinite(x) or not np.isfinite(y):
                    continue
                ax.plot([0.0, x], [0.0, y], color=COLORS[class_name], linewidth=0.8, alpha=0.22)

            for row in proto_stage[stage]:
                class_name = row["target_class"]
                x = float(row["s_prime"])
                y = float(row["d_prime"])
                if not np.isfinite(x) or not np.isfinite(y):
                    continue
                ax.plot([0.0, x], [0.0, y], color=COLORS[class_name], linewidth=2.0, alpha=0.95)
                ax.scatter([x], [y], color=COLORS[class_name], s=22)

            ax.set_title(f"{stage}%")
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(-0.05, ymax)
            ax.grid(True, alpha=0.22)

        for ax in axes_list[5:]:
            ax.set_xlabel("s'")
        for ax in (axes_list[0], axes_list[5]):
            ax.set_ylabel("d'")

        handles = [
            plt.Line2D([0], [0], color="#111111", lw=1.6, label=f"{axis_class} base"),
            *[
                plt.Line2D([0], [0], color=COLORS[class_name], lw=1.8, label=f"{class_name} prototype + nearest6")
                for class_name in CLASS_NAMES
            ],
        ]
        fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, 0.01))
        fig.suptitle(f"nearest6 stage-wise normalized segments ({method}, base = {axis_class})", y=0.99)
        fig.tight_layout(rect=(0, 0.07, 1, 0.93))
        out = self.cfg.output_root / "plots" / f"nearest6_segments_axis_{axis_class}_{method}.png"
        fig.savefig(out)
        plt.close(fig)
        return out

    def plot_demo(self) -> Path:
        out = self.cfg.output_root / "plots" / "stagewise_s_d_prime_calculation_demo.png"
        fig, ax = plt.subplots(figsize=(6.8, 4.8), dpi=150)
        target = np.array([0.68, 0.42])
        projection = np.array([target[0], 0.0])

        ax.arrow(0, 0, 1.0, 0, length_includes_head=True, head_width=0.035, head_length=0.04, color="#111111", linewidth=2)
        ax.arrow(0, 0, target[0], target[1], length_includes_head=True, head_width=0.035, head_length=0.04, color="#2563eb", linewidth=2)
        ax.plot([target[0], target[0]], [0, target[1]], linestyle="--", color="#ef4444", linewidth=1.8)
        ax.scatter([projection[0], target[0]], [projection[1], target[1]], color=["#111111", "#2563eb"], s=36)

        ax.text(0.48, -0.08, "base vector v1 normalized to 1", ha="center", fontsize=10)
        ax.text(target[0] / 2 + 0.03, target[1] / 2 + 0.04, "target vector v2", color="#2563eb", fontsize=10)
        ax.text(target[0] / 2, -0.04, "s'", ha="center", color="#111111", fontsize=12)
        ax.text(target[0] + 0.03, target[1] / 2, "d'", va="center", color="#ef4444", fontsize=12)
        ax.text(1.0, 0.04, "(1, 0)", ha="center", fontsize=9)
        ax.text(target[0] + 0.04, target[1] + 0.02, "(s', d')", color="#2563eb", fontsize=10)

        ax.set_xlim(-0.08, 1.18)
        ax.set_ylim(-0.14, 0.68)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("s'")
        ax.set_ylabel("d'")
        ax.set_title("Stage-wise s'-d' decomposition")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(out)
        plt.close(fig)
        return out

    @staticmethod
    def relpath_for_html(path: Path, report_dir: Path) -> str:
        return Path("..", "plots", path.name).as_posix()

    def relpath_projection_output_for_html(self, path: Path, report_dir: Path) -> str:
        rel = path.relative_to(self.cfg.projection_output_root)
        return Path("..", "..", rel).as_posix()

    def fixed_axis_plot_paths(self, method: str, axis_class: str) -> dict[str, Path]:
        plot_root = self.cfg.projection_output_root / "linear_axis_extension" / "plots"
        return {
            "prototype": plot_root / f"s_d_axis_{axis_class}_{method}.png",
            "all_sequences": plot_root / f"s_d_all_sequences_axis_{axis_class}_{method}.png",
            "nearest6": plot_root / f"s_d_nearest6_axis_{axis_class}_{method}.png",
        }

    def build_summary_rows(self, all_rows: list[dict]) -> list[dict]:
        summaries = []
        grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
        for row in all_rows:
            grouped[(row["method"], row["axis_class"], row["target_class"])].append(row)

        for (method, axis_class, target_class), rows in sorted(grouped.items()):
            rows = sorted(rows, key=lambda r: int(r["stage_percent"]))
            if target_class == axis_class:
                continue
            last = rows[-1]
            max_d = max(rows, key=lambda r: float(r["d_prime"]))
            min_s = min(rows, key=lambda r: float(r["s_prime"]))
            summaries.append(
                {
                    "method": method,
                    "axis_class": axis_class,
                    "target_class": target_class,
                    "s_100": float(last["s_prime"]),
                    "d_100": float(last["d_prime"]),
                    "max_d": float(max_d["d_prime"]),
                    "max_d_stage": int(max_d["stage_percent"]),
                    "min_s": float(min_s["s_prime"]),
                    "min_s_stage": int(min_s["stage_percent"]),
                }
            )
        return summaries

    def build_nearest_summary_rows(self, all_nearest_rows: list[dict]) -> list[dict]:
        summaries = []
        grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
        for row in all_nearest_rows:
            grouped[(row["method"], row["axis_class"], row["target_class"])].append(row)

        for (method, axis_class, target_class), rows in sorted(grouped.items()):
            if target_class == axis_class:
                continue
            by_stage: dict[int, list[dict]] = defaultdict(list)
            for row in rows:
                by_stage[int(row["stage_percent"])].append(row)

            final_rows = by_stage[max(by_stage)]
            s100 = np.asarray([float(r["s_prime"]) for r in final_rows], dtype=np.float64)
            d100 = np.asarray([float(r["d_prime"]) for r in final_rows], dtype=np.float64)

            mean_d_by_stage = {
                stage: float(np.mean([float(r["d_prime"]) for r in items]))
                for stage, items in by_stage.items()
            }
            max_d_stage = max(mean_d_by_stage, key=mean_d_by_stage.get)

            summaries.append(
                {
                    "method": method,
                    "axis_class": axis_class,
                    "target_class": target_class,
                    "s100_mean": float(np.mean(s100)),
                    "s100_q1": float(np.quantile(s100, 0.25)),
                    "s100_q3": float(np.quantile(s100, 0.75)),
                    "d100_mean": float(np.mean(d100)),
                    "d100_q1": float(np.quantile(d100, 0.25)),
                    "d100_q3": float(np.quantile(d100, 0.75)),
                    "max_mean_d": mean_d_by_stage[max_d_stage],
                    "max_mean_d_stage": max_d_stage,
                }
            )
        return summaries

    def write_html_report(
        self,
        all_rows: list[dict],
        all_nearest_rows: list[dict],
        plot_paths: dict[tuple[str, str], dict[str, Path]],
        nearest_plot_paths: dict[tuple[str, str], dict[str, Path]],
        demo_path: Path,
    ) -> Path:
        report_dir = self.cfg.output_root / "report"
        summaries = self.build_summary_rows(all_rows)
        nearest_summaries = self.build_nearest_summary_rows(all_nearest_rows)

        def fmt(value: float) -> str:
            return f"{value:.4f}"

        summary_table = "\n".join(
            "<tr>"
            f"<td>{html.escape(row['method'])}</td>"
            f"<td>{html.escape(row['axis_class'])}</td>"
            f"<td>{html.escape(row['target_class'])}</td>"
            f"<td>{fmt(row['s_100'])}</td>"
            f"<td>{fmt(row['d_100'])}</td>"
            f"<td>{fmt(row['max_d'])} at {row['max_d_stage']}%</td>"
            f"<td>{fmt(row['min_s'])} at {row['min_s_stage']}%</td>"
            "</tr>"
            for row in summaries
        )

        nearest_summary_table = "\n".join(
            "<tr>"
            f"<td>{html.escape(row['method'])}</td>"
            f"<td>{html.escape(row['axis_class'])}</td>"
            f"<td>{html.escape(row['target_class'])}</td>"
            f"<td>{fmt(row['s100_mean'])} [{fmt(row['s100_q1'])}, {fmt(row['s100_q3'])}]</td>"
            f"<td>{fmt(row['d100_mean'])} [{fmt(row['d100_q1'])}, {fmt(row['d100_q3'])}]</td>"
            f"<td>{fmt(row['max_mean_d'])} at {row['max_mean_d_stage']}%</td>"
            "</tr>"
            for row in nearest_summaries
        )

        sections = []
        for method in METHODS:
            for axis_class in AXIS_CLASSES:
                paths = plot_paths[(method, axis_class)]
                nearest_paths = nearest_plot_paths[(method, axis_class)]
                fixed_paths = self.fixed_axis_plot_paths(method, axis_class)
                sections.append(
                    f"""
                    <section>
                      <h2>{html.escape(method)} | base = {html.escape(axis_class)}</h2>
                      <h3>Original fixed-axis 0-100% s-d plots</h3>
                      <p>
                        These are the earlier s-d plots using one fixed base axis from 0% to 100%.
                        They should be read as continuous curves in one fixed coordinate system.
                        This is different from the stage-wise s′-d′ figures below, where each stage uses a new base vector.
                      </p>
                      <div class="figure-grid">
                        <figure>
                          <img src="{self.relpath_projection_output_for_html(fixed_paths['prototype'], report_dir)}" alt="original fixed-axis prototype s-d plot">
                          <figcaption>Original fixed-axis prototype s-d plot.</figcaption>
                        </figure>
                        <figure>
                          <img src="{self.relpath_projection_output_for_html(fixed_paths['all_sequences'], report_dir)}" alt="original fixed-axis all-sequence s-d plot">
                          <figcaption>Original fixed-axis all-sequence s-d plot.</figcaption>
                        </figure>
                        <figure>
                          <img src="{self.relpath_projection_output_for_html(fixed_paths['nearest6'], report_dir)}" alt="original fixed-axis nearest6 s-d plot">
                          <figcaption>Original fixed-axis nearest6 s-d plot.</figcaption>
                        </figure>
                      </div>
                      <h3>Stage-wise s′-d′ prototype figures</h3>
                      <p>
                        These figures use a new base vector at each stage. The segment plot should be read as
                        ten independent normalized decompositions, not as one continuous spatial trajectory.
                      </p>
                      <div class="figure-grid">
                        <figure class="wide">
                          <img src="{self.relpath_for_html(paths['segments'], report_dir)}" alt="stage-wise segments">
                          <figcaption>Ten discontinuous normalized segments. The black base is always (0,0) to (1,0).</figcaption>
                        </figure>
                        <figure>
                          <img src="{self.relpath_for_html(paths['points'], report_dir)}" alt="s prime d prime points">
                          <figcaption>Stage-wise s′-d′ endpoints. Points are not connected as one trajectory.</figcaption>
                        </figure>
                        <figure>
                          <img src="{self.relpath_for_html(paths['s_prime'], report_dir)}" alt="s prime by stage">
                          <figcaption>s′ over stage. This line shows trend over stages, not a fixed-coordinate path.</figcaption>
                        </figure>
                        <figure>
                          <img src="{self.relpath_for_html(paths['d_prime'], report_dir)}" alt="d prime by stage">
                          <figcaption>d′ over stage. Larger values mean stronger off-base-direction change.</figcaption>
                        </figure>
                      </div>
                      <h3>Nearest6 representative-neighborhood curves</h3>
                      <p>
                        The following figures add the six sequences nearest to each class prototype from the previous
                        linear-axis-extension analysis. Transparent lines/points show nearest6 samples; dashed or thicker
                        elements keep the prototype as the reference.
                      </p>
                      <div class="figure-grid">
                        <figure class="wide">
                          <img src="{self.relpath_for_html(nearest_paths['segments'], report_dir)}" alt="nearest6 stage-wise segments">
                          <figcaption>Nearest6 stage-wise normalized segments. This checks whether prototype behavior is supported by nearby samples.</figcaption>
                        </figure>
                        <figure>
                          <img src="{self.relpath_for_html(nearest_paths['points'], report_dir)}" alt="nearest6 s prime d prime points">
                          <figcaption>Nearest6 s′-d′ endpoints. Heavier overlap means weaker separation in this local neighborhood.</figcaption>
                        </figure>
                        <figure>
                          <img src="{self.relpath_for_html(nearest_paths['s_prime_band'], report_dir)}" alt="nearest6 s prime band">
                          <figcaption>Nearest6 mean and IQR for s′, with prototype shown as dashed reference.</figcaption>
                        </figure>
                        <figure>
                          <img src="{self.relpath_for_html(nearest_paths['d_prime_band'], report_dir)}" alt="nearest6 d prime band">
                          <figcaption>Nearest6 mean and IQR for d′, with prototype shown as dashed reference.</figcaption>
                        </figure>
                      </div>
                    </section>
                    """
                )

        out = report_dir / "stagewise_s_d_prime_report.html"
        html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Stage-wise s′-d′ Analysis Report</title>
  <style>
    body {{
      margin: 0;
      background: #f6f8fb;
      color: #1f2933;
      font-family: "Segoe UI", Arial, sans-serif;
      line-height: 1.6;
    }}
    main {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 36px 24px 56px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 30px;
    }}
    h2 {{
      margin-top: 34px;
      padding-left: 10px;
      border-left: 4px solid #2563eb;
      font-size: 21px;
    }}
    h3 {{
      margin-top: 22px;
      font-size: 17px;
    }}
    .meta {{
      color: #52606d;
      font-size: 14px;
    }}
    .panel {{
      background: #ffffff;
      border: 1px solid #d9e2ec;
      border-radius: 8px;
      padding: 16px 18px;
      margin: 18px 0;
    }}
    code, pre {{
      font-family: Consolas, "Liberation Mono", monospace;
    }}
    pre {{
      background: #eef2f7;
      border-radius: 8px;
      overflow-x: auto;
      padding: 14px 16px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin: 12px 0 20px;
      background: #ffffff;
    }}
    th, td {{
      border: 1px solid #d9e2ec;
      padding: 8px 10px;
      text-align: left;
      vertical-align: top;
    }}
    th {{
      background: #eef4ff;
    }}
    figure {{
      margin: 0;
      background: #ffffff;
      border: 1px solid #d9e2ec;
      border-radius: 8px;
      padding: 10px;
    }}
    figure img {{
      display: block;
      width: 100%;
      height: auto;
    }}
    figure.wide {{
      grid-column: 1 / -1;
    }}
    figcaption {{
      color: #52606d;
      font-size: 13px;
      margin-top: 8px;
    }}
    .figure-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
    }}
    @media (max-width: 820px) {{
      .figure-grid {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <h1>Stage-wise s′-d′ Analysis Report</h1>
      <p class="meta">Output root: {html.escape(str(self.cfg.output_root))}</p>
    </header>

    <section class="panel">
      <h2>Calculation Method</h2>
      <p>
        This analysis follows the interpretation that each intensity stage uses a different base vector.
        Therefore, the s′-d′ points are stage-wise decompositions, not one continuous trajectory in a fixed basis.
      </p>
      <p>For a base curve C1 and a target curve C2, at stage q:</p>
      <pre><code>v1(q) = C1(q) - C1(0)
v2(q) = C2(q) - C2(0)

s′(q) = dot(v1(q), v2(q)) / ||v1(q)||^2
d′(q) = ||v2(q) - s′(q) v1(q)|| / ||v1(q)||</code></pre>
      <p>
        Here <strong>s′</strong> is the normalized intensity along the stage-wise base transition,
        and <strong>d′</strong> is the normalized off-direction distance from that base transition.
        The 0% stage is excluded because ||v1|| = 0.
      </p>
      <figure>
        <img src="{self.relpath_for_html(demo_path, report_dir)}" alt="calculation demo">
        <figcaption>Formula demo. For each stage, the base vector is normalized to (1, 0), and the target transition becomes (s′, d′).</figcaption>
      </figure>
    </section>

    <section>
      <h2>Numerical Summary</h2>
      <h3>Prototype endpoints</h3>
      <table>
        <thead>
          <tr>
            <th>method</th>
            <th>base axis</th>
            <th>target</th>
            <th>s′ at 100%</th>
            <th>d′ at 100%</th>
            <th>max d′</th>
            <th>min s′</th>
          </tr>
        </thead>
        <tbody>
          {summary_table}
        </tbody>
      </table>
      <p>
        Reading rule: s′ close to 1 and d′ close to 0 means the target transition resembles the base transition at that stage.
        Low or negative s′ means little forward progress along the base transition. High d′ means the target changes in another direction.
      </p>
      <h3>Nearest6 endpoints</h3>
      <table>
        <thead>
          <tr>
            <th>method</th>
            <th>base axis</th>
            <th>target</th>
            <th>mean s′ at 100% [IQR]</th>
            <th>mean d′ at 100% [IQR]</th>
            <th>max mean d′</th>
          </tr>
        </thead>
        <tbody>
          {nearest_summary_table}
        </tbody>
      </table>
      <p>
        The nearest6 rows are not full-distribution statistics. They describe the representative neighborhood selected
        from the previous analysis, so they are useful for checking whether the prototype result is locally stable.
      </p>
    </section>

    {''.join(sections)}
  </main>
</body>
</html>
"""
        out.write_text(html_text, encoding="utf-8")
        return out

    def run(self) -> None:
        all_rows: list[dict] = []
        all_nearest_rows: list[dict] = []
        plot_paths: dict[tuple[str, str], dict[str, Path]] = {}
        nearest_plot_paths: dict[tuple[str, str], dict[str, Path]] = {}
        demo_path = self.plot_demo()

        for method in METHODS:
            protos = self.load_prototypes(method)
            nearest_curves = self.load_nearest6_curves(method)
            for axis_class in AXIS_CLASSES:
                rows = self.compute_rows_for_axis(method, axis_class, protos)
                nearest_rows = self.compute_nearest_rows_for_axis(method, axis_class, protos, nearest_curves)
                all_rows.extend(rows)
                all_nearest_rows.extend(nearest_rows)
                self.write_csv(
                    self.cfg.output_root / "csv" / f"stagewise_s_d_prime_axis_{axis_class}_{method}.csv",
                    rows,
                    [
                        "method",
                        "axis_class",
                        "target_class",
                        "stage_percent",
                        "s_prime",
                        "d_prime",
                        "base_norm",
                        "target_norm",
                        "target_norm_ratio",
                        "residual_norm",
                        "total_difference_ratio",
                        "angle_deg",
                    ],
                )
                self.write_csv(
                    self.cfg.output_root / "csv" / f"nearest6_stagewise_s_d_prime_axis_{axis_class}_{method}.csv",
                    nearest_rows,
                    [
                        "method",
                        "axis_class",
                        "target_class",
                        "sequence_id",
                        "stage_percent",
                        "s_prime",
                        "d_prime",
                        "base_norm",
                        "target_norm",
                        "target_norm_ratio",
                        "residual_norm",
                        "total_difference_ratio",
                        "angle_deg",
                    ],
                )
                plot_paths[(method, axis_class)] = {
                    "s_prime": self.plot_s_prime(rows, method, axis_class),
                    "d_prime": self.plot_d_prime(rows, method, axis_class),
                    "points": self.plot_s_d_points(rows, method, axis_class),
                    "segments": self.plot_stage_segments(rows, method, axis_class),
                }
                nearest_plot_paths[(method, axis_class)] = {
                    "s_prime_band": self.plot_nearest_metric_band(
                        nearest_rows,
                        rows,
                        method,
                        axis_class,
                        "s_prime",
                        "s'",
                    ),
                    "d_prime_band": self.plot_nearest_metric_band(
                        nearest_rows,
                        rows,
                        method,
                        axis_class,
                        "d_prime",
                        "d'",
                    ),
                    "points": self.plot_nearest_s_d_points(nearest_rows, rows, method, axis_class),
                    "segments": self.plot_nearest_stage_segments(nearest_rows, rows, method, axis_class),
                }

        self.write_csv(
            self.cfg.output_root / "csv" / "stagewise_s_d_prime_all.csv",
            all_rows,
            [
                "method",
                "axis_class",
                "target_class",
                "stage_percent",
                "s_prime",
                "d_prime",
                "base_norm",
                "target_norm",
                "target_norm_ratio",
                "residual_norm",
                "total_difference_ratio",
                "angle_deg",
            ],
        )
        self.write_csv(
            self.cfg.output_root / "csv" / "nearest6_stagewise_s_d_prime_all.csv",
            all_nearest_rows,
            [
                "method",
                "axis_class",
                "target_class",
                "sequence_id",
                "stage_percent",
                "s_prime",
                "d_prime",
                "base_norm",
                "target_norm",
                "target_norm_ratio",
                "residual_norm",
                "total_difference_ratio",
                "angle_deg",
            ],
        )
        report_path = self.write_html_report(all_rows, all_nearest_rows, plot_paths, nearest_plot_paths, demo_path)
        print(f"[STAGEWISE_S_D_PRIME] Finished.")
        print(f"Plots: {self.cfg.output_root / 'plots'}")
        print(f"CSV: {self.cfg.output_root / 'csv'}")
        print(f"HTML report: {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate stage-wise s'-d' decomposition plots for smile prototypes.")
    parser.add_argument("--analysis_input_root", default=r"E:\Matsuda_data\2-27meeting")
    parser.add_argument("--projection_output_root", default=r"E:\Matsuda_data\3-10meeting")
    parser.add_argument("--output_root", default=r"E:\Matsuda_data\3-10meeting\stagewise_s_d_prime")
    parser.add_argument("--stages", default="10,20,30,40,50,60,70,80,90,100")
    args = parser.parse_args()
    StagewiseSDPrimeAnalysis(StagewiseConfig.from_args(args)).run()


if __name__ == "__main__":
    main()
