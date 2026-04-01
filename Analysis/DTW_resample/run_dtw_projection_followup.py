from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


ANALYSIS_PROJECTION_COMMON = Path(__file__).resolve().parent.parent / "analysis_projection" / "common.py"
_proj_spec = importlib.util.spec_from_file_location("analysis_projection_common_reuse", ANALYSIS_PROJECTION_COMMON)
if _proj_spec is None or _proj_spec.loader is None:
    raise RuntimeError(f"Cannot load analysis_projection common module from {ANALYSIS_PROJECTION_COMMON}")
_proj_module = importlib.util.module_from_spec(_proj_spec)
sys.modules["analysis_projection_common_reuse"] = _proj_module
_proj_spec.loader.exec_module(_proj_module)
CLASS_NAMES = _proj_module.CLASS_NAMES
compute_axis_metrics = _proj_module.compute_axis_metrics
compute_summary_stats = _proj_module.compute_summary_stats


COLORS = {"polite": "#1f77b4", "truesmile": "#2ca02c", "ambiguous": "#ff7f0e"}


@dataclass
class FollowupConfig:
    dtw_resample_root: Path = Path(r"E:\Matsuda_data\DTW_resample_output")
    output_root: Path = Path(r"E:\Matsuda_data\DTW_resample_output\projection_followup")
    norm_len: int = 20

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "FollowupConfig":
        return cls(
            dtw_resample_root=Path(args.dtw_resample_root),
            output_root=Path(args.output_root),
            norm_len=int(args.norm_len),
        )


class DTWProjectionFollowup:
    def __init__(self, config: FollowupConfig):
        self.cfg = config
        for sub in ("csv", "plots", "report", "prototypes"):
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
    def read_csv(path: Path) -> list[dict[str, str]]:
        with path.open("r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))

    @staticmethod
    def load_npy(path: Path) -> np.ndarray:
        return np.load(path, allow_pickle=False)

    def representative_sequences(self) -> dict[str, str]:
        rows = self.read_csv(self.cfg.dtw_resample_root / "csv" / "representative_sequences.csv")
        return {row["class"]: row["representative_sequence_id"] for row in rows}

    def prototype_path(self, class_name: str, seq_id: str) -> Path:
        return self.cfg.dtw_resample_root / "metrics" / "resampled20_aligned" / class_name / seq_id / "aligned_resampled20.npy"

    def all_sequences(self) -> list[tuple[str, str, Path]]:
        items = []
        root = self.cfg.dtw_resample_root / "metrics" / "resampled20_aligned"
        for class_name in CLASS_NAMES:
            class_dir = root / class_name
            if not class_dir.is_dir():
                continue
            for seq_dir in sorted(class_dir.iterdir(), key=lambda p: p.name):
                if (seq_dir / "aligned_resampled20.npy").exists():
                    items.append((class_name, seq_dir.name, seq_dir / "aligned_resampled20.npy"))
        return items

    def run(self) -> None:
        reps = self.representative_sequences()
        protos = {
            class_name: self.load_npy(self.prototype_path(class_name, reps[class_name])).astype(np.float32)
            for class_name in CLASS_NAMES
        }
        true_proto = protos["truesmile"]
        axis = true_proto[-1] - true_proto[0]
        axis_norm = float(np.linalg.norm(axis))

        prototype_rows: list[dict] = []
        for class_name, proto in protos.items():
            projection_length, projection_ratio, off_axis_distance, off_axis_ratio = compute_axis_metrics(proto, axis)
            for t in range(self.cfg.norm_len):
                prototype_rows.append(
                    {
                        "class": class_name,
                        "representative_sequence_id": reps[class_name],
                        "time_index": t,
                        "projection_length": float(projection_length[t]),
                        "projection_ratio": float(projection_ratio[t]),
                        "off_axis_distance": float(off_axis_distance[t]),
                        "off_axis_ratio": float(off_axis_ratio[t]),
                    }
                )

        sequence_rows: list[dict] = []
        for class_name, seq_id, path in self.all_sequences():
            arr = self.load_npy(path).astype(np.float32)
            projection_length, projection_ratio, off_axis_distance, off_axis_ratio = compute_axis_metrics(arr, axis)
            for t in range(self.cfg.norm_len):
                sequence_rows.append(
                    {
                        "class": class_name,
                        "sequence_id": seq_id,
                        "time_index": t,
                        "projection_length": float(projection_length[t]),
                        "projection_ratio": float(projection_ratio[t]),
                        "off_axis_distance": float(off_axis_distance[t]),
                        "off_axis_ratio": float(off_axis_ratio[t]),
                    }
                )

        self.write_csv(
            self.cfg.output_root / "csv" / "projection_metrics_prototype_dtw.csv",
            prototype_rows,
            [
                "class",
                "representative_sequence_id",
                "time_index",
                "projection_length",
                "projection_ratio",
                "off_axis_distance",
                "off_axis_ratio",
            ],
        )
        self.write_csv(
            self.cfg.output_root / "csv" / "projection_metrics_per_sequence_dtw.csv",
            sequence_rows,
            [
                "class",
                "sequence_id",
                "time_index",
                "projection_length",
                "projection_ratio",
                "off_axis_distance",
                "off_axis_ratio",
            ],
        )

        grouped_stats: dict[tuple[str, str], list[float]] = defaultdict(list)
        for row in sequence_rows:
            t = int(row["time_index"])
            grouped_stats[("projection_ratio", row["class"], t)].append(float(row["projection_ratio"]))
            grouped_stats[("off_axis_ratio", row["class"], t)].append(float(row["off_axis_ratio"]))

        stats_rows: list[dict] = []
        for (metric_type, class_name, t), values in sorted(grouped_stats.items()):
            stats = compute_summary_stats(values)
            stats_rows.append(
                {
                    "metric_type": metric_type,
                    "class": class_name,
                    "time_index": t,
                    **stats,
                }
            )
        self.write_csv(
            self.cfg.output_root / "csv" / "projection_statistics_dtw.csv",
            stats_rows,
            ["metric_type", "class", "time_index", "mean", "std", "median", "q1", "q3"],
        )

        # plots
        def grouped_proto(metric_key: str) -> dict[str, list[tuple[int, float]]]:
            out: dict[str, list[tuple[int, float]]] = defaultdict(list)
            for row in prototype_rows:
                out[row["class"]].append((int(row["time_index"]), float(row[metric_key])))
            for key in out:
                out[key].sort(key=lambda x: x[0])
            return out

        proto_along = grouped_proto("projection_ratio")
        proto_off = grouped_proto("off_axis_ratio")

        def grouped_stat(metric_name: str) -> dict[str, list[dict]]:
            out: dict[str, list[dict]] = defaultdict(list)
            for row in stats_rows:
                if row["metric_type"] == metric_name:
                    out[row["class"]].append(row)
            for key in out:
                out[key].sort(key=lambda r: int(r["time_index"]))
            return out

        stat_along = grouped_stat("projection_ratio")
        stat_off = grouped_stat("off_axis_ratio")

        fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
        for class_name in CLASS_NAMES:
            items = proto_along[class_name]
            ax.plot([x[0] for x in items], [x[1] for x in items], linewidth=2.0, color=COLORS[class_name], label=class_name)
        ax.set_title("Projection Along True-Smile Axis (DTW-resampled)")
        ax.set_xlabel("Time Index")
        ax.set_ylabel("Ratio")
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(self.cfg.output_root / "plots" / "projection_along_dtw.png")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
        for class_name in CLASS_NAMES:
            items = proto_off[class_name]
            ax.plot([x[0] for x in items], [x[1] for x in items], linewidth=2.0, color=COLORS[class_name], label=class_name)
        ax.set_title("Deviation from True-Smile Axis (DTW-resampled)")
        ax.set_xlabel("Time Index")
        ax.set_ylabel("Ratio")
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(self.cfg.output_root / "plots" / "projection_off_dtw.png")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
        for class_name in CLASS_NAMES:
            x = [v for _, v in proto_along[class_name]]
            y = [v for _, v in proto_off[class_name]]
            ax.plot(x, y, linewidth=2.0, color=COLORS[class_name], label=class_name)
            ax.scatter(x[0], y[0], color=COLORS[class_name], s=20)
            ax.scatter(x[-1], y[-1], color=COLORS[class_name], s=30, marker="x")
        ax.set_title("Along vs Off Phase Plot (DTW-resampled)")
        ax.set_xlabel("Projection Ratio")
        ax.set_ylabel("Off-Axis Ratio")
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(self.cfg.output_root / "plots" / "projection_phase_dtw.png")
        plt.close(fig)

        for metric_name, proto_source, stat_source, out_name in (
            ("projection_ratio", proto_along, stat_along, "projection_along_band_dtw.png"),
            ("off_axis_ratio", proto_off, stat_off, "projection_off_band_dtw.png"),
        ):
            fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
            for class_name in CLASS_NAMES:
                proto = np.asarray([v for _, v in proto_source[class_name]], dtype=np.float64)
                items = stat_source[class_name]
                t = [int(r["time_index"]) for r in items]
                mean = np.asarray([float(r["mean"]) for r in items], dtype=np.float64)
                q1 = np.asarray([float(r["q1"]) for r in items], dtype=np.float64)
                q3 = np.asarray([float(r["q3"]) for r in items], dtype=np.float64)
                ax.fill_between(t, q1, q3, color=COLORS[class_name], alpha=0.15)
                ax.plot(t, mean, color=COLORS[class_name], linestyle="--", linewidth=1.5, label=f"{class_name} mean")
                ax.plot(t, proto, color=COLORS[class_name], linewidth=2.0, label=f"{class_name} proto")
            ax.set_title(f"Per-Sequence Band Plot ({metric_name}, DTW-resampled)")
            ax.set_xlabel("Time Index")
            ax.set_ylabel("Ratio")
            ax.legend(loc="best", ncol=2, fontsize=8)
            fig.tight_layout()
            fig.savefig(self.cfg.output_root / "plots" / out_name)
            plt.close(fig)

        summary_lines = [
            "# DTW-resampled projection follow-up",
            "",
            "## Definition",
            "- Prototype trajectory is defined by the DTW representative sequence of each class.",
            "- All participant sequences are the DTW-aligned and then resampled-to-20-point sequences.",
            "",
            "## Representative sequences",
        ]
        for class_name in CLASS_NAMES:
            summary_lines.append(f"- {class_name}: representative_sequence_id={reps[class_name]}")
        summary_lines.append("")
        summary_lines.append("## Prototype metrics")
        for class_name in CLASS_NAMES:
            along_end = proto_along[class_name][-1][1]
            along_peak = max(v for _, v in proto_along[class_name])
            off_end = proto_off[class_name][-1][1]
            off_peak = max(v for _, v in proto_off[class_name])
            summary_lines.append(
                f"- {class_name}: along_end={along_end:.4f}, along_peak={along_peak:.4f}, off_end={off_end:.4f}, off_peak={off_peak:.4f}"
            )
        (self.cfg.output_root / "report" / "dtw_projection_followup_summary.md").write_text(
            "\n".join(summary_lines) + "\n",
            encoding="utf-8",
        )

        local_result = Path(__file__).resolve().parent / "projection_followup_result.md"
        local_result.write_text(
            "# DTW_resample projection follow-up\n\n"
            + f"- output_root: {self.cfg.output_root}\n"
            + f"- report: {self.cfg.output_root / 'report' / 'dtw_projection_followup_summary.md'}\n",
            encoding="utf-8",
        )
        print(f"[DTW_RESAMPLE_PROJECTION] Finished. Report saved to: {self.cfg.output_root / 'report' / 'dtw_projection_followup_summary.md'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run projection follow-up on DTW-resampled 20-point sequences.")
    parser.add_argument("--dtw_resample_root", default=r"E:\Matsuda_data\DTW_resample_output")
    parser.add_argument("--output_root", default=r"E:\Matsuda_data\DTW_resample_output\projection_followup")
    parser.add_argument("--norm_len", type=int, default=20)
    args = parser.parse_args()
    pipeline = DTWProjectionFollowup(FollowupConfig.from_args(args))
    pipeline.run()


if __name__ == "__main__":
    main()
