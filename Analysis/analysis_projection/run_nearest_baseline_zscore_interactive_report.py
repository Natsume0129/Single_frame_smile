from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


METHODS = ("methodA", "methodB")
BASELINE_CLASSES = ("truesmile", "polite")
CLASS_NAMES = ("polite", "truesmile", "ambiguous")
TARGET_STAGES = tuple(range(5, 101, 5))
SEARCH_PERCENTS = np.arange(0.0, 101.0, 1.0, dtype=np.float64)
METRICS = ("raw", "zscore_all74", "zscore_all74_clip_k3")

POINT_COLUMNS = [
    "metric",
    "method",
    "baseline_class",
    "target_class",
    "source_type",
    "seq_id",
    "rank",
    "target_stage_percent",
    "target_stage_index",
    "nearest_baseline_progress",
    "nearest_distance",
    "nearest_baseline_grid_index",
    "curve_id",
]


@dataclass
class ZscoreNearestBaselineConfig:
    analysis_input_root: Path = Path(r"E:\Matsuda_data\2-27meeting")
    projection_output_root: Path = Path(r"E:\Matsuda_data\3-10meeting")
    existing_nearest_root: Path = Path(r"E:\Matsuda_data\3-10meeting\nearest_baseline_curve")
    output_root: Path = Path(r"E:\Matsuda_data\3-10meeting\nearest_baseline_zscore_interactive")
    clip_k: float = 3.0

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "ZscoreNearestBaselineConfig":
        return cls(
            analysis_input_root=Path(args.analysis_input_root),
            projection_output_root=Path(args.projection_output_root),
            existing_nearest_root=Path(args.existing_nearest_root),
            output_root=Path(args.output_root),
            clip_k=float(args.clip_k),
        )


class ZscoreNearestBaselineReport:
    def __init__(self, cfg: ZscoreNearestBaselineConfig):
        self.cfg = cfg
        self.normalized_root = cfg.analysis_input_root / "metrics" / "normalized"
        self.csv_dir = cfg.output_root / "csv"
        self.report_dir = cfg.output_root / "report"
        self.csv_dir.mkdir(parents=True, exist_ok=True)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self._sequence_cache: dict[tuple[str, str], np.ndarray] = {}
        self._prototype_cache: dict[tuple[str, str], np.ndarray] = {}

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
    def sort_key(value: str) -> tuple[int, int | str]:
        return (0, int(value)) if str(value).isdigit() else (1, str(value))

    @staticmethod
    def clean_for_json(value: object) -> object:
        if isinstance(value, dict):
            return {str(k): ZscoreNearestBaselineReport.clean_for_json(v) for k, v in value.items()}
        if isinstance(value, list):
            return [ZscoreNearestBaselineReport.clean_for_json(v) for v in value]
        if isinstance(value, tuple):
            return [ZscoreNearestBaselineReport.clean_for_json(v) for v in value]
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            value = float(value)
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value

    def require_file(self, path: Path) -> Path:
        if not path.is_file():
            raise FileNotFoundError(f"Required input file is missing: {path}")
        return path

    def normalized_sequence_path(self, class_name: str, seq_id: str) -> Path:
        return self.normalized_root / class_name / str(seq_id) / "normalized_sequence.npy"

    def load_sequence(self, class_name: str, seq_id: str) -> np.ndarray:
        key = (class_name, str(seq_id))
        if key not in self._sequence_cache:
            path = self.require_file(self.normalized_sequence_path(class_name, str(seq_id)))
            arr = np.load(path, allow_pickle=False).astype(np.float64)
            self.validate_curve_shape(arr, path)
            self._sequence_cache[key] = arr
        return self._sequence_cache[key]

    def prototype_path(self, method: str, class_name: str) -> Path:
        return (
            self.cfg.projection_output_root
            / method
            / "prototypes"
            / f"prototype_{class_name}_{method}.npy"
        )

    def load_prototype(self, method: str, class_name: str) -> np.ndarray:
        key = (method, class_name)
        if key not in self._prototype_cache:
            path = self.require_file(self.prototype_path(method, class_name))
            arr = np.load(path, allow_pickle=False).astype(np.float64)
            self.validate_curve_shape(arr, path)
            self._prototype_cache[key] = arr
        return self._prototype_cache[key]

    @staticmethod
    def validate_curve_shape(arr: np.ndarray, path: Path) -> None:
        if arr.shape != (20, 4096):
            raise ValueError(f"Expected shape [20, 4096], got {arr.shape}: {path}")

    def load_all_sequences_for_zscore(self) -> tuple[dict[str, list[tuple[str, np.ndarray]]], np.ndarray]:
        by_class: dict[str, list[tuple[str, np.ndarray]]] = {}
        all_curves: list[np.ndarray] = []
        for class_name in CLASS_NAMES:
            class_dir = self.normalized_root / class_name
            if not class_dir.is_dir():
                raise FileNotFoundError(f"Missing normalized class directory: {class_dir}")
            entries = []
            for seq_dir in sorted((p for p in class_dir.iterdir() if p.is_dir()), key=lambda p: self.sort_key(p.name)):
                arr = self.load_sequence(class_name, seq_dir.name)
                entries.append((seq_dir.name, arr))
                all_curves.append(arr)
            by_class[class_name] = entries

        if not all_curves:
            raise RuntimeError(f"No normalized trajectories found under {self.normalized_root}")
        x = np.vstack(all_curves)
        if x.shape[1] != 4096:
            raise ValueError(f"Expected feature dimension 4096, got {x.shape[1]}")
        return by_class, x

    def compute_zscore_params(self, x: np.ndarray) -> tuple[dict, dict[str, np.ndarray]]:
        mu = np.mean(x, axis=0)
        sigma = np.std(x, axis=0, ddof=0)
        positive = sigma[sigma > 0]
        if positive.size == 0:
            raise RuntimeError("All sigma values are zero; z-score normalization is undefined.")
        eps = max(1e-12, 1e-8 * float(np.median(positive)))
        sigma_safe = np.where(sigma > eps, sigma, eps)
        small_count = int(np.sum(sigma <= eps))

        z = (x - mu) / sigma_safe
        abs_z = np.abs(z)
        clip_rate = float(np.mean(abs_z > self.cfg.clip_k))

        summary = {
            "n_sequences_total": int(x.shape[0] // 20),
            "n_observations_M": int(x.shape[0]),
            "feature_dim_D": int(x.shape[1]),
            "sigma_eps": float(eps),
            "sigma_zero_or_small_count": small_count,
            "sigma_min": float(np.min(sigma)),
            "sigma_p01": float(np.percentile(sigma, 1)),
            "sigma_p05": float(np.percentile(sigma, 5)),
            "sigma_median": float(np.median(sigma)),
            "sigma_p95": float(np.percentile(sigma, 95)),
            "sigma_p99": float(np.percentile(sigma, 99)),
            "sigma_max": float(np.max(sigma)),
            "sigma_safe_min": float(np.min(sigma_safe)),
            "sigma_safe_median": float(np.median(sigma_safe)),
            "sigma_safe_max": float(np.max(sigma_safe)),
            "abs_z_min": float(np.min(abs_z)),
            "abs_z_median": float(np.median(abs_z)),
            "abs_z_p95": float(np.percentile(abs_z, 95)),
            "abs_z_p99": float(np.percentile(abs_z, 99)),
            "abs_z_max": float(np.max(abs_z)),
            "clip_k": float(self.cfg.clip_k),
            "clip_rate": clip_rate,
        }
        return summary, {"mu": mu, "sigma": sigma, "sigma_safe": sigma_safe}

    def transform(self, values: np.ndarray, metric: str, params: dict[str, np.ndarray]) -> np.ndarray:
        if metric == "raw":
            return values
        z = (values - params["mu"]) / params["sigma_safe"]
        if metric == "zscore_all74":
            return z
        if metric == "zscore_all74_clip_k3":
            return np.clip(z, -self.cfg.clip_k, self.cfg.clip_k)
        raise ValueError(f"Unknown metric: {metric}")

    def nearest6_csv_path(self, method: str) -> Path:
        return (
            self.cfg.projection_output_root
            / "linear_axis_extension"
            / "csv"
            / f"nearest6_to_prototype_sequences_{method}.csv"
        )

    def load_nearest6_table(self, method: str) -> pd.DataFrame:
        path = self.require_file(self.nearest6_csv_path(method))
        df = pd.read_csv(path, dtype={"sequence_id": str, "rank": str})
        required = {"method", "class", "rank", "sequence_id"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Nearest-6 CSV missing columns {sorted(missing)}: {path}")
        return df

    def compute_rows_for_curve(
        self,
        metric: str,
        method: str,
        baseline_class: str,
        target_class: str,
        source_type: str,
        seq_id: str,
        rank: str,
        baseline_curve: np.ndarray,
        target_curve: np.ndarray,
        params: dict[str, np.ndarray],
    ) -> list[dict]:
        baseline_samples = np.vstack([self.point_at_percent(baseline_curve, p) for p in SEARCH_PERCENTS])
        baseline_samples_t = self.transform(baseline_samples, metric, params)
        curve_id = "|".join([metric, method, baseline_class, target_class, source_type, str(seq_id), str(rank)])
        rows: list[dict] = []
        for stage_index, stage in enumerate(TARGET_STAGES):
            target_point = self.point_at_percent(target_curve, float(stage))
            target_point_t = self.transform(target_point, metric, params)
            diff = baseline_samples_t - target_point_t
            squared_dist = np.einsum("ij,ij->i", diff, diff)
            nearest_idx = int(np.argmin(squared_dist))
            rows.append(
                {
                    "metric": metric,
                    "method": method,
                    "baseline_class": baseline_class,
                    "target_class": target_class,
                    "source_type": source_type,
                    "seq_id": str(seq_id),
                    "rank": str(rank),
                    "target_stage_percent": float(stage),
                    "target_stage_index": int(stage_index),
                    "nearest_baseline_progress": float(SEARCH_PERCENTS[nearest_idx]),
                    "nearest_distance": float(np.sqrt(squared_dist[nearest_idx])),
                    "nearest_baseline_grid_index": int(nearest_idx),
                    "curve_id": curve_id,
                }
            )
        return rows

    def compute_all_point_rows(self, params: dict[str, np.ndarray]) -> pd.DataFrame:
        rows: list[dict] = []
        for metric in METRICS:
            print(f"computing nearest-baseline rows for metric={metric}")
            for method in METHODS:
                nearest6 = self.load_nearest6_table(method)
                for baseline_class in BASELINE_CLASSES:
                    baseline_curve = self.load_prototype(method, baseline_class)
                    for target_class in CLASS_NAMES:
                        rows.extend(
                            self.compute_rows_for_curve(
                                metric=metric,
                                method=method,
                                baseline_class=baseline_class,
                                target_class=target_class,
                                source_type="prototype",
                                seq_id="prototype",
                                rank="",
                                baseline_curve=baseline_curve,
                                target_curve=self.load_prototype(method, target_class),
                                params=params,
                            )
                        )
                    for target_class in CLASS_NAMES:
                        selected = nearest6[nearest6["class"] == target_class].copy()
                        selected["rank_int"] = selected["rank"].astype(int)
                        selected = selected.sort_values("rank_int")
                        for row in selected.itertuples(index=False):
                            rows.extend(
                                self.compute_rows_for_curve(
                                    metric=metric,
                                    method=method,
                                    baseline_class=baseline_class,
                                    target_class=target_class,
                                    source_type="nearest6",
                                    seq_id=str(row.sequence_id),
                                    rank=str(row.rank),
                                    baseline_curve=baseline_curve,
                                    target_curve=self.load_sequence(target_class, str(row.sequence_id)),
                                    params=params,
                                )
                            )
        df = pd.DataFrame(rows, columns=POINT_COLUMNS)
        return df

    @staticmethod
    def summarize_endpoint(df: pd.DataFrame) -> pd.DataFrame:
        endpoint = df[df["target_stage_percent"] == 100.0]
        group_cols = ["metric", "method", "baseline_class", "target_class", "source_type"]
        rows = []
        for key, group in endpoint.groupby(group_cols, sort=True):
            progress = group["nearest_baseline_progress"].to_numpy(dtype=np.float64)
            distance = group["nearest_distance"].to_numpy(dtype=np.float64)
            rows.append(
                dict(
                    zip(group_cols, key),
                    mean_nearest_progress=float(np.mean(progress)),
                    median_nearest_progress=float(np.median(progress)),
                    std_nearest_progress=float(np.std(progress, ddof=0)),
                    mean_nearest_distance=float(np.mean(distance)),
                    median_nearest_distance=float(np.median(distance)),
                    std_nearest_distance=float(np.std(distance, ddof=0)),
                    n_curves=int(group["curve_id"].nunique()),
                )
            )
        return pd.DataFrame(rows)

    @staticmethod
    def foldback_per_curve(df: pd.DataFrame) -> pd.DataFrame:
        group_cols = ["metric", "method", "baseline_class", "target_class", "source_type", "seq_id", "rank", "curve_id"]
        rows = []
        for key, group in df.groupby(group_cols, sort=True):
            ordered = group.sort_values("target_stage_index")
            progress = ordered["nearest_baseline_progress"].to_numpy(dtype=np.float64)
            distance = ordered["nearest_distance"].to_numpy(dtype=np.float64)
            stages = ordered["target_stage_percent"].to_numpy(dtype=np.float64)
            diff = np.diff(progress)
            negative = diff[diff < 0.0]
            jump_sizes = -negative
            rows.append(
                dict(
                    zip(group_cols, key),
                    foldback_count=int(np.sum(diff < 0.0)),
                    severe_foldback_count=int(np.sum(diff <= -10.0)),
                    mean_negative_jump_size=float(np.mean(jump_sizes)) if jump_sizes.size else 0.0,
                    max_negative_jump_size=float(np.max(jump_sizes)) if jump_sizes.size else 0.0,
                    endpoint_nearest_progress=float(progress[-1]),
                    endpoint_nearest_distance=float(distance[-1]),
                    max_nearest_progress=float(np.max(progress)),
                    mean_nearest_distance=float(np.mean(distance)),
                    distance_auc_over_target_stage=float(np.trapezoid(distance, stages)),
                )
            )
        return pd.DataFrame(rows)

    @staticmethod
    def foldback_aggregate(per_curve: pd.DataFrame) -> pd.DataFrame:
        group_cols = ["metric", "method", "baseline_class", "target_class", "source_type"]
        rows = []
        for key, group in per_curve.groupby(group_cols, sort=True):
            rows.append(
                dict(
                    zip(group_cols, key),
                    n_curves=int(len(group)),
                    total_foldback_count=int(group["foldback_count"].sum()),
                    total_severe_foldback_count=int(group["severe_foldback_count"].sum()),
                    mean_foldback_count=float(group["foldback_count"].mean()),
                    mean_severe_foldback_count=float(group["severe_foldback_count"].mean()),
                    mean_negative_jump_size=float(group["mean_negative_jump_size"].mean()),
                    max_negative_jump_size=float(group["max_negative_jump_size"].max()),
                    mean_endpoint_nearest_progress=float(group["endpoint_nearest_progress"].mean()),
                    mean_endpoint_nearest_distance=float(group["endpoint_nearest_distance"].mean()),
                    mean_max_nearest_progress=float(group["max_nearest_progress"].mean()),
                    mean_nearest_distance=float(group["mean_nearest_distance"].mean()),
                    mean_distance_auc_over_target_stage=float(group["distance_auc_over_target_stage"].mean()),
                )
            )
        return pd.DataFrame(rows)

    @staticmethod
    def foldback_method_baseline(per_curve: pd.DataFrame) -> pd.DataFrame:
        group_cols = ["metric", "method", "baseline_class"]
        rows = []
        for key, group in per_curve.groupby(group_cols, sort=True):
            rows.append(
                dict(
                    zip(group_cols, key),
                    n_curves=int(len(group)),
                    total_foldback_count=int(group["foldback_count"].sum()),
                    total_severe_foldback_count=int(group["severe_foldback_count"].sum()),
                    mean_foldback_count=float(group["foldback_count"].mean()),
                    mean_severe_foldback_count=float(group["severe_foldback_count"].mean()),
                    mean_endpoint_nearest_progress=float(group["endpoint_nearest_progress"].mean()),
                    mean_endpoint_nearest_distance=float(group["endpoint_nearest_distance"].mean()),
                    mean_nearest_distance=float(group["mean_nearest_distance"].mean()),
                )
            )
        return pd.DataFrame(rows)

    def validate_raw_endpoint(self, endpoint: pd.DataFrame) -> dict:
        existing_path = self.require_file(self.cfg.existing_nearest_root / "csv" / "endpoint_100_summary.csv")
        existing = pd.read_csv(existing_path)
        mismatches = []
        for old in existing.itertuples(index=False):
            source_type = "nearest6" if str(old.source) == "nearest6 mean" else "prototype"
            selected = endpoint[
                (endpoint["metric"] == "raw")
                & (endpoint["method"] == old.method)
                & (endpoint["baseline_class"] == old.baseline)
                & (endpoint["target_class"] == old.target)
                & (endpoint["source_type"] == source_type)
            ]
            if len(selected) != 1:
                mismatches.append(
                    {
                        "method": old.method,
                        "baseline": old.baseline,
                        "target": old.target,
                        "source": old.source,
                        "reason": f"expected 1 matching row, found {len(selected)}",
                    }
                )
                continue
            row = selected.iloc[0]
            progress_diff = abs(float(row["mean_nearest_progress"]) - float(old.progress_100))
            distance_diff = abs(float(row["mean_nearest_distance"]) - float(old.distance_100))
            if progress_diff > 1e-9 or distance_diff > 1e-9:
                mismatches.append(
                    {
                        "method": old.method,
                        "baseline": old.baseline,
                        "target": old.target,
                        "source": old.source,
                        "progress_diff": progress_diff,
                        "distance_diff": distance_diff,
                        "new_progress": float(row["mean_nearest_progress"]),
                        "old_progress": float(old.progress_100),
                        "new_distance": float(row["mean_nearest_distance"]),
                        "old_distance": float(old.distance_100),
                    }
                )
        return {
            "existing_summary": str(existing_path),
            "matched": len(mismatches) == 0,
            "mismatch_count": len(mismatches),
            "mismatches": mismatches,
        }

    @staticmethod
    def table_records(df: pd.DataFrame) -> list[dict]:
        return df.replace({np.nan: None}).to_dict(orient="records")

    def build_lines_payload(self, df: pd.DataFrame) -> list[dict]:
        group_cols = ["metric", "method", "baseline_class", "target_class", "source_type", "seq_id", "rank", "curve_id"]
        lines = []
        for key, group in df.groupby(group_cols, sort=True):
            metric, method, baseline_class, target_class, source_type, seq_id, rank, curve_id = key
            ordered = group.sort_values("target_stage_index")
            rank_text = f", rank={rank}" if str(rank) else ""
            label = (
                f"{metric} | {method} | baseline={baseline_class} | target={target_class} | "
                f"{source_type} seq={seq_id}{rank_text}"
            )
            lines.append(
                {
                    "metric": metric,
                    "method": method,
                    "baselineClass": baseline_class,
                    "targetClass": target_class,
                    "sourceType": source_type,
                    "seqId": seq_id,
                    "rank": rank,
                    "curveId": curve_id,
                    "label": label,
                    "points": [
                        {
                            "stage": float(row.target_stage_percent),
                            "stageIndex": int(row.target_stage_index),
                            "progress": float(row.nearest_baseline_progress),
                            "distance": float(row.nearest_distance),
                            "gridIndex": int(row.nearest_baseline_grid_index),
                        }
                        for row in ordered.itertuples(index=False)
                    ],
                }
            )
        return lines

    def render_html(
        self,
        points: pd.DataFrame,
        endpoint: pd.DataFrame,
        foldback_mb: pd.DataFrame,
        zscore_summary: dict,
    ) -> str:
        payload = {
            "lines": self.build_lines_payload(points),
            "zscoreSummary": zscore_summary,
            "foldbackMethodBaseline": self.table_records(foldback_mb),
            "endpointSummary": self.table_records(endpoint),
            "metrics": list(METRICS),
            "classes": list(CLASS_NAMES),
            "sources": ["prototype", "nearest6"],
            "charts": [
                {"method": "methodB", "baseline": "truesmile", "title": "Nearest-baseline new curve | methodB | baseline=truesmile"},
                {"method": "methodB", "baseline": "polite", "title": "Nearest-baseline new curve | methodB | baseline=polite"},
                {"method": "methodA", "baseline": "truesmile", "title": "Nearest-baseline new curve | methodA | baseline=truesmile"},
                {"method": "methodA", "baseline": "polite", "title": "Nearest-baseline new curve | methodA | baseline=polite"},
            ],
        }
        data_json = json.dumps(self.clean_for_json(payload), ensure_ascii=False, allow_nan=False)
        return HTML_TEMPLATE.replace("__DATA_JSON__", data_json)

    @staticmethod
    def metric_totals(per_curve: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for metric, group in per_curve.groupby("metric", sort=True):
            rows.append(
                {
                    "metric": metric,
                    "n_curves": int(len(group)),
                    "total_foldback_count": int(group["foldback_count"].sum()),
                    "total_severe_foldback_count": int(group["severe_foldback_count"].sum()),
                    "mean_endpoint_nearest_progress": float(group["endpoint_nearest_progress"].mean()),
                    "mean_endpoint_nearest_distance": float(group["endpoint_nearest_distance"].mean()),
                }
            )
        return pd.DataFrame(rows)

    def write_run_summary(
        self,
        zscore_summary: dict,
        points: pd.DataFrame,
        endpoint: pd.DataFrame,
        per_curve: pd.DataFrame,
        validation: dict,
        output_paths: dict[str, Path],
    ) -> None:
        totals = self.metric_totals(per_curve)
        raw_total = totals[totals["metric"] == "raw"].iloc[0]
        z_total = totals[totals["metric"] == "zscore_all74"].iloc[0]
        clip_total = totals[totals["metric"] == "zscore_all74_clip_k3"].iloc[0]
        lines = [
            "# Nearest-Baseline Z-Score Interactive Report Run Summary",
            "",
            f"Timestamp: {datetime.now().isoformat(timespec='seconds')}",
            "",
            f"Script path: `{Path(__file__).resolve()}`",
            f"Output root: `{self.cfg.output_root}`",
            "",
            "Input roots:",
            "",
            f"- Normalized sequences: `{self.normalized_root}`",
            f"- Projection outputs/prototypes: `{self.cfg.projection_output_root}`",
            f"- Existing raw nearest-baseline summary: `{self.cfg.existing_nearest_root}`",
            "",
            "Z-score parameter status:",
            "",
            f"- Number of sequences used for mu/sigma: {zscore_summary['n_sequences_total']}",
            f"- M observations: {zscore_summary['n_observations_M']}",
            f"- D features: {zscore_summary['feature_dim_D']}",
            f"- sigma eps: {zscore_summary['sigma_eps']:.12g}",
            f"- sigma <= eps count: {zscore_summary['sigma_zero_or_small_count']}",
            f"- sigma min/median/max: {zscore_summary['sigma_min']:.12g} / {zscore_summary['sigma_median']:.12g} / {zscore_summary['sigma_max']:.12g}",
            f"- clip rate for k={self.cfg.clip_k:g}: {zscore_summary['clip_rate']:.8f}",
            "",
            "Output counts:",
            "",
            f"- Point row count: {len(points)}",
            f"- Curve count per metric: {points.groupby('metric')['curve_id'].nunique().to_dict()}",
            f"- HTML path: `{output_paths['html']}`",
            "",
            "Commands run:",
            "",
            "```powershell",
            "python -m py_compile E:\\Single_frame_smile\\Analysis\\analysis_projection\\run_nearest_baseline_zscore_interactive_report.py",
            "cd E:\\Single_frame_smile\\Analysis\\analysis_projection",
            "python .\\run_nearest_baseline_zscore_interactive_report.py",
            "```",
            "",
            "Validation status:",
            "",
            f"- Raw endpoint validation matched existing result: {validation['matched']}",
            f"- Raw endpoint mismatch count: {validation['mismatch_count']}",
            f"- Expected point rows: 5040; actual point rows: {len(points)}",
            "",
            "Top-level foldback comparison:",
            "",
            "| metric | curves | foldbacks | severe foldbacks | mean endpoint progress | mean endpoint distance |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for row in totals.itertuples(index=False):
            lines.append(
                f"| {row.metric} | {row.n_curves} | {row.total_foldback_count} | "
                f"{row.total_severe_foldback_count} | {row.mean_endpoint_nearest_progress:.3f} | "
                f"{row.mean_endpoint_nearest_distance:.6f} |"
            )
        lines.extend(
            [
                "",
                "Short interpretation:",
                "",
                f"- zscore_all74 {'reduced' if z_total.total_foldback_count < raw_total.total_foldback_count else 'did not reduce'} total foldbacks compared with raw.",
                f"- zscore_all74_clip_k3 {'reduced' if clip_total.total_foldback_count < z_total.total_foldback_count else 'did not reduce'} total foldbacks compared with unclipped z-score.",
                "- Endpoint nearest progress patterns should be compared from `endpoint_100_summary_by_metric.csv`; absolute distance values are not directly comparable across raw and transformed metrics.",
                "- If endpoint progress or foldback counts change strongly across metrics, nearest-baseline conclusions are metric-sensitive.",
            ]
        )
        if validation["mismatches"]:
            lines.extend(["", "Raw validation mismatches:", ""])
            lines.append("```json")
            lines.append(json.dumps(validation["mismatches"], indent=2))
            lines.append("```")

        output_paths["run_summary"].write_text("\n".join(lines), encoding="utf-8")

    def run(self) -> dict[str, object]:
        print("loading all normalized trajectories for z-score parameters")
        sequences_by_class, x = self.load_all_sequences_for_zscore()
        sequence_count = sum(len(v) for v in sequences_by_class.values())
        print(f"loaded {sequence_count} sequences, X shape={x.shape}")

        print("computing global feature-wise z-score parameters")
        zscore_summary, params = self.compute_zscore_params(x)
        pd.DataFrame([zscore_summary]).to_csv(self.csv_dir / "zscore_parameter_summary.csv", index=False)

        points = self.compute_all_point_rows(params)
        points_path = self.csv_dir / "nearest_baseline_points_all_metrics.csv"
        points.to_csv(points_path, index=False)

        endpoint = self.summarize_endpoint(points)
        endpoint_path = self.csv_dir / "endpoint_100_summary_by_metric.csv"
        endpoint.to_csv(endpoint_path, index=False)

        per_curve = self.foldback_per_curve(points)
        per_curve_path = self.csv_dir / "foldback_per_curve_by_metric.csv"
        per_curve.to_csv(per_curve_path, index=False)

        aggregate = self.foldback_aggregate(per_curve)
        aggregate_path = self.csv_dir / "foldback_aggregate_by_metric.csv"
        aggregate.to_csv(aggregate_path, index=False)
        summary_path = self.csv_dir / "foldback_summary_by_metric.csv"
        aggregate.to_csv(summary_path, index=False)

        foldback_mb = self.foldback_method_baseline(per_curve)
        foldback_mb_path = self.csv_dir / "foldback_method_baseline_by_metric.csv"
        foldback_mb.to_csv(foldback_mb_path, index=False)

        print("validating raw endpoint summary against existing nearest-baseline output")
        validation = self.validate_raw_endpoint(endpoint)

        print("writing standalone HTML")
        html_path = self.cfg.output_root / "nearest_baseline_zscore_four_charts.html"
        html_path.write_text(self.render_html(points, endpoint, foldback_mb, zscore_summary), encoding="utf-8")

        run_summary_path = self.report_dir / "run_summary.md"
        paths = {
            "points": points_path,
            "endpoint": endpoint_path,
            "foldback_summary": summary_path,
            "foldback_per_curve": per_curve_path,
            "foldback_aggregate": aggregate_path,
            "foldback_method_baseline": foldback_mb_path,
            "zscore_summary": self.csv_dir / "zscore_parameter_summary.csv",
            "html": html_path,
            "run_summary": run_summary_path,
        }
        self.write_run_summary(zscore_summary, points, endpoint, per_curve, validation, paths)

        return {
            "paths": paths,
            "point_rows": int(len(points)),
            "curve_count_by_metric": points.groupby("metric")["curve_id"].nunique().to_dict(),
            "zscore_summary": zscore_summary,
            "raw_validation": validation,
            "metric_totals": self.metric_totals(per_curve).to_dict(orient="records"),
        }


HTML_TEMPLATE = r"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Nearest-Baseline Metric Sensitivity: Raw vs Z-Score fc7</title>
<style>
body{font-family:Arial,sans-serif;margin:28px;color:#222;background:#fafafa;line-height:1.55}
main{max-width:1340px;margin:0 auto}
h1{font-size:29px;margin:0 0 8px}
h2{font-size:21px;margin:30px 0 10px;border-bottom:1px solid #ddd;padding-bottom:6px}
h3{font-size:16px;margin:0 0 7px}
p{max-width:1080px}
.lead{font-size:15px;color:#333}
.panel{background:white;border:1px solid #ddd;padding:15px 17px;margin:16px 0}
.note{background:#fff7df;border-left:4px solid #d99b00;padding:10px 12px;margin:12px 0}
.formula{font-family:Consolas,monospace;background:#f6f6f6;border:1px solid #ddd;padding:10px 12px;white-space:pre-wrap}
.controls{position:sticky;top:0;z-index:10;background:#fafafa;border-bottom:1px solid #ddd;padding:10px 0;margin:14px 0 16px;display:flex;flex-wrap:wrap;gap:12px;align-items:center}
label{font-size:13px;display:inline-flex;gap:6px;align-items:center}
select,input{font-size:13px}
.chart-card{background:#fff;border:1px solid #ddd;padding:14px 16px;margin:0 0 22px}
.chart-title{font-weight:700;margin-bottom:4px}
.chart-caption{font-size:13px;color:#555;margin-bottom:7px}
.chart-svg{width:100%;height:auto;display:block;background:#fff}
.axis{stroke:#222;stroke-width:1.2}
.grid{stroke:#ddd;stroke-width:1}
.tick{font-size:12px;fill:#555}
.label{font-size:13px;fill:#333}
.curve{fill:none;stroke-width:2.3;opacity:.62;vector-effect:non-scaling-stroke;pointer-events:stroke;cursor:pointer}
.curve.prototype{stroke-width:3.1;opacity:.95;stroke-dasharray:7 5}
.curve-hit{fill:none;stroke:#000;stroke-opacity:.001;stroke-width:14;pointer-events:stroke;cursor:pointer}
.point{opacity:.42;cursor:pointer;transition:r .1s, opacity .1s}
.point.prototype{opacity:.78}
.chart-card.hovering .curve{opacity:.07;stroke-width:1.1}
.chart-card.hovering .point{opacity:.10}
.chart-card.hovering .curve.active{opacity:1;stroke-width:4.2}
.chart-card.hovering .curve.prototype.active{opacity:1;stroke-width:4.8}
.chart-card.hovering .point.active{opacity:1}
.legend{display:flex;flex-wrap:wrap;gap:12px 18px;font-size:13px;color:#444;margin-top:8px}
.legend span{display:inline-flex;align-items:center;gap:6px}
.dot{width:12px;height:12px;border-radius:50%;display:inline-block}
.active-label{font-size:13px;background:#f5f5f5;border:1px solid #ddd;padding:7px 9px;margin-top:9px;min-height:18px}
.tooltip{position:fixed;display:none;z-index:50;max-width:360px;background:#222;color:white;padding:8px 9px;font-size:12px;line-height:1.35;pointer-events:none}
table{border-collapse:collapse;width:100%;font-size:13px;background:#fff;margin:10px 0 18px}
th,td{border:1px solid #ddd;padding:6px 7px;text-align:right}
th{background:#f0f0f0}
th:first-child,td:first-child,th:nth-child(2),td:nth-child(2),th:nth-child(3),td:nth-child(3),th:nth-child(4),td:nth-child(4),th:nth-child(5),td:nth-child(5){text-align:left}
.table-wrap{overflow-x:auto}
@media(max-width:900px){.controls{position:static}.chart-card{padding:12px 10px}}
</style>
</head>
<body>
<main>
<h1>Nearest-Baseline Metric Sensitivity: Raw vs Z-Score fc7</h1>
<p class="lead">This standalone report compares the original raw fc7 nearest-baseline metric against two feature-wise z-score metrics. It does not replace the raw nearest-baseline result; it tests whether the nearest-point mapping is sensitive to feature scaling.</p>

<section class="panel">
<h2>Method</h2>
<div class="formula">x = (x_1, x_2, ..., x_4096)
X = all 74 normalized baseline-relative trajectories stacked over all 20 stages
mu_i = mean(X[:, i])
sigma_i = std(X[:, i], ddof=0)
T_raw(x) = x
T_z(x)_i = (x_i - mu_i) / sigma_safe_i
T_clip(x)_i = clip(T_z(x)_i, -3, 3)

u_i = argmin_u || T(C_target(t_i)) - T(C_baseline(u)) ||_2
new curve point = (u_i, nearest distance)</div>
<p>All 74 normalized sequences are used to estimate global feature-wise vectors <b>mu</b> and <b>sigma</b>, each with length 4096. The nearest-6 selection is not the sampling rule for estimating these parameters; nearest-6 is only the balanced representative subset plotted in the charts.</p>
<p>Z-score without clipping is feature-wise standardized Euclidean distance. Clipping tests whether high-leverage transformed coordinates drive nearest-point instability.</p>
<div class="note">Raw distance and transformed distance values are not directly comparable in absolute units. Compare curve shape, endpoint progress, foldback behavior, and relative class relationships.</div>
</section>

<section class="panel">
<h2>Interactive Charts</h2>
<div class="controls">
<label>metric <select id="metricSelect">
<option value="raw">raw</option>
<option value="zscore_all74">zscore_all74</option>
<option value="zscore_all74_clip_k3">zscore_all74_clip_k3</option>
</select></label>
<label><input type="checkbox" data-target="polite" checked> polite</label>
<label><input type="checkbox" data-target="truesmile" checked> truesmile</label>
<label><input type="checkbox" data-target="ambiguous" checked> ambiguous</label>
<label><input type="checkbox" data-source="prototype" checked> prototype</label>
<label><input type="checkbox" data-source="nearest6" checked> nearest6</label>
</div>

<section class="chart-card" data-method="methodB" data-baseline="truesmile">
<div class="chart-title">Nearest-baseline new curve | methodB | baseline=truesmile</div>
<div class="chart-caption"></div>
<div class="chart-host"></div>
<div class="legend"><span><i class="dot" style="background:#1f77b4"></i>polite</span><span><i class="dot" style="background:#2ca02c"></i>truesmile</span><span><i class="dot" style="background:#ff7f0e"></i>ambiguous</span><span>solid = nearest6</span><span>dashed = prototype</span></div>
<div class="active-label">No curve selected</div>
</section>
<section class="chart-card" data-method="methodB" data-baseline="polite">
<div class="chart-title">Nearest-baseline new curve | methodB | baseline=polite</div>
<div class="chart-caption"></div>
<div class="chart-host"></div>
<div class="legend"><span><i class="dot" style="background:#1f77b4"></i>polite</span><span><i class="dot" style="background:#2ca02c"></i>truesmile</span><span><i class="dot" style="background:#ff7f0e"></i>ambiguous</span><span>solid = nearest6</span><span>dashed = prototype</span></div>
<div class="active-label">No curve selected</div>
</section>
<section class="chart-card" data-method="methodA" data-baseline="truesmile">
<div class="chart-title">Nearest-baseline new curve | methodA | baseline=truesmile</div>
<div class="chart-caption"></div>
<div class="chart-host"></div>
<div class="legend"><span><i class="dot" style="background:#1f77b4"></i>polite</span><span><i class="dot" style="background:#2ca02c"></i>truesmile</span><span><i class="dot" style="background:#ff7f0e"></i>ambiguous</span><span>solid = nearest6</span><span>dashed = prototype</span></div>
<div class="active-label">No curve selected</div>
</section>
<section class="chart-card" data-method="methodA" data-baseline="polite">
<div class="chart-title">Nearest-baseline new curve | methodA | baseline=polite</div>
<div class="chart-caption"></div>
<div class="chart-host"></div>
<div class="legend"><span><i class="dot" style="background:#1f77b4"></i>polite</span><span><i class="dot" style="background:#2ca02c"></i>truesmile</span><span><i class="dot" style="background:#ff7f0e"></i>ambiguous</span><span>solid = nearest6</span><span>dashed = prototype</span></div>
<div class="active-label">No curve selected</div>
</section>
</section>

<section class="panel">
<h2>Z-Score Parameter Summary</h2>
<div class="table-wrap" id="zscoreTable"></div>
</section>

<section class="panel">
<h2>Foldback Summary by Metric / Method / Baseline</h2>
<p>This table aggregates over all target classes and both plotted source types.</p>
<div class="table-wrap" id="foldbackTable"></div>
</section>

<section class="panel">
<h2>Endpoint 100% Summary</h2>
<div class="table-wrap" id="endpointTable"></div>
</section>
</main>
<div class="tooltip" id="tooltip"></div>
<script>
const DATA = __DATA_JSON__;
const COLORS = {polite:'#1f77b4', truesmile:'#2ca02c', ambiguous:'#ff7f0e'};
const Y_LABELS = {
  raw: 'Nearest distance in raw fc7 space',
  zscore_all74: 'Nearest distance in z-score fc7 space',
  zscore_all74_clip_k3: 'Nearest distance in z-score clipped fc7 space'
};

function fmt(value, digits) {
  const n = Number(value);
  if (!Number.isFinite(n)) return 'N/A';
  return n.toFixed(digits);
}
function selectedValues(selector, attr) {
  return new Set(Array.from(document.querySelectorAll(selector + ':checked')).map(el => el.dataset[attr]));
}
function pathFromPoints(points, xScale, yScale) {
  return points.map((p, i) => (i === 0 ? 'M' : 'L') + xScale(p.progress).toFixed(2) + ',' + yScale(p.distance).toFixed(2)).join(' ');
}
function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
}
function currentLines(method, baseline) {
  const metric = document.getElementById('metricSelect').value;
  const targets = selectedValues('[data-target]', 'target');
  const sources = selectedValues('[data-source]', 'source');
  return DATA.lines.filter(line =>
    line.metric === metric &&
    line.method === method &&
    line.baselineClass === baseline &&
    targets.has(line.targetClass) &&
    sources.has(line.sourceType)
  );
}
function tooltipHtml(line, point) {
  const rows = [
    ['metric', line.metric],
    ['method', line.method],
    ['baseline class', line.baselineClass],
    ['target class', line.targetClass],
    ['source type', line.sourceType],
    ['seq_id', line.seqId],
    ['rank', line.rank || '-']
  ];
  if (point) {
    rows.push(['target stage percent', fmt(point.stage, 0) + '%']);
    rows.push(['nearest baseline progress', fmt(point.progress, 2) + '%']);
    rows.push(['nearest distance', fmt(point.distance, 5)]);
  }
  return rows.map(row => '<b>' + escapeHtml(row[0]) + ':</b> ' + escapeHtml(row[1])).join('<br>');
}
function showTooltip(event, html) {
  const tip = document.getElementById('tooltip');
  tip.innerHTML = html;
  tip.style.display = 'block';
  moveTooltip(event);
}
function moveTooltip(event) {
  const tip = document.getElementById('tooltip');
  tip.style.left = Math.min(event.clientX + 14, window.innerWidth - 390) + 'px';
  tip.style.top = (event.clientY + 14) + 'px';
}
function hideTooltip() {
  document.getElementById('tooltip').style.display = 'none';
}
function setActive(card, key, label) {
  card.classList.add('hovering');
  card.querySelectorAll('[data-role="curve"]').forEach(el => el.classList.remove('active'));
  card.querySelectorAll('.point').forEach(el => el.classList.remove('active'));
  card.querySelectorAll('g[data-key]').forEach(g => {
    if (g.dataset.key === key) {
      g.parentNode.appendChild(g);
      g.querySelectorAll('[data-role="curve"]').forEach(el => el.classList.add('active'));
      card.querySelector('.active-label').textContent = label;
    }
  });
}
function clearActive(card) {
  card.classList.remove('hovering');
  card.querySelectorAll('[data-role="curve"]').forEach(el => el.classList.remove('active'));
  card.querySelectorAll('.point').forEach(el => {
    el.classList.remove('active');
    el.setAttribute('r', el.dataset.baseR);
  });
  card.querySelector('.active-label').textContent = 'No curve selected';
  hideTooltip();
}
function renderChart(card) {
  const method = card.dataset.method;
  const baseline = card.dataset.baseline;
  const metric = document.getElementById('metricSelect').value;
  const lines = currentLines(method, baseline);
  card.querySelector('.chart-caption').textContent = 'Metric: ' + metric + '. Connected line order follows target stage order, not sorted x-axis order.';

  const width = 1120, height = 500;
  const left = 70, right = 28, top = 36, bottom = 58;
  const plotW = width - left - right, plotH = height - top - bottom;
  const yMax = Math.max(1, ...lines.flatMap(line => line.points.map(p => p.distance))) * 1.08;
  const xScale = x => left + (x / 100) * plotW;
  const yScale = y => top + plotH - (y / yMax) * plotH;
  const yTicks = [0, yMax/4, yMax/2, yMax*3/4, yMax];
  let svg = '<svg class="chart-svg" viewBox="0 0 ' + width + ' ' + height + '">';
  [0,20,40,60,80,100].forEach(t => {
    const x = xScale(t);
    svg += '<line class="grid" x1="' + x + '" y1="' + top + '" x2="' + x + '" y2="' + (top+plotH) + '"/>';
    svg += '<text class="tick" x="' + x + '" y="' + (top+plotH+23) + '" text-anchor="middle">' + t + '</text>';
  });
  yTicks.forEach(t => {
    const y = yScale(t);
    svg += '<line class="grid" x1="' + left + '" y1="' + y + '" x2="' + (left+plotW) + '" y2="' + y + '"/>';
    svg += '<text class="tick" x="' + (left-10) + '" y="' + (y+4) + '" text-anchor="end">' + fmt(t, 2) + '</text>';
  });
  svg += '<line class="axis" x1="' + left + '" y1="' + (top+plotH) + '" x2="' + (left+plotW) + '" y2="' + (top+plotH) + '"/>';
  svg += '<line class="axis" x1="' + left + '" y1="' + top + '" x2="' + left + '" y2="' + (top+plotH) + '"/>';
  svg += '<text class="label" x="' + (left+plotW/2) + '" y="' + (height-18) + '" text-anchor="middle">nearest baseline progress (%)</text>';
  svg += '<text class="label" x="20" y="' + (top+plotH/2) + '" text-anchor="middle" transform="rotate(-90,20,' + (top+plotH/2) + ')">' + escapeHtml(Y_LABELS[metric]) + '</text>';

  lines.forEach(line => {
    const color = COLORS[line.targetClass];
    const proto = line.sourceType === 'prototype' ? ' prototype' : '';
    const d = pathFromPoints(line.points, xScale, yScale);
    svg += '<g data-key="' + escapeHtml(line.curveId) + '" data-line-index="' + DATA.lines.indexOf(line) + '" data-label="' + escapeHtml(line.label) + '">';
    svg += '<path class="curve' + proto + '" data-role="curve" d="' + d + '" stroke="' + color + '"/>';
    svg += '<path class="curve-hit" data-role="hit" d="' + d + '"/>';
    line.points.forEach((p, pointIndex) => {
      const r = line.sourceType === 'prototype' ? 3.6 : 2.4;
      svg += '<circle class="point' + proto + '" data-point-index="' + pointIndex + '" data-base-r="' + r + '" cx="' + xScale(p.progress).toFixed(2) + '" cy="' + yScale(p.distance).toFixed(2) + '" r="' + r + '" fill="' + color + '"/>';
    });
    svg += '</g>';
  });
  svg += '</svg>';
  card.querySelector('.chart-host').innerHTML = svg;

  card.querySelectorAll('g[data-key]').forEach(g => {
    const line = DATA.lines[Number(g.dataset.lineIndex)];
    g.querySelectorAll('[data-role="curve"], [data-role="hit"]').forEach(el => {
      el.addEventListener('mouseover', event => {
        setActive(card, g.dataset.key, g.dataset.label);
        showTooltip(event, tooltipHtml(line, null));
      });
      el.addEventListener('mousemove', moveTooltip);
      el.addEventListener('mouseout', event => {
        if (!g.contains(event.relatedTarget)) clearActive(card);
      });
    });
    g.querySelectorAll('.point').forEach(pointEl => {
      pointEl.addEventListener('mouseover', event => {
        const point = line.points[Number(pointEl.dataset.pointIndex)];
        setActive(card, g.dataset.key, g.dataset.label);
        pointEl.classList.add('active');
        pointEl.setAttribute('r', '6');
        showTooltip(event, tooltipHtml(line, point));
      });
      pointEl.addEventListener('mousemove', moveTooltip);
      pointEl.addEventListener('mouseout', event => {
        pointEl.classList.remove('active');
        pointEl.setAttribute('r', pointEl.dataset.baseR);
        if (!g.contains(event.relatedTarget)) clearActive(card);
      });
    });
  });
}
function renderCharts() {
  document.querySelectorAll('.chart-card').forEach(renderChart);
}
function renderZscoreTable() {
  const s = DATA.zscoreSummary;
  const rows = [
    ['n_sequences_total', s.n_sequences_total],
    ['n_observations_M', s.n_observations_M],
    ['feature_dim_D', s.feature_dim_D],
    ['sigma_eps', fmt(s.sigma_eps, 12)],
    ['sigma_zero_or_small_count', s.sigma_zero_or_small_count],
    ['sigma_min / p01 / p05', fmt(s.sigma_min, 8) + ' / ' + fmt(s.sigma_p01, 8) + ' / ' + fmt(s.sigma_p05, 8)],
    ['sigma_median / p95 / p99 / max', fmt(s.sigma_median, 8) + ' / ' + fmt(s.sigma_p95, 8) + ' / ' + fmt(s.sigma_p99, 8) + ' / ' + fmt(s.sigma_max, 8)],
    ['sigma_safe_min / median / max', fmt(s.sigma_safe_min, 8) + ' / ' + fmt(s.sigma_safe_median, 8) + ' / ' + fmt(s.sigma_safe_max, 8)],
    ['abs_z_min / median / p95 / p99 / max', fmt(s.abs_z_min, 5) + ' / ' + fmt(s.abs_z_median, 5) + ' / ' + fmt(s.abs_z_p95, 5) + ' / ' + fmt(s.abs_z_p99, 5) + ' / ' + fmt(s.abs_z_max, 5)],
    ['clip_k / clip_rate', fmt(s.clip_k, 1) + ' / ' + fmt(s.clip_rate, 8)]
  ];
  document.getElementById('zscoreTable').innerHTML = '<table><tr><th>field</th><th>value</th></tr>' +
    rows.map(r => '<tr><td>' + escapeHtml(r[0]) + '</td><td>' + escapeHtml(r[1]) + '</td></tr>').join('') + '</table>';
}
function renderTable(containerId, rows, columns) {
  document.getElementById(containerId).innerHTML = '<table><tr>' + columns.map(c => '<th>' + escapeHtml(c.label) + '</th>').join('') + '</tr>' +
    rows.map(row => '<tr>' + columns.map(c => {
      const value = row[c.key];
      const shown = typeof value === 'number' ? fmt(value, c.digits ?? 3) : value;
      return '<td>' + escapeHtml(shown ?? '') + '</td>';
    }).join('') + '</tr>').join('') + '</table>';
}
function renderSummaryTables() {
  renderZscoreTable();
  renderTable('foldbackTable', DATA.foldbackMethodBaseline, [
    {key:'metric', label:'metric'},
    {key:'method', label:'method'},
    {key:'baseline_class', label:'baseline'},
    {key:'n_curves', label:'curves', digits:0},
    {key:'total_foldback_count', label:'foldbacks', digits:0},
    {key:'total_severe_foldback_count', label:'severe foldbacks', digits:0},
    {key:'mean_endpoint_nearest_progress', label:'mean endpoint progress', digits:2},
    {key:'mean_endpoint_nearest_distance', label:'mean endpoint distance', digits:5},
    {key:'mean_nearest_distance', label:'mean distance', digits:5}
  ]);
  renderTable('endpointTable', DATA.endpointSummary, [
    {key:'metric', label:'metric'},
    {key:'method', label:'method'},
    {key:'baseline_class', label:'baseline'},
    {key:'target_class', label:'target'},
    {key:'source_type', label:'source'},
    {key:'n_curves', label:'curves', digits:0},
    {key:'mean_nearest_progress', label:'mean progress', digits:2},
    {key:'median_nearest_progress', label:'median progress', digits:2},
    {key:'std_nearest_progress', label:'std progress', digits:2},
    {key:'mean_nearest_distance', label:'mean distance', digits:5},
    {key:'median_nearest_distance', label:'median distance', digits:5},
    {key:'std_nearest_distance', label:'std distance', digits:5}
  ]);
}
document.getElementById('metricSelect').addEventListener('change', renderCharts);
document.querySelectorAll('input').forEach(el => el.addEventListener('change', renderCharts));
document.querySelectorAll('.chart-card').forEach(card => {
  card.addEventListener('mouseleave', () => clearActive(card));
});
renderCharts();
renderSummaryTables();
</script>
</body>
</html>
"""


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate nearest-baseline interactive metric-sensitivity report for raw vs z-score fc7."
    )
    parser.add_argument("--analysis_input_root", default=str(ZscoreNearestBaselineConfig.analysis_input_root))
    parser.add_argument("--projection_output_root", default=str(ZscoreNearestBaselineConfig.projection_output_root))
    parser.add_argument("--existing_nearest_root", default=str(ZscoreNearestBaselineConfig.existing_nearest_root))
    parser.add_argument("--output_root", default=str(ZscoreNearestBaselineConfig.output_root))
    parser.add_argument("--clip-k", type=float, default=ZscoreNearestBaselineConfig.clip_k)
    return parser


def main() -> None:
    cfg = ZscoreNearestBaselineConfig.from_args(build_arg_parser().parse_args())
    result = ZscoreNearestBaselineReport(cfg).run()
    print(f"html: {result['paths']['html']}")
    print(f"point_rows: {result['point_rows']}")
    print(f"curve_count_by_metric: {result['curve_count_by_metric']}")
    print(f"raw_validation_matched: {result['raw_validation']['matched']}")
    print("metric_totals:")
    for row in result["metric_totals"]:
        print(row)


if __name__ == "__main__":
    main()
