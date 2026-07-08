from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


CLASS_NAMES = ("polite", "truesmile", "ambiguous")
METRICS = ("raw", "zscore_all74", "zscore_all74_clip_k3")
TARGET_STAGES = np.arange(5.0, 101.0, 5.0, dtype=np.float64)
SEARCH_PERCENTS = np.arange(0.0, 101.0, 1.0, dtype=np.float64)
EXPECTED_SEQUENCE_COUNTS = {"polite": 41, "truesmile": 6, "ambiguous": 27}
FALLBACK_METHODB_MEDOIDS = {"polite": "13", "truesmile": "2", "ambiguous": "27"}

POINT_COLUMNS = [
    "metric",
    "baseline_class",
    "baseline_seq_id",
    "baseline_source_type",
    "baseline_is_methodB_medoid",
    "target_class",
    "target_seq_id",
    "target_source_type",
    "target_is_same_sequence_as_baseline",
    "target_stage_percent",
    "target_stage_index",
    "nearest_baseline_progress",
    "nearest_distance",
    "nearest_baseline_grid_index",
    "curve_id",
]


@dataclass
class AllSequenceBaselineConfig:
    analysis_input_root: Path = Path(r"E:\Matsuda_data\2-27meeting")
    projection_output_root: Path = Path(r"E:\Matsuda_data\3-10meeting")
    previous_zscore_root: Path = Path(r"E:\Matsuda_data\3-10meeting\nearest_baseline_zscore_interactive")
    output_root: Path = Path(r"E:\Matsuda_data\3-10meeting\nearest_baseline_all_sequence_baselines")
    clip_k: float = 3.0

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "AllSequenceBaselineConfig":
        return cls(
            analysis_input_root=Path(args.analysis_input_root),
            projection_output_root=Path(args.projection_output_root),
            previous_zscore_root=Path(args.previous_zscore_root),
            output_root=Path(args.output_root),
            clip_k=float(args.clip_k),
        )


class AllSequenceBaselineReport:
    def __init__(self, cfg: AllSequenceBaselineConfig):
        self.cfg = cfg
        self.normalized_root = cfg.analysis_input_root / "metrics" / "normalized"
        self.frames_root = cfg.analysis_input_root / "metrics" / "normalized_frames"
        self.csv_dir = cfg.output_root / "csv"
        self.npz_dir = cfg.output_root / "npz"
        self.report_dir = cfg.output_root / "report"
        self.frame_asset_root = cfg.output_root / "assets" / "baseline_frames"
        for path in (self.csv_dir, self.npz_dir, self.report_dir, self.frame_asset_root):
            path.mkdir(parents=True, exist_ok=True)
        self._sequence_cache: dict[tuple[str, str], np.ndarray] = {}
        self._prototype_cache: dict[tuple[str, str], np.ndarray] = {}

    @staticmethod
    def sort_key(value: str) -> tuple[int, int | str]:
        return (0, int(value)) if str(value).isdigit() else (1, str(value))

    @staticmethod
    def require_file(path: Path) -> Path:
        if not path.is_file():
            raise FileNotFoundError(f"Required input file is missing: {path}")
        return path

    @staticmethod
    def point_at_percent(curve: np.ndarray, percent: float) -> np.ndarray:
        pos = (percent / 100.0) * (curve.shape[0] - 1)
        lo = int(np.floor(pos))
        hi = int(np.ceil(pos))
        if lo == hi:
            return curve[lo]
        alpha = pos - lo
        return (1.0 - alpha) * curve[lo] + alpha * curve[hi]

    @staticmethod
    def sampled_points(curve: np.ndarray, percents: np.ndarray) -> np.ndarray:
        return np.vstack([AllSequenceBaselineReport.point_at_percent(curve, float(p)) for p in percents])

    @staticmethod
    def validate_curve_shape(arr: np.ndarray, path: Path) -> None:
        if arr.shape != (20, 4096):
            raise ValueError(f"Expected shape [20, 4096], got {arr.shape}: {path}")

    @staticmethod
    def clean_for_json(value: object) -> object:
        if isinstance(value, dict):
            return {str(k): AllSequenceBaselineReport.clean_for_json(v) for k, v in value.items()}
        if isinstance(value, list):
            return [AllSequenceBaselineReport.clean_for_json(v) for v in value]
        if isinstance(value, tuple):
            return [AllSequenceBaselineReport.clean_for_json(v) for v in value]
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            value = float(value)
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value

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

    def load_methodb_medoids(self) -> dict[str, str]:
        meta_path = self.cfg.projection_output_root / "methodB" / "prototypes" / "projection_meta_methodB.json"
        if not meta_path.is_file():
            return dict(FALLBACK_METHODB_MEDOIDS)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return {class_name: str(meta[class_name]["sequence_id"]) for class_name in CLASS_NAMES}

    def load_all_real_sequences(self) -> list[dict]:
        sequences = []
        for class_name in CLASS_NAMES:
            class_dir = self.normalized_root / class_name
            if not class_dir.is_dir():
                raise FileNotFoundError(f"Missing normalized class directory: {class_dir}")
            seq_dirs = sorted((p for p in class_dir.iterdir() if p.is_dir()), key=lambda p: self.sort_key(p.name))
            if len(seq_dirs) != EXPECTED_SEQUENCE_COUNTS[class_name]:
                raise ValueError(
                    f"Expected {EXPECTED_SEQUENCE_COUNTS[class_name]} {class_name} sequences, found {len(seq_dirs)}"
                )
            for seq_dir in seq_dirs:
                sequences.append(
                    {
                        "class": class_name,
                        "seq_id": seq_dir.name,
                        "curve": self.load_sequence(class_name, seq_dir.name),
                    }
                )
        return sequences

    def build_target_curves(self, real_sequences: list[dict]) -> list[dict]:
        targets = [
            {
                "target_class": item["class"],
                "target_seq_id": item["seq_id"],
                "target_source_type": "sequence",
                "curve": item["curve"],
            }
            for item in real_sequences
        ]
        for method, source_type in (("methodA", "prototype_methodA"), ("methodB", "prototype_methodB")):
            for class_name in CLASS_NAMES:
                targets.append(
                    {
                        "target_class": class_name,
                        "target_seq_id": source_type,
                        "target_source_type": source_type,
                        "curve": self.load_prototype(method, class_name),
                    }
                )
        return targets

    def compute_zscore_params(self, real_sequences: list[dict]) -> tuple[dict, dict[str, np.ndarray]]:
        x = np.vstack([item["curve"] for item in real_sequences])
        mu = np.mean(x, axis=0)
        sigma = np.std(x, axis=0, ddof=0)
        positive = sigma[sigma > 0]
        if positive.size == 0:
            raise RuntimeError("All sigma values are zero; z-score normalization is undefined.")
        eps = max(1e-12, 1e-8 * float(np.median(positive)))
        sigma_safe = np.where(sigma > eps, sigma, eps)
        z = (x - mu) / sigma_safe
        abs_z = np.abs(z)
        summary = {
            "n_sequences_total": len(real_sequences),
            "n_observations_M": int(x.shape[0]),
            "feature_dim_D": int(x.shape[1]),
            "sigma_eps": float(eps),
            "sigma_zero_or_small_count": int(np.sum(sigma <= eps)),
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
            "clip_rate": float(np.mean(abs_z > self.cfg.clip_k)),
        }
        np.savez(
            self.npz_dir / "zscore_params_all74.npz",
            mu=mu,
            sigma=sigma,
            sigma_safe=sigma_safe,
            eps=np.asarray(eps, dtype=np.float64),
            clip_k=np.asarray(self.cfg.clip_k, dtype=np.float64),
        )
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

    def copy_baseline_frames(self, real_sequences: list[dict]) -> list[dict]:
        baseline_assets = []
        for item in real_sequences:
            class_name = item["class"]
            seq_id = item["seq_id"]
            copied = []
            for idx in range(20):
                source = self.require_file(self.frames_root / class_name / seq_id / f"{idx:03d}.png")
                destination = self.frame_asset_root / class_name / seq_id / source.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
                copied.append(destination.relative_to(self.cfg.output_root).as_posix())
            baseline_assets.append({"class": class_name, "seq_id": seq_id, "frames": copied})
        return baseline_assets

    def compute_point_rows(
        self,
        real_sequences: list[dict],
        targets: list[dict],
        methodb_medoids: dict[str, str],
        params: dict[str, np.ndarray],
    ) -> pd.DataFrame:
        target_stage_meta = []
        target_stage_points = {}
        for metric in METRICS:
            stage_points = []
            for target_index, target in enumerate(targets):
                sampled = self.sampled_points(target["curve"], TARGET_STAGES)
                sampled_t = self.transform(sampled, metric, params)
                for stage_index, stage in enumerate(TARGET_STAGES):
                    if metric == METRICS[0]:
                        target_stage_meta.append((target_index, stage_index, float(stage)))
                    stage_points.append(sampled_t[stage_index])
            target_stage_points[metric] = np.vstack(stage_points)

        rows = []
        for metric in METRICS:
            print(f"computing all-sequence baseline points for metric={metric}")
            target_matrix = target_stage_points[metric]
            target_norms = np.einsum("ij,ij->i", target_matrix, target_matrix)
            for baseline_index, baseline in enumerate(real_sequences, start=1):
                if baseline_index % 10 == 0:
                    print(f"  baseline {baseline_index}/{len(real_sequences)}")
                baseline_samples = self.sampled_points(baseline["curve"], SEARCH_PERCENTS)
                baseline_samples_t = self.transform(baseline_samples, metric, params)
                baseline_norms = np.einsum("ij,ij->i", baseline_samples_t, baseline_samples_t)
                d2 = target_norms[:, None] + baseline_norms[None, :] - 2.0 * (target_matrix @ baseline_samples_t.T)
                np.maximum(d2, 0.0, out=d2)
                nearest_indices = np.argmin(d2, axis=1)
                nearest_distances = np.sqrt(d2[np.arange(d2.shape[0]), nearest_indices])

                baseline_class = baseline["class"]
                baseline_seq_id = baseline["seq_id"]
                baseline_is_medoid = baseline_seq_id == methodb_medoids.get(baseline_class)
                use_direct_distance = baseline_is_medoid and baseline_class in {"truesmile", "polite"}
                for flat_index, (target_index, stage_index, stage) in enumerate(target_stage_meta):
                    target = targets[target_index]
                    target_same = (
                        target["target_source_type"] == "sequence"
                        and target["target_class"] == baseline_class
                        and target["target_seq_id"] == baseline_seq_id
                    )
                    curve_id = "|".join(
                        [
                            metric,
                            baseline_class,
                            baseline_seq_id,
                            target["target_class"],
                            target["target_seq_id"],
                            target["target_source_type"],
                        ]
                    )
                    if target_same:
                        nearest_idx = int(stage)
                        nearest_distance = 0.0
                    elif use_direct_distance:
                        direct_diff = baseline_samples_t - target_matrix[flat_index]
                        direct_d2 = np.einsum("ij,ij->i", direct_diff, direct_diff)
                        nearest_idx = int(np.argmin(direct_d2))
                        nearest_distance = float(np.sqrt(direct_d2[nearest_idx]))
                    else:
                        nearest_idx = int(nearest_indices[flat_index])
                        nearest_distance = float(nearest_distances[flat_index])
                    rows.append(
                        {
                            "metric": metric,
                            "baseline_class": baseline_class,
                            "baseline_seq_id": baseline_seq_id,
                            "baseline_source_type": "sequence",
                            "baseline_is_methodB_medoid": bool(baseline_is_medoid),
                            "target_class": target["target_class"],
                            "target_seq_id": target["target_seq_id"],
                            "target_source_type": target["target_source_type"],
                            "target_is_same_sequence_as_baseline": bool(target_same),
                            "target_stage_percent": stage,
                            "target_stage_index": int(stage_index),
                            "nearest_baseline_progress": float(SEARCH_PERCENTS[nearest_idx]),
                            "nearest_distance": nearest_distance,
                            "nearest_baseline_grid_index": nearest_idx,
                            "curve_id": curve_id,
                        }
                    )
        return pd.DataFrame(rows, columns=POINT_COLUMNS)

    @staticmethod
    def per_curve_summary(points: pd.DataFrame) -> pd.DataFrame:
        group_cols = [
            "metric",
            "baseline_class",
            "baseline_seq_id",
            "target_class",
            "target_seq_id",
            "target_source_type",
            "curve_id",
        ]
        rows = []
        for key, group in points.groupby(group_cols, sort=True):
            ordered = group.sort_values("target_stage_index")
            progress = ordered["nearest_baseline_progress"].to_numpy(dtype=np.float64)
            distance = ordered["nearest_distance"].to_numpy(dtype=np.float64)
            stages = ordered["target_stage_percent"].to_numpy(dtype=np.float64)
            diff = np.diff(progress)
            jumps = -diff[diff < 0.0]
            rows.append(
                dict(
                    zip(group_cols, key),
                    foldback_count=int(np.sum(diff < 0.0)),
                    severe_foldback_count=int(np.sum(diff <= -10.0)),
                    mean_negative_jump_size=float(np.mean(jumps)) if jumps.size else 0.0,
                    max_negative_jump_size=float(np.max(jumps)) if jumps.size else 0.0,
                    endpoint_nearest_progress=float(progress[-1]),
                    endpoint_nearest_distance=float(distance[-1]),
                    max_nearest_progress=float(np.max(progress)),
                    mean_nearest_progress=float(np.mean(progress)),
                    mean_nearest_distance=float(np.mean(distance)),
                    distance_auc_over_target_stage=float(np.trapezoid(distance, stages)),
                    progress_auc_over_target_stage=float(np.trapezoid(progress, stages)),
                    is_late_endpoint_70=bool(progress[-1] >= 70.0),
                    is_late_endpoint_80=bool(progress[-1] >= 80.0),
                    is_late_max_70=bool(np.max(progress) >= 70.0),
                    is_late_max_80=bool(np.max(progress) >= 80.0),
                    target_is_same_sequence_as_baseline=bool(group["target_is_same_sequence_as_baseline"].iloc[0]),
                    baseline_is_methodB_medoid=bool(group["baseline_is_methodB_medoid"].iloc[0]),
                )
            )
        return pd.DataFrame(rows)

    @staticmethod
    def class_target_summary(per_curve: pd.DataFrame) -> pd.DataFrame:
        group_cols = ["metric", "baseline_class", "target_class", "target_source_type"]
        source = per_curve[
            ~(
                (per_curve["target_source_type"] == "sequence")
                & (per_curve["target_is_same_sequence_as_baseline"])
            )
        ].copy()
        rows = []
        for key, group in source.groupby(group_cols, sort=True):
            endpoint = group["endpoint_nearest_progress"].to_numpy(dtype=np.float64)
            distance = group["endpoint_nearest_distance"].to_numpy(dtype=np.float64)
            rows.append(
                dict(
                    zip(group_cols, key),
                    n_baselines=int(group["baseline_seq_id"].nunique()),
                    n_target_curves=int(
                        group[["target_class", "target_seq_id", "target_source_type"]].drop_duplicates().shape[0]
                    ),
                    n_pair_curves=int(len(group)),
                    mean_endpoint_progress=float(np.mean(endpoint)),
                    median_endpoint_progress=float(np.median(endpoint)),
                    std_endpoint_progress=float(np.std(endpoint, ddof=0)),
                    p25_endpoint_progress=float(np.percentile(endpoint, 25)),
                    p75_endpoint_progress=float(np.percentile(endpoint, 75)),
                    mean_endpoint_distance=float(np.mean(distance)),
                    median_endpoint_distance=float(np.median(distance)),
                    std_endpoint_distance=float(np.std(distance, ddof=0)),
                    late_endpoint_70_count=int(group["is_late_endpoint_70"].sum()),
                    late_endpoint_70_rate=float(group["is_late_endpoint_70"].mean()),
                    late_endpoint_80_count=int(group["is_late_endpoint_80"].sum()),
                    late_endpoint_80_rate=float(group["is_late_endpoint_80"].mean()),
                    late_max_70_count=int(group["is_late_max_70"].sum()),
                    late_max_70_rate=float(group["is_late_max_70"].mean()),
                    late_max_80_count=int(group["is_late_max_80"].sum()),
                    late_max_80_rate=float(group["is_late_max_80"].mean()),
                    mean_foldback_count=float(group["foldback_count"].mean()),
                    mean_severe_foldback_count=float(group["severe_foldback_count"].mean()),
                )
            )
        return pd.DataFrame(rows)

    @staticmethod
    def baseline_sequence_summary(per_curve: pd.DataFrame) -> pd.DataFrame:
        group_cols = ["metric", "baseline_class", "baseline_seq_id", "target_class", "target_source_type"]
        rows = []
        for key, group in per_curve.groupby(group_cols, sort=True):
            endpoint = group["endpoint_nearest_progress"].to_numpy(dtype=np.float64)
            distance = group["endpoint_nearest_distance"].to_numpy(dtype=np.float64)
            rows.append(
                dict(
                    zip(group_cols, key),
                    n_target_curves=int(len(group)),
                    mean_endpoint_progress=float(np.mean(endpoint)),
                    median_endpoint_progress=float(np.median(endpoint)),
                    std_endpoint_progress=float(np.std(endpoint, ddof=0)),
                    late_endpoint_70_rate=float(group["is_late_endpoint_70"].mean()),
                    late_endpoint_80_rate=float(group["is_late_endpoint_80"].mean()),
                    mean_endpoint_distance=float(np.mean(distance)),
                    median_endpoint_distance=float(np.median(distance)),
                    mean_foldback_count=float(group["foldback_count"].mean()),
                    mean_severe_foldback_count=float(group["severe_foldback_count"].mean()),
                    baseline_is_methodB_medoid=bool(group["baseline_is_methodB_medoid"].iloc[0]),
                )
            )
        return pd.DataFrame(rows)

    @staticmethod
    def percentile_rank(values: np.ndarray, value: float) -> float:
        if values.size == 0 or not np.isfinite(value):
            return float("nan")
        return float(np.mean(values <= value) * 100.0)

    def medoid_vs_all_summary(self, per_curve: pd.DataFrame, methodb_medoids: dict[str, str]) -> pd.DataFrame:
        rows = []
        source = per_curve[
            ~(
                (per_curve["target_source_type"] == "sequence")
                & (per_curve["target_is_same_sequence_as_baseline"])
            )
        ].copy()
        group_cols = ["metric", "baseline_class", "target_class", "target_source_type"]
        for key, group in source.groupby(group_cols, sort=True):
            metric, baseline_class, target_class, target_source_type = key
            medoid_seq = methodb_medoids[baseline_class]
            baseline_stats = (
                group.groupby("baseline_seq_id", as_index=False)
                .agg(
                    mean_endpoint_progress=("endpoint_nearest_progress", "mean"),
                    mean_endpoint_distance=("endpoint_nearest_distance", "mean"),
                )
                .copy()
            )
            medoid_row = baseline_stats[baseline_stats["baseline_seq_id"].astype(str) == str(medoid_seq)]
            if medoid_row.empty:
                medoid_progress = float("nan")
                medoid_distance = float("nan")
            else:
                medoid_progress = float(medoid_row["mean_endpoint_progress"].iloc[0])
                medoid_distance = float(medoid_row["mean_endpoint_distance"].iloc[0])
            all_progress = baseline_stats["mean_endpoint_progress"].to_numpy(dtype=np.float64)
            all_distance = baseline_stats["mean_endpoint_distance"].to_numpy(dtype=np.float64)
            rows.append(
                {
                    "metric": metric,
                    "baseline_class": baseline_class,
                    "target_class": target_class,
                    "target_source_type": target_source_type,
                    "methodB_medoid_seq_id": medoid_seq,
                    "medoid_mean_endpoint_progress": medoid_progress,
                    "all_baselines_mean_endpoint_progress": float(np.mean(all_progress)),
                    "all_baselines_median_endpoint_progress": float(np.median(all_progress)),
                    "all_baselines_p25_endpoint_progress": float(np.percentile(all_progress, 25)),
                    "all_baselines_p75_endpoint_progress": float(np.percentile(all_progress, 75)),
                    "medoid_percentile_endpoint_progress": self.percentile_rank(all_progress, medoid_progress),
                    "medoid_late_endpoint_70": bool(medoid_progress >= 70.0),
                    "medoid_late_endpoint_80": bool(medoid_progress >= 80.0),
                    "all_baselines_late_endpoint_70_rate": float(group["is_late_endpoint_70"].mean()),
                    "all_baselines_late_endpoint_80_rate": float(group["is_late_endpoint_80"].mean()),
                    "medoid_mean_endpoint_distance": medoid_distance,
                    "all_baselines_median_endpoint_distance": float(np.median(all_distance)),
                    "medoid_percentile_endpoint_distance": self.percentile_rank(all_distance, medoid_distance),
                }
            )
        return pd.DataFrame(rows)

    @staticmethod
    def self_pair_sanity(points: pd.DataFrame) -> pd.DataFrame:
        rows = []
        source = points[points["target_is_same_sequence_as_baseline"]].copy()
        for metric, group in source.groupby("metric", sort=True):
            progress_error = np.abs(
                group["nearest_baseline_progress"].to_numpy(dtype=np.float64)
                - group["target_stage_percent"].to_numpy(dtype=np.float64)
            )
            max_progress_error = float(np.max(progress_error)) if progress_error.size else float("nan")
            max_distance = float(group["nearest_distance"].max()) if len(group) else float("nan")
            rows.append(
                {
                    "metric": metric,
                    "n_self_curves": int(group["curve_id"].nunique()),
                    "n_self_points": int(len(group)),
                    "max_abs_progress_error": max_progress_error,
                    "max_distance": max_distance,
                    "passed": bool(max_progress_error <= 1e-9 and max_distance <= 1e-9),
                }
            )
        return pd.DataFrame(rows)

    def compatibility_validation(self, points: pd.DataFrame, methodb_medoids: dict[str, str]) -> pd.DataFrame:
        previous_path = self.require_file(
            self.cfg.previous_zscore_root / "csv" / "nearest_baseline_points_all_metrics.csv"
        )
        previous = pd.read_csv(previous_path, dtype={"seq_id": str, "rank": str})
        nearest6 = pd.read_csv(
            self.cfg.projection_output_root
            / "linear_axis_extension"
            / "csv"
            / "nearest6_to_prototype_sequences_methodB.csv",
            dtype={"sequence_id": str, "rank": str},
        )

        checks = []
        for baseline_class in ("truesmile", "polite"):
            medoid_seq = methodb_medoids[baseline_class]
            subset = points[
                (points["baseline_class"] == baseline_class)
                & (points["baseline_seq_id"].astype(str) == str(medoid_seq))
            ].copy()

            proto = subset[subset["target_source_type"] == "prototype_methodB"].copy()
            proto["prev_source_type"] = "prototype"
            proto["prev_seq_id"] = "prototype"
            checks.append(proto)

            real = subset[subset["target_source_type"] == "sequence"].copy()
            selected = nearest6[["class", "sequence_id", "rank"]].rename(
                columns={"class": "target_class", "sequence_id": "target_seq_id"}
            )
            real = real.merge(selected, on=["target_class", "target_seq_id"], how="inner")
            real["prev_source_type"] = "nearest6"
            real["prev_seq_id"] = real["target_seq_id"]
            checks.append(real)

        current = pd.concat(checks, ignore_index=True)
        previous_keyed = previous[
            (previous["method"] == "methodB") & (previous["baseline_class"].isin(["truesmile", "polite"]))
        ].rename(
            columns={
                "source_type": "prev_source_type",
                "seq_id": "prev_seq_id",
                "nearest_baseline_progress": "prev_progress",
                "nearest_distance": "prev_distance",
            }
        )
        merge_cols = [
            "metric",
            "baseline_class",
            "target_class",
            "prev_source_type",
            "prev_seq_id",
            "target_stage_percent",
        ]
        merged = current.merge(previous_keyed[merge_cols + ["prev_progress", "prev_distance"]], on=merge_cols, how="left")
        missing = merged["prev_progress"].isna()
        progress_diff = np.abs(merged["nearest_baseline_progress"] - merged["prev_progress"])
        distance_diff = np.abs(merged["nearest_distance"] - merged["prev_distance"])
        passed = bool((not missing.any()) and (progress_diff.max() <= 1e-9) and (distance_diff.max() <= 1e-9))
        return pd.DataFrame(
            [
                {
                    "n_checked": int(len(merged)),
                    "n_matched": int(np.sum((~missing) & (progress_diff <= 1e-9) & (distance_diff <= 1e-9))),
                    "n_mismatched": int(len(merged) - np.sum((~missing) & (progress_diff <= 1e-9) & (distance_diff <= 1e-9))),
                    "missing_previous_rows": int(missing.sum()),
                    "max_progress_diff": float(progress_diff.max(skipna=True)),
                    "max_distance_diff": float(distance_diff.max(skipna=True)),
                    "passed": passed,
                }
            ]
        )

    @staticmethod
    def records(df: pd.DataFrame) -> list[dict]:
        return df.replace({np.nan: None}).to_dict(orient="records")

    def html_payload(
        self,
        points: pd.DataFrame,
        per_curve: pd.DataFrame,
        class_target: pd.DataFrame,
        medoid_summary: pd.DataFrame,
        self_sanity: pd.DataFrame,
        zscore_summary: dict,
        baseline_assets: list[dict],
    ) -> dict:
        summary_by_curve = per_curve.set_index("curve_id")
        lines = []
        for curve_id, group in points.groupby("curve_id", sort=True):
            ordered = group.sort_values("target_stage_index")
            first = ordered.iloc[0]
            summary = summary_by_curve.loc[curve_id]
            lines.append(
                {
                    "id": curve_id,
                    "m": first["metric"],
                    "bc": first["baseline_class"],
                    "bs": str(first["baseline_seq_id"]),
                    "bm": bool(first["baseline_is_methodB_medoid"]),
                    "tc": first["target_class"],
                    "ts": str(first["target_seq_id"]),
                    "tt": first["target_source_type"],
                    "same": bool(first["target_is_same_sequence_as_baseline"]),
                    "fb": int(summary["foldback_count"]),
                    "sfb": int(summary["severe_foldback_count"]),
                    "ep": float(summary["endpoint_nearest_progress"]),
                    "ed": float(summary["endpoint_nearest_distance"]),
                    "p": [
                        [float(row.nearest_baseline_progress), float(row.nearest_distance)]
                        for row in ordered.itertuples(index=False)
                    ],
                }
            )
        return {
            "baselines": baseline_assets,
            "lines": lines,
            "classTargetSummary": self.records(class_target),
            "medoidSummary": self.records(medoid_summary),
            "selfPairSanity": self.records(self_sanity),
            "zscoreSummary": zscore_summary,
            "classes": list(CLASS_NAMES),
            "metrics": list(METRICS),
            "targetStages": [float(x) for x in TARGET_STAGES],
        }

    def render_html(
        self,
        points: pd.DataFrame,
        per_curve: pd.DataFrame,
        class_target: pd.DataFrame,
        medoid_summary: pd.DataFrame,
        self_sanity: pd.DataFrame,
        zscore_summary: dict,
        baseline_assets: list[dict],
    ) -> str:
        payload = self.html_payload(points, per_curve, class_target, medoid_summary, self_sanity, zscore_summary, baseline_assets)
        data_json = json.dumps(self.clean_for_json(payload), ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        return HTML_TEMPLATE.replace("__DATA_JSON__", data_json)

    def write_run_summary(
        self,
        zscore_summary: dict,
        points: pd.DataFrame,
        per_curve: pd.DataFrame,
        class_target: pd.DataFrame,
        medoid_summary: pd.DataFrame,
        self_sanity: pd.DataFrame,
        compatibility: pd.DataFrame,
        paths: dict[str, Path],
    ) -> None:
        expected_points = 74 * 80 * 3 * 20
        expected_curves = 74 * 80 * 3
        actual_curves = int(points["curve_id"].nunique())
        late_sequence = class_target[class_target["target_source_type"] == "sequence"][
            ["metric", "baseline_class", "target_class", "late_endpoint_70_rate", "late_endpoint_80_rate"]
        ]
        lines = [
            "# All-Sequence Baseline Nearest-Baseline Sensitivity Run Summary",
            "",
            f"Timestamp: {datetime.now().isoformat(timespec='seconds')}",
            f"Script path: `{Path(__file__).resolve()}`",
            f"Output root: `{self.cfg.output_root}`",
            "",
            "Input roots:",
            "",
            f"- Normalized sequences: `{self.normalized_root}`",
            f"- Normalized frames: `{self.frames_root}`",
            f"- Projection outputs: `{self.cfg.projection_output_root}`",
            f"- Previous z-score report: `{self.cfg.previous_zscore_root}`",
            "",
            f"Number of baseline sequences: 74",
            f"Number of target curves: 80",
            f"Metric names: {', '.join(METRICS)}",
            f"Expected point rows: {expected_points}",
            f"Actual point rows: {len(points)}",
            f"Expected curve count: {expected_curves}",
            f"Actual curve count: {actual_curves}",
            "",
            "Z-score status:",
            "",
            f"- M: {zscore_summary['n_observations_M']}",
            f"- D: {zscore_summary['feature_dim_D']}",
            f"- sigma eps: {zscore_summary['sigma_eps']:.12g}",
            f"- sigma <= eps count: {zscore_summary['sigma_zero_or_small_count']}",
            f"- sigma min/median/max: {zscore_summary['sigma_min']:.12g} / {zscore_summary['sigma_median']:.12g} / {zscore_summary['sigma_max']:.12g}",
            f"- clip rate: {zscore_summary['clip_rate']:.8f}",
            "",
            "Validation:",
            "",
            f"- Self-pair sanity passed: {bool(self_sanity['passed'].all())}",
            f"- Compatibility validation passed: {bool(compatibility['passed'].iloc[0])}",
            f"- HTML path: `{paths['html']}`",
            "",
            "CSV paths:",
            "",
        ]
        for name, path in paths.items():
            if name != "html":
                lines.append(f"- {name}: `{path}`")
        lines.extend(
            [
                "",
                "Late endpoint >=70 rates for sequence targets:",
                "",
                "| metric | baseline_class | target_class | late>=70 | late>=80 |",
                "|---|---|---|---:|---:|",
            ]
        )
        for row in late_sequence.itertuples(index=False):
            lines.append(
                f"| {row.metric} | {row.baseline_class} | {row.target_class} | "
                f"{row.late_endpoint_70_rate:.3f} | {row.late_endpoint_80_rate:.3f} |"
            )
        atypical = medoid_summary.copy()
        atypical["distance_from_median"] = np.abs(
            atypical["medoid_mean_endpoint_progress"] - atypical["all_baselines_median_endpoint_progress"]
        )
        top_atypical = atypical.sort_values("distance_from_median", ascending=False).head(8)
        lines.extend(
            [
                "",
                "Method B medoid typicality notes:",
                "",
                "- Use `methodB_medoid_vs_all_baselines_summary.csv` for the full table.",
                "- Largest absolute differences from all-baseline median endpoint progress:",
                "",
                "| metric | baseline | target | source | medoid progress | all median | medoid percentile |",
                "|---|---|---|---|---:|---:|---:|",
            ]
        )
        for row in top_atypical.itertuples(index=False):
            lines.append(
                f"| {row.metric} | {row.baseline_class} | {row.target_class} | {row.target_source_type} | "
                f"{row.medoid_mean_endpoint_progress:.3f} | {row.all_baselines_median_endpoint_progress:.3f} | "
                f"{row.medoid_percentile_endpoint_progress:.1f} |"
            )
        lines.extend(
            [
                "",
                "Short interpretation:",
                "",
                "- This report measures baseline sensitivity: how much nearest progress changes when the baseline is every real sequence instead of one medoid.",
                "- Raw and transformed distances are not comparable in absolute units; compare endpoint progress, late-progress rates, and foldback behavior.",
                "- Strong differences between medoid rows and all-baseline distributions indicate that the Method B medoid is not typical for that target/baseline condition.",
            ]
        )
        paths["run_summary"].write_text("\n".join(lines), encoding="utf-8")

    def run(self) -> dict[str, object]:
        print("loading real sequences")
        real_sequences = self.load_all_real_sequences()
        targets = self.build_target_curves(real_sequences)
        methodb_medoids = self.load_methodb_medoids()
        print(f"baseline sequences={len(real_sequences)}, target curves={len(targets)}")

        print("computing z-score parameters")
        zscore_summary, params = self.compute_zscore_params(real_sequences)
        pd.DataFrame([zscore_summary]).to_csv(self.csv_dir / "zscore_parameter_summary.csv", index=False)

        print("copying baseline frame strips")
        baseline_assets = self.copy_baseline_frames(real_sequences)

        points = self.compute_point_rows(real_sequences, targets, methodb_medoids, params)
        points_path = self.csv_dir / "all_sequence_baseline_points_all_metrics.csv"
        points.to_csv(points_path, index=False)

        per_curve = self.per_curve_summary(points)
        per_curve_path = self.csv_dir / "all_sequence_baseline_per_curve_summary.csv"
        per_curve.to_csv(per_curve_path, index=False)

        class_target = self.class_target_summary(per_curve)
        class_target_path = self.csv_dir / "baseline_class_target_class_summary_by_metric.csv"
        class_target.to_csv(class_target_path, index=False)

        baseline_sequence = self.baseline_sequence_summary(per_curve)
        baseline_sequence_path = self.csv_dir / "baseline_sequence_summary_by_metric.csv"
        baseline_sequence.to_csv(baseline_sequence_path, index=False)

        medoid_summary = self.medoid_vs_all_summary(per_curve, methodb_medoids)
        medoid_summary_path = self.csv_dir / "methodB_medoid_vs_all_baselines_summary.csv"
        medoid_summary.to_csv(medoid_summary_path, index=False)

        self_sanity = self.self_pair_sanity(points)
        self_sanity_path = self.csv_dir / "self_pair_sanity_check.csv"
        self_sanity.to_csv(self_sanity_path, index=False)

        compatibility = self.compatibility_validation(points, methodb_medoids)
        compatibility_path = self.csv_dir / "compatibility_validation_against_previous_zscore_report.csv"
        compatibility.to_csv(compatibility_path, index=False)

        print("writing standalone HTML")
        html_path = self.cfg.output_root / "all_sequence_baseline_interactive_report.html"
        html_path.write_text(
            self.render_html(points, per_curve, class_target, medoid_summary, self_sanity, zscore_summary, baseline_assets),
            encoding="utf-8",
        )

        paths = {
            "points": points_path,
            "per_curve": per_curve_path,
            "class_target": class_target_path,
            "baseline_sequence": baseline_sequence_path,
            "medoid_summary": medoid_summary_path,
            "self_pair": self_sanity_path,
            "compatibility": compatibility_path,
            "zscore_summary": self.csv_dir / "zscore_parameter_summary.csv",
            "zscore_npz": self.npz_dir / "zscore_params_all74.npz",
            "html": html_path,
            "run_summary": self.report_dir / "run_summary.md",
        }
        self.write_run_summary(
            zscore_summary,
            points,
            per_curve,
            class_target,
            medoid_summary,
            self_sanity,
            compatibility,
            paths,
        )
        return {
            "paths": paths,
            "point_rows": int(len(points)),
            "curve_count": int(points["curve_id"].nunique()),
            "self_pair_passed": bool(self_sanity["passed"].all()),
            "compatibility_passed": bool(compatibility["passed"].iloc[0]),
            "class_target": class_target,
            "medoid_summary": medoid_summary,
        }


HTML_TEMPLATE = r"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>All-Sequence Baseline Nearest-Baseline Sensitivity</title>
<style>
body{font-family:Arial,sans-serif;margin:28px;color:#222;background:#fafafa;line-height:1.55}
main{max-width:1400px;margin:0 auto}
h1{font-size:29px;margin:0 0 8px}
h2{font-size:21px;margin:30px 0 10px;border-bottom:1px solid #ddd;padding-bottom:6px}
p{max-width:1120px}
.panel{background:white;border:1px solid #ddd;padding:15px 17px;margin:16px 0}
.note{background:#fff7df;border-left:4px solid #d99b00;padding:10px 12px;margin:12px 0}
.formula{font-family:Consolas,monospace;background:#f6f6f6;border:1px solid #ddd;padding:10px 12px;white-space:pre-wrap}
.controls{position:sticky;top:0;z-index:10;background:#fafafa;border-bottom:1px solid #ddd;padding:10px 0;margin:14px 0 16px;display:flex;flex-wrap:wrap;gap:12px;align-items:center}
label{font-size:13px;display:inline-flex;gap:6px;align-items:center}
select,input{font-size:13px}
.frame-strip{display:grid;grid-template-columns:repeat(20,minmax(42px,1fr));gap:5px;margin:10px 0}
.frame-strip img{width:100%;height:58px;object-fit:contain;background:#eee;border:1px solid #ddd}
.frame-strip span{font-size:10px;color:#555;text-align:center;display:block}
.chart-card{background:#fff;border:1px solid #ddd;padding:14px 16px;margin:0 0 22px}
.chart-title{font-weight:700;margin-bottom:4px}
.chart-caption{font-size:13px;color:#555;margin-bottom:7px}
.chart-svg{width:100%;height:auto;display:block;background:#fff}
.axis{stroke:#222;stroke-width:1.2}
.grid{stroke:#ddd;stroke-width:1}
.tick{font-size:12px;fill:#555}
.label{font-size:13px;fill:#333}
.curve{fill:none;stroke-width:2.0;opacity:.52;vector-effect:non-scaling-stroke;pointer-events:stroke;cursor:pointer}
.curve.proto{stroke-width:3.1;opacity:.95;stroke-dasharray:7 5}
.curve-hit{fill:none;stroke:#000;stroke-opacity:.001;stroke-width:12;pointer-events:stroke;cursor:pointer}
.point{opacity:.35;cursor:pointer}
.chart-card.hovering .curve{opacity:.04;stroke-width:1}
.chart-card.hovering .point{opacity:.08}
.chart-card.hovering .curve.active{opacity:1;stroke-width:4}
.chart-card.hovering .point.active{opacity:1}
.scatter{opacity:.48;cursor:pointer}
.scatter.medoid{stroke:#111;stroke-width:2;opacity:1}
.legend{display:flex;flex-wrap:wrap;gap:12px 18px;font-size:13px;color:#444;margin-top:8px}
.legend span{display:inline-flex;align-items:center;gap:6px}
.dot{width:12px;height:12px;border-radius:50%;display:inline-block}
.active-label{font-size:13px;background:#f5f5f5;border:1px solid #ddd;padding:7px 9px;margin-top:9px;min-height:18px}
.tooltip{position:fixed;display:none;z-index:50;max-width:390px;background:#222;color:white;padding:8px 9px;font-size:12px;line-height:1.35;pointer-events:none}
table{border-collapse:collapse;width:100%;font-size:13px;background:#fff;margin:10px 0 18px}
th,td{border:1px solid #ddd;padding:6px 7px;text-align:right}
th{background:#f0f0f0}
th:first-child,td:first-child,th:nth-child(2),td:nth-child(2),th:nth-child(3),td:nth-child(3),th:nth-child(4),td:nth-child(4){text-align:left}
.table-wrap{overflow-x:auto;max-height:520px}
@media(max-width:900px){.controls{position:static}.frame-strip{grid-template-columns:repeat(5,1fr)}}
</style>
</head>
<body>
<main>
<h1>All-Sequence Baseline Nearest-Baseline Sensitivity</h1>
<section class="panel">
<h2>Method Summary</h2>
<p>Every real normalized sequence is used once as a baseline. Target curves include all real sequences plus Method A and Method B class prototypes. The same three metrics are used: raw fc7, global feature-wise z-score, and signed clipped z-score.</p>
<div class="formula">u*(t) = argmin_u || T(C_target(t)) - T(B_sequence(u)) ||_2
new curve point = (nearest baseline progress u*(t), nearest distance)
target stages = 5%, 10%, ..., 100%
baseline search grid = 0%, 1%, ..., 100%</div>
<p>Z-score parameters are estimated globally from all 74 real normalized trajectories. Nearest progress is a metric-dependent correspondence measure, not a true physical time variable.</p>
<div class="note">Raw and transformed distances are not directly comparable in absolute units. Compare endpoint progress, foldback behavior, late-progress rates, and baseline sensitivity.</div>
</section>

<section class="panel">
<h2>Global Controls</h2>
<div class="controls">
<label>metric <select id="metricSelect"></select></label>
<label>baseline class <select id="baselineClassSelect"></select></label>
<label>baseline sequence <select id="baselineSeqSelect"></select></label>
<label><input type="checkbox" data-target="polite" checked> polite</label>
<label><input type="checkbox" data-target="truesmile" checked> truesmile</label>
<label><input type="checkbox" data-target="ambiguous" checked> ambiguous</label>
<label><input type="checkbox" data-source="sequence" checked> sequence</label>
<label><input type="checkbox" data-source="prototype_methodA" checked> prototype_methodA</label>
<label><input type="checkbox" data-source="prototype_methodB" checked> prototype_methodB</label>
<label><input type="checkbox" id="excludeSelf" checked> exclude self-pairs</label>
<label><input type="checkbox" id="lateOnly"> endpoint >= 70 only</label>
</div>
</section>

<section class="panel">
<h2>Baseline Frame Strip</h2>
<div id="frameStrip" class="frame-strip"></div>
</section>

<section class="chart-card" id="mainChartCard">
<div class="chart-title">Selected baseline new-curve plot</div>
<div class="chart-caption" id="mainCaption"></div>
<div id="mainChart"></div>
<div class="legend"><span><i class="dot" style="background:#1f77b4"></i>polite</span><span><i class="dot" style="background:#2ca02c"></i>truesmile</span><span><i class="dot" style="background:#ff7f0e"></i>ambiguous</span><span>solid = sequence</span><span>dashed = prototype target</span></div>
<div class="active-label">No curve selected</div>
</section>

<section class="chart-card" id="endpointChartCard">
<div class="chart-title">Endpoint distribution across baseline sequences</div>
<div class="chart-caption" id="endpointCaption"></div>
<div id="endpointChart"></div>
<div class="legend"><span><i class="dot" style="background:#1f77b4"></i>polite target</span><span><i class="dot" style="background:#2ca02c"></i>truesmile target</span><span><i class="dot" style="background:#ff7f0e"></i>ambiguous target</span><span>outlined = Method B medoid baseline</span></div>
</section>

<section class="panel">
<h2>Late-Progress Summary Tables</h2>
<h3>Baseline Class / Target Class Summary</h3>
<div class="table-wrap" id="classTargetTable"></div>
<h3>Method B Medoid vs All Baselines</h3>
<div class="table-wrap" id="medoidTable"></div>
<h3>Self-Pair Sanity Check</h3>
<div class="table-wrap" id="selfPairTable"></div>
<h3>Z-Score Parameter Summary</h3>
<div class="table-wrap" id="zscoreTable"></div>
</section>
</main>
<div class="tooltip" id="tooltip"></div>
<script>
const DATA = __DATA_JSON__;
const COLORS = {polite:'#1f77b4', truesmile:'#2ca02c', ambiguous:'#ff7f0e'};
const STAGES = DATA.targetStages;
function fmt(v,d=3){const n=Number(v); return Number.isFinite(n)?n.toFixed(d):'N/A';}
function esc(v){return String(v).replace(/[&<>"']/g,ch=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));}
function selectedValues(selector, attr){return new Set(Array.from(document.querySelectorAll(selector+':checked')).map(el=>el.dataset[attr]));}
function currentMetric(){return document.getElementById('metricSelect').value;}
function currentBaselineClass(){return document.getElementById('baselineClassSelect').value;}
function currentBaselineSeq(){return document.getElementById('baselineSeqSelect').value;}
function passesFilters(line){
  if(line.m!==currentMetric()||line.bc!==currentBaselineClass()) return false;
  if(!selectedValues('[data-target]','target').has(line.tc)) return false;
  if(!selectedValues('[data-source]','source').has(line.tt)) return false;
  if(document.getElementById('excludeSelf').checked && line.same) return false;
  if(document.getElementById('lateOnly').checked && line.ep < 70) return false;
  return true;
}
function selectedBaselineLines(){
  return DATA.lines.filter(line=>passesFilters(line)&&line.bs===currentBaselineSeq());
}
function classBaselines(cls){return DATA.baselines.filter(b=>b.class===cls).sort((a,b)=>Number(a.seq_id)-Number(b.seq_id));}
function initControls(){
  DATA.metrics.forEach(m=>document.getElementById('metricSelect').add(new Option(m,m)));
  DATA.classes.forEach(c=>document.getElementById('baselineClassSelect').add(new Option(c,c)));
  document.getElementById('baselineClassSelect').value='truesmile';
  updateBaselineSeqOptions();
}
function updateBaselineSeqOptions(){
  const select=document.getElementById('baselineSeqSelect');
  const prev=select.value;
  select.innerHTML='';
  classBaselines(currentBaselineClass()).forEach(b=>select.add(new Option(b.seq_id,b.seq_id)));
  if(Array.from(select.options).some(o=>o.value===prev)) select.value=prev;
  renderAll();
}
function renderFrameStrip(){
  const b=DATA.baselines.find(item=>item.class===currentBaselineClass()&&item.seq_id===currentBaselineSeq());
  document.getElementById('frameStrip').innerHTML=b.frames.map((src,i)=>'<div><img src="'+esc(src)+'" alt="frame '+i+'"><span>'+String(i).padStart(3,'0')+'</span></div>').join('');
}
function pathFromPoints(points,xs,ys){return points.map((p,i)=>(i?'L':'M')+xs(p[0]).toFixed(2)+','+ys(p[1]).toFixed(2)).join(' ');}
function showTip(event,html){const t=document.getElementById('tooltip'); t.innerHTML=html; t.style.display='block'; moveTip(event);}
function moveTip(event){const t=document.getElementById('tooltip'); t.style.left=Math.min(event.clientX+14,window.innerWidth-410)+'px'; t.style.top=(event.clientY+14)+'px';}
function hideTip(){document.getElementById('tooltip').style.display='none';}
function clearActive(card){card.classList.remove('hovering'); card.querySelectorAll('.active').forEach(e=>e.classList.remove('active')); card.querySelector('.active-label').textContent='No curve selected'; hideTip();}
function setActive(card,g,line){card.classList.add('hovering'); card.querySelectorAll('.active').forEach(e=>e.classList.remove('active')); g.querySelectorAll('[data-role="curve"]').forEach(e=>e.classList.add('active')); g.parentNode.appendChild(g); card.querySelector('.active-label').textContent=line.id;}
function tipHtml(line,pointIndex=null){
  const rows=[['metric',line.m],['baseline_class',line.bc],['baseline_seq_id',line.bs],['baseline_is_methodB_medoid',line.bm],['target_class',line.tc],['target_seq_id',line.ts],['target_source_type',line.tt],['foldback_count',line.fb],['endpoint_nearest_progress',fmt(line.ep,2)]];
  if(pointIndex!==null){rows.push(['target_stage_percent',STAGES[pointIndex]+'%']); rows.push(['nearest_baseline_progress',fmt(line.p[pointIndex][0],2)+'%']); rows.push(['nearest_distance',fmt(line.p[pointIndex][1],5)]);}
  return rows.map(r=>'<b>'+esc(r[0])+':</b> '+esc(r[1])).join('<br>');
}
function renderMainChart(){
  const lines=selectedBaselineLines();
  document.getElementById('mainCaption').textContent='metric='+currentMetric()+', baseline='+currentBaselineClass()+' seq='+currentBaselineSeq()+', visible curves='+lines.length;
  const width=1180,height=520,left=72,right=28,top=34,bottom=58,plotW=width-left-right,plotH=height-top-bottom;
  const yMax=Math.max(1,...lines.flatMap(l=>l.p.map(p=>p[1])))*1.08;
  const xs=x=>left+(x/100)*plotW, ys=y=>top+plotH-(y/yMax)*plotH;
  let svg='<svg class="chart-svg" viewBox="0 0 '+width+' '+height+'">';
  [0,20,40,60,80,100].forEach(t=>{const x=xs(t); svg+='<line class="grid" x1="'+x+'" y1="'+top+'" x2="'+x+'" y2="'+(top+plotH)+'"/><text class="tick" x="'+x+'" y="'+(top+plotH+23)+'" text-anchor="middle">'+t+'</text>';});
  [0,yMax/4,yMax/2,yMax*3/4,yMax].forEach(t=>{const y=ys(t); svg+='<line class="grid" x1="'+left+'" y1="'+y+'" x2="'+(left+plotW)+'" y2="'+y+'"/><text class="tick" x="'+(left-10)+'" y="'+(y+4)+'" text-anchor="end">'+fmt(t,2)+'</text>';});
  svg+='<line class="axis" x1="'+left+'" y1="'+(top+plotH)+'" x2="'+(left+plotW)+'" y2="'+(top+plotH)+'"/><line class="axis" x1="'+left+'" y1="'+top+'" x2="'+left+'" y2="'+(top+plotH)+'"/>';
  svg+='<text class="label" x="'+(left+plotW/2)+'" y="'+(height-18)+'" text-anchor="middle">nearest baseline progress (%)</text><text class="label" x="20" y="'+(top+plotH/2)+'" text-anchor="middle" transform="rotate(-90,20,'+(top+plotH/2)+')">nearest distance</text>';
  lines.forEach((line,i)=>{const color=COLORS[line.tc], proto=line.tt==='sequence'?'':' proto', d=pathFromPoints(line.p,xs,ys); svg+='<g data-i="'+i+'"><path class="curve'+proto+'" data-role="curve" d="'+d+'" stroke="'+color+'"/><path class="curve-hit" data-role="hit" d="'+d+'"/>'; line.p.forEach((p,j)=>{svg+='<circle class="point'+proto+'" data-j="'+j+'" cx="'+xs(p[0]).toFixed(2)+'" cy="'+ys(p[1]).toFixed(2)+'" r="'+(line.tt==='sequence'?2.2:3.8)+'" fill="'+color+'"/>';}); svg+='</g>';});
  svg+='</svg>';
  const host=document.getElementById('mainChart'); host.innerHTML=svg; const card=document.getElementById('mainChartCard');
  host.querySelectorAll('g[data-i]').forEach(g=>{const line=lines[Number(g.dataset.i)]; g.querySelectorAll('[data-role]').forEach(el=>{el.addEventListener('mouseover',e=>{setActive(card,g,line);showTip(e,tipHtml(line));}); el.addEventListener('mousemove',moveTip); el.addEventListener('mouseout',e=>{if(!g.contains(e.relatedTarget)) clearActive(card);});}); g.querySelectorAll('.point').forEach(pt=>{pt.addEventListener('mouseover',e=>{setActive(card,g,line);pt.classList.add('active');pt.setAttribute('r','6');showTip(e,tipHtml(line,Number(pt.dataset.j)));}); pt.addEventListener('mousemove',moveTip); pt.addEventListener('mouseout',e=>{pt.setAttribute('r',line.tt==='sequence'?2.2:3.8); if(!g.contains(e.relatedTarget)) clearActive(card);});});});
}
function renderEndpointChart(){
  const lines=DATA.lines.filter(line=>passesFilters(line)&&line.bc===currentBaselineClass());
  document.getElementById('endpointCaption').textContent='metric='+currentMetric()+', baseline class='+currentBaselineClass()+', plotted pair-curves='+lines.length;
  const baselines=classBaselines(currentBaselineClass()), xIndex=new Map(baselines.map((b,i)=>[b.seq_id,i]));
  const width=1180,height=420,left=72,right=28,top=28,bottom=72,plotW=width-left-right,plotH=height-top-bottom;
  const xs=seq=>left+(xIndex.get(seq)/(Math.max(1,baselines.length-1)))*plotW, ys=y=>top+plotH-(y/100)*plotH;
  let svg='<svg class="chart-svg" viewBox="0 0 '+width+' '+height+'">';
  [0,20,40,60,80,100].forEach(t=>{const y=ys(t); svg+='<line class="grid" x1="'+left+'" y1="'+y+'" x2="'+(left+plotW)+'" y2="'+y+'"/><text class="tick" x="'+(left-10)+'" y="'+(y+4)+'" text-anchor="end">'+t+'</text>';});
  baselines.forEach((b,i)=>{if(i%Math.ceil(baselines.length/12)===0){const x=xs(b.seq_id); svg+='<text class="tick" x="'+x+'" y="'+(top+plotH+22)+'" text-anchor="middle">'+b.seq_id+'</text>'; }});
  svg+='<line class="axis" x1="'+left+'" y1="'+(top+plotH)+'" x2="'+(left+plotW)+'" y2="'+(top+plotH)+'"/><line class="axis" x1="'+left+'" y1="'+top+'" x2="'+left+'" y2="'+(top+plotH)+'"/>';
  svg+='<text class="label" x="'+(left+plotW/2)+'" y="'+(height-18)+'" text-anchor="middle">baseline sequence id</text><text class="label" x="20" y="'+(top+plotH/2)+'" text-anchor="middle" transform="rotate(-90,20,'+(top+plotH/2)+')">endpoint nearest progress (%)</text>';
  lines.forEach((line,i)=>{const x=xs(line.bs)+(Math.random()-0.5)*10; const y=ys(line.ep); const med=line.bm?' medoid':''; svg+='<circle class="scatter'+med+'" data-i="'+i+'" cx="'+x.toFixed(2)+'" cy="'+y.toFixed(2)+'" r="'+(line.bm?4.8:3)+'" fill="'+COLORS[line.tc]+'"><title>'+esc(line.id+' endpoint='+fmt(line.ep,2))+'</title></circle>';});
  svg+='</svg>'; document.getElementById('endpointChart').innerHTML=svg;
}
function table(container,rows,cols,maxRows=200){
  document.getElementById(container).innerHTML='<table><tr>'+cols.map(c=>'<th>'+esc(c.l)+'</th>').join('')+'</tr>'+rows.slice(0,maxRows).map(r=>'<tr>'+cols.map(c=>{const v=r[c.k]; return '<td>'+esc(typeof v==='number'?fmt(v,c.d??3):(v??''))+'</td>';}).join('')+'</tr>').join('')+'</table>';
}
function renderTables(){
  table('classTargetTable',DATA.classTargetSummary,[{k:'metric',l:'metric'},{k:'baseline_class',l:'baseline'},{k:'target_class',l:'target'},{k:'target_source_type',l:'source'},{k:'n_pair_curves',l:'pairs',d:0},{k:'mean_endpoint_progress',l:'mean endpoint'},{k:'late_endpoint_70_rate',l:'late>=70'},{k:'late_endpoint_80_rate',l:'late>=80'},{k:'mean_foldback_count',l:'mean foldback'}],300);
  table('medoidTable',DATA.medoidSummary,[{k:'metric',l:'metric'},{k:'baseline_class',l:'baseline'},{k:'target_class',l:'target'},{k:'target_source_type',l:'source'},{k:'methodB_medoid_seq_id',l:'medoid'},{k:'medoid_mean_endpoint_progress',l:'medoid endpoint'},{k:'all_baselines_median_endpoint_progress',l:'all median'},{k:'medoid_percentile_endpoint_progress',l:'medoid percentile'},{k:'all_baselines_late_endpoint_70_rate',l:'all late>=70'}],300);
  table('selfPairTable',DATA.selfPairSanity,[{k:'metric',l:'metric'},{k:'n_self_curves',l:'self curves',d:0},{k:'n_self_points',l:'self points',d:0},{k:'max_abs_progress_error',l:'max progress error'},{k:'max_distance',l:'max distance',d:8},{k:'passed',l:'passed'}],20);
  const s=DATA.zscoreSummary; table('zscoreTable',[s],[{k:'n_sequences_total',l:'sequences',d:0},{k:'n_observations_M',l:'M',d:0},{k:'feature_dim_D',l:'D',d:0},{k:'sigma_eps',l:'sigma eps',d:12},{k:'sigma_zero_or_small_count',l:'sigma<=eps',d:0},{k:'sigma_median',l:'sigma median',d:8},{k:'sigma_max',l:'sigma max',d:8},{k:'clip_rate',l:'clip rate',d:8}],10);
}
function renderAll(){renderFrameStrip(); renderMainChart(); renderEndpointChart();}
document.getElementById('metricSelect').addEventListener('change',renderAll);
document.getElementById('baselineClassSelect').addEventListener('change',updateBaselineSeqOptions);
document.getElementById('baselineSeqSelect').addEventListener('change',renderAll);
document.getElementById('mainChartCard').addEventListener('mouseleave',()=>clearActive(document.getElementById('mainChartCard')));
document.querySelectorAll('input').forEach(el=>el.addEventListener('change',renderAll));
initControls(); renderTables();
</script>
</body>
</html>
"""


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate all-sequence baseline nearest-baseline sensitivity report.")
    parser.add_argument("--analysis_input_root", default=str(AllSequenceBaselineConfig.analysis_input_root))
    parser.add_argument("--projection_output_root", default=str(AllSequenceBaselineConfig.projection_output_root))
    parser.add_argument("--previous_zscore_root", default=str(AllSequenceBaselineConfig.previous_zscore_root))
    parser.add_argument("--output_root", default=str(AllSequenceBaselineConfig.output_root))
    parser.add_argument("--clip-k", type=float, default=AllSequenceBaselineConfig.clip_k)
    return parser


def main() -> None:
    cfg = AllSequenceBaselineConfig.from_args(build_arg_parser().parse_args())
    result = AllSequenceBaselineReport(cfg).run()
    print(f"html: {result['paths']['html']}")
    print(f"point_rows: {result['point_rows']}")
    print(f"curve_count: {result['curve_count']}")
    print(f"self_pair_passed: {result['self_pair_passed']}")
    print(f"compatibility_passed: {result['compatibility_passed']}")


if __name__ == "__main__":
    main()
