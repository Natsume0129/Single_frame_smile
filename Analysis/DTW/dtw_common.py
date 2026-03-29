from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tslearn.metrics import dtw


ANALYSIS_SEQUENCE_DIR = Path(__file__).resolve().parent.parent / "analysis_sequence"
BASE_PY = ANALYSIS_SEQUENCE_DIR / "common" / "base.py"
_base_spec = importlib.util.spec_from_file_location("analysis_sequence_base_dtw", BASE_PY)
if _base_spec is None or _base_spec.loader is None:
    raise RuntimeError(f"Cannot load analysis_sequence base module from {BASE_PY}")
_base_module = importlib.util.module_from_spec(_base_spec)
sys.modules["analysis_sequence_base_dtw"] = _base_module
_base_spec.loader.exec_module(_base_module)
CLASS_NAMES = _base_module.CLASS_NAMES


@dataclass(frozen=True)
class SequenceInfo:
    class_name: str
    sequence_id: str


@dataclass
class DTWConfig:
    analysis_input_root: Path = Path(r"E:\Matsuda_data\2-27meeting")
    output_root: Path = Path(r"E:\Matsuda_data\DTW_analysis")
    sakoe_chiba_ratio: float = 0.2

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "DTWConfig":
        return cls(
            analysis_input_root=Path(args.analysis_input_root),
            output_root=Path(args.output_root),
            sakoe_chiba_ratio=float(args.sakoe_chiba_ratio),
        )


class DTWTaskBase:
    def __init__(self, config: DTWConfig):
        self.cfg = config
        self._ensure_roots()

    @staticmethod
    def build_common_arg_parser(description: str) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description=description)
        parser.add_argument("--analysis_input_root", default=str(DTWConfig.analysis_input_root))
        parser.add_argument("--output_root", default=str(DTWConfig.output_root))
        parser.add_argument("--sakoe_chiba_ratio", type=float, default=DTWConfig.sakoe_chiba_ratio)
        return parser

    def _ensure_roots(self) -> None:
        for sub in ("csv", "plots", "report", "models"):
            (self.cfg.output_root / sub).mkdir(parents=True, exist_ok=True)

    def discover_sequences(self) -> list[SequenceInfo]:
        seqs: list[SequenceInfo] = []
        rel_root = self.cfg.analysis_input_root / "metrics" / "sequence_features_rel"
        for class_name in CLASS_NAMES:
            class_dir = rel_root / class_name
            if not class_dir.is_dir():
                continue
            for seq_dir in sorted(class_dir.iterdir(), key=lambda p: p.name):
                if seq_dir.is_dir():
                    seqs.append(SequenceInfo(class_name=class_name, sequence_id=seq_dir.name))
        return seqs

    def rel_seq_path(self, seq: SequenceInfo) -> Path:
        return self.cfg.analysis_input_root / "metrics" / "sequence_features_rel" / seq.class_name / seq.sequence_id / "sequence_features_rel.npy"

    def load_all_sequences(self) -> dict[tuple[str, str], np.ndarray]:
        out: dict[tuple[str, str], np.ndarray] = {}
        for seq in self.discover_sequences():
            out[(seq.class_name, seq.sequence_id)] = np.load(self.rel_seq_path(seq), allow_pickle=False).astype(np.float32)
        return out

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
    def save_json(path: Path, payload: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)


def make_magnitude_sequence(arr: np.ndarray) -> np.ndarray:
    return np.linalg.norm(arr, axis=1, keepdims=True).astype(np.float32)


def make_velocity_sequence(arr: np.ndarray) -> np.ndarray:
    if arr.shape[0] == 0:
        return np.zeros((0, 1), dtype=np.float32)
    if arr.shape[0] == 1:
        return np.zeros((1, 1), dtype=np.float32)
    diffs = arr[1:] - arr[:-1]
    v = np.linalg.norm(diffs, axis=1, keepdims=True)
    v = np.vstack([np.zeros((1, 1), dtype=np.float32), v.astype(np.float32)])
    return v.astype(np.float32)


def fit_pca_projection(
    all_sequences: dict[tuple[str, str], np.ndarray],
    n_components: int,
) -> tuple[StandardScaler, PCA, dict[tuple[str, str], np.ndarray]]:
    all_frames = np.concatenate(list(all_sequences.values()), axis=0)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(all_frames)
    pca = PCA(n_components=n_components, random_state=0)
    pca.fit(scaled)
    transformed: dict[tuple[str, str], np.ndarray] = {}
    for key, arr in all_sequences.items():
        transformed[key] = pca.transform(scaler.transform(arr)).astype(np.float32)
    return scaler, pca, transformed


def dtw_distance(seq1: np.ndarray, seq2: np.ndarray, use_band: bool, ratio: float) -> float:
    if not use_band:
        return float(dtw(seq1, seq2))
    radius = max(1, int(math.ceil(max(len(seq1), len(seq2)) * ratio)))
    return float(
        dtw(
            seq1,
            seq2,
            global_constraint="sakoe_chiba",
            sakoe_chiba_radius=radius,
        )
    )


def summary_stats(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "median": float(np.median(arr)),
        "q1": float(np.quantile(arr, 0.25)),
        "q3": float(np.quantile(arr, 0.75)),
    }
