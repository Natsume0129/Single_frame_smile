from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


ANALYSIS_SEQUENCE_DIR = Path(__file__).resolve().parent.parent / "analysis_sequence"
BASE_PY = ANALYSIS_SEQUENCE_DIR / "common" / "base.py"
_base_spec = importlib.util.spec_from_file_location("analysis_sequence_base", BASE_PY)
if _base_spec is None or _base_spec.loader is None:
    raise RuntimeError(f"Cannot load analysis_sequence base module from {BASE_PY}")
_base_module = importlib.util.module_from_spec(_base_spec)
sys.modules["analysis_sequence_base"] = _base_module
_base_spec.loader.exec_module(_base_module)
CLASS_NAMES = _base_module.CLASS_NAMES


@dataclass(frozen=True)
class SequenceInfo:
    class_name: str
    sequence_id: str


@dataclass
class ProjectionConfig:
    analysis_input_root: Path = Path(r"E:\Matsuda_data\2-27meeting")
    output_root: Path = Path(r"E:\Matsuda_data\3-10meeting")
    norm_len: int = 20

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "ProjectionConfig":
        return cls(
            analysis_input_root=Path(args.analysis_input_root),
            output_root=Path(args.output_root),
            norm_len=int(args.norm_len),
        )


class ProjectionTaskBase:
    def __init__(self, config: ProjectionConfig):
        self.cfg = config
        self._ensure_roots()

    @staticmethod
    def build_common_arg_parser(description: str) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description=description)
        parser.add_argument("--analysis_input_root", default=str(ProjectionConfig.analysis_input_root))
        parser.add_argument("--output_root", default=str(ProjectionConfig.output_root))
        parser.add_argument("--norm_len", type=int, default=ProjectionConfig.norm_len)
        return parser

    def _ensure_roots(self) -> None:
        for method in ("methodA", "methodB"):
            for sub in ("csv", "plots", "prototypes", "report"):
                (self.cfg.output_root / method / sub).mkdir(parents=True, exist_ok=True)

    def discover_sequences(self) -> list[SequenceInfo]:
        seqs: list[SequenceInfo] = []
        normalized_root = self.cfg.analysis_input_root / "metrics" / "normalized"
        for class_name in CLASS_NAMES:
            class_dir = normalized_root / class_name
            if not class_dir.is_dir():
                continue
            for seq_dir in sorted(class_dir.iterdir(), key=lambda p: p.name):
                if seq_dir.is_dir():
                    seqs.append(SequenceInfo(class_name=class_name, sequence_id=seq_dir.name))
        return seqs

    def sequences_for_class(self, class_name: str) -> list[SequenceInfo]:
        return [seq for seq in self.discover_sequences() if seq.class_name == class_name]

    def normalized_seq_path(self, seq: SequenceInfo) -> Path:
        return self.cfg.analysis_input_root / "metrics" / "normalized" / seq.class_name / seq.sequence_id / "normalized_sequence.npy"

    def normalized_frames_dir(self, seq: SequenceInfo) -> Path:
        return self.cfg.analysis_input_root / "metrics" / "normalized_frames" / seq.class_name / seq.sequence_id

    def sampled_frames_path(self, seq: SequenceInfo) -> Path:
        return self.cfg.analysis_input_root / "metrics" / "normalized" / seq.class_name / seq.sequence_id / "sampled_frames.json"

    def method_dir(self, method: str) -> Path:
        return self.cfg.output_root / method

    def method_csv(self, method: str, filename: str) -> Path:
        return self.method_dir(method) / "csv" / filename

    def method_plot(self, method: str, filename: str) -> Path:
        return self.method_dir(method) / "plots" / filename

    def method_proto(self, method: str, filename: str) -> Path:
        return self.method_dir(method) / "prototypes" / filename

    def method_report(self, method: str, filename: str) -> Path:
        return self.method_dir(method) / "report" / filename

    @staticmethod
    def save_json(path: Path, payload: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    @staticmethod
    def load_json(path: Path) -> object:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def save_npy(path: Path, arr: np.ndarray) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, arr)

    @staticmethod
    def load_npy(path: Path) -> np.ndarray:
        return np.load(path, allow_pickle=False)

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


def medoid_index_by_frobenius(seqs: np.ndarray) -> int:
    n = seqs.shape[0]
    costs = np.zeros((n,), dtype=np.float64)
    for i in range(n):
        diff = seqs[i][None, :, :] - seqs
        d = np.sqrt(np.sum(diff * diff, axis=(1, 2)))
        costs[i] = d.sum()
    return int(np.argmin(costs))


def compute_axis_metrics(sequence: np.ndarray, axis: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= 1e-12:
        raise RuntimeError("True-smile axis norm is too small for projection analysis.")
    axis_unit = axis / axis_norm
    delta = sequence - sequence[0]
    projection_length = delta @ axis_unit
    projection_vector = np.outer(projection_length, axis_unit)
    residual = delta - projection_vector
    off_axis_distance = np.linalg.norm(residual, axis=1)
    projection_ratio = projection_length / axis_norm
    off_axis_ratio = off_axis_distance / axis_norm
    return projection_length, projection_ratio, off_axis_distance, off_axis_ratio


def compute_summary_stats(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "median": float(np.median(arr)),
        "q1": float(np.quantile(arr, 0.25)),
        "q3": float(np.quantile(arr, 0.75)),
    }
