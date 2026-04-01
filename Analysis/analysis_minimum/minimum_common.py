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
_base_spec = importlib.util.spec_from_file_location("analysis_sequence_base_minimum", BASE_PY)
if _base_spec is None or _base_spec.loader is None:
    raise RuntimeError(f"Cannot load analysis_sequence base module from {BASE_PY}")
_base_module = importlib.util.module_from_spec(_base_spec)
sys.modules["analysis_sequence_base_minimum"] = _base_module
_base_spec.loader.exec_module(_base_module)
CLASS_NAMES = _base_module.CLASS_NAMES


@dataclass(frozen=True)
class SequenceInfo:
    class_name: str
    sequence_id: str


@dataclass
class MinimumConfig:
    analysis_input_root: Path = Path(r"E:\Matsuda_data\2-27meeting")
    output_root: Path = Path(r"E:\Matsuda_data\analysis_minimum_output")
    norm_len: int = 20

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "MinimumConfig":
        return cls(
            analysis_input_root=Path(args.analysis_input_root),
            output_root=Path(args.output_root),
            norm_len=int(args.norm_len),
        )


class MinimumTaskBase:
    def __init__(self, config: MinimumConfig):
        self.cfg = config
        self._ensure_roots()

    @staticmethod
    def build_common_arg_parser(description: str) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description=description)
        parser.add_argument("--analysis_input_root", default=str(MinimumConfig.analysis_input_root))
        parser.add_argument("--output_root", default=str(MinimumConfig.output_root))
        parser.add_argument("--norm_len", type=int, default=MinimumConfig.norm_len)
        return parser

    def _ensure_roots(self) -> None:
        for sub in ("csv", "plots", "report"):
            (self.cfg.output_root / sub).mkdir(parents=True, exist_ok=True)
        for method in ("methodA", "methodB"):
            for sub in ("csv", "plots", "report", "prototypes"):
                (self.cfg.output_root / method / sub).mkdir(parents=True, exist_ok=True)
        for sub in ("csv", "plots", "report"):
            (self.cfg.output_root / "shared" / sub).mkdir(parents=True, exist_ok=True)

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

    def normalized_seq_path(self, seq: SequenceInfo) -> Path:
        return self.cfg.analysis_input_root / "metrics" / "normalized" / seq.class_name / seq.sequence_id / "normalized_sequence.npy"

    def sampled_frames_path(self, seq: SequenceInfo) -> Path:
        return self.cfg.analysis_input_root / "metrics" / "normalized" / seq.class_name / seq.sequence_id / "sampled_frames.json"

    def normalized_frames_dir(self, seq: SequenceInfo) -> Path:
        return self.cfg.analysis_input_root / "metrics" / "normalized_frames" / seq.class_name / seq.sequence_id

    def prototype_dir(self) -> Path:
        return self.cfg.analysis_input_root / "prototypes"

    def method_dir(self, method: str) -> Path:
        return self.cfg.output_root / method

    def method_csv(self, method: str, filename: str) -> Path:
        return self.method_dir(method) / "csv" / filename

    def method_plot(self, method: str, filename: str) -> Path:
        return self.method_dir(method) / "plots" / filename

    def method_report(self, method: str, filename: str) -> Path:
        return self.method_dir(method) / "report" / filename

    def shared_csv(self, filename: str) -> Path:
        return self.cfg.output_root / "shared" / "csv" / filename

    def shared_plot(self, filename: str) -> Path:
        return self.cfg.output_root / "shared" / "plots" / filename

    def shared_report(self, filename: str) -> Path:
        return self.cfg.output_root / "shared" / "report" / filename

    @staticmethod
    def load_npy(path: Path) -> np.ndarray:
        return np.load(path, allow_pickle=False)

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


def compute_sync_min(curve1: np.ndarray, curve2: np.ndarray) -> tuple[int, float]:
    if curve1.shape != curve2.shape:
        raise RuntimeError(f"Shape mismatch for synchronized minimum distance: {curve1.shape} vs {curve2.shape}")
    d = np.linalg.norm(curve1 - curve2, axis=1)
    idx = int(np.argmin(d))
    return idx, float(d[idx])


def summary_stats(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "median": float(np.median(arr)),
        "q1": float(np.quantile(arr, 0.25)),
        "q3": float(np.quantile(arr, 0.75)),
    }
