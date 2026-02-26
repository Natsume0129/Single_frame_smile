from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
CLASS_NAMES = ("polite", "truesmile", "ambiguous")
IGNORED_SEQUENCE_DIRS = {"videos", "video"}


@dataclass(frozen=True)
class SequenceInfo:
    class_name: str
    sequence_id: str
    sequence_path: Path


@dataclass
class PipelineConfig:
    input_root: Path = Path(r"E:\Matsuda_data\2-18meeting")
    output_root: Path = Path(r"E:\Matsuda_data\2-27meeting")
    weights: Path = Path(r"E:\Single_frame_smile\data\models\vggface.pth")
    fps: int = 30
    norm_len: int = 20
    batch_size: int = 64
    num_workers: int = 0
    device: str = "cuda:0"

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "PipelineConfig":
        return cls(
            input_root=Path(args.input_root),
            output_root=Path(args.output_root),
            weights=Path(args.weights),
            fps=int(args.fps),
            norm_len=int(args.norm_len),
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            device=str(args.device),
        )


class SequenceTaskBase:
    def __init__(self, config: PipelineConfig):
        self.cfg = config
        self._ensure_roots()

    @staticmethod
    def build_common_arg_parser(description: str) -> argparse.ArgumentParser:
        p = argparse.ArgumentParser(description=description)
        p.add_argument("--input_root", default=str(PipelineConfig.input_root))
        p.add_argument("--output_root", default=str(PipelineConfig.output_root))
        p.add_argument("--weights", default=str(PipelineConfig.weights))
        p.add_argument("--fps", type=int, default=PipelineConfig.fps)
        p.add_argument("--norm_len", type=int, default=PipelineConfig.norm_len)
        p.add_argument("--batch_size", type=int, default=PipelineConfig.batch_size)
        p.add_argument("--num_workers", type=int, default=PipelineConfig.num_workers)
        p.add_argument("--device", default=PipelineConfig.device)
        return p

    def _ensure_roots(self) -> None:
        for sub in ("prototypes", "metrics", "plots", "csv", "report"):
            (self.cfg.output_root / sub).mkdir(parents=True, exist_ok=True)

    def discover_sequences(self) -> list[SequenceInfo]:
        seqs: list[SequenceInfo] = []
        for class_name in CLASS_NAMES:
            class_dir = self.cfg.input_root / class_name
            if not class_dir.is_dir():
                continue
            for child in sorted(class_dir.iterdir(), key=lambda p: p.name):
                if not child.is_dir():
                    continue
                if child.name.lower() in IGNORED_SEQUENCE_DIRS:
                    continue
                seqs.append(
                    SequenceInfo(
                        class_name=class_name,
                        sequence_id=child.name,
                        sequence_path=child,
                    )
                )
        return seqs

    @staticmethod
    def frame_number_from_name(path: Path) -> int:
        m = re.search(r"(\d+)(?!.*\d)", path.stem)
        if not m:
            raise RuntimeError(f"Cannot parse frame number from file name: {path.name}")
        return int(m.group(1))

    def list_sorted_frames(self, sequence_path: Path) -> list[Path]:
        frames = [
            p
            for p in sequence_path.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        ]
        if not frames:
            raise RuntimeError(f"No image frames in sequence folder: {sequence_path}")
        return sorted(frames, key=self.frame_number_from_name)

    def metrics_seq_dir(self, category: str, seq: SequenceInfo) -> Path:
        out = self.cfg.output_root / "metrics" / category / seq.class_name / seq.sequence_id
        out.mkdir(parents=True, exist_ok=True)
        return out

    def save_json(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def load_json(self, path: Path) -> dict:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def save_npy(self, path: Path, arr: np.ndarray) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, arr)

    def load_npy(self, path: Path) -> np.ndarray:
        return np.load(path, allow_pickle=False)

    def write_csv(self, path: Path, rows: Iterable[dict], fieldnames: list[str]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

