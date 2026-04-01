from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from tslearn.metrics import dtw, dtw_path


ANALYSIS_SEQUENCE_DIR = Path(__file__).resolve().parent.parent / "analysis_sequence"
BASE_PY = ANALYSIS_SEQUENCE_DIR / "common" / "base.py"
_base_spec = importlib.util.spec_from_file_location("analysis_sequence_base_dtw_resample", BASE_PY)
if _base_spec is None or _base_spec.loader is None:
    raise RuntimeError(f"Cannot load analysis_sequence base module from {BASE_PY}")
_base_module = importlib.util.module_from_spec(_base_spec)
sys.modules["analysis_sequence_base_dtw_resample"] = _base_module
_base_spec.loader.exec_module(_base_module)
CLASS_NAMES = _base_module.CLASS_NAMES


@dataclass(frozen=True)
class SequenceInfo:
    class_name: str
    sequence_id: str


@dataclass
class DTWResampleConfig:
    analysis_input_root: Path = Path(r"E:\Matsuda_data\2-27meeting")
    source_input_root: Path = Path(r"E:\Matsuda_data\2-18meeting")
    output_root: Path = Path(r"E:\Matsuda_data\DTW_resample_output")
    norm_len: int = 20
    sakoe_chiba_ratio: float = 0.2
    clip_fps: int = 30

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "DTWResampleConfig":
        return cls(
            analysis_input_root=Path(args.analysis_input_root),
            source_input_root=Path(args.source_input_root),
            output_root=Path(args.output_root),
            norm_len=int(args.norm_len),
            sakoe_chiba_ratio=float(args.sakoe_chiba_ratio),
            clip_fps=int(args.clip_fps),
        )


class DTWResampleTaskBase:
    def __init__(self, config: DTWResampleConfig):
        self.cfg = config
        self._ensure_roots()

    @staticmethod
    def build_common_arg_parser(description: str) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description=description)
        parser.add_argument("--analysis_input_root", default=str(DTWResampleConfig.analysis_input_root))
        parser.add_argument("--source_input_root", default=str(DTWResampleConfig.source_input_root))
        parser.add_argument("--output_root", default=str(DTWResampleConfig.output_root))
        parser.add_argument("--norm_len", type=int, default=DTWResampleConfig.norm_len)
        parser.add_argument("--sakoe_chiba_ratio", type=float, default=DTWResampleConfig.sakoe_chiba_ratio)
        parser.add_argument("--clip_fps", type=int, default=DTWResampleConfig.clip_fps)
        return parser

    def _ensure_roots(self) -> None:
        for sub in ("csv", "metrics", "media", "report"):
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

    def sequences_for_class(self, class_name: str) -> list[SequenceInfo]:
        return [s for s in self.discover_sequences() if s.class_name == class_name]

    def rel_seq_path(self, seq: SequenceInfo) -> Path:
        return self.cfg.analysis_input_root / "metrics" / "sequence_features_rel" / seq.class_name / seq.sequence_id / "sequence_features_rel.npy"

    def frame_names_path(self, seq: SequenceInfo) -> Path:
        return self.cfg.analysis_input_root / "metrics" / "sequence_features" / seq.class_name / seq.sequence_id / "frame_names.json"

    def source_sequence_dir(self, seq: SequenceInfo) -> Path:
        return self.cfg.source_input_root / seq.class_name / seq.sequence_id

    def source_videos_dir(self, class_name: str) -> Path:
        return self.cfg.source_input_root / class_name / "videos"

    @staticmethod
    def load_npy(path: Path) -> np.ndarray:
        return np.load(path, allow_pickle=False)

    @staticmethod
    def save_npy(path: Path, arr: np.ndarray) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, arr)

    @staticmethod
    def load_json(path: Path) -> object:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def save_json(path: Path, payload: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    @staticmethod
    def write_csv(path: Path, rows: Iterable[dict], fieldnames: list[str]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)


def dtw_distance_and_path(seq: np.ndarray, ref: np.ndarray, ratio: float) -> tuple[float, list[tuple[int, int]]]:
    radius = max(1, int(math.ceil(max(len(seq), len(ref)) * ratio)))
    path, score = dtw_path(
        seq,
        ref,
        global_constraint="sakoe_chiba",
        sakoe_chiba_radius=radius,
    )
    return float(score), list(path)


def align_sequence_to_reference(seq: np.ndarray, ref_len: int, path: list[tuple[int, int]]) -> tuple[np.ndarray, list[list[int]]]:
    dim = seq.shape[1]
    grouped_indices: list[list[int]] = [[] for _ in range(ref_len)]
    for src_idx, ref_idx in path:
        grouped_indices[ref_idx].append(src_idx)

    aligned = np.empty((ref_len, dim), dtype=np.float32)
    last_valid = 0
    for ref_idx in range(ref_len):
        src_ids = grouped_indices[ref_idx]
        if src_ids:
            aligned[ref_idx] = seq[np.asarray(src_ids, dtype=np.int32)].mean(axis=0)
            last_valid = ref_idx
        else:
            aligned[ref_idx] = aligned[last_valid] if ref_idx > 0 else seq[0]
    return aligned, grouped_indices


def resample_2d(arr: np.ndarray, target_len: int) -> np.ndarray:
    if arr.shape[0] == target_len:
        return arr.astype(np.float32)
    t_old = np.arange(arr.shape[0], dtype=np.float32)
    t_new = np.linspace(0, arr.shape[0] - 1, target_len, dtype=np.float32)
    out = np.empty((target_len, arr.shape[1]), dtype=np.float32)
    for d in range(arr.shape[1]):
        out[:, d] = np.interp(t_new, t_old, arr[:, d])
    return out


def sampled_indices(length: int, target_len: int) -> np.ndarray:
    if length == 1:
        return np.zeros((target_len,), dtype=np.int32)
    idx = np.linspace(0, length - 1, target_len)
    return np.rint(idx).astype(np.int32)


def infer_video_stem_from_frame(frame_name: str) -> str:
    stem = Path(frame_name).stem
    parts = stem.split("_")
    if len(parts) < 4:
        return stem
    return "_".join(parts[:-3])


def find_source_video(videos_dir: Path, representative_frame_name: str) -> Path | None:
    stem = infer_video_stem_from_frame(representative_frame_name)
    candidate = videos_dir / f"{stem}.mp4"
    if candidate.exists():
        return candidate
    matches = list(videos_dir.glob(f"{stem}.*"))
    return matches[0] if matches else None


def export_clip_from_frames(src_dir: Path, frame_names: list[str], dst_video: Path, fps: int) -> None:
    temp_dir = dst_video.parent / f"{dst_video.stem}_frames"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    for idx, frame_name in enumerate(frame_names):
        src = src_dir / frame_name
        dst = temp_dir / f"{idx:03d}{src.suffix.lower()}"
        shutil.copy2(src, dst)

    pattern = str(temp_dir / "%03d.png")
    vf = "scale=iw:ih:force_original_aspect_ratio=decrease,pad=ceil(max(iw\\,ih*4/3)/2)*2:ceil(max(ih\\,iw*3/4)/2)*2:(ow-iw)/2:(oh-ih)/2"
    cmd = [
        "C:\\ffmpeg\\bin\\ffmpeg.exe",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        pattern,
        "-vf",
        vf,
        "-pix_fmt",
        "yuv420p",
        str(dst_video),
    ]
    subprocess.run(cmd, check=True)
