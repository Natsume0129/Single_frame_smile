from __future__ import annotations

import csv
import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from build_region_mask_samples import (
    FINE_REGIONS,
    build_natural_masks,
    exact_area_mask,
    local_mean_fill,
)


EXPERIMENT_ROOT = Path(r"E:\Dataset\smile_ranking_region_ablation_20260818")
SMILECOMP_DIR = Path(
    r"E:\SmileAnnotation\FaceTracking-Smile_Detection"
    r"\SmileComp_SiameseNet\Pytorch-Shimonishi"
)
SEQUENCE_ROOT = Path(r"E:\Dataset\sequence")
VGGFACE_CONV = SMILECOMP_DIR / "models" / "vggface_conv.pth"
YUNET_MODEL = Path(
    r"E:\Dataset\smile_ranking_gradcam_compare\analysis_assets"
    r"\face_detection_yunet_2023mar.onnx"
)

MODEL_CLASSES = ("true", "polite", "bitter")
INPUT_CLASSES = ("true", "polite", "bitter")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}


@dataclass(frozen=True)
class DatasetSpec:
    label: str
    root: Path
    images_dir: Path
    ranking_items: Path
    manual_pairs: Path
    baseline_weights: Path
    baseline_scale: Path
    baseline_frame_scores: Path
    batch_size: int


DATASETS = {
    "true": DatasetSpec(
        label="true",
        root=Path(r"E:\Dataset\smile_ranking_true"),
        images_dir=Path(r"E:\Dataset\smile_ranking_true\images"),
        ranking_items=Path(r"E:\Dataset\smile_ranking_true\ranking_items.csv"),
        manual_pairs=Path(
            r"E:\Dataset\smile_ranking_true\train_manual_cross_sequence_198.dat"
        ),
        baseline_weights=Path(
            r"E:\Dataset\smile_ranking_true\model"
            r"\smile_rank_true_cross_sequence_100ep.pth"
        ),
        baseline_scale=Path(r"E:\Dataset\smile_ranking_true\scale10"),
        baseline_frame_scores=Path(
            r"E:\Dataset\smileranking_plot\frame_smile_ranking_scores.csv"
        ),
        batch_size=16,
    ),
    "polite": DatasetSpec(
        label="polite",
        root=Path(r"E:\Dataset\smile_ranking_polite"),
        images_dir=Path(r"E:\Dataset\smile_ranking_polite\images"),
        ranking_items=Path(r"E:\Dataset\smile_ranking_polite\ranking_items.csv"),
        manual_pairs=Path(
            r"E:\Dataset\smile_ranking_polite\train_manual_cross_sequence_192.dat"
        ),
        baseline_weights=Path(
            r"E:\Dataset\smile_ranking_polite\model"
            r"\smile_rank_polite_cross_sequence_100ep.pth"
        ),
        baseline_scale=Path(r"E:\Dataset\smile_ranking_polite\scale10"),
        baseline_frame_scores=Path(
            r"E:\Dataset\politesmile_ranking_plot\frame_smile_ranking_scores.csv"
        ),
        batch_size=64,
    ),
    "bitter": DatasetSpec(
        label="bitter",
        root=Path(r"E:\Dataset\smile_ranking_bitter"),
        images_dir=Path(r"E:\Dataset\smile_ranking_bitter\images"),
        ranking_items=Path(r"E:\Dataset\smile_ranking_bitter\ranking_items.csv"),
        manual_pairs=Path(
            r"E:\Dataset\smile_ranking_bitter\train_manual_cross_sequence_198.dat"
        ),
        baseline_weights=Path(
            r"E:\Dataset\smile_ranking_bitter\model"
            r"\smile_rank_bitter_cross_sequence_100ep.pth"
        ),
        baseline_scale=Path(r"E:\Dataset\smile_ranking_bitter\scale10"),
        baseline_frame_scores=Path(
            r"E:\Dataset\bittersmile_ranking_plot\frame_smile_ranking_scores.csv"
        ),
        batch_size=64,
    ),
}


@dataclass(frozen=True)
class MaskCondition:
    condition_id: str
    variant: str
    region: str


CONDITIONS = tuple(
    [
        MaskCondition(f"natural_{region}", "natural", region)
        for region in (
            "periocular",
            "mouth_perioral",
            "nose",
            "cheeks",
            "upper_face",
            "lower_face",
        )
    ]
    + [
        MaskCondition(f"area_matched_{region}", "area_matched", region)
        for region in FINE_REGIONS
    ]
)
CONTROL_CONDITION = MaskCondition("control_unmasked", "control", "none")
CONDITIONS_BY_ID = {
    condition.condition_id: condition
    for condition in (*CONDITIONS, CONTROL_CONDITION)
}


def natural_key(value: str) -> list[object]:
    return [
        int(part) if part.isdigit() else part.casefold()
        for part in re.split(r"(\d+)", value)
    ]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def list_ranking_images() -> list[Path]:
    paths: list[Path] = []
    for label in MODEL_CLASSES:
        spec = DATASETS[label]
        items = read_csv(spec.ranking_items)
        for item in items:
            path = spec.images_dir / item["image_file"]
            if not path.is_file():
                raise FileNotFoundError(path)
            paths.append(path)
    return paths


def list_sequence_frames() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for label in INPUT_CLASSES:
        label_dir = SEQUENCE_ROOT / label
        if not label_dir.is_dir():
            raise FileNotFoundError(label_dir)
        sequence_dirs = sorted(
            (path for path in label_dir.iterdir() if path.is_dir()),
            key=lambda path: natural_key(path.name),
        )
        for sequence_dir in sequence_dirs:
            frames = sorted(
                (
                    path
                    for path in sequence_dir.iterdir()
                    if path.is_file()
                    and path.suffix.casefold() in IMAGE_EXTENSIONS
                ),
                key=lambda path: natural_key(path.name),
            )
            for frame_index, path in enumerate(frames):
                rows.append(
                    {
                        "input_class": label,
                        "sequence_id": sequence_dir.name,
                        "frame_index_from_onset": frame_index,
                        "frame_file": path.name,
                        "frame_path": path,
                    }
                )
    return rows


def all_landmark_source_paths() -> list[Path]:
    unique: dict[str, Path] = {}
    for path in list_ranking_images():
        unique[str(path.resolve()).casefold()] = path.resolve()
    for row in list_sequence_frames():
        path = Path(row["frame_path"]).resolve()
        unique[str(path).casefold()] = path
    return sorted(unique.values(), key=lambda path: str(path).casefold())


def connect_landmark_cache(path: Path) -> sqlite3.Connection:
    if not path.is_file():
        raise FileNotFoundError(path)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    return connection


def fetch_landmarks(
    connection: sqlite3.Connection, image_path: Path
) -> np.ndarray:
    resolved = str(image_path.resolve())
    row = connection.execute(
        "SELECT status, landmarks FROM landmarks WHERE path = ?", (resolved,)
    ).fetchone()
    if row is None:
        raise KeyError(f"No landmark cache row: {resolved}")
    if row["status"] != "ok":
        raise RuntimeError(f"Landmark status is {row['status']}: {resolved}")
    landmarks = np.frombuffer(row["landmarks"], dtype=np.float32).copy()
    if landmarks.size != 68 * 2:
        raise ValueError(f"Invalid landmark payload: {resolved}")
    return landmarks.reshape(68, 2)


def load_rgb_224(path: Path) -> np.ndarray:
    with Image.open(path) as source:
        image = source.convert("RGB")
        if image.size != (224, 224):
            image = image.resize((224, 224), Image.Resampling.LANCZOS)
        return np.asarray(image)


def apply_condition(
    image: np.ndarray,
    landmarks: np.ndarray,
    condition: MaskCondition,
    area_target_fraction: float,
) -> tuple[np.ndarray, dict[str, float | int | list[float]]]:
    if condition.variant == "control":
        metrics: dict[str, float | int | list[float]] = {
            "face_pixels": 224 * 224,
            "natural_mask_pixels": 0,
            "mask_pixels": 0,
            "mask_fraction_of_face": 0.0,
            "split_y": float("nan"),
            "fill_rgb": [float("nan"), float("nan"), float("nan")],
        }
        return image.copy(), metrics
    face_mask, natural_masks, split_y = build_natural_masks(
        image.shape[:2], landmarks
    )
    natural_mask = natural_masks[condition.region]
    if condition.variant == "natural":
        mask = natural_mask
    elif condition.variant == "area_matched":
        target_pixels = int(round(area_target_fraction * int(face_mask.sum())))
        mask = exact_area_mask(natural_mask, face_mask, target_pixels)
    else:
        raise ValueError(condition.variant)
    masked, color = local_mean_fill(image, mask, face_mask)
    metrics: dict[str, float | int | list[float]] = {
        "face_pixels": int(face_mask.sum()),
        "natural_mask_pixels": int(natural_mask.sum()),
        "mask_pixels": int(mask.sum()),
        "mask_fraction_of_face": float(mask.sum()) / float(face_mask.sum()),
        "split_y": split_y,
        "fill_rgb": [float(value) for value in color],
    }
    return masked, metrics


def validate_inputs() -> None:
    required = [SMILECOMP_DIR, SEQUENCE_ROOT, VGGFACE_CONV, YUNET_MODEL]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    for spec in DATASETS.values():
        for path in (
            spec.images_dir,
            spec.ranking_items,
            spec.manual_pairs,
            spec.baseline_weights,
            spec.baseline_scale,
            spec.baseline_frame_scores,
        ):
            if not path.exists():
                raise FileNotFoundError(path)
