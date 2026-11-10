from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

from build_region_mask_samples import build_natural_masks, feather_alpha, morphology
from region_ablation_common import (
    DATASETS,
    INPUT_CLASSES,
    MODEL_CLASSES,
    SMILECOMP_DIR,
    connect_landmark_cache,
    fetch_landmarks,
    list_sequence_frames,
    load_rgb_224,
)


DEFAULT_OUTPUT_ROOT = Path(r"E:\Dataset\smile_ranking_roi_pattern_20260825")
LANDMARK_CACHE = Path(
    r"E:\Dataset\smile_ranking_region_ablation_20260818\cache\landmarks.sqlite3"
)
CLEAN_SEQUENCE_MANIFEST = Path(
    r"E:\Dataset\smile_ranking_true_likeness\selected_sequences.csv"
)
SEQUENCE_MANIFEST = Path(r"E:\Dataset\sequence\sequence_manifest.csv")

ROI_ORDER = (
    "all",
    "upper_face",
    "lower_face",
    "periocular",
    "cheeks",
    "mouth_perioral",
    "nose",
)
FINE_ROIS = ("periocular", "cheeks", "mouth_perioral", "nose")
COARSE_ROIS = ("upper_face", "lower_face")
FILL_METHODS = ("blur", "local_mean")
FOLLOWUP_FILL_METHODS = ("downsample16",)
BLUR_SIGMA = 11.0
FEATHER_SIGMA = 3.0
DOWNSAMPLE_SIZE = 16
DUPLICATE_CLEAN_SEQUENCE = "polite_037"

MODEL_WEIGHTS = {
    label: DATASETS[label].baseline_weights for label in MODEL_CLASSES
}
TRUE_SCALE_DIR = DATASETS["true"].baseline_scale


@dataclass(frozen=True)
class RoiCondition:
    condition_id: str
    roi: str
    fill_method: str


ALL_CONDITION = RoiCondition("all_unprocessed", "all", "unprocessed")
CANDIDATE_CONDITIONS = (ALL_CONDITION,) + tuple(
    RoiCondition(f"{fill_method}_{roi}", roi, fill_method)
    for fill_method in FILL_METHODS
    for roi in ROI_ORDER
    if roi != "all"
)
CONDITIONS_BY_ID = {
    condition.condition_id: condition for condition in CANDIDATE_CONDITIONS
}


def natural_key(value: str) -> list[object]:
    import re

    return [
        int(part) if part.isdigit() else part.casefold()
        for part in re.split(r"(\d+)", value)
    ]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    materialized = list(rows)
    if not materialized:
        raise ValueError(f"No rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    with partial.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(materialized[0]))
        writer.writeheader()
        writer.writerows(materialized)
    os.replace(partial, path)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(partial, path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        Path(r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf"),
        Path(r"C:\Windows\Fonts\segoeuib.ttf" if bold else r"C:\Windows\Fonts\segoeui.ttf"),
    ]
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def build_frame_manifest() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    clean_keys = {
        str(row["sequence_key"]) for row in read_csv(CLEAN_SEQUENCE_MANIFEST)
    }
    manifest_rows = read_csv(SEQUENCE_MANIFEST)
    manifest_by_key = {
        f"{row['label']}_{int(row['sequence_id']):03d}": row
        for row in manifest_rows
        if row.get("label") in INPUT_CLASSES
    }
    frame_rows: list[dict[str, object]] = []
    sequence_counts: dict[tuple[str, str], int] = {}
    for row in list_sequence_frames():
        label = str(row["input_class"])
        sequence_id = str(row["sequence_id"])
        sequence_key = f"{label}_{int(sequence_id):03d}"
        key = (label, sequence_id)
        sequence_counts[key] = sequence_counts.get(key, 0) + 1
        source = manifest_by_key.get(sequence_key, {})
        frame_rows.append(
            {
                **row,
                "sequence_key": sequence_key,
                "clean84": int(sequence_key in clean_keys),
                "clean83_deduplicated": int(
                    sequence_key in clean_keys
                    and sequence_key != DUPLICATE_CLEAN_SEQUENCE
                ),
                "manifest_filled_count": source.get("filled_count", ""),
            }
        )
    sequence_rows: list[dict[str, object]] = []
    for (label, sequence_id), frame_count in sorted(
        sequence_counts.items(), key=lambda item: (INPUT_CLASSES.index(item[0][0]), natural_key(item[0][1]))
    ):
        sequence_key = f"{label}_{int(sequence_id):03d}"
        source = manifest_by_key.get(sequence_key, {})
        sequence_rows.append(
            {
                "input_class": label,
                "sequence_id": sequence_id,
                "sequence_key": sequence_key,
                "frame_count": frame_count,
                "clean84": int(sequence_key in clean_keys),
                "clean83_deduplicated": int(
                    sequence_key in clean_keys
                    and sequence_key != DUPLICATE_CLEAN_SEQUENCE
                ),
                "manifest_filled_count": source.get("filled_count", ""),
                "sequence_dir": str(Path(r"E:\Dataset\sequence") / label / sequence_id),
            }
        )
    return frame_rows, sequence_rows


def load_true_anchor_records() -> list[dict[str, object]]:
    rows = read_csv(TRUE_SCALE_DIR / "scale10_items.csv")
    rows.sort(key=lambda row: int(row["scale_level"]))
    if [int(row["scale_level"]) for row in rows] != list(range(10)):
        raise ValueError("True scale does not contain levels 0..9")
    result: list[dict[str, object]] = []
    for row in rows:
        source_path = Path(row["source_frame_path"])
        reference_path = TRUE_SCALE_DIR / row["reference_image"]
        if not source_path.is_file() or not reference_path.is_file():
            raise FileNotFoundError(source_path if not source_path.is_file() else reference_path)
        result.append(
            {
                **row,
                "source_frame_path": source_path,
                "reference_path": reference_path,
                "source_sha256": file_sha256(source_path),
                "reference_sha256": file_sha256(reference_path),
            }
        )
    return result


def inference_transform(fcmodel):
    return transforms.Compose(
        [
            transforms.Resize((fcmodel.IMG_HEIGHT_VGG16, fcmodel.IMG_WIDTH_VGG16)),
            transforms.ToTensor(),
            transforms.Normalize(
                (129.1863 / 255, 104.7624 / 255, 93.5940 / 255),
                (1.0, 1.0, 1.0),
            ),
        ]
    )


def load_models(device: torch.device, model_labels: Iterable[str]):
    sys.path.insert(0, str(SMILECOMP_DIR))
    import face_comp_torch as fcmodel

    models = {}
    extractor_states = []
    for label in model_labels:
        model = fcmodel.face_comp_siamese_model_vgg16based_for_predict(
            str(MODEL_WEIGHTS[label]), on_device=device
        ).to(device)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        models[label] = model
        extractor_states.append(
            {
                key: value.detach().cpu()
                for key, value in model.extractor.state_dict().items()
            }
        )
    reference = extractor_states[0]
    for state in extractor_states[1:]:
        for key in reference:
            if not torch.equal(reference[key], state[key]):
                raise RuntimeError(f"Frozen extractor mismatch: {key}")
    return fcmodel, models


def build_keep_roi(
    image: np.ndarray,
    landmarks: np.ndarray,
    condition: RoiCondition,
) -> tuple[np.ndarray, dict[str, float | int | list[float]]]:
    if condition.roi == "all":
        return image.copy(), {
            "face_pixels": 224 * 224,
            "roi_pixels": 224 * 224,
            "roi_fraction_of_face": 1.0,
            "split_y": float("nan"),
            "fill_rgb": [float("nan")] * 3,
            "roi_core_mae": 0.0,
            "outside_detail_ratio": float("nan"),
            "boundary_gradient_ratio": float("nan"),
        }

    face_mask, masks, split_y = build_natural_masks(image.shape[:2], landmarks)
    roi_mask = masks[condition.roi]
    alpha = feather_alpha(roi_mask, sigma=FEATHER_SIGMA)
    fill_rgb = [float("nan")] * 3
    if condition.fill_method == "blur":
        replacement = cv2.GaussianBlur(
            image, (0, 0), sigmaX=BLUR_SIGMA, sigmaY=BLUR_SIGMA
        ).astype(np.float32)
    elif condition.fill_method == "downsample16":
        low_resolution = cv2.resize(
            image,
            (DOWNSAMPLE_SIZE, DOWNSAMPLE_SIZE),
            interpolation=cv2.INTER_AREA,
        )
        replacement = cv2.resize(
            low_resolution,
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_CUBIC,
        ).astype(np.float32)
    elif condition.fill_method == "local_mean":
        source_mask = face_mask & ~roi_mask
        if int(source_mask.sum()) < 30:
            source_mask = face_mask
        color = image[source_mask].astype(np.float32).mean(axis=0)
        replacement = np.empty_like(image, dtype=np.float32)
        replacement[:] = color
        fill_rgb = [float(value) for value in color]
    else:
        raise ValueError(condition.fill_method)
    processed = (
        image.astype(np.float32) * alpha + replacement * (1.0 - alpha)
    )
    processed = np.clip(np.rint(processed), 0, 255).astype(np.uint8)

    core = morphology(roi_mask, 4, "erode")
    if not core.any():
        core = roi_mask
    roi_core_mae = float(
        np.abs(processed[core].astype(np.float32) - image[core].astype(np.float32)).mean()
    )
    outside = face_mask & ~roi_mask
    original_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    processed_gray = cv2.cvtColor(processed, cv2.COLOR_RGB2GRAY).astype(np.float32)
    original_lap = np.abs(cv2.Laplacian(original_gray, cv2.CV_32F))
    processed_lap = np.abs(cv2.Laplacian(processed_gray, cv2.CV_32F))
    denominator = float(original_lap[outside].mean()) if outside.any() else 0.0
    outside_detail_ratio = (
        float(processed_lap[outside].mean()) / denominator
        if denominator > 1e-8
        else float("nan")
    )
    dilated = morphology(roi_mask, 4, "dilate")
    eroded = morphology(roi_mask, 4, "erode")
    boundary = (dilated ^ eroded) & face_mask
    original_gradient = np.hypot(
        cv2.Sobel(original_gray, cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(original_gray, cv2.CV_32F, 0, 1, ksize=3),
    )
    processed_gradient = np.hypot(
        cv2.Sobel(processed_gray, cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(processed_gray, cv2.CV_32F, 0, 1, ksize=3),
    )
    boundary_denominator = (
        float(original_gradient[boundary].mean()) if boundary.any() else 0.0
    )
    boundary_gradient_ratio = (
        float(processed_gradient[boundary].mean()) / boundary_denominator
        if boundary_denominator > 1e-8
        else float("nan")
    )
    return processed, {
        "face_pixels": int(face_mask.sum()),
        "roi_pixels": int(roi_mask.sum()),
        "roi_fraction_of_face": float(roi_mask.sum()) / float(face_mask.sum()),
        "split_y": split_y,
        "fill_rgb": fill_rgb,
        "roi_core_mae": roi_core_mae,
        "outside_detail_ratio": outside_detail_ratio,
        "boundary_gradient_ratio": boundary_gradient_ratio,
    }


@torch.inference_mode()
def extract_features(model, tensors: list[torch.Tensor], device: torch.device) -> torch.Tensor:
    return model.extractor(torch.stack(tensors).to(device)).detach()


@torch.inference_mode()
def score_feature_batch(model, features: torch.Tensor, anchors: torch.Tensor):
    frame_count = int(features.shape[0])
    anchor_count = int(anchors.shape[0])
    repeated_frames = features.repeat_interleave(anchor_count, dim=0)
    repeated_anchors = anchors.repeat((frame_count, 1))
    logits_original = model.comp_layer(
        torch.cat((repeated_frames, repeated_anchors), dim=1)
    )
    logits_swapped = model.comp_layer(
        torch.cat((repeated_anchors, repeated_frames), dim=1)
    )
    probs_original = F.softmax(logits_original, dim=1)
    probs_swapped = F.softmax(logits_swapped, dim=1)
    probabilities = (
        (probs_original[:, 0] + probs_swapped[:, 1]) * 0.5
    ).reshape(frame_count, anchor_count)
    consistency = (
        torch.minimum(probs_original[:, 0], probs_swapped[:, 1])
        + torch.minimum(probs_original[:, 1], probs_swapped[:, 0])
    ).reshape(frame_count, anchor_count)
    symmetric_logit_margin = (
        0.5
        * (
            (logits_original[:, 0] - logits_original[:, 1])
            + (logits_swapped[:, 1] - logits_swapped[:, 0])
        )
    ).reshape(frame_count, anchor_count)
    raw_score = probabilities.sum(dim=1) - 0.5
    score = torch.clamp(raw_score, 0.0, float(anchor_count - 1))
    signed_response = 2.0 * probabilities.mean(dim=1) - 1.0
    return {
        "mean_probability": probabilities.mean(dim=1).cpu().numpy(),
        "signed_response": signed_response.cpu().numpy(),
        "mean_logit_margin": symmetric_logit_margin.mean(dim=1).cpu().numpy(),
        "mean_anchor_consistency": consistency.mean(dim=1).cpu().numpy(),
        "raw_score_unclamped": raw_score.cpu().numpy(),
        "score_0_9": score.cpu().numpy(),
    }


def representative_indices(frame_rows: list[dict[str, object]]) -> list[int]:
    allocations = {"true": 4, "polite": 3, "bitter": 3}
    selected: list[int] = []
    for label in INPUT_CLASSES:
        indices = [
            index
            for index, row in enumerate(frame_rows)
            if row["input_class"] == label
        ]
        positions = np.linspace(0, len(indices) - 1, allocations[label], dtype=int)
        selected.extend(indices[int(position)] for position in positions)
    return selected


def save_montage(
    image_paths: list[Path],
    labels: list[str],
    output_path: Path,
    title: str,
) -> None:
    if len(image_paths) != len(labels):
        raise ValueError("Image and label counts differ")
    cell_width, image_height, label_height = 224, 224, 34
    columns = 5
    rows = int(np.ceil(len(image_paths) / columns))
    canvas = Image.new(
        "RGB", (columns * cell_width, 44 + rows * (image_height + label_height)), (245, 247, 250)
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 10), title, font=load_font(20, bold=True), fill=(28, 49, 68))
    for index, (path, label) in enumerate(zip(image_paths, labels)):
        with Image.open(path) as source:
            image = source.convert("RGB").resize((224, 224), Image.Resampling.LANCZOS)
        x = (index % columns) * cell_width
        y = 44 + (index // columns) * (image_height + label_height)
        canvas.paste(image, (x, y))
        draw.rectangle((x, y + image_height, x + cell_width, y + image_height + label_height), fill=(28, 49, 68))
        draw.text((x + 7, y + image_height + 8), label, font=load_font(13), fill=(255, 255, 255))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
