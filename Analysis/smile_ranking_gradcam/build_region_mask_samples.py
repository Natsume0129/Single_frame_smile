from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import face_alignment
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import distance_transform_edt


DEFAULT_OUTPUT_DIR = Path(
    r"E:\Dataset\smile_ranking_region_mask_samples_20260817"
)

DEFAULT_SAMPLES = {
    "true": Path(
        r"E:\Dataset\smile_ranking_true\images\true_seq000_level05.png"
    ),
    "polite": Path(
        r"E:\Dataset\smile_ranking_polite\images\polite_seq035_level05.png"
    ),
    "bitter": Path(
        r"E:\Dataset\smile_ranking_bitter\images\bitter_seq000_level05.png"
    ),
}

FINE_REGIONS = ["periocular", "mouth_perioral", "nose", "cheeks"]
ALL_REGIONS = FINE_REGIONS + ["upper_face", "lower_face"]

DISPLAY_NAMES = {
    "original": "Original",
    "periocular": "Periocular",
    "mouth_perioral": "Mouth / perioral",
    "nose": "Nose",
    "cheeks": "Cheeks",
    "upper_face": "Upper face",
    "lower_face": "Lower face",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create reproducible facial-region masking samples from 68-point "
            "landmarks."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        Path(r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf"),
        Path(r"C:\Windows\Fonts\segoeuib.ttf" if bold else r"C:\Windows\Fonts\segoeui.ttf"),
    ]
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def convex_mask(shape: tuple[int, int], points: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    hull = cv2.convexHull(np.rint(points).astype(np.int32))
    cv2.fillConvexPoly(mask, hull, 1)
    return mask.astype(bool)


def polygon_mask(shape: tuple[int, int], points: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(mask, [np.rint(points).astype(np.int32)], 1)
    return mask.astype(bool)


def morphology(mask: np.ndarray, pixels: int, operation: str) -> np.ndarray:
    if pixels <= 0:
        return mask.copy()
    kernel_size = pixels * 2 + 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    source = mask.astype(np.uint8)
    if operation == "dilate":
        result = cv2.dilate(source, kernel)
    elif operation == "erode":
        result = cv2.erode(source, kernel)
    else:
        raise ValueError(operation)
    return result.astype(bool)


def face_outline_mask(
    shape: tuple[int, int], landmarks: np.ndarray
) -> np.ndarray:
    jaw = landmarks[0:17]
    brow = landmarks[17:27]
    face_width = float(np.linalg.norm(landmarks[16] - landmarks[0]))
    brow_top = float(brow[:, 1].min())
    chin_y = float(landmarks[8, 1])
    upper_height = max(20.0, 0.42 * (chin_y - brow_top))
    center_x = float(landmarks[27, 0])
    top_y = max(1.0, brow_top - upper_height)
    forehead = np.array(
        [
            [landmarks[16, 0], brow_top],
            [center_x + 0.46 * face_width, brow_top - 0.25 * upper_height],
            [center_x + 0.26 * face_width, top_y + 0.08 * upper_height],
            [center_x, top_y],
            [center_x - 0.26 * face_width, top_y + 0.08 * upper_height],
            [center_x - 0.46 * face_width, brow_top - 0.25 * upper_height],
            [landmarks[0, 0], brow_top],
        ],
        dtype=np.float32,
    )
    return polygon_mask(shape, np.vstack([jaw, forehead]))


def build_natural_masks(
    shape: tuple[int, int], landmarks: np.ndarray
) -> tuple[np.ndarray, dict[str, np.ndarray], float]:
    face = face_outline_mask(shape, landmarks)
    face_width = max(80.0, float(np.linalg.norm(landmarks[16] - landmarks[0])))

    eye_padding = max(4, int(round(0.038 * face_width)))
    mouth_padding = max(5, int(round(0.046 * face_width)))
    nose_padding = max(4, int(round(0.032 * face_width)))
    cheek_padding = max(2, int(round(0.018 * face_width)))

    eye_left = convex_mask(
        shape, landmarks[np.r_[17:22, 36:42]]
    )
    eye_right = convex_mask(
        shape, landmarks[np.r_[22:27, 42:48]]
    )
    periocular = morphology(eye_left | eye_right, eye_padding, "dilate")

    mouth = morphology(
        convex_mask(shape, landmarks[48:68]), mouth_padding, "dilate"
    )
    nose = morphology(
        convex_mask(shape, landmarks[27:36]), nose_padding, "dilate"
    )

    left_cheek_indices = np.array(
        [1, 2, 3, 4, 5, 31, 48, 39, 40, 41], dtype=int
    )
    right_cheek_indices = np.array(
        [11, 12, 13, 14, 15, 35, 54, 42, 43, 44], dtype=int
    )
    cheeks = morphology(
        convex_mask(shape, landmarks[left_cheek_indices])
        | convex_mask(shape, landmarks[right_cheek_indices]),
        cheek_padding,
        "dilate",
    )

    periocular &= face
    nose &= face & ~periocular
    mouth &= face & ~periocular & ~nose
    cheeks &= face & ~periocular & ~nose & ~mouth

    split_y = float(landmarks[30, 1])
    yy = np.arange(shape[0])[:, None]
    upper_face = face & (yy < split_y)
    lower_face = face & ~upper_face

    masks = {
        "periocular": periocular,
        "mouth_perioral": mouth,
        "nose": nose,
        "cheeks": cheeks,
        "upper_face": upper_face,
        "lower_face": lower_face,
    }
    if any(not mask.any() for mask in masks.values()):
        empty = [name for name, mask in masks.items() if not mask.any()]
        raise RuntimeError(f"Empty masks: {empty}")
    return face, masks, split_y


def exact_area_mask(
    mask: np.ndarray, face_mask: np.ndarray, target_pixels: int
) -> np.ndarray:
    available = np.flatnonzero(face_mask)
    target_pixels = max(1, min(int(target_pixels), len(available)))
    if int(mask.sum()) == target_pixels:
        return mask.copy()

    if int(mask.sum()) > target_pixels:
        distance = distance_transform_edt(mask)
        candidate_indices = np.flatnonzero(mask)
        order = np.argsort(distance.flat[candidate_indices])[::-1]
        selected = candidate_indices[order[:target_pixels]]
    else:
        distance = distance_transform_edt(~mask).astype(np.float64)
        distance[~face_mask] = np.inf
        candidate_indices = np.flatnonzero(face_mask)
        order = np.argsort(distance.flat[candidate_indices])
        selected = candidate_indices[order[:target_pixels]]

    result = np.zeros_like(mask, dtype=bool)
    result.flat[selected] = True
    return result


def feather_alpha(mask: np.ndarray, sigma: float = 3.0) -> np.ndarray:
    alpha = cv2.GaussianBlur(
        mask.astype(np.float32), (0, 0), sigmaX=sigma, sigmaY=sigma
    )
    return np.clip(alpha[..., None], 0.0, 1.0)


def local_mean_fill(
    image: np.ndarray, mask: np.ndarray, face_mask: np.ndarray
) -> tuple[np.ndarray, tuple[float, float, float]]:
    ring = morphology(mask, 12, "dilate") & face_mask & ~mask
    if int(ring.sum()) < 30:
        ring = face_mask & ~mask
    if int(ring.sum()) < 30:
        raise RuntimeError("Not enough valid pixels to estimate local mean.")
    color = image[ring].astype(np.float32).mean(axis=0)
    replacement = np.empty_like(image, dtype=np.float32)
    replacement[:] = color
    alpha = feather_alpha(mask)
    result = (
        image.astype(np.float32) * (1.0 - alpha) + replacement * alpha
    )
    return np.clip(np.rint(result), 0, 255).astype(np.uint8), tuple(color)


def strong_blur_fill(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=11.0, sigmaY=11.0)
    alpha = feather_alpha(mask)
    result = (
        image.astype(np.float32) * (1.0 - alpha)
        + blurred.astype(np.float32) * alpha
    )
    return np.clip(np.rint(result), 0, 255).astype(np.uint8)


def overlay_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    result = image.astype(np.float32).copy()
    overlay_color = np.array([230.0, 65.0, 65.0], dtype=np.float32)
    result[mask] = result[mask] * 0.55 + overlay_color * 0.45
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    rendered = np.clip(np.rint(result), 0, 255).astype(np.uint8)
    cv2.drawContours(rendered, contours, -1, (255, 225, 60), 2)
    return rendered


def save_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image, mode="RGB").save(path)


def make_montage(
    output_path: Path,
    title: str,
    rows: dict[str, dict[str, np.ndarray]],
    columns: list[str],
) -> None:
    image_size = 224
    left_width = 122
    gap = 12
    title_height = 54
    header_height = 48
    row_height = image_size + 28
    width = left_width + len(columns) * (image_size + gap) + gap
    height = title_height + header_height + len(rows) * row_height + gap
    canvas = Image.new("RGB", (width, height), (248, 248, 248))
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(25, bold=True)
    header_font = load_font(16, bold=True)
    row_font = load_font(22, bold=True)
    draw.text((18, 13), title, fill=(25, 25, 25), font=title_font)

    for column_index, name in enumerate(columns):
        x = left_width + column_index * (image_size + gap)
        label = DISPLAY_NAMES[name]
        box = draw.textbbox((0, 0), label, font=header_font)
        text_width = box[2] - box[0]
        draw.text(
            (x + (image_size - text_width) / 2, title_height + 14),
            label,
            fill=(45, 45, 45),
            font=header_font,
        )

    for row_index, (class_label, images) in enumerate(rows.items()):
        y = title_height + header_height + row_index * row_height
        draw.text(
            (18, y + image_size / 2 - 15),
            class_label.capitalize(),
            fill=(25, 25, 25),
            font=row_font,
        )
        for column_index, name in enumerate(columns):
            x = left_width + column_index * (image_size + gap)
            cell = Image.fromarray(images[name], mode="RGB")
            canvas.paste(cell, (x, y))
            draw.rectangle(
                (x, y, x + image_size - 1, y + image_size - 1),
                outline=(185, 185, 185),
                width=1,
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(
            f"Output directory already exists; choose a new path: {args.output_dir}"
        )
    for path in DEFAULT_SAMPLES.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    aligner = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D,
        flip_input=False,
        device=args.device,
        compile=False,
    )

    sample_data: dict[str, dict[str, object]] = {}
    natural_fractions: list[float] = []
    for class_label, source_path in DEFAULT_SAMPLES.items():
        image = np.asarray(Image.open(source_path).convert("RGB"))
        if image.shape[:2] != (224, 224):
            raise ValueError(f"Expected 224x224 input: {source_path} {image.shape}")
        predictions = aligner.get_landmarks(image)
        if not predictions:
            raise RuntimeError(f"No 68-point landmarks detected: {source_path}")
        landmarks = max(
            predictions,
            key=lambda points: float(
                np.ptp(points[:, 0]) * np.ptp(points[:, 1])
            ),
        )
        face_mask, natural_masks, split_y = build_natural_masks(
            image.shape[:2], landmarks
        )
        face_pixels = int(face_mask.sum())
        for region in FINE_REGIONS:
            natural_fractions.append(
                float(natural_masks[region].sum()) / face_pixels
            )
        sample_data[class_label] = {
            "source_path": source_path,
            "image": image,
            "landmarks": landmarks,
            "face_mask": face_mask,
            "natural_masks": natural_masks,
            "split_y": split_y,
        }

    area_target_fraction = float(np.median(natural_fractions))
    output_rows: list[dict[str, object]] = []
    natural_mean_rows: dict[str, dict[str, np.ndarray]] = {}
    natural_blur_rows: dict[str, dict[str, np.ndarray]] = {}
    natural_overlay_rows: dict[str, dict[str, np.ndarray]] = {}
    matched_mean_rows: dict[str, dict[str, np.ndarray]] = {}
    matched_overlay_rows: dict[str, dict[str, np.ndarray]] = {}

    for class_label, data in sample_data.items():
        image = data["image"]
        face_mask = data["face_mask"]
        natural_masks = data["natural_masks"]
        face_pixels = int(face_mask.sum())
        target_pixels = int(round(area_target_fraction * face_pixels))

        natural_mean_rows[class_label] = {"original": image}
        natural_blur_rows[class_label] = {"original": image}
        natural_overlay_rows[class_label] = {"original": image}
        matched_mean_rows[class_label] = {"original": image}
        matched_overlay_rows[class_label] = {"original": image}

        for region in ALL_REGIONS:
            mask = natural_masks[region]
            mean_image, color = local_mean_fill(image, mask, face_mask)
            blur_image = strong_blur_fill(image, mask)
            overlay_image = overlay_mask(image, mask)
            natural_mean_rows[class_label][region] = mean_image
            natural_blur_rows[class_label][region] = blur_image
            natural_overlay_rows[class_label][region] = overlay_image
            save_rgb(
                args.output_dir
                / "natural_local_mean"
                / f"{class_label}_{region}.png",
                mean_image,
            )
            save_rgb(
                args.output_dir / "natural_blur" / f"{class_label}_{region}.png",
                blur_image,
            )
            save_rgb(
                args.output_dir
                / "natural_region_overlays"
                / f"{class_label}_{region}.png",
                overlay_image,
            )
            row = {
                "class_label": class_label,
                "source_path": str(data["source_path"]),
                "region": region,
                "mask_variant": "natural_anatomical",
                "mask_pixels": int(mask.sum()),
                "face_pixels": face_pixels,
                "mask_fraction_of_face": float(mask.sum()) / face_pixels,
                "local_mean_r": color[0],
                "local_mean_g": color[1],
                "local_mean_b": color[2],
                "split_y_for_upper_lower": float(data["split_y"]),
            }
            output_rows.append(row)

        for region in FINE_REGIONS:
            natural_mask = natural_masks[region]
            matched_mask = exact_area_mask(
                natural_mask, face_mask, target_pixels
            )
            mean_image, color = local_mean_fill(
                image, matched_mask, face_mask
            )
            overlay_image = overlay_mask(image, matched_mask)
            matched_mean_rows[class_label][region] = mean_image
            matched_overlay_rows[class_label][region] = overlay_image
            save_rgb(
                args.output_dir
                / "area_matched_local_mean"
                / f"{class_label}_{region}.png",
                mean_image,
            )
            save_rgb(
                args.output_dir
                / "area_matched_region_overlays"
                / f"{class_label}_{region}.png",
                overlay_image,
            )
            output_rows.append(
                {
                    "class_label": class_label,
                    "source_path": str(data["source_path"]),
                    "region": region,
                    "mask_variant": "area_matched",
                    "mask_pixels": int(matched_mask.sum()),
                    "face_pixels": face_pixels,
                    "mask_fraction_of_face": (
                        float(matched_mask.sum()) / face_pixels
                    ),
                    "local_mean_r": color[0],
                    "local_mean_g": color[1],
                    "local_mean_b": color[2],
                    "split_y_for_upper_lower": float(data["split_y"]),
                }
            )

    make_montage(
        args.output_dir / "natural_local_mean_montage.png",
        "Natural anatomical masks — local mean fill with feathered edges",
        natural_mean_rows,
        ["original"] + ALL_REGIONS,
    )
    make_montage(
        args.output_dir / "natural_blur_montage.png",
        "Natural anatomical masks — strong Gaussian blur robustness check",
        natural_blur_rows,
        ["original"] + ALL_REGIONS,
    )
    make_montage(
        args.output_dir / "natural_region_overlay_montage.png",
        "Natural anatomical mask coverage — red fill with yellow boundary",
        natural_overlay_rows,
        ["original"] + ALL_REGIONS,
    )
    make_montage(
        args.output_dir / "area_matched_local_mean_montage.png",
        (
            "Area-matched fine-region masks — local mean fill "
            f"({area_target_fraction:.1%} of face)"
        ),
        matched_mean_rows,
        ["original"] + FINE_REGIONS,
    )
    make_montage(
        args.output_dir / "area_matched_region_overlay_montage.png",
        (
            "Area-matched fine-region coverage "
            f"({area_target_fraction:.1%} of face)"
        ),
        matched_overlay_rows,
        ["original"] + FINE_REGIONS,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "mask_manifest.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)

    summary = {
        "created_for": "smile-ranking facial-region occlusion pilot",
        "landmark_method": "face-alignment 1.5.0, 68-point 2D FAN",
        "input_size": [224, 224],
        "samples": {
            label: str(path) for label, path in DEFAULT_SAMPLES.items()
        },
        "natural_regions": ALL_REGIONS,
        "area_matched_regions": FINE_REGIONS,
        "area_matched_target_fraction_of_face": area_target_fraction,
        "primary_fill": "per-image local surrounding-face mean with Gaussian-feathered edge",
        "robustness_fill": "strong Gaussian blur with Gaussian-feathered edge",
        "upper_lower_rule": "complementary split inside face outline at landmark 30 y-coordinate",
        "montages": [
            "natural_local_mean_montage.png",
            "natural_blur_montage.png",
            "natural_region_overlay_montage.png",
            "area_matched_local_mean_montage.png",
            "area_matched_region_overlay_montage.png",
        ],
    }
    (args.output_dir / "mask_sample_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
