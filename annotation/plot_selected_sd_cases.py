"""Plot selected sequence s-d curves with frame thumbnails.

The common axis is defined by true/5: last feature - first feature. Each
sequence is then projected from its own first-frame feature onto that axis.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import OrderedDict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image, ImageOps
from torchvision import transforms


DEFAULT_SMILECOMP_DIR = Path(
    r"E:\SmileAnnotation\FaceTracking-Smile_Detection\SmileComp_SiameseNet\Pytorch-Shimonishi"
)
DEFAULT_SEQUENCE_ROOT = Path(r"E:\Dataset\sequence")
DEFAULT_OUTPUT_DIR = Path(r"E:\Dataset\sd_plot_selected_cases")
DEFAULT_WEIGHTS = Path(
    r"E:\Dataset\smile_ranking_true\model\smile_rank_true_cross_sequence_100ep.pth"
)
DEFAULT_CASES = OrderedDict(
    [
        ("bitter", ["8", "11", "12", "13", "15", "17", "23"]),
        ("true", ["0", "2", "5", "10"]),
        ("polite", ["8", "14", "17", "18", "21"]),
    ]
)
BASELINE_LABEL = "true"
BASELINE_ID = "5"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}
CLASS_CMAPS = {
    "bitter": "Reds",
    "true": "Blues",
    "polite": "Greens",
}
CLASS_BASE_COLORS = {
    "bitter": "#c9332b",
    "true": "#1f77b4",
    "polite": "#2f8f46",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract SmileComp features and plot selected sequences as s-d curves "
            "against the true/5 first-to-last baseline axis."
        )
    )
    parser.add_argument("--smilecomp_dir", type=Path, default=DEFAULT_SMILECOMP_DIR)
    parser.add_argument("--sequence_root", type=Path, default=DEFAULT_SEQUENCE_ROOT)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--baseline_label", default=BASELINE_LABEL)
    parser.add_argument("--baseline_id", default=BASELINE_ID)
    parser.add_argument(
        "--cases",
        default="",
        help="Optional format: bitter:8,11;true:0,2,5,10;polite:8,14",
    )
    parser.add_argument("--batch_size", type=int, default=96)
    parser.add_argument("--thumbnail_stride", type=int, default=8)
    parser.add_argument("--thumbnail_size", type=int, default=76)
    parser.add_argument("--thumbnail_zoom", type=float, default=0.72)
    parser.add_argument("--figure_width", type=float, default=32.0)
    parser.add_argument("--figure_height", type=float, default=24.0)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def parse_cases(case_spec: str) -> OrderedDict[str, list[str]]:
    if not case_spec.strip():
        return OrderedDict((label, list(ids)) for label, ids in DEFAULT_CASES.items())

    parsed: OrderedDict[str, list[str]] = OrderedDict()
    for block in case_spec.split(";"):
        block = block.strip()
        if not block:
            continue
        if ":" not in block:
            raise ValueError(f"Invalid case block: {block!r}")
        label, raw_ids = block.split(":", 1)
        ids = [value.strip() for value in raw_ids.split(",") if value.strip()]
        if not label.strip() or not ids:
            raise ValueError(f"Invalid case block: {block!r}")
        parsed[label.strip()] = ids
    return parsed


def natural_key(value: str):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)]


def list_frame_paths(sequence_dir: Path) -> list[Path]:
    frames = [p for p in sequence_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    return sorted(frames, key=lambda p: natural_key(p.name))


def prepare_transform(fcmodel):
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


def load_image_tensor(path: Path, transform) -> torch.Tensor:
    with Image.open(path) as img:
        return transform(img.convert("RGB"))


def extract_features(
    model,
    frame_paths: list[Path],
    transform,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    features = []
    with torch.no_grad():
        for start in range(0, len(frame_paths), batch_size):
            batch_paths = frame_paths[start : start + batch_size]
            batch = torch.stack([load_image_tensor(path, transform) for path in batch_paths]).to(device)
            features.append(model.extractor(batch).detach().cpu().float())
    return torch.cat(features, dim=0).numpy()


def compute_sd_coordinates(features: np.ndarray, axis_unit: np.ndarray, axis_norm: float) -> dict[str, np.ndarray]:
    delta = features - features[0:1]
    s = delta @ axis_unit
    residual = delta - s[:, None] * axis_unit[None, :]
    d = np.linalg.norm(residual, axis=1)
    return {
        "s": s,
        "d": d,
        "s_ratio": s / axis_norm,
        "d_ratio": d / axis_norm,
    }


def choose_thumbnail_indices(num_frames: int, stride: int) -> list[int]:
    if stride <= 0:
        raise ValueError("--thumbnail_stride must be > 0")
    indices = list(range(0, num_frames, stride))
    last = num_frames - 1
    if indices[-1] != last:
        indices.append(last)
    return indices


def load_thumbnail(path: Path, size: int) -> np.ndarray:
    with Image.open(path) as img:
        thumb = ImageOps.fit(img.convert("RGB"), (size, size), method=Image.Resampling.LANCZOS)
    return np.asarray(thumb)


def sequence_color(label: str, index: int, count: int):
    cmap_name = CLASS_CMAPS.get(label)
    if not cmap_name:
        return CLASS_BASE_COLORS.get(label, "#333333")
    position = 0.48 if count <= 1 else 0.42 + 0.48 * (index / (count - 1))
    return plt.get_cmap(cmap_name)(position)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_sd_curves(
    output_path: Path,
    results: list[dict],
    baseline_label: str,
    baseline_id: str,
    thumbnail_stride: int,
    thumbnail_size: int,
    thumbnail_zoom: float,
    figure_width: float,
    figure_height: float,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(figure_width, figure_height), dpi=dpi)
    label_counts = {label: sum(1 for item in results if item["label"] == label) for label in CLASS_CMAPS}
    label_seen = {label: 0 for label in CLASS_CMAPS}

    for result in results:
        label = result["label"]
        label_index = label_seen.get(label, 0)
        label_seen[label] = label_index + 1
        color = sequence_color(label, label_index, max(label_counts.get(label, 1), 1))
        line_width = 3.0 if label == baseline_label and result["sequence_id"] == baseline_id else 1.8
        alpha = 0.98 if line_width > 2.0 else 0.78

        ax.plot(
            result["s"],
            result["d"],
            color=color,
            linewidth=line_width,
            alpha=alpha,
            marker="o",
            markersize=2.2,
            label=f"{label}/{result['sequence_id']} (n={len(result['frame_paths'])})",
            zorder=2,
        )
        ax.scatter(result["s"][0], result["d"][0], color=color, s=32, marker="o", zorder=4)
        ax.scatter(result["s"][-1], result["d"][-1], color=color, s=52, marker="s", zorder=4)
        ax.text(
            float(result["s"][-1]),
            float(result["d"][-1]),
            f" {label}/{result['sequence_id']}",
            color=color,
            fontsize=9,
            weight="bold" if line_width > 2.0 else "normal",
            zorder=5,
        )

        for frame_idx in result["thumbnail_indices"]:
            imagebox = OffsetImage(
                load_thumbnail(result["frame_paths"][frame_idx], thumbnail_size),
                zoom=thumbnail_zoom,
            )
            ab = AnnotationBbox(
                imagebox,
                (float(result["s"][frame_idx]), float(result["d"][frame_idx])),
                frameon=True,
                pad=0.02,
                bboxprops={
                    "edgecolor": color,
                    "linewidth": 1.1,
                    "alpha": 0.88,
                },
                zorder=6,
            )
            ax.add_artist(ab)

    ax.axhline(0.0, color="#555555", linewidth=1.0, linestyle="--", alpha=0.55)
    ax.set_xlabel("s: progress along true/5 first-to-last feature axis")
    ax.set_ylabel("d: deviation from the true/5 feature axis")
    ax.set_title(
        "Selected sequence s-d plot using SmileComp feature vectors\n"
        f"Baseline axis: {baseline_label}/{baseline_id}; thumbnails every {thumbnail_stride} frames"
    )
    ax.grid(True, color="#d5d9dd", linewidth=0.65, alpha=0.8)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=9, frameon=True)
    fig.tight_layout(rect=(0.0, 0.0, 0.84, 1.0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    cases = parse_cases(args.cases)

    if not args.smilecomp_dir.is_dir():
        raise FileNotFoundError(args.smilecomp_dir)
    if not args.sequence_root.is_dir():
        raise FileNotFoundError(args.sequence_root)
    if not args.weights.is_file():
        raise FileNotFoundError(args.weights)

    sys.path.insert(0, str(args.smilecomp_dir))
    import face_comp_torch as fcmodel  # noqa: PLC0415

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    transform = prepare_transform(fcmodel)
    model = fcmodel.face_comp_siamese_model_vgg16based_for_predict(
        str(args.weights), on_device=device
    ).to(device)
    model.eval()

    selected = []
    manifest_rows = []
    feature_cache: dict[tuple[str, str], tuple[list[Path], np.ndarray]] = {}

    for label, ids in cases.items():
        for sequence_id in ids:
            sequence_dir = args.sequence_root / label / sequence_id
            row = {
                "label": label,
                "sequence_id": sequence_id,
                "sequence_dir": str(sequence_dir),
                "status": "ok",
                "reason": "",
                "frame_count": 0,
            }
            if not sequence_dir.is_dir():
                row["status"] = "missing"
                row["reason"] = "sequence directory does not exist"
                manifest_rows.append(row)
                continue
            frame_paths = list_frame_paths(sequence_dir)
            if len(frame_paths) < 2:
                row["status"] = "skipped"
                row["reason"] = "needs at least 2 frames"
                row["frame_count"] = len(frame_paths)
                manifest_rows.append(row)
                continue
            row["frame_count"] = len(frame_paths)
            manifest_rows.append(row)
            selected.append((label, sequence_id, frame_paths))

    baseline_key = (args.baseline_label, args.baseline_id)
    baseline_dir = args.sequence_root / args.baseline_label / args.baseline_id
    if not baseline_dir.is_dir():
        raise FileNotFoundError(f"Baseline sequence does not exist: {baseline_dir}")
    baseline_frames = list_frame_paths(baseline_dir)
    if len(baseline_frames) < 2:
        raise RuntimeError(f"Baseline sequence needs at least 2 frames: {baseline_dir}")

    all_keys = OrderedDict()
    all_keys[baseline_key] = baseline_frames
    for label, sequence_id, frame_paths in selected:
        all_keys[(label, sequence_id)] = frame_paths

    for (label, sequence_id), frame_paths in all_keys.items():
        features = extract_features(model, frame_paths, transform, device, args.batch_size)
        feature_cache[(label, sequence_id)] = (frame_paths, features)
        print(f"[features] {label}/{sequence_id}: {len(frame_paths)} frames")

    _, baseline_features = feature_cache[baseline_key]
    axis = baseline_features[-1] - baseline_features[0]
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= 1e-12:
        raise RuntimeError("Baseline axis norm is too small for s-d projection.")
    axis_unit = axis / axis_norm

    results = []
    coordinate_rows = []
    thumbnail_rows = []
    for label, sequence_id, _ in selected:
        frame_paths, features = feature_cache[(label, sequence_id)]
        coords = compute_sd_coordinates(features, axis_unit, axis_norm)
        thumbnail_indices = choose_thumbnail_indices(len(frame_paths), args.thumbnail_stride)
        result = {
            "label": label,
            "sequence_id": sequence_id,
            "frame_paths": frame_paths,
            "s": coords["s"],
            "d": coords["d"],
            "s_ratio": coords["s_ratio"],
            "d_ratio": coords["d_ratio"],
            "thumbnail_indices": thumbnail_indices,
        }
        results.append(result)

        thumbnail_index_set = set(thumbnail_indices)
        for frame_idx, frame_path in enumerate(frame_paths):
            is_thumbnail = frame_idx in thumbnail_index_set
            coordinate_rows.append(
                {
                    "label": label,
                    "sequence_id": sequence_id,
                    "frame_index_from_onset": frame_idx,
                    "frame_file": frame_path.name,
                    "frame_path": str(frame_path),
                    "s": f"{float(coords['s'][frame_idx]):.8f}",
                    "d": f"{float(coords['d'][frame_idx]):.8f}",
                    "s_ratio": f"{float(coords['s_ratio'][frame_idx]):.8f}",
                    "d_ratio": f"{float(coords['d_ratio'][frame_idx]):.8f}",
                    "is_thumbnail": int(is_thumbnail),
                }
            )
            if is_thumbnail:
                thumbnail_rows.append(
                    {
                        "label": label,
                        "sequence_id": sequence_id,
                        "frame_index_from_onset": frame_idx,
                        "frame_file": frame_path.name,
                        "frame_path": str(frame_path),
                        "s": f"{float(coords['s'][frame_idx]):.8f}",
                        "d": f"{float(coords['d'][frame_idx]):.8f}",
                    }
                )

    if not results:
        raise RuntimeError("No valid selected sequences were available for plotting.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = args.output_dir / "selected_cases_sd_plot.png"
    plot_sd_curves(
        plot_path,
        results,
        args.baseline_label,
        args.baseline_id,
        args.thumbnail_stride,
        args.thumbnail_size,
        args.thumbnail_zoom,
        args.figure_width,
        args.figure_height,
        args.dpi,
    )

    manifest_fields = ["label", "sequence_id", "sequence_dir", "status", "reason", "frame_count"]
    write_csv(args.output_dir / "selected_cases_manifest.csv", manifest_fields, manifest_rows)
    coordinate_fields = [
        "label",
        "sequence_id",
        "frame_index_from_onset",
        "frame_file",
        "frame_path",
        "s",
        "d",
        "s_ratio",
        "d_ratio",
        "is_thumbnail",
    ]
    write_csv(args.output_dir / "selected_cases_sd_coordinates.csv", coordinate_fields, coordinate_rows)
    thumbnail_fields = ["label", "sequence_id", "frame_index_from_onset", "frame_file", "frame_path", "s", "d"]
    write_csv(args.output_dir / "selected_cases_thumbnail_frames.csv", thumbnail_fields, thumbnail_rows)

    summary = {
        "sequence_root": str(args.sequence_root),
        "output_dir": str(args.output_dir),
        "weights": str(args.weights),
        "device": str(device),
        "baseline_label": args.baseline_label,
        "baseline_id": args.baseline_id,
        "baseline_sequence_dir": str(baseline_dir),
        "baseline_frame_count": len(baseline_frames),
        "baseline_axis_norm": axis_norm,
        "selected_cases": cases,
        "valid_sequence_count": len(results),
        "total_frame_count": len(coordinate_rows),
        "thumbnail_stride": args.thumbnail_stride,
        "total_thumbnail_count": len(thumbnail_rows),
        "method": (
            "Features are SmileComp extractor outputs. The baseline axis is "
            "feature(true/5 last frame) - feature(true/5 first frame). For each "
            "sequence, delta_t = feature_t - feature_sequence_first; "
            "s = dot(delta_t, axis_unit); d = norm(delta_t - s * axis_unit)."
        ),
        "plot_path": str(plot_path),
        "manifest_csv": str(args.output_dir / "selected_cases_manifest.csv"),
        "coordinates_csv": str(args.output_dir / "selected_cases_sd_coordinates.csv"),
        "thumbnail_frames_csv": str(args.output_dir / "selected_cases_thumbnail_frames.csv"),
    }
    with (args.output_dir / "selected_cases_sd_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
