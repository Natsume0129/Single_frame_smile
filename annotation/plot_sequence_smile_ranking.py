"""Plot frame-level smile ranking curves for every sequence."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.lines import Line2D
from PIL import Image
from torchvision import transforms


DEFAULT_SMILECOMP_DIR = Path(
    r"E:\SmileAnnotation\FaceTracking-Smile_Detection\SmileComp_SiameseNet\Pytorch-Shimonishi"
)
DEFAULT_SEQUENCE_ROOT = Path(r"E:\Dataset\sequence")
DEFAULT_OUTPUT_ROOT = Path(r"E:\Dataset\smileranking_plot")
DEFAULT_SCALE_DIR = Path(r"E:\Dataset\smile_ranking_true\scale10")
DEFAULT_WEIGHTS = Path(
    r"E:\Dataset\smile_ranking_true\model\smile_rank_true_cross_sequence_100ep.pth"
)
LABELS = ("true", "polite", "bitter")
COLORS = {
    "true": "#1f77b4",
    "polite": "#2ca02c",
    "bitter": "#d62728",
}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use a trained SmileComp model and 10-level scale to plot smile ranking curves."
    )
    parser.add_argument("--smilecomp_dir", type=Path, default=DEFAULT_SMILECOMP_DIR)
    parser.add_argument("--sequence_root", type=Path, default=DEFAULT_SEQUENCE_ROOT)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--scale_dir", type=Path, default=DEFAULT_SCALE_DIR)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--labels", nargs="+", default=list(LABELS))
    return parser.parse_args()


def natural_key(value: str):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)]


def safe_sequence_name(label: str, sequence_id: str) -> str:
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", sequence_id)
    if sequence_id.isdigit():
        safe_id = f"{int(sequence_id):03d}"
    return f"{label}_seq{safe_id}"


def parse_frame_number_from_name(path: Path) -> int | None:
    return int(path.stem) if path.stem.isdigit() else None


def format_optional_int(value: int | None) -> str:
    return "" if value is None else str(value)


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


def load_scale_anchor_paths(scale_dir: Path) -> list[Path]:
    items_csv = scale_dir / "scale10_items.csv"
    if items_csv.is_file():
        rows = []
        with items_csv.open(newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                rows.append(row)
        rows.sort(key=lambda row: int(row["scale_level"]))
        paths = [scale_dir / row["reference_image"] for row in rows]
    else:
        paths = [scale_dir / f"level_{level:02d}.png" for level in range(10)]

    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing scale anchor image: {missing[0]}")
    return paths


def list_sequence_dirs(sequence_root: Path, label: str) -> list[Path]:
    label_dir = sequence_root / label
    if not label_dir.is_dir():
        return []
    return sorted([p for p in label_dir.iterdir() if p.is_dir()], key=lambda p: natural_key(p.name))


def list_frame_paths(sequence_dir: Path) -> list[Path]:
    frames = [p for p in sequence_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    return sorted(frames, key=lambda p: natural_key(p.name))


def extract_features(model, tensors: list[torch.Tensor], device, batch_size: int) -> torch.Tensor:
    features = []
    with torch.no_grad():
        for start in range(0, len(tensors), batch_size):
            batch = torch.stack(tensors[start : start + batch_size]).to(device)
            features.append(model.extractor(batch).detach())
    return torch.cat(features, dim=0)


def score_features_against_anchors(
    model,
    frame_features: torch.Tensor,
    anchor_features: torch.Tensor,
    batch_size: int,
):
    num_frames = frame_features.shape[0]
    num_anchors = anchor_features.shape[0]
    all_probs = []
    all_consistency = []

    with torch.no_grad():
        for start in range(0, num_frames, batch_size):
            frame_batch = frame_features[start : start + batch_size]
            current_batch = frame_batch.shape[0]
            repeated_frames = frame_batch.repeat_interleave(num_anchors, dim=0)
            repeated_anchors = anchor_features.repeat((current_batch, 1))

            logits_org = model.comp_layer(torch.cat((repeated_frames, repeated_anchors), dim=1))
            logits_swp = model.comp_layer(torch.cat((repeated_anchors, repeated_frames), dim=1))
            probs_org = F.softmax(logits_org, dim=1)
            probs_swp = F.softmax(logits_swp, dim=1)

            # Class 0 means first image is stronger; class 1 in swapped order means anchor is weaker.
            p_frame_stronger = ((probs_org[:, 0] + probs_swp[:, 1]) * 0.5).reshape(
                current_batch, num_anchors
            )
            consistency = (
                torch.minimum(probs_org[:, 0], probs_swp[:, 1])
                + torch.minimum(probs_org[:, 1], probs_swp[:, 0])
            ).reshape(current_batch, num_anchors)

            all_probs.append(p_frame_stronger.detach().cpu().numpy())
            all_consistency.append(consistency.detach().cpu().numpy())

    probs = np.vstack(all_probs)
    consistency = np.vstack(all_consistency)
    soft_win_sum = probs.sum(axis=1)
    raw_score = soft_win_sum - 0.5
    score_0_9 = np.clip(raw_score, 0.0, float(num_anchors - 1))
    nearest_level = np.rint(score_0_9).astype(int)
    return probs, consistency, soft_win_sum, raw_score, score_0_9, nearest_level


def plot_single_sequence(
    output_path: Path,
    label: str,
    sequence_id: str,
    frame_indices: list[int],
    scores: np.ndarray,
) -> None:
    width = min(14.0, max(7.0, len(frame_indices) / 14.0))
    fig, ax = plt.subplots(figsize=(width, 4.2), dpi=150)
    color = COLORS.get(label, "#333333")
    ax.plot(frame_indices, scores, color=color, linewidth=1.5, marker="o", markersize=2.4)
    ax.set_title(f"{label} sequence {sequence_id} ({len(frame_indices)} frames)")
    ax.set_xlabel("Frame index from onset")
    ax.set_ylabel("Smile ranking (0-9)")
    ax.set_ylim(-0.25, 9.25)
    ax.set_yticks(range(10))
    ax.grid(True, color="#d0d4d8", linewidth=0.6, alpha=0.8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_all_sequences(output_path: Path, sequence_results: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(18, 10), dpi=160)
    counts = {label: 0 for label in LABELS}

    for result in sequence_results:
        label = result["label"]
        counts[label] = counts.get(label, 0) + 1
        ax.plot(
            result["frame_numbers"],
            result["scores"],
            color=COLORS.get(label, "#333333"),
            alpha=0.34,
            linewidth=0.85,
            marker="o",
            markersize=1.2,
        )

    handles = [
        Line2D([0], [0], color=COLORS[label], lw=2.2, label=f"{label} ({counts.get(label, 0)})")
        for label in LABELS
        if counts.get(label, 0)
    ]
    ax.legend(handles=handles, title="Class", loc="upper right")
    ax.set_title("All sequence smile ranking curves")
    ax.set_xlabel("Frame index from onset")
    ax.set_ylabel("Smile ranking (0-9)")
    ax.set_ylim(-0.25, 9.25)
    ax.set_yticks(range(10))
    ax.grid(True, color="#d0d4d8", linewidth=0.6, alpha=0.8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    fig.savefig(output_path.with_suffix(".svg"))
    plt.close(fig)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
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

    anchor_paths = load_scale_anchor_paths(args.scale_dir)
    anchor_tensors = [load_image_tensor(path, transform) for path in anchor_paths]

    model = fcmodel.face_comp_siamese_model_vgg16based_for_predict(
        str(args.weights), on_device=device
    ).to(device)
    model.eval()
    anchor_features = extract_features(model, anchor_tensors, device, args.batch_size)

    for label in args.labels:
        (args.output_root / label).mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    frame_rows = []
    sequence_results = []

    for label in args.labels:
        sequence_dirs = list_sequence_dirs(args.sequence_root, label)
        for seq_idx, sequence_dir in enumerate(sequence_dirs, start=1):
            frame_paths = list_frame_paths(sequence_dir)
            if not frame_paths:
                continue

            frame_tensors = [load_image_tensor(path, transform) for path in frame_paths]
            frame_features = extract_features(model, frame_tensors, device, args.batch_size)
            probs, consistency, soft_win_sum, raw_score, score_0_9, nearest_level = (
                score_features_against_anchors(
                    model, frame_features, anchor_features, args.batch_size
                )
            )

            frame_indices = list(range(len(frame_paths)))
            frame_numbers_from_name = [parse_frame_number_from_name(frame_path) for frame_path in frame_paths]
            sequence_name = safe_sequence_name(label, sequence_dir.name)
            plot_path = args.output_root / label / f"{sequence_name}_smileranking.png"
            plot_single_sequence(plot_path, label, sequence_dir.name, frame_indices, score_0_9)

            manifest_rows.append(
                {
                    "label": label,
                    "sequence_id": sequence_dir.name,
                    "sequence_dir": str(sequence_dir),
                    "frame_count": len(frame_paths),
                    "plot_path": str(plot_path),
                    "first_frame_file": frame_paths[0].name,
                    "last_frame_file": frame_paths[-1].name,
                    "min_score_0_9": f"{float(np.min(score_0_9)):.6f}",
                    "max_score_0_9": f"{float(np.max(score_0_9)):.6f}",
                    "mean_score_0_9": f"{float(np.mean(score_0_9)):.6f}",
                }
            )

            for i, frame_path in enumerate(frame_paths):
                row = {
                    "label": label,
                    "sequence_id": sequence_dir.name,
                    "sequence_dir": str(sequence_dir),
                    "plot_frame_index_from_onset": frame_indices[i],
                    "frame_number_from_name": format_optional_int(frame_numbers_from_name[i]),
                    "frame_file": frame_path.name,
                    "frame_path": str(frame_path),
                    "score_0_9": f"{float(score_0_9[i]):.8f}",
                    "raw_score_unclamped": f"{float(raw_score[i]):.8f}",
                    "soft_anchor_win_sum": f"{float(soft_win_sum[i]):.8f}",
                    "nearest_level": int(nearest_level[i]),
                    "mean_anchor_consistency": f"{float(np.mean(consistency[i])):.8f}",
                    "plot_path": str(plot_path),
                }
                for level in range(len(anchor_paths)):
                    row[f"p_stronger_than_level_{level:02d}"] = f"{float(probs[i, level]):.8f}"
                    row[f"consistency_level_{level:02d}"] = f"{float(consistency[i, level]):.8f}"
                frame_rows.append(row)

            sequence_results.append(
                {
                    "label": label,
                    "sequence_id": sequence_dir.name,
                    "frame_numbers": frame_indices,
                    "scores": score_0_9,
                    "plot_path": str(plot_path),
                }
            )
            print(
                f"[{label}] {seq_idx}/{len(sequence_dirs)} seq {sequence_dir.name}: "
                f"{len(frame_paths)} frames -> {plot_path}"
            )

    if not sequence_results:
        raise RuntimeError("No sequence plots were generated.")

    all_plot_path = args.output_root / "all_sequences_smileranking_plot.png"
    plot_all_sequences(all_plot_path, sequence_results)

    manifest_fields = [
        "label",
        "sequence_id",
        "sequence_dir",
        "frame_count",
        "plot_path",
        "first_frame_file",
        "last_frame_file",
        "min_score_0_9",
        "max_score_0_9",
        "mean_score_0_9",
    ]
    write_csv(args.output_root / "plot_manifest.csv", manifest_fields, manifest_rows)

    frame_fields = [
        "label",
        "sequence_id",
        "sequence_dir",
        "plot_frame_index_from_onset",
        "frame_number_from_name",
        "frame_file",
        "frame_path",
        "score_0_9",
        "raw_score_unclamped",
        "soft_anchor_win_sum",
        "nearest_level",
        "mean_anchor_consistency",
        "plot_path",
    ]
    for level in range(len(anchor_paths)):
        frame_fields.append(f"p_stronger_than_level_{level:02d}")
        frame_fields.append(f"consistency_level_{level:02d}")
    write_csv(args.output_root / "frame_smile_ranking_scores.csv", frame_fields, frame_rows)

    summary = {
        "sequence_root": str(args.sequence_root),
        "output_root": str(args.output_root),
        "weights": str(args.weights),
        "scale_dir": str(args.scale_dir),
        "scale_anchor_images": [str(path) for path in anchor_paths],
        "device": str(device),
        "labels": args.labels,
        "num_sequences": len(sequence_results),
        "num_frames": len(frame_rows),
        "scoring_method": (
            "score_0_9 = clamp(sum_k P(frame stronger than scale_anchor_k) - 0.5, 0, 9)"
        ),
        "x_axis": "plot_frame_index_from_onset; no temporal resampling",
        "all_sequences_plot": str(all_plot_path),
        "all_sequences_plot_svg": str(all_plot_path.with_suffix(".svg")),
        "plot_manifest": str(args.output_root / "plot_manifest.csv"),
        "frame_scores": str(args.output_root / "frame_smile_ranking_scores.csv"),
    }
    with (args.output_root / "smileranking_plot_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
