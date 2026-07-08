"""Build a 0-9 true-smile intensity scale from a trained SmileComp model."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont, ImageOps
from torchvision import transforms


DEFAULT_PROJECT_DIR = Path(
    r"E:\SmileAnnotation\FaceTracking-Smile_Detection\SmileComp_SiameseNet\Pytorch-Shimonishi"
)
DEFAULT_RANKING_ROOT = Path(r"E:\Dataset\smile_ranking_true")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score true-smile images pairwise and export a 10-level scale."
    )
    parser.add_argument("--smilecomp_dir", type=Path, default=DEFAULT_PROJECT_DIR)
    parser.add_argument(
        "--ranking_items",
        type=Path,
        default=DEFAULT_RANKING_ROOT / "ranking_items.csv",
    )
    parser.add_argument(
        "--images_dir",
        type=Path,
        default=DEFAULT_RANKING_ROOT / "images",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_RANKING_ROOT
        / "model"
        / "smile_rank_true_cross_sequence_100ep.pth",
    )
    parser.add_argument(
        "--manual_pairs",
        type=Path,
        default=DEFAULT_RANKING_ROOT / "train_manual_cross_sequence_198.dat",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_RANKING_ROOT / "scale10",
    )
    parser.add_argument("--levels", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=256)
    return parser.parse_args()


def read_ranking_items(path: Path, images_dir: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError(f"No rows found in {path}")

    missing = [row["image_file"] for row in rows if not (images_dir / row["image_file"]).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} images, first missing: {missing[0]}")

    return rows


def load_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        Path(r"C:\Windows\Fonts\arial.ttf"),
        Path(r"C:\Windows\Fonts\segoeui.ttf"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


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


def extract_features(model, image_tensors: list[torch.Tensor], device, batch_size: int) -> torch.Tensor:
    features = []
    with torch.no_grad():
        for start in range(0, len(image_tensors), batch_size):
            batch = torch.stack(image_tensors[start : start + batch_size]).to(device)
            features.append(model.extractor(batch).detach())
    return torch.cat(features, dim=0)


def score_all_pairs(model, features: torch.Tensor, batch_size: int):
    n_items = features.shape[0]
    pair_indices = [(i, j) for i in range(n_items) for j in range(i + 1, n_items)]

    strong_prob = np.full((n_items, n_items), np.nan, dtype=np.float32)
    consistency = np.full((n_items, n_items), np.nan, dtype=np.float32)
    score_sum = np.zeros(n_items, dtype=np.float64)
    hard_wins = np.zeros(n_items, dtype=np.int32)
    pair_rows = []

    with torch.no_grad():
        for start in range(0, len(pair_indices), batch_size):
            pairs = pair_indices[start : start + batch_size]
            idx_i = torch.tensor([pair[0] for pair in pairs], device=features.device)
            idx_j = torch.tensor([pair[1] for pair in pairs], device=features.device)

            feat_i = features.index_select(0, idx_i)
            feat_j = features.index_select(0, idx_j)
            probs_org = F.softmax(model.comp_layer(torch.cat((feat_i, feat_j), dim=1)), dim=1)
            probs_swp = F.softmax(model.comp_layer(torch.cat((feat_j, feat_i), dim=1)), dim=1)

            p_i_gt_j = ((probs_org[:, 0] + probs_swp[:, 1]) * 0.5).detach().cpu().numpy()
            p_i_lt_j = ((probs_org[:, 1] + probs_swp[:, 0]) * 0.5).detach().cpu().numpy()
            cons = (
                torch.minimum(probs_org[:, 0], probs_swp[:, 1])
                + torch.minimum(probs_org[:, 1], probs_swp[:, 0])
            ).detach().cpu().numpy()

            for local_idx, (i, j) in enumerate(pairs):
                p = float(p_i_gt_j[local_idx])
                c = float(cons[local_idx])
                strong_prob[i, j] = p
                strong_prob[j, i] = 1.0 - p
                consistency[i, j] = c
                consistency[j, i] = c
                score_sum[i] += p
                score_sum[j] += 1.0 - p
                hard_wins[i] += int(p >= 0.5)
                hard_wins[j] += int(p < 0.5)
                pair_rows.append(
                    {
                        "image_i_index": i,
                        "image_j_index": j,
                        "p_i_stronger_than_j": p,
                        "p_i_weaker_than_j": float(p_i_lt_j[local_idx]),
                        "consistency": c,
                    }
                )

    np.fill_diagonal(strong_prob, np.nan)
    np.fill_diagonal(consistency, np.nan)
    mean_stronger_prob = score_sum / (n_items - 1)
    # Same ranking idea as the original scripts: hard voting dominates, soft score breaks ties.
    vote_score = hard_wins.astype(np.float64) + (score_sum / (n_items - 1))
    return strong_prob, consistency, mean_stronger_prob, hard_wins, vote_score, pair_rows


def write_matrix(path: Path, matrix: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for row in matrix:
            writer.writerow(["" if math.isnan(float(value)) else f"{float(value):.8f}" for value in row])


def assign_equal_population_levels(n_items: int, levels: int) -> list[int]:
    return [min(levels - 1, int(rank * levels / n_items)) for rank in range(n_items)]


def equal_interval_positions(n_items: int, levels: int) -> list[int]:
    if levels == 1:
        return [0]
    return [round(k * (n_items - 1) / (levels - 1)) for k in range(levels)]


def make_montage(
    output_path: Path,
    items: list[dict[str, str]],
    rep_indices: list[int],
    scores: np.ndarray,
    images_dir: Path,
) -> None:
    tile = 176
    gap = 10
    margin = 18
    title_h = 38
    label_h = 52
    width = margin * 2 + tile * len(rep_indices) + gap * (len(rep_indices) - 1)
    height = margin * 2 + title_h + tile + label_h

    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    font_title = load_font(22)
    font_label = load_font(17)
    font_small = load_font(13)

    title = "True smile scale: 0 weakest -> 9 strongest"
    draw.text((margin, margin), title, fill=(20, 24, 28), font=font_title)

    y_img = margin + title_h
    y_label = y_img + tile + 7
    for level, idx in enumerate(rep_indices):
        item = items[idx]
        x = margin + level * (tile + gap)
        with Image.open(images_dir / item["image_file"]) as img:
            thumb = ImageOps.fit(img.convert("RGB"), (tile, tile), method=Image.Resampling.LANCZOS)
        canvas.paste(thumb, (x, y_img))
        draw.rectangle((x, y_img, x + tile - 1, y_img + tile - 1), outline=(70, 76, 82), width=1)

        label = f"Level {level}"
        score = f"score {scores[idx]:.3f}"
        source = f"seq {item['sequence_id']} / prior {item['provisional_level']}"
        draw.text((x, y_label), label, fill=(0, 0, 0), font=font_label)
        draw.text((x, y_label + 20), score, fill=(44, 50, 56), font=font_small)
        draw.text((x, y_label + 36), source, fill=(70, 76, 82), font=font_small)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def make_grid_montage(
    output_path: Path,
    items: list[dict[str, str]],
    rep_indices: list[int],
    scores: np.ndarray,
    images_dir: Path,
) -> None:
    cols = 5
    tile = 192
    gap = 12
    margin = 18
    title_h = 34
    label_h = 48
    rows = math.ceil(len(rep_indices) / cols)
    width = margin * 2 + cols * tile + (cols - 1) * gap
    height = margin * 2 + title_h + rows * (tile + label_h) + (rows - 1) * gap

    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    font_title = load_font(21)
    font_label = load_font(17)
    font_small = load_font(13)
    draw.text((margin, margin), "True smile scale: 0 weakest -> 9 strongest", fill=(20, 24, 28), font=font_title)

    for level, idx in enumerate(rep_indices):
        row = level // cols
        col = level % cols
        x = margin + col * (tile + gap)
        y_img = margin + title_h + row * (tile + label_h + gap)
        item = items[idx]
        with Image.open(images_dir / item["image_file"]) as img:
            thumb = ImageOps.fit(img.convert("RGB"), (tile, tile), method=Image.Resampling.LANCZOS)
        canvas.paste(thumb, (x, y_img))
        draw.rectangle((x, y_img, x + tile - 1, y_img + tile - 1), outline=(70, 76, 82), width=1)
        y_label = y_img + tile + 6
        draw.text((x, y_label), f"Level {level}  score {scores[idx]:.3f}", fill=(0, 0, 0), font=font_label)
        draw.text(
            (x, y_label + 22),
            f"seq {item['sequence_id']} / prior {item['provisional_level']}",
            fill=(70, 76, 82),
            font=font_small,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def evaluate_manual_pairs(
    manual_pairs: Path,
    items_by_name: dict[str, int],
    strong_prob: np.ndarray,
    output_path: Path,
) -> dict[str, float | int]:
    if not manual_pairs.is_file():
        return {"manual_pair_rows": 0}

    rows = []
    total = 0
    correct = 0
    ambiguous = 0
    with manual_pairs.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for image1, image2, raw_label, *rest in reader:
            label = raw_label.strip()
            if label == "0":
                ambiguous += 1
                continue
            if label not in {"1", "-1"}:
                continue
            i = items_by_name[image1]
            j = items_by_name[image2]
            p = float(strong_prob[i, j])
            pred_label = "1" if p >= 0.5 else "-1"
            is_correct = pred_label == label
            total += 1
            correct += int(is_correct)
            rows.append(
                {
                    "image1": image1,
                    "image2": image2,
                    "manual_label": label,
                    "p_image1_stronger": f"{p:.8f}",
                    "pred_label": pred_label,
                    "correct": int(is_correct),
                }
            )

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image1",
                "image2",
                "manual_label",
                "p_image1_stronger",
                "pred_label",
                "correct",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    return {
        "manual_pair_rows_used": total,
        "manual_pair_ambiguous_skipped": ambiguous,
        "manual_pair_accuracy": correct / total if total else None,
        "manual_pair_correct": correct,
    }


def main() -> None:
    args = parse_args()
    if not args.smilecomp_dir.is_dir():
        raise FileNotFoundError(args.smilecomp_dir)
    if not args.weights.is_file():
        raise FileNotFoundError(args.weights)

    sys.path.insert(0, str(args.smilecomp_dir))
    import face_comp_torch as fcmodel  # noqa: PLC0415

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    items = read_ranking_items(args.ranking_items, args.images_dir)
    transform = prepare_transform(fcmodel)
    image_tensors = [load_image_tensor(args.images_dir / item["image_file"], transform) for item in items]

    model = fcmodel.face_comp_siamese_model_vgg16based_for_predict(
        str(args.weights), on_device=device
    ).to(device)
    model.eval()

    features = extract_features(model, image_tensors, device, args.batch_size)
    strong_prob, consistency, mean_prob, hard_wins, vote_score, pair_rows = score_all_pairs(
        model, features, args.batch_size
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_matrix(args.output_dir / "strong_probability_matrix.csv", strong_prob)
    write_matrix(args.output_dir / "stream_consistency_matrix.csv", consistency)

    with (args.output_dir / "pairwise_predictions.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "image_i",
            "image_j",
            "p_i_stronger_than_j",
            "p_i_weaker_than_j",
            "consistency",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in pair_rows:
            i = int(row["image_i_index"])
            j = int(row["image_j_index"])
            writer.writerow(
                {
                    "image_i": items[i]["image_file"],
                    "image_j": items[j]["image_file"],
                    "p_i_stronger_than_j": f"{row['p_i_stronger_than_j']:.8f}",
                    "p_i_weaker_than_j": f"{row['p_i_weaker_than_j']:.8f}",
                    "consistency": f"{row['consistency']:.8f}",
                }
            )

    ranked_indices_weak_to_strong = sorted(
        range(len(items)),
        key=lambda idx: (vote_score[idx], mean_prob[idx], items[idx]["image_file"]),
    )
    level_by_rank = assign_equal_population_levels(len(items), args.levels)
    rank_position = {idx: rank for rank, idx in enumerate(ranked_indices_weak_to_strong)}
    scale_level = {idx: level_by_rank[rank_position[idx]] for idx in range(len(items))}

    ranked_rows = []
    for rank, idx in enumerate(ranked_indices_weak_to_strong):
        item = items[idx]
        ranked_rows.append(
            {
                "rank_weak_to_strong": rank,
                "scale_level_equal_population": scale_level[idx],
                "image_file": item["image_file"],
                "vote_score": f"{vote_score[idx]:.8f}",
                "mean_stronger_probability": f"{mean_prob[idx]:.8f}",
                "hard_wins": int(hard_wins[idx]),
                "sequence_id": item["sequence_id"],
                "provisional_level": item["provisional_level"],
                "source_frame_index": item["source_frame_index"],
                "source_frame_name": item["source_frame_name"],
                "source_frame_path": item["source_frame_path"],
            }
        )

    with (args.output_dir / "scale10_ranked_all.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(ranked_rows[0].keys()))
        writer.writeheader()
        writer.writerows(ranked_rows)

    rep_positions = equal_interval_positions(len(items), args.levels)
    rep_indices = [ranked_indices_weak_to_strong[pos] for pos in rep_positions]

    scale_rows = []
    for level, (pos, idx) in enumerate(zip(rep_positions, rep_indices)):
        item = items[idx]
        dst_name = f"level_{level:02d}.png"
        shutil.copy2(args.images_dir / item["image_file"], args.output_dir / dst_name)
        scale_rows.append(
            {
                "scale_level": level,
                "rank_position_weak_to_strong": pos,
                "reference_image": dst_name,
                "source_image_file": item["image_file"],
                "vote_score": f"{vote_score[idx]:.8f}",
                "mean_stronger_probability": f"{mean_prob[idx]:.8f}",
                "hard_wins": int(hard_wins[idx]),
                "sequence_id": item["sequence_id"],
                "provisional_level": item["provisional_level"],
                "source_frame_index": item["source_frame_index"],
                "source_frame_name": item["source_frame_name"],
                "source_frame_path": item["source_frame_path"],
            }
        )

    with (args.output_dir / "scale10_items.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(scale_rows[0].keys()))
        writer.writeheader()
        writer.writerows(scale_rows)

    make_montage(args.output_dir / "scale10_montage.png", items, rep_indices, vote_score, args.images_dir)
    make_grid_montage(
        args.output_dir / "scale10_montage_grid.png", items, rep_indices, vote_score, args.images_dir
    )

    items_by_name = {item["image_file"]: idx for idx, item in enumerate(items)}
    manual_summary = evaluate_manual_pairs(
        args.manual_pairs,
        items_by_name,
        strong_prob,
        args.output_dir / "manual_pair_eval.csv",
    )

    summary = {
        "weights": str(args.weights),
        "ranking_items": str(args.ranking_items),
        "images_dir": str(args.images_dir),
        "output_dir": str(args.output_dir),
        "device": str(device),
        "num_images": len(items),
        "num_pairwise_predictions": len(pair_rows),
        "levels": args.levels,
        "level_direction": "0 weakest, 9 strongest",
        "representative_selection": "equal interval over weak-to-strong ranked list",
        "equal_population_assignment": "scale_level_equal_population in scale10_ranked_all.csv",
        "mean_stream_consistency": float(np.nanmean(consistency)),
        "min_stream_consistency": float(np.nanmin(consistency)),
        "max_stream_consistency": float(np.nanmax(consistency)),
        **manual_summary,
    }
    with (args.output_dir / "scale10_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
