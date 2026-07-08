from __future__ import annotations

import argparse
import csv
import re
import shutil
from itertools import combinations
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


DEFAULT_TRUE_ROOT = Path(r"E:\Dataset\sequence\true")
DEFAULT_OUTPUT_ROOT = Path(r"E:\Dataset\smile_ranking_true")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
LEVEL_COUNT = 10
TILE_SIZE = (224, 224)
PADDING = 6
BACKGROUND = (255, 255, 255)


def last_int_key(path: Path) -> tuple[int, str]:
    match = re.search(r"(\d+)(?!.*\d)", path.stem)
    if match:
        return int(match.group(1)), path.name
    return 0, path.name


def numeric_dir_key(path: Path) -> tuple[int, str]:
    try:
        return int(path.name), path.name
    except ValueError:
        return 0, path.name


def sorted_image_files(sequence_dir: Path) -> list[Path]:
    files = [
        path
        for path in sequence_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(files, key=last_int_key)


def sample_indices(total: int, count: int) -> list[int]:
    if total <= 0:
        raise ValueError("Cannot sample from an empty sequence.")
    if count == 1:
        return [0]
    return [round(i * (total - 1) / (count - 1)) for i in range(count)]


def sequence_token(sequence_id: str) -> str:
    return sequence_id.zfill(3) if sequence_id.isdigit() else sequence_id


def copy_level_images(true_root: Path, images_dir: Path) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    images_dir.mkdir(parents=True, exist_ok=True)
    sequence_dirs = sorted((p for p in true_root.iterdir() if p.is_dir()), key=numeric_dir_key)
    for sequence_dir in sequence_dirs:
        frame_paths = sorted_image_files(sequence_dir)
        if not frame_paths:
            continue
        sampled = sample_indices(len(frame_paths), LEVEL_COUNT)
        token = sequence_token(sequence_dir.name)
        for level, source_index in enumerate(sampled):
            source_path = frame_paths[source_index]
            image_file = f"true_seq{token}_level{level:02d}.png"
            output_path = images_dir / image_file
            shutil.copy2(source_path, output_path)
            items.append(
                {
                    "image_file": image_file,
                    "provisional_level": level,
                    "sequence_id": sequence_dir.name,
                    "source_frame_index": source_index,
                    "source_frame_name": source_path.name,
                    "source_frame_path": str(source_path),
                    "sequence_frame_count": len(frame_paths),
                    "status": "auto_temporal_prior",
                }
            )
    return items


def write_items(output_root: Path, items: list[dict[str, object]]) -> None:
    fields = [
        "image_file",
        "provisional_level",
        "sequence_id",
        "source_frame_index",
        "source_frame_name",
        "source_frame_path",
        "sequence_frame_count",
        "status",
    ]
    with (output_root / "ranking_items.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(items)


def build_pair_rows(items: list[dict[str, object]]) -> tuple[list[tuple[str, str, int]], list[tuple[str, str, int]], list[dict[str, object]]]:
    by_sequence: dict[str, list[dict[str, object]]] = {}
    for item in items:
        by_sequence.setdefault(str(item["sequence_id"]), []).append(item)

    all_pairs: list[tuple[str, str, int]] = []
    gap2_pairs: list[tuple[str, str, int]] = []
    pair_meta: list[dict[str, object]] = []

    for sequence_id, sequence_items in sorted(by_sequence.items(), key=lambda kv: numeric_dir_key(Path(kv[0]))):
        ordered = sorted(sequence_items, key=lambda item: int(item["provisional_level"]))
        for weaker, stronger in combinations(ordered, 2):
            low = int(weaker["provisional_level"])
            high = int(stronger["provisional_level"])
            gap = high - low
            forward = (str(weaker["image_file"]), str(stronger["image_file"]), -1)
            reverse = (str(stronger["image_file"]), str(weaker["image_file"]), 1)
            all_pairs.extend([forward, reverse])
            if gap >= 2:
                gap2_pairs.extend([forward, reverse])
            pair_meta.append(
                {
                    "sequence_id": sequence_id,
                    "weaker_image": weaker["image_file"],
                    "stronger_image": stronger["image_file"],
                    "weaker_level": low,
                    "stronger_level": high,
                    "level_gap": gap,
                    "source": "within_sequence_temporal_prior",
                }
            )
    return all_pairs, gap2_pairs, pair_meta


def write_training_pairs(path: Path, pairs: list[tuple[str, str, int]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerows(pairs)


def write_pair_meta(output_root: Path, rows: list[dict[str, object]]) -> None:
    fields = [
        "sequence_id",
        "weaker_image",
        "stronger_image",
        "weaker_level",
        "stronger_level",
        "level_gap",
        "source",
    ]
    with (output_root / "pair_meta.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_review_csvs(output_root: Path, items: list[dict[str, object]]) -> None:
    adjacent_path = output_root / "review_adjacent_pairs.csv"
    adjacent_fields = [
        "image1_file",
        "image2_file",
        "expected_relation",
        "sequence_id",
        "level1",
        "level2",
        "human_relation",
        "note",
    ]
    by_sequence: dict[str, list[dict[str, object]]] = {}
    by_level: dict[int, list[dict[str, object]]] = {}
    for item in items:
        by_sequence.setdefault(str(item["sequence_id"]), []).append(item)
        by_level.setdefault(int(item["provisional_level"]), []).append(item)

    with adjacent_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=adjacent_fields)
        writer.writeheader()
        for sequence_id, sequence_items in sorted(by_sequence.items(), key=lambda kv: numeric_dir_key(Path(kv[0]))):
            ordered = sorted(sequence_items, key=lambda item: int(item["provisional_level"]))
            for left, right in zip(ordered, ordered[1:]):
                writer.writerow(
                    {
                        "image1_file": left["image_file"],
                        "image2_file": right["image_file"],
                        "expected_relation": "image2_stronger",
                        "sequence_id": sequence_id,
                        "level1": left["provisional_level"],
                        "level2": right["provisional_level"],
                        "human_relation": "",
                        "note": "",
                    }
                )

    cross_path = output_root / "review_same_level_cross_sequence_pairs.csv"
    cross_fields = [
        "image1_file",
        "image2_file",
        "provisional_level",
        "sequence1_id",
        "sequence2_id",
        "human_relation",
        "note",
    ]
    with cross_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=cross_fields)
        writer.writeheader()
        for level, level_items in sorted(by_level.items()):
            ordered = sorted(level_items, key=lambda item: numeric_dir_key(Path(str(item["sequence_id"]))))
            for item1, item2 in combinations(ordered, 2):
                writer.writerow(
                    {
                        "image1_file": item1["image_file"],
                        "image2_file": item2["image_file"],
                        "provisional_level": level,
                        "sequence1_id": item1["sequence_id"],
                        "sequence2_id": item2["sequence_id"],
                        "human_relation": "",
                        "note": "",
                    }
                )


def labeled_tile(image_path: Path, label: str) -> Image.Image:
    with Image.open(image_path) as image:
        face = ImageOps.pad(image.convert("RGB"), TILE_SIZE, color=BACKGROUND)
    tile = Image.new("RGB", (TILE_SIZE[0], TILE_SIZE[1] + 18), BACKGROUND)
    tile.paste(face, (0, 0))
    draw = ImageDraw.Draw(tile)
    draw.text((4, TILE_SIZE[1] + 2), label, fill=(0, 0, 0))
    return tile


def save_grid(tiles: list[Image.Image], cols: int, out_path: Path) -> None:
    if not tiles:
        return
    rows = (len(tiles) + cols - 1) // cols
    tile_w, tile_h = tiles[0].size
    canvas_w = cols * tile_w + (cols + 1) * PADDING
    canvas_h = rows * tile_h + (rows + 1) * PADDING
    canvas = Image.new("RGB", (canvas_w, canvas_h), BACKGROUND)
    for idx, tile in enumerate(tiles):
        row = idx // cols
        col = idx % cols
        x = PADDING + col * (tile_w + PADDING)
        y = PADDING + row * (tile_h + PADDING)
        canvas.paste(tile, (x, y))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def build_review_montages(output_root: Path, items: list[dict[str, object]]) -> None:
    images_dir = output_root / "images"
    by_sequence: dict[str, list[dict[str, object]]] = {}
    by_level: dict[int, list[dict[str, object]]] = {}
    for item in items:
        by_sequence.setdefault(str(item["sequence_id"]), []).append(item)
        by_level.setdefault(int(item["provisional_level"]), []).append(item)

    sequence_out = output_root / "review_sequence_montages"
    for sequence_id, sequence_items in sorted(by_sequence.items(), key=lambda kv: numeric_dir_key(Path(kv[0]))):
        ordered = sorted(sequence_items, key=lambda item: int(item["provisional_level"]))
        tiles = [
            labeled_tile(images_dir / str(item["image_file"]), f"seq {sequence_id} L{int(item['provisional_level']):02d}")
            for item in ordered
        ]
        save_grid(tiles, 5, sequence_out / f"true_seq{sequence_token(sequence_id)}_levels.png")

    level_out = output_root / "review_level_montages"
    for level, level_items in sorted(by_level.items()):
        ordered = sorted(level_items, key=lambda item: numeric_dir_key(Path(str(item["sequence_id"]))))
        tiles = [
            labeled_tile(images_dir / str(item["image_file"]), f"seq {item['sequence_id']} L{level:02d}")
            for item in ordered
        ]
        save_grid(tiles, 4, level_out / f"level_{level:02d}.png")


def write_readme(output_root: Path, item_count: int, all_pair_count: int, gap2_pair_count: int) -> None:
    text = f"""# True Smile Ranking Dataset

This folder is an auto-generated first-pass dataset for building a true-smile-only 0-9 smiling-intensity scale.

Data source:
- `E:\\Dataset\\sequence\\true`

Generated files:
- `images/`: copied true-smile candidate frames, one image per provisional sequence level.
- `ranking_items.csv`: item manifest with source frame paths and provisional level 0-9.
- `train_pairs_gap2.dat`: recommended initial SiameseNet training pairs. Adjacent levels are excluded.
- `train_pairs_all.dat`: all within-sequence temporal pairs, including adjacent levels.
- `pair_meta.csv`: metadata for all within-sequence temporal pairs.
- `review_adjacent_pairs.csv`: adjacent within-sequence pairs for human confirmation.
- `review_same_level_cross_sequence_pairs.csv`: same-level cross-sequence pairs for checking whether provisional levels align across people.
- `review_sequence_montages/`: one 10-level montage per true sequence.
- `review_level_montages/`: one montage per provisional level across all true sequences.

Counts:
- ranking images: {item_count}
- directed all pairs: {all_pair_count}
- directed gap>=2 pairs: {gap2_pair_count}

SiameseNet convention used here:
- `dataset_dir`: `E:\\Dataset\\smile_ranking_true\\images`
- `label = 1`: image1 is stronger than image2
- `label = -1`: image1 is weaker than image2

Important limitation:
- The 0-9 level is a temporal prior, not a verified absolute intensity label. Each true-smile sequence is assumed to run from onset to peak, so later sampled frames are treated as stronger within the same sequence. Cross-person absolute ordering still needs human confirmation.
"""
    (output_root / "README.md").write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a true-smile-only ranking dataset from sequence frames.")
    parser.add_argument("--true-root", type=Path, default=DEFAULT_TRUE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    images_dir = args.output_root / "images"

    items = copy_level_images(args.true_root, images_dir)
    write_items(args.output_root, items)

    all_pairs, gap2_pairs, pair_meta = build_pair_rows(items)
    write_training_pairs(args.output_root / "train_pairs_all.dat", all_pairs)
    write_training_pairs(args.output_root / "train_pairs_gap2.dat", gap2_pairs)
    write_pair_meta(args.output_root, pair_meta)
    write_review_csvs(args.output_root, items)
    build_review_montages(args.output_root, items)
    write_readme(args.output_root, len(items), len(all_pairs), len(gap2_pairs))

    print(f"[RANKING] wrote {len(items)} ranking images to {images_dir}")
    print(f"[RANKING] all directed pairs: {len(all_pairs)}")
    print(f"[RANKING] gap>=2 directed pairs: {len(gap2_pairs)}")
    print(f"[RANKING] output root: {args.output_root}")


if __name__ == "__main__":
    main()
