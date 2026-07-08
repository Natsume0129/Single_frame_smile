from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

from PIL import Image, ImageOps


DEFAULT_SEQUENCE_ROOT = Path(r"E:\Dataset\sequence")
DEFAULT_OUTPUT_ROOT = Path(r"E:\Dataset\stillimages")
LABELS = ("true", "polite", "bitter")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
FRAME_COUNT = 20
GRID_COLS = 5
GRID_ROWS = 4
TILE_SIZE = (224, 224)
PADDING = 6
BACKGROUND = (255, 255, 255)


def last_int_key(path: Path) -> tuple[int, str]:
    match = re.search(r"(\d+)(?!.*\d)", path.stem)
    if match:
        return int(match.group(1)), path.name
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


def build_stillimage(frame_paths: list[Path], out_path: Path) -> None:
    indices = sample_indices(len(frame_paths), FRAME_COUNT)
    tiles: list[Image.Image] = []
    for index in indices:
        with Image.open(frame_paths[index]) as image:
            tile = ImageOps.pad(image.convert("RGB"), TILE_SIZE, color=BACKGROUND)
        tiles.append(tile)

    canvas_w = GRID_COLS * TILE_SIZE[0] + (GRID_COLS + 1) * PADDING
    canvas_h = GRID_ROWS * TILE_SIZE[1] + (GRID_ROWS + 1) * PADDING
    canvas = Image.new("RGB", (canvas_w, canvas_h), BACKGROUND)

    for idx, tile in enumerate(tiles):
        row = idx // GRID_COLS
        col = idx % GRID_COLS
        x = PADDING + col * (TILE_SIZE[0] + PADDING)
        y = PADDING + row * (TILE_SIZE[1] + PADDING)
        canvas.paste(tile, (x, y))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def numeric_dir_key(path: Path) -> tuple[int, str]:
    try:
        return int(path.name), path.name
    except ValueError:
        return 0, path.name


def build_all(sequence_root: Path, output_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for label in LABELS:
        label_dir = sequence_root / label
        if not label_dir.is_dir():
            continue

        out_label_dir = output_root / label
        out_label_dir.mkdir(parents=True, exist_ok=True)
        for sequence_dir in sorted((p for p in label_dir.iterdir() if p.is_dir()), key=numeric_dir_key):
            frame_paths = sorted_image_files(sequence_dir)
            if not frame_paths:
                rows.append(
                    {
                        "label": label,
                        "sequence_id": sequence_dir.name,
                        "source_dir": str(sequence_dir),
                        "output_file": "",
                        "source_frame_count": 0,
                        "sampled_source_files": "",
                        "status": "skipped_no_images",
                    }
                )
                continue

            out_path = out_label_dir / f"{sequence_dir.name}.png"
            build_stillimage(frame_paths, out_path)
            sampled = [frame_paths[i].name for i in sample_indices(len(frame_paths), FRAME_COUNT)]
            rows.append(
                {
                    "label": label,
                    "sequence_id": sequence_dir.name,
                    "source_dir": str(sequence_dir),
                    "output_file": str(out_path),
                    "source_frame_count": len(frame_paths),
                    "sampled_source_files": "|".join(sampled),
                    "status": "ok",
                }
            )
    return rows


def write_manifest(output_root: Path, rows: list[dict[str, object]]) -> None:
    manifest_path = output_root / "stillimages_manifest.csv"
    fieldnames = [
        "label",
        "sequence_id",
        "source_dir",
        "output_file",
        "source_frame_count",
        "sampled_source_files",
        "status",
    ]
    output_root.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build 20-frame stillimages from sequence image folders.")
    parser.add_argument("--sequence-root", type=Path, default=DEFAULT_SEQUENCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_all(args.sequence_root, args.output_root)
    write_manifest(args.output_root, rows)
    ok_rows = [row for row in rows if row["status"] == "ok"]
    print(f"[STILL] wrote {len(ok_rows)} stillimages to {args.output_root}")
    for label in LABELS:
        count = sum(1 for row in ok_rows if row["label"] == label)
        print(f"[STILL] {label}: {count}")


if __name__ == "__main__":
    main()
