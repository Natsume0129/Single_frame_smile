from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


FOLDER_PATTERN = re.compile(r"^(?P<date>\d{8})_(?P<seg_start>\d+)-(?P<seg_end>\d+)$")
IMAGE_TS_PATTERN = re.compile(r"_(\d+)\.[^.]+$")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_image_ts(file_path: Path) -> int:
    m = IMAGE_TS_PATTERN.search(file_path.name)
    if not m:
        raise ValueError(f"Cannot parse image timestamp from filename: {file_path.name}")
    return int(m.group(1))


def build_rows(root_dir: Path) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []

    subdirs = sorted([p for p in root_dir.iterdir() if p.is_dir()], key=lambda p: p.name)
    if not subdirs:
        raise RuntimeError(f"No subfolders found under: {root_dir}")

    for folder in subdirs:
        fm = FOLDER_PATTERN.match(folder.name)
        if not fm:
            raise RuntimeError(
                f"Unexpected folder name format: {folder.name} "
                f"(expected YYYYMMDD_start-end)"
            )

        date_str = fm.group("date")
        seg_start = int(fm.group("seg_start"))
        seg_end = int(fm.group("seg_end"))

        images = sorted(
            [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS],
            key=lambda p: p.name,
        )
        if len(images) != 2:
            raise RuntimeError(
                f"Folder must contain exactly 2 images, got {len(images)}: {folder}"
            )

        img_with_ts = [(img, parse_image_ts(img)) for img in images]
        img_with_ts.sort(key=lambda x: x[1])

        start_img, start_ts = img_with_ts[0]
        end_img, end_ts = img_with_ts[1]

        rows.append(
            {
                "date": date_str,
                "segment_folder": folder.name,
                "segment_start": seg_start,
                "segment_end": seg_end,
                "start_image": start_img.name,
                "end_image": end_img.name,
                "start_ts": start_ts,
                "end_ts": end_ts,
                "segment_span": seg_end - seg_start,
                "pair_span": end_ts - start_ts,
                "folder_path": str(folder),
            }
        )

    return rows


def write_csv(rows: list[dict[str, str | int]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "date",
        "segment_folder",
        "segment_start",
        "segment_end",
        "start_image",
        "end_image",
        "start_ts",
        "end_ts",
        "segment_span",
        "pair_span",
        "folder_path",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a CSV manifest from key_frames pair folders."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(r"E:\Matsuda_data\2-12meeting\key_frames"),
        help="Root folder containing pair subfolders.",
    )
    parser.add_argument(
        "--out_csv",
        type=Path,
        default=Path(r"E:\Matsuda_data\2-12meeting\key_frames_manifest.csv"),
        help="Output CSV path.",
    )
    args = parser.parse_args()

    rows = build_rows(args.root)
    write_csv(rows, args.out_csv)

    print(f"Done. rows={len(rows)}")
    print(f"Saved CSV: {args.out_csv}")


if __name__ == "__main__":
    main()
