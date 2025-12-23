#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Split matched.dat into 3 parts based on date=20251019 and frame order:

1) test.dat              : all dates except 20251019
2) 20251019before.dat    : first 50% of 20251019 samples (sorted by frame index)
3) 20251019after.dat     : last 50% of 20251019 samples (sorted by frame index)

Also copy corresponding image files into:
- images/test
- images/20251019_before
- images/20251019_after

Input:
- MATCHED_DAT: full dat file, each line: "<path> <label>"
  Example path: "images/dataset/20251029_0_0_6676.png"
- DATASET_DIR: directory containing the actual images (your un-split dataset folder)

Notes:
- The script extracts date from filename prefix (first 8 digits) and frame index from the last '_' number.
- It resolves each dat path's filename, then searches the image under DATASET_DIR by that filename.
  (This avoids path conflicts and ignores the "images/dataset/" prefix.)
"""

from __future__ import annotations
import os
import re
import shutil
from pathlib import Path
from typing import List, Tuple, Optional

# =========================
# Config: edit here
# =========================
MATCHED_DAT = r"E:\Single_frame_smile\data\Isornot\matched.dat"
DATASET_DIR = r"E:\Single_frame_smile\data\Isornot\images\dataset"
OUT_ROOT = r"E:\Single_frame_smile\data\Isornot"

DATE_TARGET = "20251019"

# Output .dat names (no extension change requested; use .dat)
OUT_DAT_TEST = os.path.join(OUT_ROOT, "test.dat")
OUT_DAT_BEFORE = os.path.join(OUT_ROOT, "20251019before.dat")
OUT_DAT_AFTER = os.path.join(OUT_ROOT, "20251019after.dat")

# Output image folders
OUT_IMG_TEST = os.path.join(OUT_ROOT, r"images\test")
OUT_IMG_BEFORE = os.path.join(OUT_ROOT, r"images\20251019_before")
OUT_IMG_AFTER = os.path.join(OUT_ROOT, r"images\20251019_after")

# If True, keep filenames only (flat) in output folders (no subdirs)
FLAT_COPY = True

# If True, shuffle "test" lines? (default False: keep original order)
SHUFFLE_TEST = False
SEED = 42
# =========================


DATE_RE = re.compile(r"(\d{8})_")
FRAME_RE = re.compile(r"_(\d+)\.(png|jpg|jpeg|bmp|webp)$", re.IGNORECASE)

def parse_line(line: str) -> Optional[Tuple[str, int]]:
    line = line.strip()
    if not line:
        return None
    parts = line.split()
    if len(parts) != 2:
        raise ValueError(f"Bad line (expected '<path> <label>'): {line!r}")
    p, y = parts
    return p, int(y)

def extract_date_and_frame(filename: str) -> Tuple[str, int]:
    m = DATE_RE.search(filename)
    if not m:
        raise ValueError(f"Cannot parse date from filename: {filename}")
    date = m.group(1)

    m2 = FRAME_RE.search(filename)
    if not m2:
        raise ValueError(f"Cannot parse frame index from filename: {filename}")
    frame = int(m2.group(1))
    return date, frame

def ensure_dir(p: str) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)

def resolve_src_image(dat_path: str, dataset_dir: str) -> Path:
    """
    dat_path example: images/dataset/20251029_0_0_6676.png
    dataset_dir: E:\\...\\images\\dataset (contains the files)

    We resolve by filename under dataset_dir.
    """
    fname = Path(dat_path).name
    src = Path(dataset_dir) / fname
    return src

def copy_image(src: Path, dst_dir: str) -> Path:
    ensure_dir(dst_dir)
    if FLAT_COPY:
        dst = Path(dst_dir) / src.name
    else:
        # If needed later: keep some structure
        dst = Path(dst_dir) / src.name
    shutil.copy2(src, dst)
    return dst

def write_dat(lines: List[Tuple[str, int]], out_dat: str, out_img_dir: str, dataset_dir: str) -> Tuple[int, int]:
    """
    Copy images and write dat lines as: "<relative_path_from_OUT_ROOT> <label>"
    e.g. "images/test/20251029_0_0_6676.png 0"
    """
    out_root = Path(OUT_ROOT)
    out_img_dir_path = Path(out_img_dir)

    ensure_dir(str(out_img_dir_path))
    Path(out_dat).parent.mkdir(parents=True, exist_ok=True)

    wrote = 0
    missing = 0

    with open(out_dat, "w", encoding="utf-8") as w:
        for p, y in lines:
            src = resolve_src_image(p, dataset_dir)
            if not src.exists():
                missing += 1
                continue
            dst = copy_image(src, out_img_dir)
            rel = dst.relative_to(out_root).as_posix()
            w.write(f"{rel} {y}\n")
            wrote += 1

    return wrote, missing


def main():
    # Read all
    raw_lines = Path(MATCHED_DAT).read_text(encoding="utf-8").splitlines()
    items = []
    for ln in raw_lines:
        parsed = parse_line(ln)
        if parsed is not None:
            items.append(parsed)

    # Split by date
    target = []
    others = []

    # For ordering within 20251019, sort by frame index
    for p, y in items:
        fname = Path(p).name
        date, frame = extract_date_and_frame(fname)
        if date == DATE_TARGET:
            target.append((p, y, frame))
        else:
            others.append((p, y))

    target.sort(key=lambda x: x[2])  # sort by frame

    n = len(target)
    half = n // 2  # first half = n//2, second half = rest
    before = [(p, y) for (p, y, _) in target[:half]]
    after  = [(p, y) for (p, y, _) in target[half:]]

    # Optional shuffle for test
    if SHUFFLE_TEST:
        import random
        random.seed(SEED)
        random.shuffle(others)

    # Write outputs + copy
    w_test, miss_test = write_dat(others, OUT_DAT_TEST, OUT_IMG_TEST, DATASET_DIR)
    w_bef,  miss_bef  = write_dat(before, OUT_DAT_BEFORE, OUT_IMG_BEFORE, DATASET_DIR)
    w_aft,  miss_aft  = write_dat(after, OUT_DAT_AFTER, OUT_IMG_AFTER, DATASET_DIR)

    print("Done.")
    print(f"Total items: {len(items)}")
    print(f"{DATE_TARGET} items: {n} -> before: {len(before)}, after: {len(after)}")
    print(f"Other-date items (test): {len(others)}")
    print("")
    print(f"Wrote test.dat lines: {w_test}, missing files: {miss_test}")
    print(f"Wrote 20251019before.dat lines: {w_bef}, missing files: {miss_bef}")
    print(f"Wrote 20251019after.dat lines: {w_aft}, missing files: {miss_aft}")
    print("")
    print("Outputs:")
    print(OUT_DAT_TEST)
    print(OUT_DAT_BEFORE)
    print(OUT_DAT_AFTER)
    print("")
    print("Image folders:")
    print(OUT_IMG_TEST)
    print(OUT_IMG_BEFORE)
    print(OUT_IMG_AFTER)


if __name__ == "__main__":
    main()
