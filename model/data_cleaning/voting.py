#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Keep only samples whose labels match in two .dat files, copy matched images,
and also output conflict.dat + agreement%.

Input dat format (each line):
  <relative_or_absolute_path> <label>

Matching is by normalized path string (order does NOT matter).
If one side is missing (discarded), it is excluded from agreement calculation by default.

Outputs (under OUT_DIR):
  - matched.dat     : only matched labels
  - conflict.dat    : only conflicts (both exist but labels differ)
  - images/         : copied matched images
"""

from __future__ import annotations
import os
import shutil
from pathlib import Path
from typing import Dict, Tuple


# =========================
# Config: edit here
# =========================
DATA_ROOT = r"E:\Single_frame_smile\data"               # base directory for resolving relative paths in dat
DAT1 = r"E:\Single_frame_smile\data\Isornot\labels.dat"
DAT2 = r"E:\Single_frame_smile\data\Isornot\labels1.dat"

OUT_DIR = r"E:\Single_frame_smile\data\Isornot"
OUT_IMG_DIR = os.path.join(OUT_DIR, "images")
OUT_MATCHED_DAT = os.path.join(OUT_DIR, "matched.dat")
OUT_CONFLICT_DAT = os.path.join(OUT_DIR, "conflict.dat")

# Agreement definition:
# - If True: agreement% computed over ONLY items present in BOTH dat files (recommended).
# - If False: compute over UNION, treating "missing on one side" as disagreement.
AGREE_ON_INTERSECTION_ONLY = True

# Copying:
# - If True: keep relative structure under OUT_IMG_DIR (recommended).
# - If False: flatten to filenames (risk of overwriting if names collide).
KEEP_RELATIVE_STRUCTURE = True

ALLOW_DUPLICATE_PATHS = True
# =========================


def parse_dat(dat_path: str, data_root: str) -> Dict[str, Tuple[str, int]]:
    dat_path = str(Path(dat_path).resolve())
    root = Path(data_root).resolve()

    mapping: Dict[str, Tuple[str, int]] = {}

    with open(dat_path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                raise ValueError(f"[{dat_path}] Line {ln} must be: <path> <label>, got: {line!r}")

            p_str, y_str = parts
            try:
                y = int(y_str)
            except ValueError:
                raise ValueError(f"[{dat_path}] Line {ln} label must be int, got: {y_str!r}")

            p = Path(p_str)

            # Resolve absolute image path
            if p.is_absolute():
                abs_p = p
            else:
                abs_p = (root / p).resolve()

            # Normalize key for matching
            if p.is_absolute():
                try:
                    rel = abs_p.relative_to(root)
                    key = rel.as_posix()
                except Exception:
                    key = abs_p.as_posix()
            else:
                key = p.as_posix().lstrip("./")

            if (key in mapping) and (not ALLOW_DUPLICATE_PATHS):
                raise ValueError(f"Duplicate path key in {dat_path}: {key}")

            mapping[key] = (str(abs_p), y)

    return mapping


def ensure_parent(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def copy_matched_image(src_path: str, key: str, out_dir: Path, out_img_root: Path) -> Path:
    src = Path(src_path)
    if not src.exists():
        raise FileNotFoundError(src_path)

    if KEEP_RELATIVE_STRUCTURE:
        rel = Path(key)
        if rel.is_absolute():
            rel = Path(src.name)
        dest = out_img_root / rel
    else:
        dest = out_img_root / src.name

    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)

    # Return path relative to OUT_DIR for writing into dat
    return dest.relative_to(out_dir)


def main() -> None:
    root = Path(DATA_ROOT).resolve()
    out_dir = Path(OUT_DIR).resolve()
    out_img_root = Path(OUT_IMG_DIR).resolve()
    out_img_root.mkdir(parents=True, exist_ok=True)

    out_matched = Path(OUT_MATCHED_DAT).resolve()
    out_conflict = Path(OUT_CONFLICT_DAT).resolve()
    ensure_parent(str(out_matched))
    ensure_parent(str(out_conflict))

    m1 = parse_dat(DAT1, str(root))
    m2 = parse_dat(DAT2, str(root))

    keys_intersection = sorted(set(m1.keys()) & set(m2.keys()))
    keys_union = sorted(set(m1.keys()) | set(m2.keys()))

    matched_count = 0
    conflict_count = 0
    missing_on_one_side = 0
    copied_count = 0
    missing_file_on_disk = 0

    with open(out_matched, "w", encoding="utf-8") as w_match, \
         open(out_conflict, "w", encoding="utf-8") as w_conf:

        # Iterate over union for full accounting, but only write matched/conflict when both exist.
        for key in keys_union:
            in1 = key in m1
            in2 = key in m2

            if not (in1 and in2):
                missing_on_one_side += 1
                continue

            img1, y1 = m1[key]
            img2, y2 = m2[key]

            if y1 == y2:
                matched_count += 1

                # Copy: prefer dat1 path; fallback dat2
                src = img1 if Path(img1).exists() else img2
                if not Path(src).exists():
                    missing_file_on_disk += 1
                    continue

                rel_dest = copy_matched_image(src, key, out_dir, out_img_root)
                w_match.write(f"{rel_dest.as_posix()} {y1}\n")
                copied_count += 1
            else:
                conflict_count += 1
                # Write conflict with both labels for audit
                # Format: <path> <label_in_dat1> <label_in_dat2>
                w_conf.write(f"{key} {y1} {y2}\n")

    # Agreement%
    if AGREE_ON_INTERSECTION_ONLY:
        denom = len(keys_intersection)
        numer = matched_count
        basis = "intersection_only"
    else:
        denom = len(keys_union)
        numer = matched_count
        basis = "union_missing_as_disagree"

    agreement = (numer / denom * 100.0) if denom > 0 else 0.0

    print("Done.")
    print(f"Dat1 entries: {len(m1)}")
    print(f"Dat2 entries: {len(m2)}")
    print(f"Intersection (both present): {len(keys_intersection)}")
    print(f"Union (either present): {len(keys_union)}")
    print(f"Matched labels (both present): {matched_count}")
    print(f"Conflicts (both present): {conflict_count}")
    print(f"Missing on one side: {missing_on_one_side}")
    print(f"Copied matched images: {copied_count}")
    print(f"Missing files on disk (matched items): {missing_file_on_disk}")
    print(f"Agreement% ({basis}): {agreement:.2f}%")
    print(f"Output matched dat: {out_matched}")
    print(f"Output conflict dat: {out_conflict}")
    print(f"Output images dir: {out_img_root}")


if __name__ == "__main__":
    main()
