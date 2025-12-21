# sample_random_images.py
# Edit CONFIG then run: python sample_random_images.py
#
# What it does:
# - Collect images from multiple source folders (any number)
# - Randomly sample N images (default 1000)
# - Copy (or move) them into a single output folder
# - Avoid filename collisions by auto-renaming if needed

from __future__ import annotations
import random
import shutil
from pathlib import Path


# =========================
# CONFIG (edit these)
# =========================
SOURCE_DIRS = [
    Path(r"E:\Single_frame_smile\data\frame_data\20250926"),
    Path(r"E:\Single_frame_smile\data\frame_data\20251008"),
    Path(r"E:\Single_frame_smile\data\frame_data\20251019"),
    Path(r"E:\Single_frame_smile\data\frame_data\20251029"),
    # Add/remove folders freely
]

OUTPUT_DIR = Path(r"E:\Single_frame_smile\data\dataset")

SAMPLE_N = 1000

# Which files count as images
EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

RECURSIVE = False         # True: include subfolders
MOVE_FILES = False        # False=copy, True=move
OVERWRITE = False         # overwrite if name exists (if False, auto-rename)
DRY_RUN = False           # True: print only, no file ops

SEED = 42                 # change for different random samples; set None for random each run
# =========================


def ensure_dir(p: Path) -> None:
    if p.exists():
        return
    if DRY_RUN:
        print(f"[DRY] mkdir: {p}")
    else:
        p.mkdir(parents=True, exist_ok=True)


def list_images(folder: Path) -> list[Path]:
    if not folder.exists():
        raise FileNotFoundError(f"Source folder not found: {folder}")

    if RECURSIVE:
        files = [p for p in folder.rglob("*") if p.is_file()]
    else:
        files = [p for p in folder.iterdir() if p.is_file()]

    return [p for p in files if p.suffix.lower() in EXTENSIONS]


def unique_dest_path(dst_dir: Path, src: Path) -> Path:
    """
    If OVERWRITE is False and the filename already exists, append a numeric suffix.
    Example: img.png -> img__2.png
    """
    dst = dst_dir / src.name
    if OVERWRITE or not dst.exists():
        return dst

    stem = src.stem
    suffix = src.suffix
    k = 2
    while True:
        cand = dst_dir / f"{stem}__{k}{suffix}"
        if not cand.exists():
            return cand
        k += 1


def transfer(src: Path, dst: Path) -> None:
    if DRY_RUN:
        op = "MOVE" if MOVE_FILES else "COPY"
        print(f"[DRY] {op}: {src} -> {dst}")
        return

    if OVERWRITE and dst.exists():
        dst.unlink()

    if MOVE_FILES:
        shutil.move(str(src), str(dst))
    else:
        shutil.copy2(str(src), str(dst))


def main() -> None:
    if SEED is not None:
        random.seed(SEED)

    all_images: list[Path] = []
    for d in SOURCE_DIRS:
        all_images.extend(list_images(d))

    if not all_images:
        print("No images found in SOURCE_DIRS.")
        return

    if SAMPLE_N > len(all_images):
        raise ValueError(
            f"SAMPLE_N={SAMPLE_N} is larger than available images={len(all_images)}"
        )

    ensure_dir(OUTPUT_DIR)

    chosen = random.sample(all_images, SAMPLE_N)

    for src in chosen:
        dst = unique_dest_path(OUTPUT_DIR, src)
        transfer(src, dst)

    print("Done.")
    print(f"Source folders: {len(SOURCE_DIRS)}")
    print(f"Available images: {len(all_images)}")
    print(f"Sampled: {len(chosen)}")
    print(f"Output: {OUTPUT_DIR}")
    if DRY_RUN:
        print("DRY_RUN=True: no files were actually moved/copied.")


if __name__ == "__main__":
    main()
