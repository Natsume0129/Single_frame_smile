from __future__ import annotations

import math
import re
from pathlib import Path

from PIL import Image, ImageOps


ROOT = Path(r"E:\Single_frame_smile\Analysis\Heatmap\output")
OUT_DIR = ROOT / "still_images"
CLASSES = ("polite", "truesmile", "ambiguous")
TARGET_RATIO = 4 / 3
PADDING = 6
BG = (255, 255, 255)


def extract_last_int(path: Path) -> int:
    m = re.search(r"(\d+)(?!.*\d)", path.stem)
    if not m:
        return 0
    return int(m.group(1))


def list_images(class_dir: Path, suffix: str) -> list[Path]:
    imgs = [p for p in class_dir.iterdir() if p.is_file() and p.name.endswith(suffix)]
    return sorted(imgs, key=extract_last_int)


def choose_grid(n: int) -> tuple[int, int]:
    best = None
    for cols in range(1, n + 1):
        rows = math.ceil(n / cols)
        ratio = cols / rows
        diff = abs(ratio - TARGET_RATIO)
        score = (diff, rows * cols)
        if best is None or score < best[0]:
            best = (score, cols, rows)
    assert best is not None
    return best[1], best[2]


def build_montage(images: list[Path], out_path: Path) -> None:
    if not images:
        return
    pil_images = [Image.open(p).convert("RGB") for p in images]
    w = max(img.width for img in pil_images)
    h = max(img.height for img in pil_images)
    cols, rows = choose_grid(len(pil_images))

    canvas_w = cols * w + (cols + 1) * PADDING
    canvas_h = rows * h + (rows + 1) * PADDING
    canvas = Image.new("RGB", (canvas_w, canvas_h), BG)

    for idx, img in enumerate(pil_images):
        r = idx // cols
        c = idx % cols
        x = PADDING + c * (w + PADDING)
        y = PADDING + r * (h + PADDING)
        if img.size != (w, h):
            img = ImageOps.pad(img, (w, h), color=BG)
        canvas.paste(img, (x, y))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for class_name in CLASSES:
        class_dir = ROOT / class_name
        heatmaps = list_images(class_dir, "_heatmap.png")
        overlays = list_images(class_dir, "_overlay.png")
        build_montage(heatmaps, OUT_DIR / f"{class_name}_heatmap_still.png")
        build_montage(overlays, OUT_DIR / f"{class_name}_overlay_still.png")
        print(f"[STILL] {class_name}: heatmaps={len(heatmaps)}, overlays={len(overlays)}")


if __name__ == "__main__":
    main()
