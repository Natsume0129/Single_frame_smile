"""
功能说明：
1. 读取 dat 文件中的帧区间（前两列：start_frame end_frame）。
2. 在单帧图片目录中按帧号匹配图片文件（例如：20250926_0_0_155.png -> 帧号 155）。
3. 按每个区间创建输出子文件夹（命名为 start-end，如 155-299）。
4. 将该区间内存在的图片复制到对应子文件夹中，并打印每个区间的复制/缺失统计。

使用方式：
python extract_window.py [--dat DAT路径] [--src 图片目录] [--out 输出目录]

命令行示例：
1) 使用脚本内默认路径：
python e:\Single_frame_smile\Analysis\analysis2-12\extract_window.py

2) 指定 dat、源目录、输出目录：
python e:\Single_frame_smile\Analysis\analysis2-12\extract_window.py 
  --dat "E:\Matsuda_data\输出\output20251019.dat"
  --src "E:\chrome-downloads\chrome-downloads\DetectedFaces\20251019\0\0"
  --out "E:\Matsuda_data\2-12meeting\output"
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path


FRAME_PATTERN = re.compile(r"_(\d+)\.[^.]+$")


def parse_dat_ranges(dat_path: Path) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    with dat_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 2:
                continue

            try:
                start = int(parts[0])
                end = int(parts[1])
            except ValueError:
                continue

            if start > end:
                start, end = end, start

            ranges.append((start, end))
    return ranges


def build_frame_index(src_dir: Path) -> dict[int, Path]:
    frame_to_file: dict[int, Path] = {}
    for file_path in src_dir.iterdir():
        if not file_path.is_file():
            continue
        match = FRAME_PATTERN.search(file_path.name)
        if not match:
            continue

        frame_id = int(match.group(1))
        frame_to_file.setdefault(frame_id, file_path)
    return frame_to_file


def extract_windows(dat_path: Path, src_dir: Path, out_dir: Path) -> None:
    ranges = parse_dat_ranges(dat_path)
    if not ranges:
        raise RuntimeError(f"No valid ranges found in dat file: {dat_path}")

    frame_to_file = build_frame_index(src_dir)
    if not frame_to_file:
        raise RuntimeError(f"No frame files found in source dir: {src_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    total_copied = 0
    total_missing = 0
    for start, end in ranges:
        window_dir = out_dir / f"{start}-{end}"
        window_dir.mkdir(parents=True, exist_ok=True)

        copied = 0
        missing = 0
        for frame_id in range(start, end + 1):
            src_file = frame_to_file.get(frame_id)
            if src_file is None:
                missing += 1
                continue
            shutil.copy2(src_file, window_dir / src_file.name)
            copied += 1

        total_copied += copied
        total_missing += missing
        print(f"[{start}-{end}] copied={copied}, missing={missing}")

    print(f"Done. windows={len(ranges)}, copied={total_copied}, missing={total_missing}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract frame windows from dat ranges into separate folders."
    )
    parser.add_argument(
        "--dat",
        type=Path,
        default=Path("E:/Matsuda_data/\u8f93\u51fa/20250926.dat"),
        help="Path to dat file containing frame ranges.",
    )
    parser.add_argument(
        "--src",
        type=Path,
        default=Path(r"E:\Matsuda_data\single_frame\20250926\DetectedFaces\20250926\0\0"),
        help="Directory containing source single-frame images.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(r"E:\Matsuda_data\single_frame\20250926\DetectedFaces\20250926\0\windows"),
        help="Output directory for extracted range folders.",
    )

    args = parser.parse_args()
    extract_windows(args.dat, args.src, args.out)


if __name__ == "__main__":
    main()
