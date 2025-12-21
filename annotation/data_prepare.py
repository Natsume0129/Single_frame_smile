# extract_frames_by_ranges_v2.py
# Edit CONFIG then run: python extract_frames_by_ranges_v2.py

from __future__ import annotations
import shutil
from pathlib import Path


# =========================
# CONFIG (edit these)
# =========================
SRC_DIR = Path(r"E:\Matsuda_data\single_frame\20251029\DetectedFaces\20251029\0\0")
DAT_PATH = Path(r"E:\Single_frame_smile\data\segments\output20251029.dat")
DST_DIR = Path(r"E:\Single_frame_smile\data\frame_data\20251029")

PREFIX = "20251029_0_0_"
EXT = ".png"

MOVE_FILES = True         # True=move, False=copy
OVERWRITE = False         # overwrite files in DST_DIR if they already exist
DRY_RUN = False           # print only, no file ops

PAD_WIDTH = None          # e.g., 6 -> 000155 ; None -> 155

LOG_MISSING = True
MISSING_LOG_PATH = DST_DIR / "missing_frames.txt"
# =========================


def parse_ranges(dat_path: Path) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    with dat_path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                s = int(parts[0])
                e = int(parts[1])
            except ValueError:
                raise ValueError(f"Line {line_no}: cannot parse start/end as int: {raw!r}")
            if e < s:
                s, e = e, s
            ranges.append((s, e))
    return ranges


def build_unique_frames(ranges: list[tuple[int, int]]) -> list[int]:
    frames = set()
    for s, e in ranges:
        frames.update(range(s, e + 1))
    return sorted(frames)


def frame_to_name(frame: int) -> str:
    if PAD_WIDTH is None:
        return f"{PREFIX}{frame}{EXT}"
    return f"{PREFIX}{frame:0{PAD_WIDTH}d}{EXT}"


def ensure_dir(p: Path) -> None:
    if p.exists():
        return
    if DRY_RUN:
        print(f"[DRY] mkdir: {p}")
    else:
        p.mkdir(parents=True, exist_ok=True)


def transfer(src: Path, dst: Path) -> None:
    if dst.exists() and not OVERWRITE:
        # already extracted before
        print(f"[SKIP] exists: {dst.name}")
        return

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
    if not SRC_DIR.exists():
        raise FileNotFoundError(f"SRC_DIR not found: {SRC_DIR}")
    if not DAT_PATH.exists():
        raise FileNotFoundError(f"DAT_PATH not found: {DAT_PATH}")

    ensure_dir(DST_DIR)

    ranges = parse_ranges(DAT_PATH)
    if not ranges:
        print("No ranges found. Nothing to do.")
        return

    frames = build_unique_frames(ranges)

    missing_frames: list[int] = []
    extracted = 0

    for frame in frames:
        name = frame_to_name(frame)
        src = SRC_DIR / name
        dst = DST_DIR / name

        if not src.exists():
            missing_frames.append(frame)
            continue

        transfer(src, dst)
        extracted += 1

    print("Done.")
    print(f"Ranges: {len(ranges)}")
    print(f"Unique target frames: {len(frames)}")
    print(f"{'Moved' if MOVE_FILES else 'Copied'}: {extracted}")
    print(f"Missing (FaceTracking filtered out etc.): {len(missing_frames)}")

    if LOG_MISSING and missing_frames:
        if DRY_RUN:
            print(f"[DRY] write missing log: {MISSING_LOG_PATH}")
        else:
            # write as plain list + also keep the expected filename
            with MISSING_LOG_PATH.open("w", encoding="utf-8") as f:
                f.write("# frame\tfilename_expected\n")
                for fr in missing_frames:
                    f.write(f"{fr}\t{frame_to_name(fr)}\n")
            print(f"Missing log written: {MISSING_LOG_PATH}")


if __name__ == "__main__":
    main()
