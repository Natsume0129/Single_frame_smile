from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi"}
MAPPING_COLUMNS = [
    "source_file",
    "source_path",
    "part_file",
    "part_path",
    "part_index",
    "start_time_sec",
    "end_time_sec",
    "theoretical_duration_sec",
    "actual_duration_sec",
    "overlap_time_sec",
    "size_bytes",
    "size_mb",
    "size_gb",
    "status",
    "warning",
]
DURATION_DIFF_WARNING_SEC = 2.0


@dataclass(frozen=True)
class SegmentPlan:
    part_index: int
    start_time_sec: float
    end_time_sec: float

    @property
    def duration_sec(self) -> float:
        return self.end_time_sec - self.start_time_sec


@dataclass
class MappingRow:
    source_file: str
    source_path: str
    part_file: str = ""
    part_path: str = ""
    part_index: str = ""
    start_time_sec: str = ""
    end_time_sec: str = ""
    theoretical_duration_sec: str = ""
    actual_duration_sec: str = ""
    overlap_time_sec: str = ""
    size_bytes: str = ""
    size_mb: str = ""
    size_gb: str = ""
    status: str = ""
    warning: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Losslessly split videos with overlap around split boundaries."
    )
    parser.add_argument("--input_dir", required=True, help="Folder containing source videos.")
    parser.add_argument("--output_dir", required=True, help="Folder for split videos and mapping files.")
    parser.add_argument("--segment_time", type=float, default=300.0, help="Target segment duration in seconds.")
    parser.add_argument("--overlap_time", type=float, default=10.0, help="Overlap duration in seconds.")
    parser.add_argument("--max_size_gb", type=float, default=2.0, help="Warning threshold per part.")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate existing parts for each source video.")
    parser.add_argument("--dry_run", action="store_true", help="Print planned ffmpeg commands without writing files.")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> tuple[Path, Path]:
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    if args.segment_time <= 0:
        raise ValueError("--segment_time must be greater than 0.")
    if args.overlap_time < 0:
        raise ValueError("--overlap_time must be greater than or equal to 0.")
    if args.overlap_time >= args.segment_time:
        raise ValueError("--overlap_time must be smaller than --segment_time.")
    if args.max_size_gb <= 0:
        raise ValueError("--max_size_gb must be greater than 0.")
    return input_dir, output_dir


def require_tool(tool_name: str) -> str:
    tool_path = shutil.which(tool_name)
    if tool_path is None:
        raise RuntimeError(f"[ERROR] {tool_name} not found. Please install ffmpeg and add it to PATH.")
    return tool_path


def find_video_files(input_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTS
    )


def build_segment_plan(
    video_duration: float,
    segment_time: float,
    overlap_time: float,
) -> list[SegmentPlan]:
    if video_duration <= 0:
        return []
    if overlap_time >= segment_time:
        raise ValueError("overlap_time must be smaller than segment_time.")

    plans: list[SegmentPlan] = []
    step_time = segment_time - overlap_time
    start = 0.0
    part_index = 0
    while start < video_duration:
        end = min(start + segment_time, video_duration)
        plans.append(SegmentPlan(part_index, start, end))
        if end >= video_duration:
            break
        start += step_time
        part_index += 1
    return plans


def part_filename(source_path: Path, plan: SegmentPlan) -> str:
    start_sec = int(round(plan.start_time_sec))
    end_sec = int(round(plan.end_time_sec))
    return (
        f"{source_path.stem}_part{plan.part_index:03d}_"
        f"start{start_sec:06d}s_end{end_sec:06d}s{source_path.suffix}"
    )


def existing_part_files(output_dir: Path, source_stem: str) -> list[Path]:
    if not output_dir.exists():
        return []
    prefix = f"{source_stem}_part"
    return sorted(path for path in output_dir.iterdir() if path.is_file() and path.name.startswith(prefix))


def ffprobe_duration(ffprobe_path: str, video_path: Path) -> float | None:
    command = [
        ffprobe_path,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except (subprocess.CalledProcessError, OSError):
        return None
    try:
        return float(result.stdout.strip())
    except ValueError:
        return None


def ffmpeg_command(ffmpeg_path: str, source_path: Path, output_path: Path, plan: SegmentPlan) -> list[str]:
    return [
        ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        format_seconds(plan.start_time_sec),
        "-i",
        str(source_path),
        "-t",
        format_seconds(plan.duration_sec),
        "-map",
        "0",
        "-c",
        "copy",
        "-dn",
        "-avoid_negative_ts",
        "make_zero",
        str(output_path),
    ]


def format_seconds(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".")


def format_mapping_time(value: float) -> str:
    return f"{value:.3f}"


def make_mapping_row(
    source_path: Path,
    output_path: Path,
    plan: SegmentPlan,
    overlap_time: float,
    max_size_gb: float,
    ffprobe_path: str,
) -> MappingRow:
    warnings: list[str] = []
    status = "OK"

    size_bytes = output_path.stat().st_size if output_path.exists() else 0
    size_mb = size_bytes / (1024**2)
    size_gb = size_bytes / (1024**3)
    if size_gb > max_size_gb:
        status = "WARNING_OVER_SIZE"
        warnings.append("File size exceeds max_size_gb. Consider reducing --segment_time.")

    actual_duration = ffprobe_duration(ffprobe_path, output_path) if output_path.exists() else None
    if actual_duration is None:
        warnings.append("FFPROBE_DURATION_FAILED")
    elif abs(actual_duration - plan.duration_sec) > DURATION_DIFF_WARNING_SEC:
        warnings.append("ACTUAL_DURATION_DIFF")

    return MappingRow(
        source_file=source_path.name,
        source_path=str(source_path.resolve()),
        part_file=output_path.name,
        part_path=str(output_path.resolve()),
        part_index=str(plan.part_index),
        start_time_sec=format_mapping_time(plan.start_time_sec),
        end_time_sec=format_mapping_time(plan.end_time_sec),
        theoretical_duration_sec=format_mapping_time(plan.duration_sec),
        actual_duration_sec="" if actual_duration is None else format_mapping_time(actual_duration),
        overlap_time_sec=format_mapping_time(overlap_time),
        size_bytes=str(size_bytes),
        size_mb=f"{size_mb:.3f}",
        size_gb=f"{size_gb:.6f}",
        status=status,
        warning=";".join(warnings),
    )


def error_row(source_path: Path, warning: str) -> MappingRow:
    return MappingRow(
        source_file=source_path.name,
        source_path=str(source_path.resolve()),
        status="ERROR",
        warning=warning,
    )


def copied_under_size_row(source_path: Path, output_path: Path, max_size_gb: float) -> MappingRow:
    size_bytes = output_path.stat().st_size if output_path.exists() else source_path.stat().st_size
    size_mb = size_bytes / (1024**2)
    size_gb = size_bytes / (1024**3)
    return MappingRow(
        source_file=source_path.name,
        source_path=str(source_path.resolve()),
        part_file=output_path.name,
        part_path=str(output_path.resolve()),
        part_index="0",
        size_bytes=str(size_bytes),
        size_mb=f"{size_mb:.3f}",
        size_gb=f"{size_gb:.6f}",
        status="COPIED_UNDER_SIZE",
        warning=f"Source file is not larger than {max_size_gb:g} GB; copied without splitting.",
    )


def write_mapping(output_dir: Path, rows: list[MappingRow]) -> None:
    csv_path = output_dir / "split_mapping.csv"
    json_path = output_dir / "split_mapping.json"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MAPPING_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    with json_path.open("w", encoding="utf-8") as f:
        json.dump([asdict(row) for row in rows], f, ensure_ascii=False, indent=2)
    print(f"[INFO] Mapping saved to: {csv_path}")
    print(f"[INFO] Mapping saved to: {json_path}")


def process_video(
    source_path: Path,
    output_dir: Path,
    args: argparse.Namespace,
    ffmpeg_path: str,
    ffprobe_path: str,
) -> list[MappingRow]:
    existing_parts = existing_part_files(output_dir, source_path.stem)
    source_size_gb = source_path.stat().st_size / (1024**3)
    if source_size_gb <= args.max_size_gb:
        print(
            f"[SKIP] {source_path.name} is {source_size_gb:.6f} GB, "
            f"not larger than {args.max_size_gb:g} GB; copying without splitting."
        )
        if existing_parts and args.overwrite:
            if args.dry_run:
                for path in existing_parts:
                    print(f"[DRY_RUN] Would remove old part: {path}")
            else:
                for path in existing_parts:
                    path.unlink()
        elif existing_parts:
            print("[INFO] Existing split parts were left in place. Use --overwrite to remove them.")

        output_path = output_dir / source_path.name
        if output_path.exists() and not args.overwrite:
            print(f"[SKIP] Copied output already exists for {source_path.name}. Use --overwrite to replace it.")
        elif args.dry_run:
            print(f"[DRY_RUN] Would copy {source_path} -> {output_path}")
        else:
            shutil.copy2(source_path, output_path)
        return [copied_under_size_row(source_path, output_path, args.max_size_gb)]

    if existing_parts and not args.overwrite:
        print(f"[SKIP] Output already exists for {source_path.stem}. Use --overwrite to regenerate.")
        return []
    if existing_parts and args.overwrite and not args.dry_run:
        for path in existing_parts:
            path.unlink()

    duration = ffprobe_duration(ffprobe_path, source_path)
    if duration is None:
        print(f"[ERROR] Could not read duration: {source_path.name}")
        return [error_row(source_path, "FFPROBE_SOURCE_DURATION_FAILED")]

    step_time = args.segment_time - args.overlap_time
    plans = build_segment_plan(duration, args.segment_time, args.overlap_time)
    print(f"[INFO] Video duration: {duration:.2f} sec")
    print(
        f"[INFO] segment_time={args.segment_time:g}, "
        f"overlap_time={args.overlap_time:g}, step_time={step_time:g}"
    )

    rows: list[MappingRow] = []
    for plan in plans:
        output_path = output_dir / part_filename(source_path, plan)
        command = ffmpeg_command(ffmpeg_path, source_path, output_path, plan)
        print(
            f"[INFO] Exporting part{plan.part_index:03d}: "
            f"{plan.start_time_sec:.2f}s -> {plan.end_time_sec:.2f}s"
        )
        if args.dry_run:
            print("[DRY_RUN] " + subprocess.list2cmdline(command))
            continue
        try:
            subprocess.run(command, check=True)
        except (subprocess.CalledProcessError, OSError) as exc:
            print(f"[ERROR] ffmpeg failed for {output_path.name}: {exc}")
            row = error_row(source_path, "FFMPEG_SPLIT_FAILED")
            row.part_file = output_path.name
            row.part_path = str(output_path.resolve())
            row.part_index = str(plan.part_index)
            row.start_time_sec = format_mapping_time(plan.start_time_sec)
            row.end_time_sec = format_mapping_time(plan.end_time_sec)
            row.theoretical_duration_sec = format_mapping_time(plan.duration_sec)
            row.overlap_time_sec = format_mapping_time(args.overlap_time)
            rows.append(row)
            continue

        row = make_mapping_row(source_path, output_path, plan, args.overlap_time, args.max_size_gb, ffprobe_path)
        if row.status == "WARNING_OVER_SIZE":
            print(
                f"[WARNING] {output_path.name} is {row.size_gb} GB, "
                f"larger than {args.max_size_gb:g} GB. Consider reducing --segment_time."
            )
        rows.append(row)
    return rows


def main() -> int:
    args = parse_args()
    try:
        input_dir, output_dir = validate_args(args)
        ffmpeg_path = require_tool("ffmpeg")
        ffprobe_path = require_tool("ffprobe")
    except (ValueError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)
    videos = find_video_files(input_dir)
    if not videos:
        print(f"[INFO] Found 0 video files in: {input_dir}")
        return 0

    print(f"[INFO] Found {len(videos)} video files.")
    all_rows: list[MappingRow] = []
    warning_count = 0
    for index, source_path in enumerate(videos, start=1):
        print(f"[INFO] Processing {index}/{len(videos)}: {source_path.name}")
        rows = process_video(source_path, output_dir, args, ffmpeg_path, ffprobe_path)
        warning_count += sum(1 for row in rows if row.status == "WARNING_OVER_SIZE")
        all_rows.extend(rows)
        print(f"[OK] Finished: {source_path.name}")

    if warning_count:
        print(f"[WARNING] {warning_count} part(s) are larger than {args.max_size_gb:g} GB.")
    if args.dry_run:
        print("[INFO] Dry run complete. No files or mapping were written.")
    else:
        write_mapping(output_dir, all_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
