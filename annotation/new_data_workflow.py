from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path


VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi"}
DEFAULT_RVM_PYTHON = Path(r"C:\Program Files\Python310\python.exe")
DEFAULT_RVM_SCRIPT = Path(r"E:\toolkit\greenbackground\rvm_extract.py")
DEFAULT_FACETRACKING_PYTHON = Path(r"E:\SmileAnnotation\FaceTracking-Smile_Detection\venv\Scripts\python.exe")
DEFAULT_FACETRACKING_SCRIPT = Path(
    r"E:\SmileAnnotation\FaceTracking-Smile_Detection\FaceTracking\CUI-pyplot\face_detection.py"
)


@dataclass(frozen=True)
class DatEvent:
    row_index: int
    time_text: str
    time_sec: float
    label: str


@dataclass(frozen=True)
class ClipPlan:
    seq_id: str
    event: DatEvent
    start_sec: float
    end_sec: float

    @property
    def duration_sec(self) -> float:
        return self.end_sec - self.start_sec


@dataclass
class ManifestRow:
    seq_id: str
    label: str
    source_video: str
    event_time_text: str
    event_time_sec: str
    window_start_sec: str
    window_end_sec: str
    window_duration_sec: str
    raw_clip_path: str
    rvm_video_path: str
    facetracking_output_dir: str
    status: str


def parse_time(value: str) -> float:
    value = value.strip()
    if not value:
        raise ValueError("Empty time value.")
    if ":" not in value:
        return float(value)

    parts = value.split(":")
    if len(parts) == 2:
        minutes = int(parts[0])
        seconds = float(parts[1])
        return minutes * 60 + seconds
    if len(parts) == 3:
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = float(parts[2])
        return hours * 3600 + minutes * 60 + seconds
    raise ValueError(f"Unsupported time format: {value!r}")


def parse_dat(dat_path: Path) -> list[DatEvent]:
    events: list[DatEvent] = []
    with dat_path.open("r", encoding="utf-8-sig") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) == 2 and parts[0].lower() == "time" and parts[1].lower() == "class":
                continue
            if len(parts) != 2:
                raise ValueError(f"{dat_path}:{line_no}: expected '<time> <class>', got {line!r}")

            time_text, label = parts
            if any(ch in label for ch in "\\/:*?\"<>|"):
                raise ValueError(f"{dat_path}:{line_no}: label cannot be used as a folder name: {label!r}")

            events.append(
                DatEvent(
                    row_index=len(events),
                    time_text=time_text,
                    time_sec=parse_time(time_text),
                    label=label,
                )
            )

    if not events:
        raise ValueError(f"No events found in dat file: {dat_path}")
    return events


def fmt_sec(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".")


def fmt_for_name(value: float) -> str:
    return f"{int(round(value * 1000)):09d}ms"


def resolve_tool(name: str, explicit_path: str | None) -> str:
    if explicit_path:
        path = Path(explicit_path)
        if not path.exists():
            raise FileNotFoundError(f"{name} not found: {path}")
        return str(path)

    found = shutil.which(name)
    if not found:
        raise RuntimeError(f"{name} not found. Install it or pass --{name}.")
    return found


def discover_video(video_dir: Path, video_path: Path | None) -> Path:
    if video_path is not None:
        if not video_path.exists() or not video_path.is_file():
            raise FileNotFoundError(f"Video not found: {video_path}")
        return video_path

    videos = sorted(path for path in video_dir.iterdir() if path.is_file() and path.suffix.lower() in VIDEO_EXTS)
    if len(videos) != 1:
        raise ValueError(
            f"Expected exactly one video in {video_dir}, found {len(videos)}. "
            "Pass --video when the directory contains multiple videos."
        )
    return videos[0]


def ffprobe_duration(ffprobe: str, video_path: Path) -> float:
    command = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return float(result.stdout.strip())


def ffprobe_frame_count(ffprobe: str, video_path: Path) -> int:
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=nb_read_frames",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    value = result.stdout.strip()
    if not value or value.upper() == "N/A":
        raise RuntimeError(f"Could not read frame count from: {video_path}")
    return int(value)


def collect_face_png_indices(detected_dir: Path) -> tuple[int, list[int], int | None, int | None]:
    indices: list[int] = []
    if not detected_dir.exists():
        return 0, [], None, None

    for path in detected_dir.rglob("*.png"):
        try:
            frame_idx = int(path.stem.rsplit("_", 1)[-1])
        except ValueError:
            continue
        indices.append(frame_idx)

    if not indices:
        return 0, [], None, None

    sorted_indices = sorted(indices)
    seen = set(sorted_indices)
    missing = [idx for idx in range(sorted_indices[0], sorted_indices[-1] + 1) if idx not in seen]
    return len(sorted_indices), missing, sorted_indices[0], sorted_indices[-1]


def validate_stage_outputs(ffprobe: str, raw_clip: Path, rvm_video: Path, facetracking_output_dir: Path) -> None:
    raw_frames = ffprobe_frame_count(ffprobe, raw_clip)
    rvm_frames = ffprobe_frame_count(ffprobe, rvm_video)
    if raw_frames != rvm_frames:
        raise RuntimeError(f"RVM frame mismatch: raw={raw_frames}, rvm={rvm_frames}, video={rvm_video}")

    detected_dir = facetracking_output_dir / "DetectedFaces" / rvm_video.stem
    png_count, missing, min_idx, max_idx = collect_face_png_indices(detected_dir)
    if png_count != rvm_frames or missing or min_idx != 0 or max_idx != rvm_frames - 1:
        missing_text = ",".join(str(x) for x in missing[:20])
        if len(missing) > 20:
            missing_text += ",..."
        raise RuntimeError(
            "FaceTracking frame mismatch: "
            f"png={png_count}, expected={rvm_frames}, min={min_idx}, max={max_idx}, "
            f"missing={missing_text}, detected_dir={detected_dir}"
        )


def make_window(time_sec: float, video_duration: float, pre_sec: float, post_sec: float) -> tuple[float, float]:
    desired_duration = pre_sec + post_sec
    start = time_sec - pre_sec
    end = time_sec + post_sec

    if start < 0:
        end += -start
        start = 0.0
    if end > video_duration:
        shift = end - video_duration
        start = max(0.0, start - shift)
        end = video_duration

    if end <= start:
        raise ValueError(f"Invalid window around {time_sec}: start={start}, end={end}")

    if end - start > desired_duration + 0.001:
        end = start + desired_duration
    return start, end


def build_clip_plans(events: list[DatEvent], video_duration: float, pre_sec: float, post_sec: float) -> list[ClipPlan]:
    plans: list[ClipPlan] = []
    for event in events:
        if event.time_sec < 0:
            raise ValueError(f"Negative event time: {event.time_text}")
        if event.time_sec > video_duration:
            raise ValueError(
                f"Event time {event.time_text} ({event.time_sec:.3f}s) exceeds video duration "
                f"{video_duration:.3f}s."
            )
        start_sec, end_sec = make_window(event.time_sec, video_duration, pre_sec, post_sec)
        plans.append(ClipPlan(f"{event.row_index:03d}", event, start_sec, end_sec))
    return plans


def cut_clip(
    ffmpeg: str,
    source_video: Path,
    output_path: Path,
    plan: ClipPlan,
    overwrite: bool,
    copy_codec: bool,
    dry_run: bool,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Raw clip already exists: {output_path}")

    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if overwrite else "-n",
        "-ss",
        fmt_sec(plan.start_sec),
        "-i",
        str(source_video),
        "-t",
        fmt_sec(plan.duration_sec),
        "-an",
    ]
    if copy_codec:
        command.extend(["-c:v", "copy"])
    else:
        command.extend(["-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p"])
    command.append(str(output_path))

    run_list_command(command, dry_run)


def run_list_command(command: list[str], dry_run: bool) -> None:
    if dry_run:
        print("[DRY]", " ".join(f'"{part}"' if " " in part else part for part in command))
        return
    subprocess.run(command, check=True)


def run_template_command(template: str, values: dict[str, str], dry_run: bool) -> None:
    command = template.format(**values)
    if dry_run:
        print("[DRY]", command)
        return
    subprocess.run(command, shell=True, check=True)


def validate_file_arg(name: str, path: Path) -> None:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"{name} not found: {path}")


def run_rvm(args: argparse.Namespace, values: dict[str, str]) -> None:
    if args.rvm_cmd:
        run_template_command(args.rvm_cmd, values, args.dry_run)
        return

    command = [
        str(args.rvm_python),
        str(args.rvm_script),
        "--input",
        values["input"],
        "--output-video",
        values["output"],
        "--output-frames",
        "",
        "--device",
        args.rvm_device,
        "--downsample-ratio",
        str(args.rvm_downsample_ratio),
        "--fourcc",
        args.rvm_fourcc,
    ]
    if args.overwrite:
        command.append("--overwrite")
    run_list_command(command, args.dry_run)


def run_facetracking(args: argparse.Namespace, values: dict[str, str]) -> None:
    if args.facetracking_cmd:
        run_template_command(args.facetracking_cmd, values, args.dry_run)
        return

    command = [
        str(args.facetracking_python),
        str(args.facetracking_script),
        "--movie_file",
        values["input"],
        "-o",
        values["output_dir"],
        "-r",
        str(args.facetracking_resolution),
    ]
    run_list_command(command, args.dry_run)


def clip_base_name(video_stem: str, plan: ClipPlan) -> str:
    event_ms = fmt_for_name(plan.event.time_sec)
    start_ms = fmt_for_name(plan.start_sec)
    end_ms = fmt_for_name(plan.end_sec)
    return f"{video_stem}_{plan.seq_id}_{plan.event.label}_t{event_ms}_win{start_ms}-{end_ms}"


def write_manifest(manifest_path: Path, rows: list[ManifestRow]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_config(config_path: Path, args: argparse.Namespace, source_video: Path, video_duration: float) -> None:
    payload = {
        "source_video": str(source_video.resolve()),
        "video_duration_sec": video_duration,
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    }
    config_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create smile-event clips from a '<time> <class>' dat file, then run external "
            "RVM and FaceTracking commands."
        )
    )
    parser.add_argument("--video_dir", required=True, type=Path, help="Directory containing one source video.")
    parser.add_argument("--video", type=Path, default=None, help="Source video path when video_dir has multiple videos.")
    parser.add_argument("--dat", required=True, type=Path, help="Dat file with lines like '0:11 polite'.")
    parser.add_argument("--output_root", type=Path, default=Path(r"E:\Matsuda_data\new_data_workflow"))
    parser.add_argument("--pre_sec", type=float, default=5.0, help="Seconds before the event time.")
    parser.add_argument("--post_sec", type=float, default=5.0, help="Seconds after the event time.")
    parser.add_argument("--ffmpeg", default=None, help="Optional explicit ffmpeg path.")
    parser.add_argument("--ffprobe", default=None, help="Optional explicit ffprobe path.")
    parser.add_argument("--copy_cut", action="store_true", help="Use stream copy for cutting instead of x264 re-encode.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument(
        "--allow_incomplete_facetracking",
        action="store_true",
        help=(
            "Continue when FaceTracking produces fewer face crops than video frames. "
            "RVM/raw frame mismatches still fail."
        ),
    )
    parser.add_argument(
        "--stop_after",
        choices=("clips", "rvm", "facetracking"),
        default="facetracking",
        help="Stop after a specific workflow stage.",
    )
    parser.add_argument(
        "--rvm_cmd",
        default=None,
        help=(
            "Command template for RVM. Use placeholders such as {input}, {output}, "
            "{seq_id}, {label}, {output_root}. Quote placeholders in the template."
        ),
    )
    parser.add_argument("--rvm_python", type=Path, default=DEFAULT_RVM_PYTHON)
    parser.add_argument("--rvm_script", type=Path, default=DEFAULT_RVM_SCRIPT)
    parser.add_argument("--rvm_device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--rvm_downsample_ratio", type=float, default=0.25)
    parser.add_argument("--rvm_fourcc", default="mp4v")
    parser.add_argument(
        "--facetracking_cmd",
        default=None,
        help=(
            "Command template for FaceTracking. Use {input} for the RVM video and "
            "{output_dir} for the target output directory."
        ),
    )
    parser.add_argument("--facetracking_python", type=Path, default=DEFAULT_FACETRACKING_PYTHON)
    parser.add_argument("--facetracking_script", type=Path, default=DEFAULT_FACETRACKING_SCRIPT)
    parser.add_argument("--facetracking_resolution", type=int, default=224)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.video_dir.exists() or not args.video_dir.is_dir():
        raise FileNotFoundError(f"video_dir not found: {args.video_dir}")
    if not args.dat.exists() or not args.dat.is_file():
        raise FileNotFoundError(f"dat not found: {args.dat}")
    if args.pre_sec < 0 or args.post_sec < 0:
        raise ValueError("--pre_sec and --post_sec must be non-negative.")
    if args.pre_sec + args.post_sec <= 0:
        raise ValueError("The window duration must be greater than zero.")
    if args.stop_after in {"rvm", "facetracking"} and not args.rvm_cmd:
        validate_file_arg("--rvm_python", args.rvm_python)
        validate_file_arg("--rvm_script", args.rvm_script)
    if args.stop_after == "facetracking" and not args.facetracking_cmd:
        validate_file_arg("--facetracking_python", args.facetracking_python)
        validate_file_arg("--facetracking_script", args.facetracking_script)
    if len(args.rvm_fourcc) != 4:
        raise ValueError("--rvm_fourcc must be exactly four characters.")


def main() -> None:
    args = parse_args()
    validate_args(args)

    ffmpeg = resolve_tool("ffmpeg", args.ffmpeg)
    ffprobe = resolve_tool("ffprobe", args.ffprobe)
    source_video = discover_video(args.video_dir, args.video)
    events = parse_dat(args.dat)
    video_duration = ffprobe_duration(ffprobe, source_video)
    plans = build_clip_plans(events, video_duration, args.pre_sec, args.post_sec)

    args.output_root.mkdir(parents=True, exist_ok=True)
    write_config(args.output_root / "workflow_config.json", args, source_video, video_duration)

    rows: list[ManifestRow] = []
    for plan in plans:
        base_name = clip_base_name(source_video.stem, plan)
        label_dir = plan.event.label
        seq_dir_name = plan.seq_id

        raw_clip = args.output_root / "clips_raw" / label_dir / seq_dir_name / f"{base_name}_raw.mp4"
        rvm_video = args.output_root / "rvm_greenbg" / label_dir / seq_dir_name / f"{base_name}_rvm_green.mp4"
        facetracking_output_dir = args.output_root / "facetracking" / label_dir / seq_dir_name

        cut_clip(ffmpeg, source_video, raw_clip, plan, args.overwrite, args.copy_cut, args.dry_run)

        values = {
            "input": str(raw_clip),
            "output": str(rvm_video),
            "output_dir": str(facetracking_output_dir),
            "output_root": str(args.output_root),
            "seq_id": plan.seq_id,
            "label": plan.event.label,
            "event_time_sec": fmt_sec(plan.event.time_sec),
            "window_start_sec": fmt_sec(plan.start_sec),
            "window_end_sec": fmt_sec(plan.end_sec),
        }
        if args.stop_after == "clips":
            stage_status = "clips_only"
        else:
            rvm_video.parent.mkdir(parents=True, exist_ok=True)
            run_rvm(args, values)

            if args.stop_after == "rvm":
                stage_status = "rvm_only"
            else:
                if args.overwrite and facetracking_output_dir.exists() and not args.dry_run:
                    shutil.rmtree(facetracking_output_dir)
                facetracking_output_dir.mkdir(parents=True, exist_ok=True)
                values["input"] = str(rvm_video)
                run_facetracking(args, values)
                if not args.dry_run:
                    try:
                        validate_stage_outputs(ffprobe, raw_clip, rvm_video, facetracking_output_dir)
                        stage_status = "ok"
                    except RuntimeError as exc:
                        if args.allow_incomplete_facetracking and str(exc).startswith("FaceTracking frame mismatch:"):
                            print(f"[WARN] {plan.seq_id} {plan.event.label}: {exc}")
                            stage_status = "facetracking_incomplete"
                        else:
                            raise
                else:
                    stage_status = "dry_run"

        rows.append(
            ManifestRow(
                seq_id=plan.seq_id,
                label=plan.event.label,
                source_video=str(source_video.resolve()),
                event_time_text=plan.event.time_text,
                event_time_sec=fmt_sec(plan.event.time_sec),
                window_start_sec=fmt_sec(plan.start_sec),
                window_end_sec=fmt_sec(plan.end_sec),
                window_duration_sec=fmt_sec(plan.duration_sec),
                raw_clip_path=str(raw_clip.resolve()),
                rvm_video_path=str(rvm_video.resolve()),
                facetracking_output_dir=str(facetracking_output_dir.resolve()),
                status=stage_status,
            )
        )

    write_manifest(args.output_root / "manifest.csv", rows)
    print(f"Done. events={len(rows)} manifest={args.output_root / 'manifest.csv'}")


if __name__ == "__main__":
    main()
