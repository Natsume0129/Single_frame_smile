from __future__ import annotations

import csv
import os
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable


CSV_COLUMNS = [
    "episode_id",
    "video_id",
    "clip_path",
    "person_id",
    "start_frame",
    "peak_frame",
    "end_frame",
    "start_time",
    "peak_time",
    "end_time",
    "main_label",
    "confidence",
    "intensity",
    "eye_involvement",
    "mouth_movement",
    "cheek_raise",
    "symmetry",
    "visible_quality",
    "usable_for_training",
    "note",
]

MAIN_LABELS = [
    "genuine_like_smile",
    "polite_like_smile",
    "bitter_awkward_like_smile",
    "ambiguous_smile",
    "neutral_or_no_smile",
    "unclear",
]

PEAK_REQUIRED_LABELS = {
    "genuine_like_smile",
    "polite_like_smile",
    "bitter_awkward_like_smile",
    "ambiguous_smile",
}

SYMMETRY_VALUES = [
    "symmetric",
    "slightly_asymmetric",
    "asymmetric",
    "unknown",
]

VISIBLE_QUALITY_VALUES = [
    "good",
    "medium",
    "poor",
]

USABLE_VALUES = [
    "yes",
    "no",
]

_EPISODE_ID_RE = re.compile(r"^E(\d+)$")


@dataclass(frozen=True)
class EpisodeDraft:
    video_path: str
    person_id: str
    start_frame: int
    peak_frame: int | None
    end_frame: int
    fps: float
    main_label: str
    confidence: int
    intensity: int
    eye_involvement: int
    mouth_movement: int
    cheek_raise: int
    symmetry: str
    visible_quality: str
    usable_for_training: str
    note: str = ""


def video_id_from_path(video_path: str) -> str:
    return Path(video_path).stem


def default_usable_for_training(
    confidence: int,
    visible_quality: str,
    main_label: str,
) -> str:
    if confidence >= 4 and visible_quality in {"good", "medium"} and main_label != "unclear":
        return "yes"
    return "no"


def label_requires_peak(main_label: str) -> bool:
    return main_label in PEAK_REQUIRED_LABELS


def validate_frame_order(
    start_frame: int,
    peak_frame: int | None,
    end_frame: int,
    *,
    peak_required: bool = True,
) -> list[str]:
    errors: list[str] = []
    if start_frame < 0 or end_frame < 0 or (peak_frame is not None and peak_frame < 0):
        errors.append("Frame indices must be non-negative.")
    if not (start_frame < end_frame):
        errors.append("Frame order must satisfy start_frame < end_frame.")
    if peak_frame is None:
        if peak_required:
            errors.append("peak_frame is required for smile labels.")
        return errors
    if not (start_frame < peak_frame < end_frame):
        errors.append(
            "Frame order must satisfy start_frame < peak_frame < end_frame when peak_frame is set."
        )
    return errors


def validate_episode_draft(draft: EpisodeDraft) -> list[str]:
    effective_peak_frame = draft.peak_frame if label_requires_peak(draft.main_label) else None
    errors = validate_frame_order(
        draft.start_frame,
        effective_peak_frame,
        draft.end_frame,
        peak_required=label_requires_peak(draft.main_label),
    )

    if not draft.video_path:
        errors.append("A video must be loaded before saving an episode.")
    if draft.fps <= 0:
        errors.append("Video FPS must be greater than 0.")
    if draft.main_label not in MAIN_LABELS:
        errors.append("A valid main_label must be selected.")
    for field_name in [
        "confidence",
        "intensity",
        "eye_involvement",
        "mouth_movement",
        "cheek_raise",
    ]:
        value = getattr(draft, field_name)
        if value not in range(1, 6):
            errors.append(f"{field_name} must be an integer from 1 to 5.")
    if draft.symmetry not in SYMMETRY_VALUES:
        errors.append("A valid symmetry value must be selected.")
    if draft.visible_quality not in VISIBLE_QUALITY_VALUES:
        errors.append("A valid visible_quality value must be selected.")
    if draft.usable_for_training not in USABLE_VALUES:
        errors.append("usable_for_training must be yes or no.")

    return errors


def _normal_path(path_value: str) -> str:
    if not path_value:
        return ""
    try:
        return os.path.normcase(str(Path(path_value).expanduser().resolve()))
    except OSError:
        return os.path.normcase(os.path.abspath(os.path.expanduser(path_value)))


def _parse_episode_number(episode_id: str) -> int | None:
    match = _EPISODE_ID_RE.match(episode_id.strip())
    if not match:
        return None
    return int(match.group(1))


class AnnotationStore:
    def __init__(self, csv_path: str | Path) -> None:
        self.csv_path = Path(csv_path)

    def read_rows(self) -> list[dict[str, str]]:
        if not self.csv_path.exists() or self.csv_path.stat().st_size == 0:
            return []

        with self.csv_path.open("r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            self._validate_header(reader.fieldnames)
            return [dict(row) for row in reader]

    def next_episode_id(self) -> str:
        max_number = 0
        for row in self.read_rows():
            number = _parse_episode_number(row.get("episode_id", ""))
            if number is not None:
                max_number = max(max_number, number)
        return f"E{max_number + 1:06d}"

    def episodes_for_video(self, video_path: str) -> list[dict[str, str]]:
        normal_video_path = _normal_path(video_path)
        video_id = video_id_from_path(video_path)
        rows = []
        for row in self.read_rows():
            row_path = row.get("clip_path", "")
            if _normal_path(row_path) == normal_video_path or row.get("video_id") == video_id:
                rows.append(row)
        rows.sort(key=lambda r: int(r.get("start_frame") or 0))
        return rows

    def append_episode(self, draft: EpisodeDraft) -> dict[str, str]:
        draft = _normalize_draft(draft)
        errors = validate_episode_draft(draft)
        if errors:
            raise ValueError("\n".join(errors))
        if self.has_duplicate(draft):
            raise ValueError(
                "An episode with the same video path and start/peak/end frames already exists."
            )

        episode_id = self.next_episode_id()
        row = self._row_from_draft(episode_id, draft)

        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        file_exists = self.csv_path.exists() and self.csv_path.stat().st_size > 0
        if file_exists:
            self._validate_existing_file()

        with self.csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

        return row

    def update_episode(self, episode_id: str, draft: EpisodeDraft) -> dict[str, str] | None:
        draft = _normalize_draft(draft)
        errors = validate_episode_draft(draft)
        if errors:
            raise ValueError("\n".join(errors))

        rows = self.read_rows()
        if not any(row.get("episode_id") == episode_id for row in rows):
            return None
        if self.has_duplicate(draft, ignore_episode_id=episode_id):
            raise ValueError(
                "Another episode with the same video path and start/peak/end frames already exists."
            )

        updated_row = self._row_from_draft(episode_id, draft)
        updated_rows = [
            updated_row if row.get("episode_id") == episode_id else row
            for row in rows
        ]
        self._write_rows(updated_rows)
        return updated_row

    def delete_episode(self, episode_id: str) -> dict[str, str] | None:
        rows = self.read_rows()
        kept_rows: list[dict[str, str]] = []
        deleted_row: dict[str, str] | None = None

        for row in rows:
            if row.get("episode_id") == episode_id and deleted_row is None:
                deleted_row = row
                continue
            kept_rows.append(row)

        if deleted_row is None:
            return None

        self._write_rows(kept_rows)
        return deleted_row

    def has_duplicate(self, draft: EpisodeDraft, ignore_episode_id: str | None = None) -> bool:
        draft = _normalize_draft(draft)
        normal_video_path = _normal_path(draft.video_path)
        for row in self.read_rows():
            if ignore_episode_id is not None and row.get("episode_id") == ignore_episode_id:
                continue
            if _normal_path(row.get("clip_path", "")) != normal_video_path:
                continue
            try:
                row_peak = _optional_int(row.get("peak_frame", ""))
                same_frames = (
                    int(row.get("start_frame", -1)) == draft.start_frame
                    and row_peak == draft.peak_frame
                    and int(row.get("end_frame", -1)) == draft.end_frame
                )
            except ValueError:
                same_frames = False
            if same_frames:
                return True
        return False

    def _row_from_draft(self, episode_id: str, draft: EpisodeDraft) -> dict[str, str]:
        return {
            "episode_id": episode_id,
            "video_id": video_id_from_path(draft.video_path),
            "clip_path": str(Path(draft.video_path).resolve()),
            "person_id": draft.person_id,
            "start_frame": str(draft.start_frame),
            "peak_frame": "" if draft.peak_frame is None else str(draft.peak_frame),
            "end_frame": str(draft.end_frame),
            "start_time": _format_seconds(draft.start_frame, draft.fps),
            "peak_time": "" if draft.peak_frame is None else _format_seconds(draft.peak_frame, draft.fps),
            "end_time": _format_seconds(draft.end_frame, draft.fps),
            "main_label": draft.main_label,
            "confidence": str(draft.confidence),
            "intensity": str(draft.intensity),
            "eye_involvement": str(draft.eye_involvement),
            "mouth_movement": str(draft.mouth_movement),
            "cheek_raise": str(draft.cheek_raise),
            "symmetry": draft.symmetry,
            "visible_quality": draft.visible_quality,
            "usable_for_training": draft.usable_for_training,
            "note": draft.note,
        }

    def _validate_existing_file(self) -> None:
        with self.csv_path.open("r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            self._validate_header(reader.fieldnames)

    def _write_rows(self, rows: list[dict[str, str]]) -> None:
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        with self.csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            for row in rows:
                writer.writerow({column: row.get(column, "") for column in CSV_COLUMNS})

    def _validate_header(self, fieldnames: Iterable[str] | None) -> None:
        if list(fieldnames or []) != CSV_COLUMNS:
            raise ValueError(
                f"{self.csv_path} does not use the expected annotation CSV columns."
            )


def _format_seconds(frame_index: int, fps: float) -> str:
    return f"{frame_index / fps:.3f}"


def _optional_int(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _normalize_draft(draft: EpisodeDraft) -> EpisodeDraft:
    if label_requires_peak(draft.main_label):
        return draft
    if draft.peak_frame is None:
        return draft
    return replace(draft, peak_frame=None)
