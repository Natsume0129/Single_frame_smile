from __future__ import annotations

import csv
import json
import os
import re
import shutil
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Iterable


REQUIRED_BASE_COLUMNS = [
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

OCCLUSION_COLUMNS_WITH_DEFAULTS = {
    "occlusion_type": "none",
    "occlusion_start_frame": "",
    "occlusion_end_frame": "",
    "occlusion_severity": "none",
    "occlusion_note": "",
    "occlusion_segments": "[]",
}

CSV_COLUMNS = REQUIRED_BASE_COLUMNS + list(OCCLUSION_COLUMNS_WITH_DEFAULTS)

MAIN_LABELS = [
    "truesmile",
    "polite_smile",
    "bitter_smile",
    "smiling_but_ambiguous",
    "neutral",
    "discard",
]

PEAK_REQUIRED_LABELS = {
    "truesmile",
    "polite_smile",
    "bitter_smile",
    "smiling_but_ambiguous",
}

NO_PEAK_LABELS = {
    "neutral",
    "discard",
}

OLD_TO_NEW_LABEL = {
    "genuine_like_smile": "truesmile",
    "polite_like_smile": "polite_smile",
    "bitter_awkward_like_smile": "bitter_smile",
    "ambiguous_smile": "smiling_but_ambiguous",
    "neutral_or_no_smile": "neutral",
    "unclear": "discard",
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

OCCLUSION_TYPES = [
    "none",
    "mouth_partial",
    "mouth_severe",
    "lower_face_occluded",
    "hand_near_face_but_not_occluding",
]

OCCLUSION_SEVERITY_VALUES = [
    "none",
    "mild",
    "moderate",
    "severe",
]

SEVERITY_RANK = {
    "none": 0,
    "mild": 1,
    "moderate": 2,
    "severe": 3,
}

OCCLUSION_TYPE_RANK = {
    "none": 0,
    "hand_near_face_but_not_occluding": 1,
    "mouth_partial": 2,
    "mouth_severe": 3,
    "lower_face_occluded": 4,
}

_EPISODE_ID_RE = re.compile(r"^E(\d+)$")


@dataclass(frozen=True)
class OcclusionSegment:
    start: int
    end: int
    type: str
    severity: str
    note: str = ""


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
    occlusion_type: str = "none"
    occlusion_start_frame: int | None = None
    occlusion_end_frame: int | None = None
    occlusion_severity: str = "none"
    occlusion_note: str = ""
    occlusion_segments: list[OcclusionSegment] = field(default_factory=list)


def video_id_from_path(video_path: str) -> str:
    return Path(video_path).stem


def default_usable_for_training(
    confidence: int,
    visible_quality: str,
    main_label: str,
) -> str:
    normalized_label = main_label if main_label in MAIN_LABELS else OLD_TO_NEW_LABEL.get(main_label, "discard")
    return "no" if normalized_label == "discard" else "yes"


def label_requires_peak(main_label: str) -> bool:
    return main_label in PEAK_REQUIRED_LABELS


def migrate_main_label(old_label: str) -> str:
    if old_label in MAIN_LABELS:
        return old_label
    return OLD_TO_NEW_LABEL.get(old_label, "discard")


def validate_occlusion_segment(segment: OcclusionSegment) -> None:
    if type(segment.start) is not int or type(segment.end) is not int:
        raise ValueError("Occlusion segment start and end must be integers.")
    if segment.start < 0 or segment.end < 0:
        raise ValueError("Occlusion segment frames must be non-negative.")
    if segment.start > segment.end:
        raise ValueError("Occlusion segment frame order must satisfy start <= end.")
    if segment.type not in OCCLUSION_TYPES:
        raise ValueError("A valid occlusion segment type must be selected.")
    if segment.severity not in OCCLUSION_SEVERITY_VALUES:
        raise ValueError("A valid occlusion segment severity must be selected.")


def serialize_occlusion_segments(segments: list[OcclusionSegment]) -> str:
    for segment in segments:
        validate_occlusion_segment(segment)
    return json.dumps(
        [
            {
                "start": segment.start,
                "end": segment.end,
                "type": segment.type,
                "severity": segment.severity,
                "note": segment.note,
            }
            for segment in segments
        ],
        ensure_ascii=False,
    )


def parse_occlusion_segments(value: str | None) -> list[OcclusionSegment]:
    if value is None or value == "":
        return []
    try:
        raw_segments = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError("occlusion_segments must be valid JSON.") from exc
    if not isinstance(raw_segments, list):
        raise ValueError("occlusion_segments must be a JSON list.")

    segments: list[OcclusionSegment] = []
    for raw_segment in raw_segments:
        if not isinstance(raw_segment, dict):
            raise ValueError("Each occlusion segment must be a JSON object.")
        try:
            segment = OcclusionSegment(
                start=raw_segment["start"],
                end=raw_segment["end"],
                type=raw_segment["type"],
                severity=raw_segment["severity"],
                note=str(raw_segment.get("note", "")),
            )
        except KeyError as exc:
            raise ValueError("Each occlusion segment must contain start, end, type, and severity.") from exc
        validate_occlusion_segment(segment)
        segments.append(segment)
    return segments


def summarize_occlusion_segments(segments: list[OcclusionSegment]) -> dict[str, str]:
    for segment in segments:
        validate_occlusion_segment(segment)
    if not segments:
        return {
            "occlusion_type": "none",
            "occlusion_start_frame": "",
            "occlusion_end_frame": "",
            "occlusion_severity": "none",
            "occlusion_note": "",
        }

    most_severe_type = max(
        segments,
        key=lambda segment: OCCLUSION_TYPE_RANK.get(segment.type, -1),
    ).type
    highest_severity = max(
        segments,
        key=lambda segment: SEVERITY_RANK.get(segment.severity, -1),
    ).severity
    return {
        "occlusion_type": most_severe_type,
        "occlusion_start_frame": str(min(segment.start for segment in segments)),
        "occlusion_end_frame": str(max(segment.end for segment in segments)),
        "occlusion_severity": highest_severity,
        "occlusion_note": segments[0].note if len(segments) == 1 else "multiple occlusion segments",
    }


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


def ensure_csv_schema_columns(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    migrated_rows: list[dict[str, str]] = []
    for row in rows:
        migrated_row = dict(row)
        for column, default_value in OCCLUSION_COLUMNS_WITH_DEFAULTS.items():
            if column == "occlusion_segments":
                continue
            if migrated_row.get(column) is None:
                migrated_row[column] = default_value
            else:
                migrated_row.setdefault(column, default_value)
        if migrated_row.get("occlusion_segments") is None:
            segments = _occlusion_segments_from_summary_row(migrated_row)
            migrated_row["occlusion_segments"] = serialize_occlusion_segments(segments)
        else:
            migrated_row.setdefault("occlusion_segments", "[]")
            segments = parse_occlusion_segments(migrated_row["occlusion_segments"])
        if _row_has_occlusion(migrated_row, segments):
            migrated_row["main_label"] = "discard"
        else:
            migrated_row["main_label"] = migrate_main_label(migrated_row.get("main_label", ""))
        if not label_requires_peak(migrated_row["main_label"]):
            migrated_row["peak_frame"] = ""
            migrated_row["peak_time"] = ""
        migrated_rows.append(migrated_row)
    return migrated_rows


class AnnotationStore:
    def __init__(self, csv_path: str | Path, backup_dir: str | Path | None = None) -> None:
        self.csv_path = Path(csv_path)
        self.backup_dir = Path(backup_dir) if backup_dir is not None else _default_backup_dir(self.csv_path)
        self.last_read_was_old_schema = False
        self.last_read_fieldnames: list[str] = []

    def read_rows(self) -> list[dict[str, str]]:
        if not self.csv_path.exists() or self.csv_path.stat().st_size == 0:
            self.last_read_was_old_schema = False
            self.last_read_fieldnames = []
            return []

        with self.csv_path.open("r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            fieldnames = list(reader.fieldnames or [])
            self._validate_header(fieldnames)
            self.last_read_fieldnames = fieldnames
            self.last_read_was_old_schema = self._has_old_schema(fieldnames)
            return ensure_csv_schema_columns([dict(row) for row in reader])

    def is_old_schema(self) -> bool:
        fieldnames = self._read_header()
        if not fieldnames:
            return False
        return self._has_old_schema(fieldnames)

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
            self._ensure_current_schema_before_append()
            fieldnames = self._read_header()
        else:
            fieldnames = CSV_COLUMNS

        with self.csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow({column: row.get(column, "") for column in fieldnames})

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

        replacement_row = self._row_from_draft(episode_id, draft)
        updated_rows = []
        for row in rows:
            if row.get("episode_id") == episode_id:
                updated_row = dict(row)
                updated_row.update(replacement_row)
                updated_rows.append(updated_row)
            else:
                updated_rows.append(row)
        self._write_rows(updated_rows, source_fieldnames=self.last_read_fieldnames)
        updated_row = next(row for row in updated_rows if row.get("episode_id") == episode_id)
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

        self._write_rows(kept_rows, source_fieldnames=self.last_read_fieldnames)
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
        occlusion_summary = summarize_occlusion_segments([])
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
            **occlusion_summary,
            "occlusion_segments": serialize_occlusion_segments([]),
        }

    def _validate_existing_file(self) -> None:
        self._read_header()

    def _write_rows(
        self,
        rows: list[dict[str, str]],
        source_fieldnames: Iterable[str] | None = None,
    ) -> None:
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._backup_csv_before_rewrite()
        fieldnames = self._output_fieldnames(rows, source_fieldnames)
        with self.csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({column: row.get(column, "") for column in fieldnames})

    def _validate_header(self, fieldnames: Iterable[str] | None) -> None:
        columns = list(fieldnames or [])
        missing_base_columns = [
            column for column in REQUIRED_BASE_COLUMNS if column not in columns
        ]
        if missing_base_columns:
            raise ValueError(
                f"{self.csv_path} is missing required annotation CSV columns: "
                f"{', '.join(missing_base_columns)}"
            )

    def _read_header(self) -> list[str]:
        if not self.csv_path.exists() or self.csv_path.stat().st_size == 0:
            return []
        with self.csv_path.open("r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            fieldnames = list(reader.fieldnames or [])
        self._validate_header(fieldnames)
        return fieldnames

    def _has_old_schema(self, fieldnames: Iterable[str]) -> bool:
        columns = set(fieldnames)
        return any(column not in columns for column in OCCLUSION_COLUMNS_WITH_DEFAULTS)

    def _ensure_current_schema_before_append(self) -> None:
        fieldnames = self._read_header()
        if not fieldnames or not self._has_old_schema(fieldnames):
            return
        rows = self.read_rows()
        self._write_rows(rows, source_fieldnames=fieldnames)

    def _output_fieldnames(
        self,
        rows: list[dict[str, str]],
        source_fieldnames: Iterable[str] | None = None,
    ) -> list[str]:
        fieldnames = list(CSV_COLUMNS)
        for column in list(source_fieldnames or []):
            if column is not None and column not in fieldnames:
                fieldnames.append(column)
        for row in rows:
            for column in row:
                if column is not None and column not in fieldnames:
                    fieldnames.append(column)
        return fieldnames

    def _backup_csv_before_rewrite(self) -> Path | None:
        if not self.csv_path.exists() or self.csv_path.stat().st_size == 0:
            return None
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        backup_path = self.backup_dir / (
            f"{self.csv_path.stem}.backup.{timestamp}{self.csv_path.suffix}"
        )
        shutil.copy2(self.csv_path, backup_path)
        return backup_path


def _default_backup_dir(csv_path: Path) -> Path:
    if csv_path.parent.name == "dataset-annotation":
        return csv_path.parent.parent / "backups"
    return csv_path.parent / "backups"


def _format_seconds(frame_index: int, fps: float) -> str:
    return f"{frame_index / fps:.3f}"


def _optional_int(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _row_has_occlusion(row: dict[str, str], segments: list[OcclusionSegment]) -> bool:
    occlusion_type = (row.get("occlusion_type") or "").strip()
    return bool(segments) or occlusion_type not in {"", "none"}


def _effective_occlusion_segments(draft: EpisodeDraft) -> list[OcclusionSegment]:
    if draft.occlusion_segments:
        return list(draft.occlusion_segments)
    return _occlusion_segments_from_summary_values(
        draft.occlusion_type,
        draft.occlusion_start_frame,
        draft.occlusion_end_frame,
        draft.occlusion_severity,
        draft.occlusion_note,
    )


def _occlusion_segments_from_summary_row(row: dict[str, str]) -> list[OcclusionSegment]:
    return _occlusion_segments_from_summary_values(
        row.get("occlusion_type", "none") or "none",
        _optional_int(row.get("occlusion_start_frame", "")),
        _optional_int(row.get("occlusion_end_frame", "")),
        row.get("occlusion_severity", "none") or "none",
        row.get("occlusion_note", ""),
    )


def _occlusion_segments_from_summary_values(
    occlusion_type: str,
    occlusion_start_frame: int | None,
    occlusion_end_frame: int | None,
    occlusion_severity: str,
    occlusion_note: str,
) -> list[OcclusionSegment]:
    if occlusion_type not in OCCLUSION_TYPES:
        raise ValueError("A valid occlusion_type must be selected.")
    if occlusion_severity not in OCCLUSION_SEVERITY_VALUES:
        raise ValueError("A valid occlusion_severity value must be selected.")
    if occlusion_type == "none":
        if occlusion_start_frame is not None or occlusion_end_frame is not None:
            raise ValueError("Occlusion frames must be empty when occlusion_type is none.")
        if occlusion_severity != "none":
            raise ValueError("occlusion_severity must be none when occlusion_type is none.")
        return []
    if occlusion_start_frame is None or occlusion_end_frame is None:
        return []
    return [
        OcclusionSegment(
            start=occlusion_start_frame,
            end=occlusion_end_frame,
            type=occlusion_type,
            severity=occlusion_severity,
            note=occlusion_note,
        )
    ]


def _normalize_draft(draft: EpisodeDraft) -> EpisodeDraft:
    main_label = migrate_main_label(draft.main_label)
    peak_frame = draft.peak_frame if label_requires_peak(main_label) else None
    if main_label == draft.main_label and peak_frame == draft.peak_frame:
        return draft
    return replace(draft, main_label=main_label, peak_frame=peak_frame)
