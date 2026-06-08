from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))

from annotation_store import (  # noqa: E402
    AnnotationStore,
    CSV_COLUMNS,
    EpisodeDraft,
    MAIN_LABELS,
    OcclusionSegment,
    REQUIRED_BASE_COLUMNS,
    default_usable_for_training,
    label_requires_peak,
    migrate_main_label,
    parse_occlusion_segments,
)


SMILE_LABELS = [
    "truesmile",
    "polite_smile",
    "bitter_smile",
    "smiling_but_ambiguous",
]


class AnnotationStoreTest(unittest.TestCase):
    def _draft(
        self,
        directory: Path,
        *,
        start_frame: int = 10,
        peak_frame: int | None = 15,
        end_frame: int = 20,
        main_label: str = "truesmile",
        video_name: str = "clip_001.mp4",
        occlusion_type: str = "none",
        occlusion_start_frame: int | None = None,
        occlusion_end_frame: int | None = None,
        occlusion_severity: str = "none",
        occlusion_note: str = "",
        occlusion_segments: list[OcclusionSegment] | None = None,
    ) -> EpisodeDraft:
        video_path = directory / video_name
        video_path.write_bytes(b"")
        return EpisodeDraft(
            video_path=str(video_path),
            person_id="P01",
            start_frame=start_frame,
            peak_frame=peak_frame,
            end_frame=end_frame,
            fps=30.0,
            main_label=main_label,
            confidence=5,
            intensity=4,
            eye_involvement=4,
            mouth_movement=4,
            cheek_raise=4,
            symmetry="symmetric",
            visible_quality="good",
            usable_for_training="yes",
            note="clear segment",
            occlusion_type=occlusion_type,
            occlusion_start_frame=occlusion_start_frame,
            occlusion_end_frame=occlusion_end_frame,
            occlusion_severity=occlusion_severity,
            occlusion_note=occlusion_note,
            occlusion_segments=[] if occlusion_segments is None else occlusion_segments,
        )

    def _base_row(
        self,
        directory: Path,
        *,
        episode_id: str = "E000123",
        main_label: str = "genuine_like_smile",
        peak_frame: str = "15",
        occlusion_type: str = "none",
        occlusion_segments: str | None = None,
    ) -> dict[str, str]:
        draft = self._draft(directory)
        row = {
            "episode_id": episode_id,
            "video_id": "clip_001",
            "clip_path": str(Path(draft.video_path).resolve()),
            "person_id": "P01",
            "start_frame": "10",
            "peak_frame": peak_frame,
            "end_frame": "20",
            "start_time": "0.333",
            "peak_time": "0.500" if peak_frame else "",
            "end_time": "0.667",
            "main_label": main_label,
            "confidence": "5",
            "intensity": "4",
            "eye_involvement": "4",
            "mouth_movement": "4",
            "cheek_raise": "4",
            "symmetry": "symmetric",
            "visible_quality": "good",
            "usable_for_training": "yes",
            "note": "old row",
            "occlusion_type": occlusion_type,
            "occlusion_start_frame": "12" if occlusion_type != "none" else "",
            "occlusion_end_frame": "18" if occlusion_type != "none" else "",
            "occlusion_severity": "severe" if occlusion_type != "none" else "none",
            "occlusion_note": "legacy occlusion" if occlusion_type != "none" else "",
        }
        if occlusion_segments is not None:
            row["occlusion_segments"] = occlusion_segments
        return row

    def _write_rows(self, csv_path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    def test_append_segment_creates_expected_csv_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            csv_path = tmp_path / "annotations.csv"
            store = AnnotationStore(csv_path)

            row = store.append_episode(self._draft(tmp_path))

            self.assertEqual("E000001", row["episode_id"])
            self.assertEqual("clip_001", row["video_id"])
            self.assertEqual("0.333", row["start_time"])
            self.assertEqual("0.500", row["peak_time"])
            self.assertEqual("0.667", row["end_time"])
            self.assertEqual("truesmile", row["main_label"])

            with csv_path.open("r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            self.assertEqual(CSV_COLUMNS, reader.fieldnames)
            self.assertEqual(1, len(rows))
            self.assertEqual("truesmile", rows[0]["main_label"])

    def test_new_label_set_and_peak_requirements(self) -> None:
        self.assertEqual(
            [
                "truesmile",
                "polite_smile",
                "bitter_smile",
                "smiling_but_ambiguous",
                "neutral",
                "discard",
            ],
            MAIN_LABELS,
        )
        for label in SMILE_LABELS:
            self.assertTrue(label_requires_peak(label))
        for label in ["neutral", "discard"]:
            self.assertFalse(label_requires_peak(label))

    def test_default_usable_depends_only_on_discard_label(self) -> None:
        self.assertEqual("no", default_usable_for_training(5, "good", "discard"))
        self.assertEqual("yes", default_usable_for_training(1, "poor", "neutral"))
        for label in SMILE_LABELS:
            with self.subTest(label=label):
                self.assertEqual("yes", default_usable_for_training(1, "poor", label))

    def test_smile_labels_require_peak_frame(self) -> None:
        for label in SMILE_LABELS:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                store = AnnotationStore(tmp_path / "annotations.csv")

                with self.assertRaisesRegex(ValueError, "peak_frame is required"):
                    store.append_episode(self._draft(tmp_path, peak_frame=None, main_label=label))

    def test_neutral_and_discard_allow_missing_peak_frame(self) -> None:
        for index, label in enumerate(["neutral", "discard"]):
            with self.subTest(label=label), tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                store = AnnotationStore(tmp_path / "annotations.csv")

                row = store.append_episode(
                    self._draft(
                        tmp_path,
                        start_frame=10 + index * 20,
                        peak_frame=None,
                        end_frame=20 + index * 20,
                        main_label=label,
                    )
                )

                self.assertEqual(label, row["main_label"])
                self.assertEqual("", row["peak_frame"])
                self.assertEqual("", row["peak_time"])

    def test_neutral_and_discard_clear_existing_peak_frame(self) -> None:
        for index, label in enumerate(["neutral", "discard"]):
            with self.subTest(label=label), tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                store = AnnotationStore(tmp_path / "annotations.csv")

                row = store.append_episode(
                    self._draft(
                        tmp_path,
                        start_frame=10 + index * 20,
                        peak_frame=15 + index * 20,
                        end_frame=20 + index * 20,
                        main_label=label,
                    )
                )

                self.assertEqual("", row["peak_frame"])
                self.assertEqual("", row["peak_time"])

    def test_old_labels_migrate_to_new_labels_before_saving(self) -> None:
        cases = {
            "genuine_like_smile": "truesmile",
            "polite_like_smile": "polite_smile",
            "bitter_awkward_like_smile": "bitter_smile",
            "ambiguous_smile": "smiling_but_ambiguous",
            "neutral_or_no_smile": "neutral",
            "unclear": "discard",
            "unknown_old_label": "discard",
        }
        for old_label, new_label in cases.items():
            with self.subTest(old_label=old_label), tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                store = AnnotationStore(tmp_path / "annotations.csv")
                peak_frame = 15 if label_requires_peak(new_label) else None

                row = store.append_episode(
                    self._draft(tmp_path, main_label=old_label, peak_frame=peak_frame)
                )

                self.assertEqual(new_label, row["main_label"])
                self.assertEqual(new_label, migrate_main_label(old_label))

    def test_read_rows_migrates_old_labels_without_rewriting_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            csv_path = tmp_path / "annotations.csv"
            self._write_rows(
                csv_path,
                REQUIRED_BASE_COLUMNS,
                [self._base_row(tmp_path, main_label="polite_like_smile")],
            )
            original_text = csv_path.read_text(encoding="utf-8")

            rows = AnnotationStore(csv_path).read_rows()

            self.assertEqual("polite_smile", rows[0]["main_label"])
            self.assertEqual(original_text, csv_path.read_text(encoding="utf-8"))

    def test_read_rows_migrates_legacy_occlusion_to_discard_in_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            csv_path = tmp_path / "annotations.csv"
            self._write_rows(
                csv_path,
                CSV_COLUMNS[:-1],
                [self._base_row(tmp_path, occlusion_type="mouth_severe")],
            )
            original_text = csv_path.read_text(encoding="utf-8")

            rows = AnnotationStore(csv_path).read_rows()

            self.assertEqual("discard", rows[0]["main_label"])
            self.assertEqual("", rows[0]["peak_frame"])
            self.assertEqual("", rows[0]["peak_time"])
            self.assertEqual(original_text, csv_path.read_text(encoding="utf-8"))

    def test_read_rows_migrates_existing_occlusion_segments_to_discard_in_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            csv_path = tmp_path / "annotations.csv"
            segments = json.dumps(
                [
                    {
                        "start": 12,
                        "end": 18,
                        "type": "mouth_severe",
                        "severity": "severe",
                        "note": "hand covers mouth",
                    }
                ]
            )
            self._write_rows(
                csv_path,
                CSV_COLUMNS,
                [self._base_row(tmp_path, occlusion_segments=segments)],
            )
            original_text = csv_path.read_text(encoding="utf-8")

            rows = AnnotationStore(csv_path).read_rows()

            self.assertEqual("discard", rows[0]["main_label"])
            self.assertEqual("", rows[0]["peak_frame"])
            self.assertEqual(1, len(parse_occlusion_segments(rows[0]["occlusion_segments"])))
            self.assertEqual(original_text, csv_path.read_text(encoding="utf-8"))

    def test_new_saves_ignore_occlusion_draft_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            store = AnnotationStore(tmp_path / "annotations.csv")

            row = store.append_episode(
                self._draft(
                    tmp_path,
                    occlusion_type="mouth_partial",
                    occlusion_start_frame=12,
                    occlusion_end_frame=18,
                    occlusion_severity="moderate",
                    occlusion_note="legacy draft should be ignored",
                    occlusion_segments=[
                        OcclusionSegment(12, 18, "mouth_partial", "moderate", "ignored")
                    ],
                )
            )

            self.assertEqual("none", row["occlusion_type"])
            self.assertEqual("", row["occlusion_start_frame"])
            self.assertEqual("", row["occlusion_end_frame"])
            self.assertEqual("none", row["occlusion_severity"])
            self.assertEqual("", row["occlusion_note"])
            self.assertEqual([], parse_occlusion_segments(row["occlusion_segments"]))

    def test_update_episode_preserves_id_and_updates_only_matching_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            store = AnnotationStore(tmp_path / "annotations.csv")
            original = store.append_episode(
                self._draft(tmp_path, start_frame=10, peak_frame=15, end_frame=20)
            )
            store.append_episode(
                self._draft(tmp_path, start_frame=30, peak_frame=35, end_frame=40)
            )

            updated = store.update_episode(
                original["episode_id"],
                self._draft(
                    tmp_path,
                    start_frame=10,
                    peak_frame=None,
                    end_frame=20,
                    main_label="neutral",
                ),
            )
            rows = store.read_rows()

            self.assertEqual(original["episode_id"], updated["episode_id"])
            self.assertEqual(2, len(rows))
            self.assertEqual("neutral", rows[0]["main_label"])
            self.assertEqual("10", rows[0]["start_frame"])
            self.assertEqual("20", rows[0]["end_frame"])
            self.assertEqual("", rows[0]["peak_frame"])
            self.assertEqual("E000003", store.next_episode_id())

    def test_duplicate_segment_range_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            store = AnnotationStore(tmp_path / "annotations.csv")
            draft = self._draft(tmp_path, main_label="neutral", peak_frame=None)

            store.append_episode(draft)

            with self.assertRaisesRegex(ValueError, "same video path"):
                store.append_episode(draft)

    def test_invalid_frame_order_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            store = AnnotationStore(tmp_path / "annotations.csv")

            with self.assertRaisesRegex(ValueError, "start_frame < peak_frame < end_frame"):
                store.append_episode(
                    self._draft(tmp_path, start_frame=20, peak_frame=15, end_frame=10)
                )

    def test_append_to_old_schema_creates_backup_and_migrates_header(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            csv_path = tmp_path / "annotations.csv"
            self._write_rows(
                csv_path,
                REQUIRED_BASE_COLUMNS,
                [self._base_row(tmp_path, main_label="genuine_like_smile")],
            )
            before_append = csv_path.read_text(encoding="utf-8")
            store = AnnotationStore(csv_path)

            row = store.append_episode(
                self._draft(tmp_path, start_frame=30, peak_frame=None, end_frame=40, main_label="neutral")
            )
            backups = sorted((tmp_path / "backups").glob("annotations.backup.*.csv"))

            with csv_path.open("r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            self.assertEqual("E000124", row["episode_id"])
            self.assertEqual(1, len(backups))
            self.assertEqual(before_append, backups[0].read_text(encoding="utf-8"))
            self.assertEqual(CSV_COLUMNS, reader.fieldnames)
            self.assertEqual("truesmile", rows[0]["main_label"])
            self.assertEqual("neutral", rows[1]["main_label"])

    def test_update_preserves_unknown_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            csv_path = tmp_path / "annotations.csv"
            row = self._base_row(tmp_path, episode_id="E000001", main_label="truesmile")
            row["reviewer"] = "annotator_a"
            self._write_rows(csv_path, CSV_COLUMNS + ["reviewer"], [row])
            store = AnnotationStore(csv_path)

            store.update_episode(
                "E000001",
                self._draft(tmp_path, start_frame=30, peak_frame=None, end_frame=40, main_label="discard"),
            )

            with csv_path.open("r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            self.assertEqual(CSV_COLUMNS + ["reviewer"], reader.fieldnames)
            self.assertEqual("annotator_a", rows[0]["reviewer"])
            self.assertEqual("discard", rows[0]["main_label"])
            self.assertEqual("", rows[0]["peak_frame"])

    def test_update_and_delete_create_backups_before_rewrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            csv_path = tmp_path / "annotations.csv"
            store = AnnotationStore(csv_path)
            first = store.append_episode(
                self._draft(tmp_path, start_frame=10, peak_frame=15, end_frame=20)
            )
            second = store.append_episode(
                self._draft(tmp_path, start_frame=30, peak_frame=35, end_frame=40)
            )
            before_update = csv_path.read_text(encoding="utf-8")

            store.update_episode(
                first["episode_id"],
                self._draft(tmp_path, start_frame=50, peak_frame=None, end_frame=60, main_label="neutral"),
            )
            backups_after_update = sorted((tmp_path / "backups").glob("annotations.backup.*.csv"))

            self.assertEqual(1, len(backups_after_update))
            self.assertEqual(before_update, backups_after_update[0].read_text(encoding="utf-8"))

            before_delete = csv_path.read_text(encoding="utf-8")
            store.delete_episode(second["episode_id"])
            backups_after_delete = sorted((tmp_path / "backups").glob("annotations.backup.*.csv"))

            self.assertEqual(2, len(backups_after_delete))
            self.assertEqual(before_delete, backups_after_delete[-1].read_text(encoding="utf-8"))
            self.assertEqual(1, len(store.read_rows()))


if __name__ == "__main__":
    unittest.main()
