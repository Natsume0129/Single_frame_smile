from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))

from annotation_store import AnnotationStore, CSV_COLUMNS, EpisodeDraft  # noqa: E402


class AnnotationStoreTest(unittest.TestCase):
    def _draft(
        self,
        directory: Path,
        *,
        start_frame: int = 10,
        peak_frame: int | None = 15,
        end_frame: int = 20,
        main_label: str = "genuine_like_smile",
        video_name: str = "clip_001.mp4",
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
            note="clear smile",
        )

    def test_append_episode_creates_expected_csv_row(self) -> None:
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

            with csv_path.open("r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                self.assertEqual(CSV_COLUMNS, reader.fieldnames)
                rows = list(reader)

            self.assertEqual(1, len(rows))
            self.assertEqual("genuine_like_smile", rows[0]["main_label"])

    def test_episode_id_continues_from_existing_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            store = AnnotationStore(tmp_path / "annotations.csv")

            store.append_episode(self._draft(tmp_path, start_frame=10, peak_frame=15, end_frame=20))
            second = store.append_episode(
                self._draft(tmp_path, start_frame=30, peak_frame=35, end_frame=40)
            )

            self.assertEqual("E000002", second["episode_id"])
            self.assertEqual("E000003", store.next_episode_id())

    def test_duplicate_episode_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            store = AnnotationStore(tmp_path / "annotations.csv")
            draft = self._draft(tmp_path)

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

    def test_smile_label_requires_peak_frame(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            store = AnnotationStore(tmp_path / "annotations.csv")

            with self.assertRaisesRegex(ValueError, "peak_frame is required"):
                store.append_episode(self._draft(tmp_path, peak_frame=None))

    def test_neutral_label_allows_missing_peak_frame(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            store = AnnotationStore(tmp_path / "annotations.csv")

            row = store.append_episode(
                self._draft(
                    tmp_path,
                    peak_frame=None,
                    main_label="neutral_or_no_smile",
                )
            )

            self.assertEqual("", row["peak_frame"])
            self.assertEqual("", row["peak_time"])

    def test_neutral_label_ignores_peak_frame_when_set(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            store = AnnotationStore(tmp_path / "annotations.csv")

            row = store.append_episode(
                self._draft(
                    tmp_path,
                    start_frame=10,
                    peak_frame=30,
                    end_frame=20,
                    main_label="neutral_or_no_smile",
                )
            )

            self.assertEqual("", row["peak_frame"])
            self.assertEqual("", row["peak_time"])

    def test_delete_episode_removes_only_matching_episode_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            store = AnnotationStore(tmp_path / "annotations.csv")

            first = store.append_episode(
                self._draft(tmp_path, start_frame=10, peak_frame=15, end_frame=20)
            )
            second = store.append_episode(
                self._draft(tmp_path, start_frame=30, peak_frame=35, end_frame=40)
            )

            deleted = store.delete_episode(first["episode_id"])
            rows = store.read_rows()

            self.assertEqual(first["episode_id"], deleted["episode_id"])
            self.assertEqual(1, len(rows))
            self.assertEqual(second["episode_id"], rows[0]["episode_id"])

    def test_delete_missing_episode_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            store = AnnotationStore(tmp_path / "annotations.csv")

            store.append_episode(self._draft(tmp_path))

            self.assertIsNone(store.delete_episode("E999999"))

    def test_update_episode_overwrites_existing_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            store = AnnotationStore(tmp_path / "annotations.csv")

            original = store.append_episode(
                self._draft(tmp_path, start_frame=10, peak_frame=15, end_frame=20)
            )
            updated = store.update_episode(
                original["episode_id"],
                self._draft(
                    tmp_path,
                    start_frame=40,
                    peak_frame=45,
                    end_frame=50,
                    main_label="polite_like_smile",
                ),
            )
            rows = store.read_rows()

            self.assertEqual(original["episode_id"], updated["episode_id"])
            self.assertEqual(1, len(rows))
            self.assertEqual("40", rows[0]["start_frame"])
            self.assertEqual("polite_like_smile", rows[0]["main_label"])
            self.assertEqual("E000002", store.next_episode_id())

    def test_update_episode_rejects_duplicate_range_from_other_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            store = AnnotationStore(tmp_path / "annotations.csv")

            first = store.append_episode(
                self._draft(tmp_path, start_frame=10, peak_frame=15, end_frame=20)
            )
            store.append_episode(
                self._draft(tmp_path, start_frame=30, peak_frame=35, end_frame=40)
            )

            with self.assertRaisesRegex(ValueError, "Another episode"):
                store.update_episode(
                    first["episode_id"],
                    self._draft(tmp_path, start_frame=30, peak_frame=35, end_frame=40),
                )

    def test_update_missing_episode_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            store = AnnotationStore(tmp_path / "annotations.csv")

            store.append_episode(self._draft(tmp_path))

            self.assertIsNone(store.update_episode("E999999", self._draft(tmp_path)))


if __name__ == "__main__":
    unittest.main()
