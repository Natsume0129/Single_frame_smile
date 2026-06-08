from __future__ import annotations

import argparse
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))

from split_videos_overlap import (  # noqa: E402
    SegmentPlan,
    build_segment_plan,
    existing_part_files,
    ffmpeg_command,
    part_filename,
    process_video,
    validate_args,
)


class SplitVideosOverlapTest(unittest.TestCase):
    def test_build_segment_plan_uses_fixed_overlap(self) -> None:
        plans = build_segment_plan(
            video_duration=1800.0,
            segment_time=300.0,
            overlap_time=10.0,
        )

        self.assertEqual(
            [
                (0, 0.0, 300.0),
                (1, 290.0, 590.0),
                (2, 580.0, 880.0),
                (3, 870.0, 1170.0),
                (4, 1160.0, 1460.0),
                (5, 1450.0, 1750.0),
                (6, 1740.0, 1800.0),
            ],
            [(plan.part_index, plan.start_time_sec, plan.end_time_sec) for plan in plans],
        )

    def test_build_segment_plan_rejects_invalid_overlap(self) -> None:
        with self.assertRaisesRegex(ValueError, "overlap_time"):
            build_segment_plan(
                video_duration=100.0,
                segment_time=10.0,
                overlap_time=10.0,
            )

    def test_part_filename_keeps_source_stem_and_times(self) -> None:
        source_path = Path("会議 video 01.MP4")
        plan = SegmentPlan(part_index=3, start_time_sec=870.0, end_time_sec=1170.0)

        self.assertEqual(
            "会議 video 01_part003_start000870s_end001170s.MP4",
            part_filename(source_path, plan),
        )

    def test_existing_part_files_only_matches_current_source_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            expected = tmp_path / "source_a_part000_start000000s_end000300s.mp4"
            expected.write_bytes(b"part")
            (tmp_path / "source_b_part000_start000000s_end000300s.mp4").write_bytes(b"part")
            (tmp_path / "source_a_notes.txt").write_text("not a part", encoding="utf-8")

            self.assertEqual([expected], existing_part_files(tmp_path, "source_a"))

    def test_ffmpeg_command_uses_copy_mode_and_list_arguments(self) -> None:
        command = ffmpeg_command(
            "ffmpeg",
            Path("E:/raw videos/入力.mp4"),
            Path("E:/split videos/入力_part000_start000000s_end000300s.mp4"),
            SegmentPlan(part_index=0, start_time_sec=0.0, end_time_sec=300.0),
        )

        self.assertIsInstance(command, list)
        self.assertIn("-c", command)
        self.assertIn("copy", command)
        self.assertIn("-dn", command)
        self.assertIn("-loglevel", command)
        self.assertIn("error", command)
        self.assertIn("-map", command)
        self.assertIn("0", command)
        self.assertIn("-avoid_negative_ts", command)
        self.assertIn("make_zero", command)

    def test_validate_args_uses_output_dir_without_requiring_it_to_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            input_dir = Path(tmp) / "raw"
            input_dir.mkdir()
            output_dir = Path(tmp) / "split"
            args = argparse.Namespace(
                input_dir=str(input_dir),
                output_dir=str(output_dir),
                segment_time=300.0,
                overlap_time=10.0,
                max_size_gb=2.0,
            )

            validated_input, validated_output = validate_args(args)

            self.assertEqual(input_dir, validated_input)
            self.assertEqual(output_dir, validated_output)

    def test_process_video_copies_source_under_max_size(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            output_dir = tmp_path / "split"
            output_dir.mkdir()
            source_path = tmp_path / "small.mp4"
            source_path.write_bytes(b"small video placeholder")
            old_part = output_dir / "small_part000_start000000s_end000300s.mp4"
            old_part.write_bytes(b"old")
            other_part = output_dir / "other_part000_start000000s_end000300s.mp4"
            other_part.write_bytes(b"other")
            args = argparse.Namespace(
                segment_time=300.0,
                overlap_time=10.0,
                max_size_gb=2.0,
                overwrite=True,
                dry_run=False,
            )

            rows = process_video(source_path, output_dir, args, "ffmpeg-not-called", "ffprobe-not-called")

            self.assertEqual(1, len(rows))
            self.assertEqual("COPIED_UNDER_SIZE", rows[0].status)
            self.assertEqual("small.mp4", rows[0].part_file)
            self.assertTrue((output_dir / "small.mp4").exists())
            self.assertEqual(source_path.read_bytes(), (output_dir / "small.mp4").read_bytes())
            self.assertFalse(old_part.exists())
            self.assertTrue(other_part.exists())


if __name__ == "__main__":
    unittest.main()
