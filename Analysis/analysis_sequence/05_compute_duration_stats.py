from __future__ import annotations

from collections import defaultdict

from common.base import PipelineConfig, SequenceTaskBase


class DurationStatsTask(SequenceTaskBase):
    def run(self) -> None:
        grouped: dict[str, list[float]] = defaultdict(list)
        detail_rows: list[dict] = []

        for seq in self.discover_sequences():
            feat = self.load_npy(self.metrics_seq_dir("sequence_features", seq) / "sequence_features.npy")
            frames = int(feat.shape[0])
            duration = frames / float(self.cfg.fps)
            grouped[seq.class_name].append(duration)
            detail_rows.append(
                {
                    "class": seq.class_name,
                    "sequence_id": seq.sequence_id,
                    "duration_frames": frames,
                    "duration_seconds": duration,
                }
            )

        avg_rows = []
        for class_name, durations in grouped.items():
            avg_rows.append(
                {
                    "class": class_name,
                    "num_sequences": len(durations),
                    "mean_duration_seconds": sum(durations) / len(durations),
                }
            )

        self.write_csv(
            self.cfg.output_root / "csv" / "duration_stats.csv",
            avg_rows,
            ["class", "num_sequences", "mean_duration_seconds"],
        )
        self.write_csv(
            self.cfg.output_root / "csv" / "duration_per_sequence.csv",
            detail_rows,
            ["class", "sequence_id", "duration_frames", "duration_seconds"],
        )
        print(f"[STEP5] Saved {len(avg_rows)} class rows and {len(detail_rows)} sequence rows.")


def main() -> None:
    parser = SequenceTaskBase.build_common_arg_parser("Step 5: compute duration statistics.")
    args = parser.parse_args()
    task = DurationStatsTask(PipelineConfig.from_args(args))
    task.run()


if __name__ == "__main__":
    main()

