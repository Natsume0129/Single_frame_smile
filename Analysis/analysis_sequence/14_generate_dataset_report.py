from __future__ import annotations

from common.base import PipelineConfig, SequenceTaskBase


class DatasetReportTask(SequenceTaskBase):
    def run(self) -> None:
        rows = []
        for seq in self.discover_sequences():
            feat = self.load_npy(self.metrics_seq_dir("sequence_features", seq) / "sequence_features.npy")
            mag = self.load_json(self.metrics_seq_dir("distance", seq) / "metrics.json")
            vel = self.load_json(self.metrics_seq_dir("velocity", seq) / "metrics.json")
            frames = int(feat.shape[0])
            rows.append(
                {
                    "class": seq.class_name,
                    "sequence_id": seq.sequence_id,
                    "frames": frames,
                    "duration_sec": frames / float(self.cfg.fps),
                    "peak_magnitude": float(mag["peak_magnitude"]),
                    "mean_velocity": float(vel["mean_velocity"]),
                }
            )

        self.write_csv(
            self.cfg.output_root / "csv" / "dataset_report.csv",
            rows,
            ["class", "sequence_id", "frames", "duration_sec", "peak_magnitude", "mean_velocity"],
        )
        print(f"[STEP14] Saved dataset report rows={len(rows)}")


def main() -> None:
    parser = SequenceTaskBase.build_common_arg_parser("Step 14: generate dataset report CSV.")
    args = parser.parse_args()
    task = DatasetReportTask(PipelineConfig.from_args(args))
    task.run()


if __name__ == "__main__":
    main()

