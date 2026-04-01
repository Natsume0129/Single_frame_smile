from __future__ import annotations

from pathlib import Path

from dtw_resample_common import (
    CLASS_NAMES,
    DTWResampleConfig,
    DTWResampleTaskBase,
    SequenceInfo,
    align_sequence_to_reference,
    dtw_distance_and_path,
    export_clip_from_frames,
    find_source_video,
    resample_2d,
    sampled_indices,
)


class DTWResamplePipeline(DTWResampleTaskBase):
    def run(self) -> None:
        representative_rows: list[dict] = []
        alignment_rows: list[dict] = []
        all_sequence_rows: list[dict] = []

        for class_name in CLASS_NAMES:
            seqs = self.sequences_for_class(class_name)
            arrays = {seq.sequence_id: self.load_npy(self.rel_seq_path(seq)).astype("float32") for seq in seqs}
            frame_names = {seq.sequence_id: self.load_json(self.frame_names_path(seq)) for seq in seqs}

            # Build class-wise DTW matrix
            matrix_rows: list[dict] = []
            centrality = {seq.sequence_id: 0.0 for seq in seqs}
            for i, seq_i in enumerate(seqs):
                for j, seq_j in enumerate(seqs):
                    if j < i:
                        continue
                    score, _ = dtw_distance_and_path(
                        arrays[seq_i.sequence_id],
                        arrays[seq_j.sequence_id],
                        self.cfg.sakoe_chiba_ratio,
                    )
                    matrix_rows.append(
                        {
                            "class": class_name,
                            "sequence1_id": seq_i.sequence_id,
                            "sequence2_id": seq_j.sequence_id,
                            "dtw_distance": score,
                        }
                    )
                    if i != j:
                        centrality[seq_i.sequence_id] += score
                        centrality[seq_j.sequence_id] += score

            ranked = sorted(centrality.items(), key=lambda item: item[1])
            rep_id, rep_score = ranked[0]
            second_score = ranked[1][1] if len(ranked) > 1 else rep_score
            rep_seq = SequenceInfo(class_name=class_name, sequence_id=rep_id)
            rep_arr = arrays[rep_id]
            rep_frame_names = frame_names[rep_id]
            assert isinstance(rep_frame_names, list)

            matrix_path = self.cfg.output_root / "csv" / f"intra_class_dtw_matrix_{class_name}.csv"
            self.write_csv(matrix_path, matrix_rows, ["class", "sequence1_id", "sequence2_id", "dtw_distance"])

            rep_media_dir = self.cfg.output_root / "media" / class_name / rep_id
            rep_media_dir.mkdir(parents=True, exist_ok=True)
            source_video = find_source_video(self.source_videos_dir(class_name), str(rep_frame_names[0]))
            copied_video_path = ""
            if source_video is not None:
                copied = rep_media_dir / source_video.name
                if not copied.exists():
                    copied.write_bytes(source_video.read_bytes())
                copied_video_path = str(copied)

            clip_path = rep_media_dir / f"{class_name}_{rep_id}_clip.mp4"
            export_clip_from_frames(
                self.source_sequence_dir(rep_seq),
                [str(name) for name in rep_frame_names],
                clip_path,
                self.cfg.clip_fps,
            )

            # Representative sequence also outputs its own 20-point resampled version
            rep_resampled = resample_2d(rep_arr, self.cfg.norm_len)
            rep_out_dir = self.cfg.output_root / "metrics" / "resampled20_aligned" / class_name / rep_id
            self.save_npy(rep_out_dir / "aligned_resampled20.npy", rep_resampled)
            rep_idx = sampled_indices(len(rep_frame_names), self.cfg.norm_len)
            rep_mapping = []
            for out_idx, src_idx in enumerate(rep_idx.tolist()):
                rep_mapping.append(
                    {
                        "resampled_index": out_idx,
                        "reference_time_index": int(src_idx),
                        "source_time_indices": [int(src_idx)],
                        "source_file": str(rep_frame_names[src_idx]),
                    }
                )
            self.save_json(rep_out_dir / "alignment_mapping.json", rep_mapping)

            representative_rows.append(
                {
                    "class": class_name,
                    "representative_sequence_id": rep_id,
                    "centrality_score": float(rep_score),
                    "second_best_centrality_score": float(second_score),
                    "source_video_path": copied_video_path,
                    "clip_video_path": str(clip_path),
                    "resampled20_path": str(rep_out_dir / "aligned_resampled20.npy"),
                }
            )

            # Align all class sequences to representative sequence timeline
            for seq in seqs:
                seq_arr = arrays[seq.sequence_id]
                seq_frames = frame_names[seq.sequence_id]
                assert isinstance(seq_frames, list)
                if seq.sequence_id == rep_id:
                    score = 0.0
                    path = [(i, i) for i in range(rep_arr.shape[0])]
                    aligned = rep_arr.copy()
                    grouped = [[i] for i in range(rep_arr.shape[0])]
                else:
                    score, path = dtw_distance_and_path(seq_arr, rep_arr, self.cfg.sakoe_chiba_ratio)
                    aligned, grouped = align_sequence_to_reference(seq_arr, rep_arr.shape[0], path)

                aligned_out_dir = self.cfg.output_root / "metrics" / "aligned_to_representative" / class_name / seq.sequence_id
                self.save_npy(aligned_out_dir / "aligned_sequence.npy", aligned)
                self.save_json(
                    aligned_out_dir / "alignment_path.json",
                    [{"source_time_index": int(src), "reference_time_index": int(ref)} for src, ref in path],
                )

                resampled = resample_2d(aligned, self.cfg.norm_len)
                resampled_out_dir = self.cfg.output_root / "metrics" / "resampled20_aligned" / class_name / seq.sequence_id
                self.save_npy(resampled_out_dir / "aligned_resampled20.npy", resampled)

                ref_idx = sampled_indices(aligned.shape[0], self.cfg.norm_len)
                mapping = []
                for out_idx, ref_idx_val in enumerate(ref_idx.tolist()):
                    src_ids = grouped[ref_idx_val] if ref_idx_val < len(grouped) else []
                    src_files = [str(seq_frames[src]) for src in src_ids if src < len(seq_frames)]
                    mapping.append(
                        {
                            "resampled_index": out_idx,
                            "reference_time_index": int(ref_idx_val),
                            "source_time_indices": [int(v) for v in src_ids],
                            "source_files": src_files,
                        }
                    )
                self.save_json(resampled_out_dir / "alignment_mapping.json", mapping)

                all_sequence_rows.append(
                    {
                        "class": class_name,
                        "sequence_id": seq.sequence_id,
                        "representative_sequence_id": rep_id,
                        "dtw_distance_to_representative": float(score),
                        "aligned_sequence_path": str(aligned_out_dir / "aligned_sequence.npy"),
                        "resampled20_path": str(resampled_out_dir / "aligned_resampled20.npy"),
                    }
                )

                for src, ref in path:
                    alignment_rows.append(
                        {
                            "class": class_name,
                            "sequence_id": seq.sequence_id,
                            "reference_sequence_id": rep_id,
                            "reference_time_index": int(ref),
                            "source_time_index": int(src),
                        }
                    )

        self.write_csv(
            self.cfg.output_root / "csv" / "representative_sequences.csv",
            representative_rows,
            [
                "class",
                "representative_sequence_id",
                "centrality_score",
                "second_best_centrality_score",
                "source_video_path",
                "clip_video_path",
                "resampled20_path",
            ],
        )
        self.write_csv(
            self.cfg.output_root / "csv" / "all_sequences_to_representative.csv",
            all_sequence_rows,
            [
                "class",
                "sequence_id",
                "representative_sequence_id",
                "dtw_distance_to_representative",
                "aligned_sequence_path",
                "resampled20_path",
            ],
        )
        self.write_csv(
            self.cfg.output_root / "csv" / "dtw_alignment_paths.csv",
            alignment_rows,
            [
                "class",
                "sequence_id",
                "reference_sequence_id",
                "reference_time_index",
                "source_time_index",
            ],
        )

        report_lines = [
            "# DTW Resample Summary",
            "",
            "## Representative Sequences",
        ]
        for row in representative_rows:
            report_lines.append(
                f"- {row['class']}: representative={row['representative_sequence_id']}, centrality={row['centrality_score']:.4f}, "
                f"video={row['source_video_path']}, clip={row['clip_video_path']}"
            )
        report_lines.append("")
        report_lines.append("## Outputs")
        report_lines.append(f"- representative csv: {self.cfg.output_root / 'csv' / 'representative_sequences.csv'}")
        report_lines.append(f"- all sequence csv: {self.cfg.output_root / 'csv' / 'all_sequences_to_representative.csv'}")
        report_lines.append(f"- alignment path csv: {self.cfg.output_root / 'csv' / 'dtw_alignment_paths.csv'}")
        (self.cfg.output_root / "report" / "dtw_resample_summary.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

        local_result = Path(__file__).resolve().parent / "result.md"
        local_result.write_text(
            "# DTW_resample result\n\n"
            f"- output_root: {self.cfg.output_root}\n"
            f"- representative_csv: {self.cfg.output_root / 'csv' / 'representative_sequences.csv'}\n"
            f"- report: {self.cfg.output_root / 'report' / 'dtw_resample_summary.md'}\n",
            encoding="utf-8",
        )
        print(f"[DTW_RESAMPLE] Finished. Report saved to: {self.cfg.output_root / 'report' / 'dtw_resample_summary.md'}")


def main() -> None:
    parser = DTWResampleTaskBase.build_common_arg_parser("Run DTW representative sequence alignment and resample pipeline.")
    args = parser.parse_args()
    pipeline = DTWResamplePipeline(DTWResampleConfig.from_args(args))
    pipeline.run()


if __name__ == "__main__":
    main()
