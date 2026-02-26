from __future__ import annotations

import json
import shutil

import numpy as np

from common.base import PipelineConfig, SequenceTaskBase


def interp_2d(arr: np.ndarray, target_len: int) -> np.ndarray:
    t_old = np.arange(arr.shape[0], dtype=np.float32)
    t_new = np.linspace(0, arr.shape[0] - 1, target_len, dtype=np.float32)
    out = np.empty((target_len, arr.shape[1]), dtype=np.float32)
    for d in range(arr.shape[1]):
        out[:, d] = np.interp(t_new, t_old, arr[:, d])
    return out


def interp_1d(arr: np.ndarray, target_len: int) -> np.ndarray:
    t_old = np.arange(arr.shape[0], dtype=np.float32)
    t_new = np.linspace(0, arr.shape[0] - 1, target_len, dtype=np.float32)
    return np.interp(t_new, t_old, arr).astype(np.float32)


class TimeNormalizeTask(SequenceTaskBase):
    def run(self) -> None:
        n = self.cfg.norm_len
        for seq in self.discover_sequences():
            feat_rel = self.load_npy(
                self.metrics_seq_dir("sequence_features_rel", seq) / "sequence_features_rel.npy"
            )
            distance = self.load_npy(self.metrics_seq_dir("distance", seq) / "distance_curve.npy")
            velocity = self.load_npy(self.metrics_seq_dir("velocity", seq) / "velocity_curve.npy")

            f_norm = interp_2d(feat_rel, n)
            d_norm = interp_1d(distance, n)
            v_norm = interp_1d(velocity, n)

            out_dir = self.metrics_seq_dir("normalized", seq)
            self.save_npy(out_dir / "normalized_sequence.npy", f_norm)
            self.save_npy(out_dir / "distance_norm.npy", d_norm)
            self.save_npy(out_dir / "velocity_norm.npy", v_norm)

            src_frames = self.list_sorted_frames(seq.sequence_path)
            src_idx = np.linspace(0, len(src_frames) - 1, n)
            src_idx = np.rint(src_idx).astype(int)

            frame_out_dir = self.cfg.output_root / "metrics" / "normalized_frames" / seq.class_name / seq.sequence_id
            frame_out_dir.mkdir(parents=True, exist_ok=True)
            mapping: list[dict] = []
            for i, idx in enumerate(src_idx.tolist()):
                src = src_frames[idx]
                dst = frame_out_dir / f"{i:03d}{src.suffix.lower()}"
                shutil.copy2(src, dst)
                mapping.append(
                    {
                        "normalized_index": i,
                        "source_index": idx,
                        "source_file": src.name,
                        "normalized_file": dst.name,
                    }
                )
            with (out_dir / "sampled_frames.json").open("w", encoding="utf-8") as f:
                json.dump(mapping, f, ensure_ascii=False, indent=2)

            print(f"[STEP6] {seq.class_name}/{seq.sequence_id}: normalized to {n}")


def main() -> None:
    parser = SequenceTaskBase.build_common_arg_parser("Step 6: normalize sequence length to fixed N.")
    args = parser.parse_args()
    task = TimeNormalizeTask(PipelineConfig.from_args(args))
    task.run()


if __name__ == "__main__":
    main()

