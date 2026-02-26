from __future__ import annotations

import numpy as np

from common.base import PipelineConfig, SequenceTaskBase


class ClassDistanceCurveTask(SequenceTaskBase):
    def run(self) -> None:
        proto_dir = self.cfg.output_root / "prototypes"
        polite = np.load(proto_dir / "prototype_polite.npy")
        true = np.load(proto_dir / "prototype_truesmile.npy")
        amb = np.load(proto_dir / "prototype_ambiguous.npy")

        pairs = {
            "polite_vs_truesmile": (polite, true),
            "polite_vs_ambiguous": (polite, amb),
            "truesmile_vs_ambiguous": (true, amb),
        }

        rows: list[dict] = []
        for name, (a, b) in pairs.items():
            curve = np.linalg.norm(a - b, axis=1)
            peak_idx = int(np.argmax(curve))
            for t, v in enumerate(curve.tolist()):
                rows.append(
                    {
                        "pair": name,
                        "t_index": t,
                        "time_percent": float(t / (len(curve) - 1) * 100.0),
                        "diff_norm": float(v),
                        "is_peak": int(t == peak_idx),
                    }
                )

        self.write_csv(
            self.cfg.output_root / "csv" / "class_distance_curve.csv",
            rows,
            ["pair", "t_index", "time_percent", "diff_norm", "is_peak"],
        )
        print(f"[STEP11] Saved class distance curve with {len(rows)} rows.")


def main() -> None:
    parser = SequenceTaskBase.build_common_arg_parser("Step 11: class distance curve.")
    args = parser.parse_args()
    task = ClassDistanceCurveTask(PipelineConfig.from_args(args))
    task.run()


if __name__ == "__main__":
    main()

