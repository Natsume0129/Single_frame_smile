from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from minimum_common import CLASS_NAMES, MinimumConfig, MinimumTaskBase, SequenceInfo, summary_stats


COLORS = {
    "polite_vs_polite": "#1f77b4",
    "truesmile_vs_truesmile": "#2ca02c",
    "ambiguous_vs_ambiguous": "#ff7f0e",
    "ambiguous_vs_polite": "#9467bd",
    "polite_vs_truesmile": "#d62728",
    "ambiguous_vs_truesmile": "#8c564b",
}


class RawMinimumDebugPipeline(MinimumTaskBase):
    def raw_seq_path(self, seq: SequenceInfo) -> Path:
        return self.cfg.analysis_input_root / "metrics" / "sequence_features" / seq.class_name / seq.sequence_id / "sequence_features.npy"

    def frame_names_path(self, seq: SequenceInfo) -> Path:
        return self.cfg.analysis_input_root / "metrics" / "sequence_features" / seq.class_name / seq.sequence_id / "frame_names.json"

    @staticmethod
    def pairwise_distance_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a2 = np.sum(a * a, axis=1, keepdims=True)
        b2 = np.sum(b * b, axis=1, keepdims=True).T
        dist2 = np.maximum(a2 + b2 - 2.0 * (a @ b.T), 0.0)
        return np.sqrt(dist2)

    @staticmethod
    def progress_percent(idx: int, length: int) -> float:
        if length <= 1:
            return 0.0
        return 100.0 * idx / float(length - 1)

    def load_all_sequences(self) -> dict[tuple[str, str], dict]:
        cache: dict[tuple[str, str], dict] = {}
        for seq in self.discover_sequences():
            arr = self.load_npy(self.raw_seq_path(seq)).astype(np.float32)
            frames = self.load_json(self.frame_names_path(seq))
            assert isinstance(frames, list)
            cache[(seq.class_name, seq.sequence_id)] = {
                "seq": seq,
                "arr": arr,
                "frames": frames,
                "length": arr.shape[0],
            }
        return cache

    def compute_all_pairs(self) -> list[dict]:
        cache = self.load_all_sequences()
        seqs = [v["seq"] for v in cache.values()]
        rows: list[dict] = []
        for i, seq1 in enumerate(seqs):
            data1 = cache[(seq1.class_name, seq1.sequence_id)]
            arr1 = data1["arr"]
            frames1 = data1["frames"]
            len1 = data1["length"]
            for j in range(i + 1, len(seqs)):
                seq2 = seqs[j]
                data2 = cache[(seq2.class_name, seq2.sequence_id)]
                arr2 = data2["arr"]
                frames2 = data2["frames"]
                len2 = data2["length"]

                dist_mat = self.pairwise_distance_matrix(arr1, arr2)
                flat_idx = int(np.argmin(dist_mat))
                t1, t2 = np.unravel_index(flat_idx, dist_mat.shape)
                pair = "_vs_".join(sorted((seq1.class_name, seq2.class_name)))
                rows.append(
                    {
                        "relation_type": "intra_class" if seq1.class_name == seq2.class_name else "inter_class",
                        "pair": pair,
                        "sequence1_class": seq1.class_name,
                        "sequence1_id": seq1.sequence_id,
                        "sequence1_time_index": int(t1),
                        "sequence1_frame_name": frames1[t1],
                        "sequence1_progress_percent": self.progress_percent(int(t1), len1),
                        "sequence2_class": seq2.class_name,
                        "sequence2_id": seq2.sequence_id,
                        "sequence2_time_index": int(t2),
                        "sequence2_frame_name": frames2[t2],
                        "sequence2_progress_percent": self.progress_percent(int(t2), len2),
                        "minimum_distance": float(dist_mat[t1, t2]),
                    }
                )
        return rows

    def write_stats(self, rows: list[dict], csv_dir: Path) -> list[dict]:
        grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
        for row in rows:
            grouped[(row["pair"], row["relation_type"])].append(float(row["minimum_distance"]))
        stats_rows: list[dict] = []
        for (pair, relation), values in sorted(grouped.items()):
            class_a, class_b = pair.split("_vs_")
            stats_rows.append(
                {
                    "pair": pair,
                    "relation_type": relation,
                    "class_a": class_a,
                    "class_b": class_b,
                    "count": len(values),
                    **summary_stats(values),
                }
            )
        self.write_csv(
            csv_dir / "raw_minimum_distance_statistics.csv",
            stats_rows,
            ["pair", "relation_type", "class_a", "class_b", "count", "mean", "std", "median", "q1", "q3"],
        )
        return stats_rows

    def plot_all_scatter(self, rows: list[dict], plot_dir: Path) -> None:
        fig, ax = plt.subplots(figsize=(7.5, 7), dpi=160)
        for pair, color in COLORS.items():
            pair_rows = [r for r in rows if r["pair"] == pair]
            if not pair_rows:
                continue
            ax.scatter(
                [float(r["sequence1_progress_percent"]) for r in pair_rows],
                [float(r["sequence2_progress_percent"]) for r in pair_rows],
                s=14,
                alpha=0.45,
                color=color,
                label=pair,
            )
        ax.plot([0, 100], [0, 100], linestyle="--", color="gray", linewidth=1.0)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_xlabel("Sequence 1 progress (%)")
        ax.set_ylabel("Sequence 2 progress (%)")
        ax.set_title("Minimum-distance positions for all sequence pairs")
        ax.legend(loc="best", fontsize=8)
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(plot_dir / "raw_minimum_distance_all_pairs_scatter.png")
        plt.close(fig)

    def plot_pair_scatter(self, rows: list[dict], plot_dir: Path, pair: str) -> None:
        pair_rows = [r for r in rows if r["pair"] == pair]
        if not pair_rows:
            return
        fig, ax = plt.subplots(figsize=(6.5, 6), dpi=160)
        ax.scatter(
            [float(r["sequence1_progress_percent"]) for r in pair_rows],
            [float(r["sequence2_progress_percent"]) for r in pair_rows],
            s=18,
            alpha=0.55,
            color=COLORS[pair],
        )
        ax.plot([0, 100], [0, 100], linestyle="--", color="gray", linewidth=1.0)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_xlabel("Sequence 1 progress (%)")
        ax.set_ylabel("Sequence 2 progress (%)")
        ax.set_title(f"Minimum-distance positions: {pair}")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(plot_dir / f"raw_minimum_distance_scatter_{pair}.png")
        plt.close(fig)

    def build_report(self, stats_rows: list[dict]) -> str:
        lines = ["# Raw Minimum Distance Debug Summary", ""]
        lines.append("## Definition")
        lines.append("- We use the original feature trajectory f(t), not f_rel(t).")
        lines.append("- For two sequences, distance is defined as min_{t1,t2} ||f1(t1) - f2(t2)||.")
        lines.append("- The minimum-distance position is recorded as two progress percentages: (x%, y%).")
        lines.append("")
        lines.append("## Intra-class results")
        for pair in ("polite_vs_polite", "ambiguous_vs_ambiguous", "truesmile_vs_truesmile"):
            row = next(r for r in stats_rows if r["pair"] == pair)
            lines.append(
                f"- {pair}: mean={float(row['mean']):.4f}, median={float(row['median']):.4f}, q1-q3=({float(row['q1']):.4f}, {float(row['q3']):.4f})"
            )
        lines.append("")
        lines.append("## Inter-class results")
        for pair in ("ambiguous_vs_polite", "polite_vs_truesmile", "ambiguous_vs_truesmile"):
            row = next(r for r in stats_rows if r["pair"] == pair)
            lines.append(
                f"- {pair}: mean={float(row['mean']):.4f}, median={float(row['median']):.4f}, q1-q3=({float(row['q1']):.4f}, {float(row['q3']):.4f})"
            )
        return "\n".join(lines) + "\n"

    def run(self) -> None:
        output_root = Path(r"E:\Matsuda_data\minimum_distace_debug")
        csv_dir = output_root / "csv"
        plot_dir = output_root / "plots"
        report_dir = output_root / "report"
        for p in (csv_dir, plot_dir, report_dir):
            p.mkdir(parents=True, exist_ok=True)

        rows = self.compute_all_pairs()
        self.write_csv(
            csv_dir / "raw_minimum_distance_all_pairs.csv",
            rows,
            [
                "relation_type",
                "pair",
                "sequence1_class",
                "sequence1_id",
                "sequence1_time_index",
                "sequence1_frame_name",
                "sequence1_progress_percent",
                "sequence2_class",
                "sequence2_id",
                "sequence2_time_index",
                "sequence2_frame_name",
                "sequence2_progress_percent",
                "minimum_distance",
            ],
        )
        stats_rows = self.write_stats(rows, csv_dir)

        self.plot_all_scatter(rows, plot_dir)
        for pair in COLORS:
            self.plot_pair_scatter(rows, plot_dir, pair)

        report_text = self.build_report(stats_rows)
        (report_dir / "raw_minimum_distance_summary.md").write_text(report_text, encoding="utf-8")
        print(f"[MINIMUM_DEBUG] Finished. Summary saved to: {report_dir / 'raw_minimum_distance_summary.md'}")


def main() -> None:
    parser = MinimumTaskBase.build_common_arg_parser("Run raw f(t) minimum-distance debug analysis.")
    args = parser.parse_args()
    pipeline = RawMinimumDebugPipeline(MinimumConfig.from_args(args))
    pipeline.run()


if __name__ == "__main__":
    main()
