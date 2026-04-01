from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from minimum_common import (
    CLASS_NAMES,
    MinimumConfig,
    MinimumTaskBase,
    SequenceInfo,
    compute_sync_min,
    summary_stats,
)


COLORS = {"polite": "#1f77b4", "truesmile": "#2ca02c", "ambiguous": "#ff7f0e"}


class MinimumPipeline(MinimumTaskBase):
    def load_method_prototypes(self, method: str) -> dict[str, np.ndarray]:
        proto_root = self.prototype_dir()
        if method == "methodA":
            return {
                class_name: self.load_npy(proto_root / f"prototype_{class_name}.npy").astype(np.float32)
                for class_name in CLASS_NAMES
            }
        return {
            class_name: self.load_npy(proto_root / f"prototype_{class_name}_medoid.npy").astype(np.float32)
            for class_name in CLASS_NAMES
        }

    def medoid_sequence_ids(self) -> dict[str, str]:
        meta = self.load_json(self.prototype_dir() / "prototype_meta.json")
        assert isinstance(meta, dict)
        return {class_name: str(meta[class_name]["medoid_sequence_id"]) for class_name in CLASS_NAMES}

    def sampled_source_map(self, seq: SequenceInfo) -> dict[int, str]:
        payload = self.load_json(self.sampled_frames_path(seq))
        assert isinstance(payload, list)
        return {int(item["normalized_index"]): str(item["source_file"]) for item in payload if isinstance(item, dict)}

    def prototype_rows(self, method: str) -> list[dict]:
        protos = self.load_method_prototypes(method)
        medoid_ids = self.medoid_sequence_ids()
        rows: list[dict] = []
        for i, class1 in enumerate(CLASS_NAMES):
            for j in range(i + 1, len(CLASS_NAMES)):
                class2 = CLASS_NAMES[j]
                idx, dist = compute_sync_min(protos[class1], protos[class2])
                frame1 = ""
                frame2 = ""
                if method == "methodB":
                    seq1 = SequenceInfo(class_name=class1, sequence_id=medoid_ids[class1])
                    seq2 = SequenceInfo(class_name=class2, sequence_id=medoid_ids[class2])
                    frame1 = self.sampled_source_map(seq1).get(idx, "")
                    frame2 = self.sampled_source_map(seq2).get(idx, "")
                rows.append(
                    {
                        "method": method,
                        "relation_type": "inter_class",
                        "curve1_class": class1,
                        "curve1_sequence_id": "prototype" if method == "methodA" else medoid_ids[class1],
                        "curve2_class": class2,
                        "curve2_sequence_id": "prototype" if method == "methodA" else medoid_ids[class2],
                        "argmin_time_index": idx,
                        "curve1_frame_name": frame1,
                        "curve2_frame_name": frame2,
                        "minimum_distance": dist,
                    }
                )
        return rows

    def sequence_rows(self) -> list[dict]:
        seqs = self.discover_sequences()
        cache = { (s.class_name, s.sequence_id): self.load_npy(self.normalized_seq_path(s)).astype(np.float32) for s in seqs }
        frame_maps = { (s.class_name, s.sequence_id): self.sampled_source_map(s) for s in seqs }
        rows: list[dict] = []
        for i, seq1 in enumerate(seqs):
            arr1 = cache[(seq1.class_name, seq1.sequence_id)]
            for j in range(i + 1, len(seqs)):
                seq2 = seqs[j]
                arr2 = cache[(seq2.class_name, seq2.sequence_id)]
                idx, dist = compute_sync_min(arr1, arr2)
                rows.append(
                    {
                        "relation_type": "intra_class" if seq1.class_name == seq2.class_name else "inter_class",
                        "sequence1_class": seq1.class_name,
                        "sequence1_id": seq1.sequence_id,
                        "sequence2_class": seq2.class_name,
                        "sequence2_id": seq2.sequence_id,
                        "argmin_time_index": idx,
                        "sequence1_frame_name": frame_maps[(seq1.class_name, seq1.sequence_id)].get(idx, ""),
                        "sequence2_frame_name": frame_maps[(seq2.class_name, seq2.sequence_id)].get(idx, ""),
                        "minimum_distance": dist,
                    }
                )
        return rows

    def write_sequence_outputs(self, rows: list[dict]) -> list[dict]:
        self.write_csv(
            self.shared_csv("sync_min_distance_all_pairs.csv"),
            rows,
            [
                "relation_type",
                "sequence1_class",
                "sequence1_id",
                "sequence2_class",
                "sequence2_id",
                "argmin_time_index",
                "sequence1_frame_name",
                "sequence2_frame_name",
                "minimum_distance",
            ],
        )
        grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
        grouped_time: dict[tuple[str, str], list[int]] = defaultdict(list)
        for row in rows:
            pair = tuple(sorted((row["sequence1_class"], row["sequence2_class"])))
            grouped[pair].append(float(row["minimum_distance"]))
            grouped_time[pair].append(int(row["argmin_time_index"]))

        stats_rows: list[dict] = []
        for pair, values in sorted(grouped.items()):
            relation = "intra_class" if pair[0] == pair[1] else "inter_class"
            time_counter = Counter(grouped_time[pair])
            top_t = time_counter.most_common(1)[0][0]
            stats_rows.append(
                {
                    "pair": f"{pair[0]}_vs_{pair[1]}",
                    "relation_type": relation,
                    "class_a": pair[0],
                    "class_b": pair[1],
                    "count": len(values),
                    "most_common_argmin_time_index": top_t,
                    **summary_stats(values),
                }
            )
        self.write_csv(
            self.shared_csv("sync_min_distance_statistics.csv"),
            stats_rows,
            ["pair", "relation_type", "class_a", "class_b", "count", "most_common_argmin_time_index", "mean", "std", "median", "q1", "q3"],
        )
        return stats_rows

    def write_prototype_outputs(self, method: str, rows: list[dict]) -> None:
        self.write_csv(
            self.method_csv(method, f"sync_min_distance_{method}.csv"),
            rows,
            [
                "method",
                "relation_type",
                "curve1_class",
                "curve1_sequence_id",
                "curve2_class",
                "curve2_sequence_id",
                "argmin_time_index",
                "curve1_frame_name",
                "curve2_frame_name",
                "minimum_distance",
            ],
        )

    def plot_distribution(self, grouped_rows: list[dict], path: Path, title: str) -> None:
        ordered_pairs = [
            ("polite", "polite"),
            ("truesmile", "truesmile"),
            ("ambiguous", "ambiguous"),
            ("ambiguous", "polite"),
            ("polite", "truesmile"),
            ("ambiguous", "truesmile"),
        ]
        row_map = { (row["class_a"], row["class_b"]): row for row in grouped_rows }
        labels = []
        medians = []
        q1 = []
        q3 = []
        for pair in ordered_pairs:
            pair_sorted = tuple(sorted(pair))
            if pair_sorted not in row_map:
                continue
            row = row_map[pair_sorted]
            labels.append(f"{pair_sorted[0]} vs {pair_sorted[1]}")
            medians.append(float(row["median"]))
            q1.append(float(row["q1"]))
            q3.append(float(row["q3"]))

        x = np.arange(len(labels))
        fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
        ax.scatter(x, medians, color="#d62728", s=45, zorder=3)
        ax.vlines(x, q1, q3, color="#1f77b4", linewidth=3, alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylabel("Synchronized Minimum Distance")
        ax.set_title(title)
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)

    def plot_time_distribution(self, rows: list[dict], path: Path, title: str) -> None:
        counts = Counter(int(row["argmin_time_index"]) for row in rows)
        x = list(range(self.cfg.norm_len))
        y = [counts[i] for i in x]
        fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
        ax.bar(x, y, color="#4c78a8")
        ax.set_xlabel("argmin_time_index")
        ax.set_ylabel("Count")
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)

    def plot_time_heatmap(self, rows: list[dict], path: Path, title: str) -> None:
        pair_order = [
            ("ambiguous", "polite"),
            ("polite", "truesmile"),
            ("ambiguous", "truesmile"),
            ("polite", "polite"),
            ("truesmile", "truesmile"),
            ("ambiguous", "ambiguous"),
        ]
        mat = np.zeros((len(pair_order), self.cfg.norm_len), dtype=np.int32)
        pair_to_idx = {tuple(sorted(pair)): i for i, pair in enumerate(pair_order)}
        for row in rows:
            pair = tuple(sorted((row.get("sequence1_class", row.get("curve1_class")), row.get("sequence2_class", row.get("curve2_class")))))
            if pair not in pair_to_idx:
                continue
            mat[pair_to_idx[pair], int(row["argmin_time_index"])] += 1

        fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
        im = ax.imshow(mat, aspect="auto", cmap="YlGnBu")
        ax.set_yticks(range(len(pair_order)))
        ax.set_yticklabels([f"{a} vs {b}" for a, b in pair_order])
        ax.set_xticks(range(self.cfg.norm_len))
        ax.set_xlabel("argmin_time_index")
        ax.set_title(title)
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)

    def plot_cdf(self, rows: list[dict], path: Path, title: str) -> None:
        grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
        for row in rows:
            pair = tuple(sorted((row.get("sequence1_class", row.get("curve1_class")), row.get("sequence2_class", row.get("curve2_class")))))
            grouped[pair].append(int(row["argmin_time_index"]))

        fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
        for pair in [("ambiguous", "polite"), ("polite", "truesmile"), ("ambiguous", "truesmile")]:
            vals = np.sort(np.asarray(grouped[pair], dtype=np.float64))
            if vals.size == 0:
                continue
            y = np.arange(1, vals.size + 1) / vals.size
            color = COLORS[pair[0]]
            ax.step(vals, y, where="post", label=f"{pair[0]} vs {pair[1]}", color=color)
        ax.set_xlabel("argmin_time_index")
        ax.set_ylabel("CDF")
        ax.set_title(title)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)

    def plot_examples(self, rows: list[dict], path: Path, title: str, key1: str, key2: str, id1: str, id2: str, frame1: str, frame2: str) -> None:
        top_rows = sorted(rows, key=lambda r: float(r["minimum_distance"]))[:6]
        if not top_rows:
            return
        fig, axes = plt.subplots(len(top_rows), 2, figsize=(8, 3 * len(top_rows)), dpi=140)
        axes_arr = np.atleast_2d(axes)
        for row_idx, row in enumerate(top_rows):
            ax1 = axes_arr[row_idx, 0]
            ax2 = axes_arr[row_idx, 1]
            for ax in (ax1, ax2):
                ax.axis("off")
            seq1 = SequenceInfo(class_name=row[key1], sequence_id=row[id1])
            seq2 = SequenceInfo(class_name=row[key2], sequence_id=row[id2])
            img1 = self.normalized_frames_dir(seq1) / f"{int(row['argmin_time_index']):03d}.png"
            img2 = self.normalized_frames_dir(seq2) / f"{int(row['argmin_time_index']):03d}.png"
            if img1.exists():
                ax1.imshow(plt.imread(img1))
            if img2.exists():
                ax2.imshow(plt.imread(img2))
            ax1.set_title(f"{row[key1]} / {row[id1]} / {row[frame1]}", fontsize=8)
            ax2.set_title(f"{row[key2]} / {row[id2]} / {row[frame2]}", fontsize=8)
        fig.suptitle(title, fontsize=12)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)

    def build_summary(self, method_rows: dict[str, list[dict]], stats_rows: list[dict]) -> str:
        lines = ["# synchronized minimum distance Summary", ""]
        lines.append("## Definition")
        lines.append("- For two time-aligned curves C1 and C2, we compute min_t ||C1(t) - C2(t)||.")
        lines.append("- This differs from the previous cross-time definition min_{t1,t2} ||C1(t1) - C2(t2)||.")
        lines.append("")
        lines.append("## Sequence-level statistics")
        for row in stats_rows:
            lines.append(
                f"- {row['pair']} ({row['relation_type']}): mean={float(row['mean']):.4f}, median={float(row['median']):.4f}, "
                f"q1-q3=({float(row['q1']):.4f}, {float(row['q3']):.4f}), most_common_t={row['most_common_argmin_time_index']}"
            )
        lines.append("")
        lines.append("## Prototype-level statistics")
        for method, rows in method_rows.items():
            lines.append(f"### {method}")
            for row in rows:
                lines.append(
                    f"- {row['curve1_class']} vs {row['curve2_class']}: min={float(row['minimum_distance']):.4f} at t={row['argmin_time_index']}"
                )
        return "\n".join(lines) + "\n"

    def run(self) -> None:
        sequence_rows = self.sequence_rows()
        stats_rows = self.write_sequence_outputs(sequence_rows)

        method_proto_rows: dict[str, list[dict]] = {}
        for method in ("methodA", "methodB"):
            rows = self.prototype_rows(method)
            self.write_prototype_outputs(method, rows)
            method_proto_rows[method] = rows

        self.plot_distribution(
            stats_rows,
            self.shared_plot("sync_min_distance_distribution_shared.png"),
            "Synchronized Minimum Distance Distribution",
        )
        self.plot_time_distribution(
            sequence_rows,
            self.shared_plot("sync_min_distance_time_distribution_shared.png"),
            "argmin_time_index Distribution",
        )
        self.plot_time_heatmap(
            sequence_rows,
            self.shared_plot("sync_min_distance_time_heatmap_shared.png"),
            "argmin_time_index Heatmap by Pair",
        )
        self.plot_cdf(
            sequence_rows,
            self.shared_plot("sync_min_distance_time_cdf_shared.png"),
            "CDF of argmin_time_index by Inter-class Pair",
        )
        self.plot_examples(
            sequence_rows,
            self.shared_plot("sync_min_distance_examples_sorted_shared.png"),
            "Top synchronized minimum-distance examples",
            "sequence1_class",
            "sequence2_class",
            "sequence1_id",
            "sequence2_id",
            "sequence1_frame_name",
            "sequence2_frame_name",
        )

        for method, rows in method_proto_rows.items():
            self.plot_time_distribution(
                rows,
                self.method_plot(method, f"sync_min_distance_time_distribution_{method}.png"),
                f"Prototype argmin_time_index Distribution ({method})",
            )

        summary_text = self.build_summary(method_proto_rows, stats_rows)
        report_path = self.cfg.output_root / "report" / "sync_min_distance_summary.md"
        report_path.write_text(summary_text, encoding="utf-8")
        local_result = Path(__file__).resolve().parent / "result.md"
        local_result.write_text(
            "# analysis_minimum result\n\n"
            f"- output_root: {self.cfg.output_root}\n"
            f"- report: {report_path}\n"
            f"- sequence_pairs: {len(sequence_rows)}\n",
            encoding="utf-8",
        )
        print(f"[MINIMUM] Finished. Report saved to: {report_path}")


def main() -> None:
    parser = MinimumTaskBase.build_common_arg_parser("Run synchronized minimum distance pipeline.")
    args = parser.parse_args()
    pipeline = MinimumPipeline(MinimumConfig.from_args(args))
    pipeline.run()


if __name__ == "__main__":
    main()
