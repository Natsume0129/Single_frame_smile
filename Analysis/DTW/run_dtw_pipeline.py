from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from dtw_common import (
    CLASS_NAMES,
    DTWConfig,
    DTWTaskBase,
    SequenceInfo,
    dtw_distance,
    fit_pca_projection,
    make_magnitude_sequence,
    make_velocity_sequence,
    summary_stats,
)


COLORS = {"polite": "#1f77b4", "truesmile": "#2ca02c", "ambiguous": "#ff7f0e"}


class DTWPipeline(DTWTaskBase):
    def __init__(self, config: DTWConfig):
        super().__init__(config)
        self.sequences = self.discover_sequences()
        self.raw_sequences = self.load_all_sequences()
        self.branches: list[tuple[str, str, bool]] = [
            ("magnitude_dtw", "magnitude", False),
            ("magnitude_dtw_band", "magnitude", True),
            ("velocity_dtw", "velocity", False),
            ("velocity_dtw_band", "velocity", True),
            ("pca10_dtw", "pca10", False),
            ("pca10_dtw_band", "pca10", True),
            ("pca20_dtw", "pca20", False),
            ("pca20_dtw_band", "pca20", True),
            ("pca30_dtw", "pca30", False),
            ("pca30_dtw_band", "pca30", True),
        ]

    def build_feature_sequences(self) -> dict[str, dict[tuple[str, str], np.ndarray]]:
        transformed: dict[str, dict[tuple[str, str], np.ndarray]] = {}
        transformed["magnitude"] = {
            key: make_magnitude_sequence(arr) for key, arr in self.raw_sequences.items()
        }
        transformed["velocity"] = {
            key: make_velocity_sequence(arr) for key, arr in self.raw_sequences.items()
        }

        for n_components in (10, 20, 30):
            scaler, pca, seq_map = fit_pca_projection(self.raw_sequences, n_components)
            transformed[f"pca{n_components}"] = seq_map
            model_dir = self.cfg.output_root / "models"
            np.savez(
                model_dir / f"pca_model_{n_components}.npz",
                mean_=scaler.mean_,
                scale_=scaler.scale_,
                components_=pca.components_,
                explained_variance_ratio_=pca.explained_variance_ratio_,
            )
        return transformed

    def compute_branch_distances(
        self,
        branch_name: str,
        feature_name: str,
        use_band: bool,
        feature_sequences: dict[tuple[str, str], np.ndarray],
    ) -> tuple[list[dict], list[dict], list[dict]]:
        pair_rows: list[dict] = []
        intra_distances_by_class: dict[str, dict[tuple[str, str], float]] = defaultdict(dict)

        for i, seq1 in enumerate(self.sequences):
            key1 = (seq1.class_name, seq1.sequence_id)
            s1 = feature_sequences[key1]
            for j in range(i + 1, len(self.sequences)):
                seq2 = self.sequences[j]
                key2 = (seq2.class_name, seq2.sequence_id)
                s2 = feature_sequences[key2]
                dist = dtw_distance(s1, s2, use_band=use_band, ratio=self.cfg.sakoe_chiba_ratio)
                relation_type = "intra_class" if seq1.class_name == seq2.class_name else "inter_class"
                pair_rows.append(
                    {
                        "branch": branch_name,
                        "feature_type": feature_name,
                        "sequence1_class": seq1.class_name,
                        "sequence1_id": seq1.sequence_id,
                        "sequence2_class": seq2.class_name,
                        "sequence2_id": seq2.sequence_id,
                        "relation_type": relation_type,
                        "dtw_distance": dist,
                    }
                )
                if relation_type == "intra_class":
                    intra_distances_by_class[seq1.class_name][(seq1.sequence_id, seq2.sequence_id)] = dist

        stats_grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
        for row in pair_rows:
            pair = tuple(sorted((row["sequence1_class"], row["sequence2_class"])))
            stats_grouped[(row["relation_type"], f"{pair[0]}_vs_{pair[1]}")].append(float(row["dtw_distance"]))

        stats_rows: list[dict] = []
        for (relation_type, pair_name), values in sorted(stats_grouped.items()):
            class_a, class_b = pair_name.split("_vs_")
            stats_rows.append(
                {
                    "branch": branch_name,
                    "feature_type": feature_name,
                    "pair": pair_name,
                    "relation_type": relation_type,
                    "class_a": class_a,
                    "class_b": class_b,
                    "count": len(values),
                    **summary_stats(values),
                }
            )

        representative_rows: list[dict] = []
        for class_name in CLASS_NAMES:
            seq_ids = [seq.sequence_id for seq in self.sequences if seq.class_name == class_name]
            centrality: dict[str, float] = {seq_id: 0.0 for seq_id in seq_ids}
            for (id1, id2), dist in intra_distances_by_class[class_name].items():
                centrality[id1] += dist
                centrality[id2] += dist
            ranked = sorted(centrality.items(), key=lambda item: item[1])
            rep_id, rep_score = ranked[0]
            second_score = ranked[1][1] if len(ranked) > 1 else rep_score
            representative_rows.append(
                {
                    "branch": branch_name,
                    "feature_type": feature_name,
                    "class": class_name,
                    "representative_sequence_id": rep_id,
                    "centrality_score": rep_score,
                    "second_best_centrality_score": second_score,
                }
            )

        representative_pair_rows: list[dict] = []
        pair_rows_grouped: dict[str, list[dict]] = defaultdict(list)
        for row in pair_rows:
            pair = "_vs_".join(sorted((row["sequence1_class"], row["sequence2_class"])))
            pair_rows_grouped[pair].append(row)
        for pair_name, rows in sorted(pair_rows_grouped.items()):
            ordered = sorted(rows, key=lambda r: float(r["dtw_distance"]))
            best = ordered[0]
            distances = [float(r["dtw_distance"]) for r in ordered]
            median_value = float(np.median(distances))
            representative = min(ordered, key=lambda r: abs(float(r["dtw_distance"]) - median_value))
            representative_pair_rows.extend(
                [
                    {
                        "branch": branch_name,
                        "feature_type": feature_name,
                        "pair": pair_name,
                        "pair_type": "best_match",
                        "sequence1_class": best["sequence1_class"],
                        "sequence1_id": best["sequence1_id"],
                        "sequence2_class": best["sequence2_class"],
                        "sequence2_id": best["sequence2_id"],
                        "dtw_distance": float(best["dtw_distance"]),
                    },
                    {
                        "branch": branch_name,
                        "feature_type": feature_name,
                        "pair": pair_name,
                        "pair_type": "representative",
                        "sequence1_class": representative["sequence1_class"],
                        "sequence1_id": representative["sequence1_id"],
                        "sequence2_class": representative["sequence2_class"],
                        "sequence2_id": representative["sequence2_id"],
                        "dtw_distance": float(representative["dtw_distance"]),
                    },
                ]
            )

        return pair_rows, stats_rows, representative_rows, representative_pair_rows

    def write_branch_outputs(
        self,
        branch_name: str,
        pair_rows: list[dict],
        stats_rows: list[dict],
        representative_rows: list[dict],
        representative_pair_rows: list[dict],
    ) -> None:
        csv_dir = self.cfg.output_root / "csv"
        self.write_csv(
            csv_dir / f"dtw_all_pairs_{branch_name}.csv",
            pair_rows,
            ["branch", "feature_type", "sequence1_class", "sequence1_id", "sequence2_class", "sequence2_id", "relation_type", "dtw_distance"],
        )
        self.write_csv(
            csv_dir / f"dtw_statistics_{branch_name}.csv",
            stats_rows,
            ["branch", "feature_type", "pair", "relation_type", "class_a", "class_b", "count", "mean", "std", "median", "q1", "q3"],
        )
        self.write_csv(
            csv_dir / f"dtw_representative_sequences_{branch_name}.csv",
            representative_rows,
            ["branch", "feature_type", "class", "representative_sequence_id", "centrality_score", "second_best_centrality_score"],
        )
        self.write_csv(
            csv_dir / f"dtw_representative_pairs_{branch_name}.csv",
            representative_pair_rows,
            ["branch", "feature_type", "pair", "pair_type", "sequence1_class", "sequence1_id", "sequence2_class", "sequence2_id", "dtw_distance"],
        )

    def plot_branch_distribution(self, branch_name: str, stats_rows: list[dict], pair_rows: list[dict]) -> None:
        ordered_pairs = [
            ("polite", "polite"),
            ("truesmile", "truesmile"),
            ("ambiguous", "ambiguous"),
            ("polite", "truesmile"),
            ("ambiguous", "truesmile"),
            ("ambiguous", "polite"),
        ]
        grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
        for row in pair_rows:
            pair = tuple(sorted((row["sequence1_class"], row["sequence2_class"])))
            grouped[pair].append(float(row["dtw_distance"]))

        labels = []
        data = []
        for a, b in ordered_pairs:
            pair = tuple(sorted((a, b)))
            if pair in grouped:
                labels.append(f"{pair[0]} vs {pair[1]}")
                data.append(grouped[pair])

        fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
        box = ax.boxplot(data, patch_artist=True, tick_labels=labels, widths=0.55)
        for patch, label in zip(box["boxes"], labels):
            if label == "polite vs polite":
                patch.set_facecolor(COLORS["polite"])
            elif label == "truesmile vs truesmile":
                patch.set_facecolor(COLORS["truesmile"])
            elif label == "ambiguous vs ambiguous":
                patch.set_facecolor(COLORS["ambiguous"])
            elif "truesmile" in label:
                patch.set_facecolor(COLORS["truesmile"])
            elif "ambiguous" in label and "polite" in label:
                patch.set_facecolor(COLORS["ambiguous"])
            else:
                patch.set_facecolor("#999999")
            patch.set_alpha(0.25)
        ax.set_title(f"DTW Distance Distribution: {branch_name}")
        ax.set_ylabel("DTW Distance")
        ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        fig.savefig(self.cfg.output_root / "plots" / f"dtw_distribution_{branch_name}.png")
        plt.close(fig)

    def plot_branch_comparison(self, all_stats_rows: list[dict]) -> None:
        branch_medians: dict[str, dict[str, float]] = defaultdict(dict)
        for row in all_stats_rows:
            branch_medians[row["branch"]][row["pair"]] = float(row["median"])

        ordered_pairs = [
            "polite_vs_polite",
            "truesmile_vs_truesmile",
            "ambiguous_vs_ambiguous",
            "polite_vs_truesmile",
            "ambiguous_vs_truesmile",
            "ambiguous_vs_polite",
        ]
        branches = [b for b, _, _ in self.branches]
        mat = np.zeros((len(branches), len(ordered_pairs)), dtype=np.float64)
        for i, branch in enumerate(branches):
            for j, pair in enumerate(ordered_pairs):
                mat[i, j] = branch_medians.get(branch, {}).get(pair, np.nan)

        fig, ax = plt.subplots(figsize=(11, 6), dpi=150)
        im = ax.imshow(mat, aspect="auto", cmap="YlGnBu")
        ax.set_xticks(range(len(ordered_pairs)))
        ax.set_xticklabels(ordered_pairs, rotation=30, ha="right")
        ax.set_yticks(range(len(branches)))
        ax.set_yticklabels(branches)
        ax.set_title("Median DTW Distance by Branch and Pair")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(self.cfg.output_root / "plots" / "dtw_branch_comparison.png")
        plt.close(fig)

    def build_report(self, all_stats_rows: list[dict], all_rep_rows: list[dict], all_rep_pair_rows: list[dict]) -> str:
        lines = ["# DTW Result Report", ""]
        lines.append("## Similarity Definition")
        lines.append("- In this analysis, we compute **DTW distance**.")
        lines.append("- Smaller DTW distance means higher similarity.")
        lines.append("- Larger DTW distance means lower similarity.")
        lines.append("")
        lines.append("## Brief Summary")

        # summarize closest inter-class pair per branch
        grouped_branch = defaultdict(list)
        for row in all_stats_rows:
            grouped_branch[row["branch"]].append(row)
        for branch, rows in grouped_branch.items():
            inter_rows = [r for r in rows if r["relation_type"] == "inter_class"]
            intra_rows = [r for r in rows if r["relation_type"] == "intra_class"]
            best_inter = min(inter_rows, key=lambda r: float(r["median"]))
            best_intra = min(intra_rows, key=lambda r: float(r["median"]))
            lines.append(
                f"- {branch}: closest inter-class pair by median DTW is {best_inter['pair']} ({float(best_inter['median']):.4f}); "
                f"most compact intra-class pair is {best_intra['pair']} ({float(best_intra['median']):.4f})."
            )
        lines.append("")

        for branch in [b for b, _, _ in self.branches]:
            lines.append(f"## Branch: {branch}")
            stats_rows = [r for r in all_stats_rows if r["branch"] == branch]
            rep_rows = [r for r in all_rep_rows if r["branch"] == branch]
            rep_pair_rows = [r for r in all_rep_pair_rows if r["branch"] == branch]

            lines.append("### Intra-class similarity")
            for pair in ("polite_vs_polite", "truesmile_vs_truesmile", "ambiguous_vs_ambiguous"):
                row = next(r for r in stats_rows if r["pair"] == pair)
                lines.append(
                    f"- {pair}: mean={float(row['mean']):.4f}, median={float(row['median']):.4f}, q1-q3=({float(row['q1']):.4f}, {float(row['q3']):.4f})"
                )

            lines.append("### Inter-class similarity")
            for pair in ("polite_vs_truesmile", "ambiguous_vs_truesmile", "ambiguous_vs_polite"):
                row = next(r for r in stats_rows if r["pair"] == pair)
                lines.append(
                    f"- {pair}: mean={float(row['mean']):.4f}, median={float(row['median']):.4f}, q1-q3=({float(row['q1']):.4f}, {float(row['q3']):.4f})"
                )

            lines.append("### Representative sequence by class")
            for class_name in CLASS_NAMES:
                row = next(r for r in rep_rows if r["class"] == class_name)
                lines.append(
                    f"- {class_name}: representative_sequence_id={row['representative_sequence_id']}, centrality_score={float(row['centrality_score']):.4f}"
                )

            lines.append("### Representative pairs")
            for pair in ("polite_vs_truesmile", "ambiguous_vs_truesmile", "ambiguous_vs_polite"):
                best = next(r for r in rep_pair_rows if r["pair"] == pair and r["pair_type"] == "best_match")
                representative = next(r for r in rep_pair_rows if r["pair"] == pair and r["pair_type"] == "representative")
                lines.append(
                    f"- {pair}: best_match=({best['sequence1_class']}:{best['sequence1_id']} vs {best['sequence2_class']}:{best['sequence2_id']}, {float(best['dtw_distance']):.4f}); "
                    f"representative=({representative['sequence1_class']}:{representative['sequence1_id']} vs {representative['sequence2_class']}:{representative['sequence2_id']}, {float(representative['dtw_distance']):.4f})"
                )
            lines.append("")

        lines.append("## Notes")
        lines.append("- Magnitude DTW uses d(t) = ||f_rel(t)||.")
        lines.append("- Velocity DTW uses v(t) = ||f_rel(t) - f_rel(t-1)||.")
        lines.append("- PCA branches use the original variable-length f_rel sequences projected to 10, 20, or 30 dimensions.")
        lines.append("- Band branches use Sakoe-Chiba constraint with radius = 20% of the longer sequence length.")
        return "\n".join(lines) + "\n"

    def run(self) -> None:
        feature_sequences_by_name = self.build_feature_sequences()
        all_stats_rows: list[dict] = []
        all_rep_rows: list[dict] = []
        all_rep_pair_rows: list[dict] = []

        for branch_name, feature_name, use_band in self.branches:
            pair_rows, stats_rows, representative_rows, representative_pair_rows = self.compute_branch_distances(
                branch_name,
                feature_name,
                use_band,
                feature_sequences_by_name[feature_name],
            )
            self.write_branch_outputs(branch_name, pair_rows, stats_rows, representative_rows, representative_pair_rows)
            self.plot_branch_distribution(branch_name, stats_rows, pair_rows)
            all_stats_rows.extend(stats_rows)
            all_rep_rows.extend(representative_rows)
            all_rep_pair_rows.extend(representative_pair_rows)

        self.write_csv(
            self.cfg.output_root / "csv" / "dtw_representative_sequences_all_branches.csv",
            all_rep_rows,
            ["branch", "feature_type", "class", "representative_sequence_id", "centrality_score", "second_best_centrality_score"],
        )
        self.write_csv(
            self.cfg.output_root / "csv" / "dtw_representative_pairs_all_branches.csv",
            all_rep_pair_rows,
            ["branch", "feature_type", "pair", "pair_type", "sequence1_class", "sequence1_id", "sequence2_class", "sequence2_id", "dtw_distance"],
        )

        self.plot_branch_comparison(all_stats_rows)
        report_text = self.build_report(all_stats_rows, all_rep_rows, all_rep_pair_rows)
        report_path = self.cfg.output_root / "report" / "dtw_result_report.md"
        report_path.write_text(report_text, encoding="utf-8")
        local_result = Path(__file__).resolve().parent / "result.md"
        local_result.write_text(
            "# DTW Result\n\n"
            f"- output_root: {self.cfg.output_root}\n"
            f"- report: {report_path}\n"
            f"- branches: {', '.join([b for b, _, _ in self.branches])}\n",
            encoding="utf-8",
        )
        print(f"[DTW] Finished. Report saved to: {report_path}")


def main() -> None:
    parser = DTWTaskBase.build_common_arg_parser("Run full DTW analysis pipeline.")
    args = parser.parse_args()
    pipeline = DTWPipeline(DTWConfig.from_args(args))
    pipeline.run()


if __name__ == "__main__":
    main()
