from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(n, eps, None)


def plot_scatter(points: np.ndarray, labels: np.ndarray, title: str, save_path: Path) -> None:
    fig = plt.figure(figsize=(8, 6), dpi=140)
    ax = fig.add_subplot(111)
    sc = ax.scatter(points[:, 0], points[:, 1], c=labels, s=28, cmap="tab10")
    ax.set_title(title)
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    legend = ax.legend(*sc.legend_elements(), title="Cluster", loc="best")
    ax.add_artist(legend)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def cluster_resultant_strength(x_unit: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {}
    for c in sorted(np.unique(labels)):
        idx = np.where(labels == c)[0]
        mean_vec = np.mean(x_unit[idx], axis=0)
        # Mean resultant length in [0,1], larger means stronger directional agreement.
        out[str(int(c))] = float(np.linalg.norm(mean_vec))
    return out


def pick_representatives(
    x_unit: np.ndarray,
    labels: np.ndarray,
    top_n: int,
) -> pd.DataFrame:
    rows: list[dict] = []
    for c in sorted(np.unique(labels)):
        idx = np.where(labels == c)[0]
        center = l2_normalize(np.mean(x_unit[idx], axis=0, keepdims=True))[0]
        sims = x_unit[idx] @ center
        ord_idx = idx[np.argsort(-sims)[:top_n]]
        for rank, sample_i in enumerate(ord_idx, start=1):
            rows.append(
                {
                    "cluster": int(c),
                    "rank_in_cluster": rank,
                    "sample_index_filtered": int(sample_i),
                    "cos_to_cluster_center": float(x_unit[sample_i] @ center),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Directional analysis for fc7 diff vectors (vp-vs): filter weak norms and cluster by direction."
    )
    ap.add_argument(
        "--input_pt",
        type=Path,
        default=Path(r"E:\Matsuda_data\2-12meeting\feature_vectors\fc7_pair_diff.pt"),
    )
    ap.add_argument(
        "--out_dir",
        type=Path,
        default=Path(r"E:\Matsuda_data\2-12meeting\analysis_result\directional"),
    )
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--min_norm_quantile", type=float, default=0.2)
    ap.add_argument("--pca_dim", type=int, default=50)
    ap.add_argument("--top_n", type=int, default=12)
    ap.add_argument("--random_state", type=int, default=42)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    data = torch.load(args.input_pt, map_location="cpu")
    diff = data["diff"]
    if torch.is_tensor(diff):
        diff = diff.detach().cpu().numpy()
    diff = np.asarray(diff, dtype=np.float32)

    n, d = diff.shape
    norms = np.linalg.norm(diff, axis=1)
    thr = float(np.quantile(norms, args.min_norm_quantile))
    keep_mask = norms > thr
    keep_idx = np.where(keep_mask)[0]
    if keep_idx.size < args.k + 2:
        raise RuntimeError("Too few samples after norm filtering. Lower --min_norm_quantile.")

    diff_keep = diff[keep_idx]
    norms_keep = norms[keep_idx]
    x_unit = l2_normalize(diff_keep)

    pca_dim = int(min(args.pca_dim, x_unit.shape[0] - 1, x_unit.shape[1]))
    x_pca = PCA(n_components=pca_dim, random_state=args.random_state).fit_transform(x_unit)
    x_pca2 = PCA(n_components=2, random_state=args.random_state).fit_transform(x_unit)

    # Spherical-kmeans approximation: kmeans on l2-normalized vectors.
    km = KMeans(n_clusters=args.k, n_init=30, random_state=args.random_state)
    labels_km = km.fit_predict(x_unit)

    # Cosine agglomerative clustering for directional grouping.
    agg = AgglomerativeClustering(n_clusters=args.k, metric="cosine", linkage="average")
    labels_agg = agg.fit_predict(x_unit)

    metrics = {
        "kmeans_unit": {
            "silhouette_cosine": float(silhouette_score(x_unit, labels_km, metric="cosine")),
            "silhouette_euclidean": float(silhouette_score(x_unit, labels_km, metric="euclidean")),
        },
        "agg_cosine": {
            "silhouette_cosine": float(silhouette_score(x_unit, labels_agg, metric="cosine")),
            "silhouette_euclidean": float(silhouette_score(x_unit, labels_agg, metric="euclidean")),
        },
    }

    meta_df = pd.DataFrame(
        {
            "original_index": np.arange(n, dtype=int),
            "date": data.get("date", [""] * n),
            "segment_folder": data.get("segment_folder", [""] * n),
            "start_image": data.get("start_image", [""] * n),
            "end_image": data.get("end_image", [""] * n),
            "start_ts": data.get("start_ts", [None] * n),
            "end_ts": data.get("end_ts", [None] * n),
            "norm": norms,
            "kept_for_directional": keep_mask,
        }
    )
    kept_df = meta_df.iloc[keep_idx].copy().reset_index(drop=True)
    kept_df["sample_index_filtered"] = np.arange(len(keep_idx), dtype=int)
    kept_df["kmeans_unit_cluster"] = labels_km
    kept_df["agg_cosine_cluster"] = labels_agg
    kept_df["pca2_x"] = x_pca2[:, 0]
    kept_df["pca2_y"] = x_pca2[:, 1]

    kept_df.to_csv(args.out_dir / "directional_cluster_assignments.csv", index=False, encoding="utf-8")
    meta_df.to_csv(args.out_dir / "all_samples_norms_and_filter.csv", index=False, encoding="utf-8")

    rep_km = pick_representatives(x_unit, labels_km, args.top_n).merge(
        kept_df, on="sample_index_filtered", how="left"
    )
    rep_km.to_csv(args.out_dir / "kmeans_unit_representatives.csv", index=False, encoding="utf-8")

    rep_agg = pick_representatives(x_unit, labels_agg, args.top_n).merge(
        kept_df, on="sample_index_filtered", how="left"
    )
    rep_agg.to_csv(args.out_dir / "agg_cosine_representatives.csv", index=False, encoding="utf-8")

    by_date_km = (
        kept_df.groupby(["date", "kmeans_unit_cluster"], dropna=False)
        .size()
        .reset_index(name="count")
        .rename(columns={"kmeans_unit_cluster": "cluster"})
    )
    by_date_km["method"] = "kmeans_unit"

    by_date_agg = (
        kept_df.groupby(["date", "agg_cosine_cluster"], dropna=False)
        .size()
        .reset_index(name="count")
        .rename(columns={"agg_cosine_cluster": "cluster"})
    )
    by_date_agg["method"] = "agg_cosine"

    pd.concat([by_date_km, by_date_agg], ignore_index=True).to_csv(
        args.out_dir / "directional_cluster_counts_by_date.csv", index=False, encoding="utf-8"
    )

    fig = plt.figure(figsize=(8, 5), dpi=140)
    ax = fig.add_subplot(111)
    ax.hist(norms, bins=24, alpha=0.75, color="#4e79a7")
    ax.axvline(thr, color="red", linestyle="--", linewidth=1.2, label=f"q={args.min_norm_quantile:.2f}")
    ax.set_title("Diff Norm Distribution")
    ax.set_xlabel("||vp-vs||")
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out_dir / "diff_norm_histogram.png")
    plt.close(fig)

    plot_scatter(
        x_pca2, labels_km, "Directional Clustering: KMeans on Unit Diff (PCA2)", args.out_dir / "scatter_pca2_kmeans_unit.png"
    )
    plot_scatter(
        x_pca2, labels_agg, "Directional Clustering: Agg Cosine (PCA2)", args.out_dir / "scatter_pca2_agg_cosine.png"
    )

    tsne = TSNE(
        n_components=2,
        perplexity=min(20.0, max(5.0, (len(keep_idx) - 1) / 3)),
        learning_rate="auto",
        init="pca",
        random_state=args.random_state,
    )
    x_tsne = tsne.fit_transform(x_pca)
    plot_scatter(
        x_tsne, labels_km, "Directional Clustering: KMeans on Unit Diff (t-SNE)", args.out_dir / "scatter_tsne_kmeans_unit.png"
    )
    plot_scatter(
        x_tsne, labels_agg, "Directional Clustering: Agg Cosine (t-SNE)", args.out_dir / "scatter_tsne_agg_cosine.png"
    )

    # Inter-method agreement: how similar the two directional clusterings are.
    agree = float(np.mean(labels_km == labels_agg))

    # Cluster direction consistency
    km_strength = cluster_resultant_strength(x_unit, labels_km)
    agg_strength = cluster_resultant_strength(x_unit, labels_agg)

    summary = {
        "input_pt": str(args.input_pt),
        "n_total": int(n),
        "feature_dim": int(d),
        "norm_filter": {
            "min_norm_quantile": float(args.min_norm_quantile),
            "threshold": thr,
            "n_kept": int(len(keep_idx)),
            "n_filtered_out": int(n - len(keep_idx)),
        },
        "clustering": {
            "k": int(args.k),
            "methods": ["kmeans_unit", "agg_cosine"],
            "agreement_rate_raw_label": agree,
            "kmeans_unit_cluster_size": pd.Series(labels_km).value_counts().sort_index().to_dict(),
            "agg_cosine_cluster_size": pd.Series(labels_agg).value_counts().sort_index().to_dict(),
        },
        "metrics": metrics,
        "direction_consistency_mean_resultant_length": {
            "kmeans_unit": km_strength,
            "agg_cosine": agg_strength,
        },
        "mean_cross_cluster_cosine_kmeans_unit": float(
            cosine_similarity(
                l2_normalize(np.vstack([np.mean(x_unit[labels_km == c], axis=0) for c in sorted(np.unique(labels_km))]))
            )[0, 1]
        )
        if args.k == 2
        else None,
    }

    with (args.out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Done.")
    print(f"total={n}, kept={len(keep_idx)}, filtered={n-len(keep_idx)}, threshold={thr:.6f}")
    print(json.dumps(metrics, indent=2))
    print(f"Saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
