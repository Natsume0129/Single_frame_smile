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
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler


def plot_scatter(
    points: np.ndarray,
    labels: np.ndarray,
    title: str,
    x_label: str,
    y_label: str,
    save_path: Path,
) -> None:
    fig = plt.figure(figsize=(8, 6), dpi=140)
    ax = fig.add_subplot(111)
    scatter = ax.scatter(points[:, 0], points[:, 1], c=labels, s=24, cmap="tab10")
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    legend = ax.legend(*scatter.legend_elements(), title="Cluster", loc="best")
    ax.add_artist(legend)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def collect_metrics(x: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    return {
        "silhouette": float(silhouette_score(x, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(x, labels)),
        "davies_bouldin": float(davies_bouldin_score(x, labels)),
    }


def representatives_for_kmeans(
    x: np.ndarray, labels: np.ndarray, centers: np.ndarray, top_n: int
) -> pd.DataFrame:
    rows: list[dict] = []
    for cluster_id in sorted(np.unique(labels)):
        idx = np.where(labels == cluster_id)[0]
        d = np.linalg.norm(x[idx] - centers[cluster_id], axis=1)
        order = idx[np.argsort(d)[:top_n]]
        for rank, pair_idx in enumerate(order, start=1):
            rows.append(
                {
                    "cluster": int(cluster_id),
                    "rank_in_cluster": rank,
                    "pair_index": int(pair_idx),
                    "distance_to_center": float(np.linalg.norm(x[pair_idx] - centers[cluster_id])),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cluster analysis for fc7 pair diff vectors.")
    parser.add_argument(
        "--input_pt",
        type=Path,
        default=Path(r"E:\Matsuda_data\2-12meeting\feature_vectors\fc7_pair_diff.pt"),
        help="Input .pt with key 'diff'.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path(r"E:\Matsuda_data\2-12meeting\analysis_result"),
        help="Output directory.",
    )
    parser.add_argument("--k", type=int, default=2, help="Number of clusters.")
    parser.add_argument("--pca_dim", type=int, default=50, help="PCA dimension before clustering.")
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--top_n", type=int, default=10, help="Top representatives per cluster.")
    parser.add_argument("--tsne_perplexity", type=float, default=20.0)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    data = torch.load(args.input_pt, map_location="cpu")
    if "diff" not in data:
        raise RuntimeError("Input .pt does not contain 'diff'.")

    x = data["diff"]
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    x = np.asarray(x, dtype=np.float32)
    n_samples, n_features = x.shape
    if n_samples < 3:
        raise RuntimeError("Need at least 3 samples for clustering metrics.")

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    pca_full = PCA(svd_solver="full", random_state=args.random_state)
    pca_full.fit(x_scaled)
    cumsum = np.cumsum(pca_full.explained_variance_ratio_)
    n95 = int(np.searchsorted(cumsum, 0.95) + 1)
    pca_dim = int(min(args.pca_dim, n_samples - 1, n_features))
    x_pca = PCA(n_components=pca_dim, random_state=args.random_state).fit_transform(x_scaled)
    x_pca2 = PCA(n_components=2, random_state=args.random_state).fit_transform(x_scaled)

    kmeans = KMeans(n_clusters=args.k, random_state=args.random_state, n_init=20)
    labels_kmeans = kmeans.fit_predict(x_pca)

    gmm = GaussianMixture(n_components=args.k, covariance_type="full", random_state=args.random_state)
    labels_gmm = gmm.fit_predict(x_pca)

    agg = AgglomerativeClustering(n_clusters=args.k, linkage="ward")
    labels_agg = agg.fit_predict(x_pca)

    metrics = {
        "kmeans": collect_metrics(x_pca, labels_kmeans),
        "gmm": collect_metrics(x_pca, labels_gmm),
        "agglomerative": collect_metrics(x_pca, labels_agg),
    }

    manifest_df = pd.DataFrame(
        {
            "pair_index": np.arange(n_samples, dtype=int),
            "date": data.get("date", [""] * n_samples),
            "segment_folder": data.get("segment_folder", [""] * n_samples),
            "segment_start": data.get("segment_start", [None] * n_samples),
            "segment_end": data.get("segment_end", [None] * n_samples),
            "start_image": data.get("start_image", [""] * n_samples),
            "end_image": data.get("end_image", [""] * n_samples),
            "start_ts": data.get("start_ts", [None] * n_samples),
            "end_ts": data.get("end_ts", [None] * n_samples),
        }
    )
    manifest_df["kmeans_cluster"] = labels_kmeans
    manifest_df["gmm_cluster"] = labels_gmm
    manifest_df["agg_cluster"] = labels_agg
    manifest_df["pca2_x"] = x_pca2[:, 0]
    manifest_df["pca2_y"] = x_pca2[:, 1]

    assignments_csv = args.out_dir / "cluster_assignments.csv"
    manifest_df.to_csv(assignments_csv, index=False, encoding="utf-8")

    date_cluster_rows: list[pd.DataFrame] = []
    for method in ["kmeans_cluster", "gmm_cluster", "agg_cluster"]:
        grouped = (
            manifest_df.groupby(["date", method], dropna=False)
            .size()
            .reset_index(name="count")
            .rename(columns={method: "cluster"})
        )
        grouped["method"] = method.replace("_cluster", "")
        date_cluster_rows.append(grouped[["method", "date", "cluster", "count"]])
    counts_by_date = pd.concat(date_cluster_rows, ignore_index=True)
    counts_by_date.to_csv(args.out_dir / "cluster_counts_by_date.csv", index=False, encoding="utf-8")

    rep_df = representatives_for_kmeans(x_pca, labels_kmeans, kmeans.cluster_centers_, top_n=args.top_n)
    rep_df = rep_df.merge(manifest_df, on="pair_index", how="left")
    rep_df.to_csv(args.out_dir / "kmeans_representatives.csv", index=False, encoding="utf-8")

    var_df = pd.DataFrame(
        {
            "pc_index": np.arange(1, len(pca_full.explained_variance_ratio_) + 1),
            "explained_variance_ratio": pca_full.explained_variance_ratio_,
            "cumulative_explained_variance": cumsum,
        }
    )
    var_df.to_csv(args.out_dir / "pca_explained_variance.csv", index=False, encoding="utf-8")

    fig = plt.figure(figsize=(8, 5), dpi=140)
    ax = fig.add_subplot(111)
    ax.plot(var_df["pc_index"], var_df["cumulative_explained_variance"], linewidth=1.6)
    ax.axhline(0.95, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Number of Principal Components")
    ax.set_ylabel("Cumulative Explained Variance")
    ax.set_title("PCA Cumulative Explained Variance")
    fig.tight_layout()
    fig.savefig(args.out_dir / "pca_variance_curve.png")
    plt.close(fig)

    plot_scatter(
        x_pca2,
        labels_kmeans,
        "KMeans (k=2) on Scaled Diff",
        "PCA1",
        "PCA2",
        args.out_dir / "scatter_pca2_kmeans.png",
    )
    plot_scatter(
        x_pca2,
        labels_gmm,
        "GMM (k=2) on Scaled Diff",
        "PCA1",
        "PCA2",
        args.out_dir / "scatter_pca2_gmm.png",
    )
    plot_scatter(
        x_pca2,
        labels_agg,
        "Agglomerative Ward (k=2) on Scaled Diff",
        "PCA1",
        "PCA2",
        args.out_dir / "scatter_pca2_agglomerative.png",
    )

    tsne = TSNE(
        n_components=2,
        perplexity=min(args.tsne_perplexity, max(5.0, (n_samples - 1) / 3)),
        learning_rate="auto",
        init="pca",
        random_state=args.random_state,
    )
    x_tsne = tsne.fit_transform(x_pca)
    plot_scatter(
        x_tsne,
        labels_kmeans,
        "KMeans (k=2) t-SNE",
        "tSNE1",
        "tSNE2",
        args.out_dir / "scatter_tsne_kmeans.png",
    )
    plot_scatter(
        x_tsne,
        labels_gmm,
        "GMM (k=2) t-SNE",
        "tSNE1",
        "tSNE2",
        args.out_dir / "scatter_tsne_gmm.png",
    )
    plot_scatter(
        x_tsne,
        labels_agg,
        "Agglomerative Ward (k=2) t-SNE",
        "tSNE1",
        "tSNE2",
        args.out_dir / "scatter_tsne_agglomerative.png",
    )

    summary = {
        "input_pt": str(args.input_pt),
        "n_samples": int(n_samples),
        "n_features": int(n_features),
        "scaler": {
            "mean_abs_avg": float(np.mean(np.abs(scaler.mean_))),
            "scale_avg": float(np.mean(scaler.scale_)),
        },
        "pca": {
            "n95": n95,
            "pca_dim_used_for_clustering": pca_dim,
            "explained_variance_sum_pca_dim": float(
                np.sum(var_df["explained_variance_ratio"].values[:pca_dim])
            ),
        },
        "cluster_size": {
            "kmeans": pd.Series(labels_kmeans).value_counts().sort_index().to_dict(),
            "gmm": pd.Series(labels_gmm).value_counts().sort_index().to_dict(),
            "agglomerative": pd.Series(labels_agg).value_counts().sort_index().to_dict(),
        },
        "metrics": metrics,
        "outputs": {
            "cluster_assignments_csv": str(assignments_csv),
            "cluster_counts_by_date_csv": str(args.out_dir / "cluster_counts_by_date.csv"),
            "kmeans_representatives_csv": str(args.out_dir / "kmeans_representatives.csv"),
        },
    }
    with (args.out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Done.")
    print(f"samples={n_samples}, feature_dim={n_features}")
    print(f"PCA n95={n95}, clustering_pca_dim={pca_dim}")
    print("Metrics:")
    print(json.dumps(metrics, indent=2))
    print(f"Saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
