from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from common.base import CLASS_NAMES, PipelineConfig, SequenceTaskBase


COLORS = {"polite": "#1f77b4", "truesmile": "#2ca02c", "ambiguous": "#ff7f0e"}


class ProjectionPCATask(SequenceTaskBase):
    def __init__(self, config: PipelineConfig, pca_mid_dim: int):
        super().__init__(config)
        self.pca_mid_dim = int(pca_mid_dim)

    def _collect_norm_sequences(self) -> dict[str, list[tuple[str, np.ndarray]]]:
        by_class: dict[str, list[tuple[str, np.ndarray]]] = {c: [] for c in CLASS_NAMES}
        for seq in self.discover_sequences():
            arr = self.load_npy(self.metrics_seq_dir("normalized", seq) / "normalized_sequence.npy")
            by_class[seq.class_name].append((seq.sequence_id, arr))
        return by_class

    def _collect_points_for_fit(self, by_class: dict[str, list[tuple[str, np.ndarray]]]) -> np.ndarray:
        points = []
        for class_items in by_class.values():
            for _, arr in class_items:
                points.append(arr)
        for c in CLASS_NAMES:
            points.append(np.load(self.cfg.output_root / "prototypes" / f"prototype_{c}.npy"))
        return np.concatenate(points, axis=0)

    @staticmethod
    def _project_track(track: np.ndarray, scaler: StandardScaler, pca_mid: PCA, pca_2d: PCA) -> np.ndarray:
        x = scaler.transform(track)
        x = pca_mid.transform(x)
        return pca_2d.transform(x)

    def _plot_class_only(
        self,
        class_name: str,
        by_class: dict[str, list[tuple[str, np.ndarray]]],
        scaler: StandardScaler,
        pca_mid: PCA,
        pca_2d: PCA,
        save_path,
    ) -> None:
        fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
        for _, arr in by_class[class_name]:
            xy = self._project_track(arr, scaler, pca_mid, pca_2d)
            ax.plot(xy[:, 0], xy[:, 1], color=COLORS[class_name], alpha=0.25, linewidth=1.0)
        proto = np.load(self.cfg.output_root / "prototypes" / f"prototype_{class_name}.npy")
        proto_xy = self._project_track(proto, scaler, pca_mid, pca_2d)
        ax.plot(proto_xy[:, 0], proto_xy[:, 1], color=COLORS[class_name], linewidth=3.0, label=f"{class_name} prototype")
        ax.scatter(proto_xy[0, 0], proto_xy[0, 1], color="black", s=24, marker="o", label="start")
        ax.scatter(proto_xy[-1, 0], proto_xy[-1, 1], color="black", s=24, marker="x", label="end")
        ax.set_title(f"{class_name} trajectories + prototype (PCA2)")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(save_path)
        plt.close(fig)

    def run(self) -> None:
        by_class = self._collect_norm_sequences()
        fit_points = self._collect_points_for_fit(by_class)
        scaler = StandardScaler()
        fit_points_scaled = scaler.fit_transform(fit_points)

        mid_dim = int(min(self.pca_mid_dim, fit_points_scaled.shape[0] - 1, fit_points_scaled.shape[1]))
        if mid_dim < 2:
            raise RuntimeError("Not enough data to run two-stage PCA. Need at least 2 effective dimensions.")

        pca_mid = PCA(n_components=mid_dim, random_state=42)
        fit_mid = pca_mid.fit_transform(fit_points_scaled)

        pca_2d = PCA(n_components=2, random_state=42)
        pca_2d.fit(fit_mid)

        np.savez(
            self.cfg.output_root / "prototypes" / "pca_model_2d.npz",
            scaler_mean=scaler.mean_.astype(np.float32),
            scaler_scale=scaler.scale_.astype(np.float32),
            pca_mid_components=pca_mid.components_.astype(np.float32),
            pca_mid_mean=pca_mid.mean_.astype(np.float32),
            pca_mid_explained_variance_ratio=pca_mid.explained_variance_ratio_.astype(np.float32),
            pca_2d_components=pca_2d.components_.astype(np.float32),
            pca_2d_mean=pca_2d.mean_.astype(np.float32),
            pca_2d_explained_variance_ratio=pca_2d.explained_variance_ratio_.astype(np.float32),
            pca_mid_dim=np.array([mid_dim], dtype=np.int32),
        )

        plots_dir = self.cfg.output_root / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(9, 7), dpi=150)
        for class_name in CLASS_NAMES:
            for _, arr in by_class[class_name]:
                xy = self._project_track(arr, scaler, pca_mid, pca_2d)
                ax.plot(xy[:, 0], xy[:, 1], color=COLORS[class_name], alpha=0.15, linewidth=0.9)
            proto = np.load(self.cfg.output_root / "prototypes" / f"prototype_{class_name}.npy")
            pxy = self._project_track(proto, scaler, pca_mid, pca_2d)
            ax.plot(pxy[:, 0], pxy[:, 1], color=COLORS[class_name], linewidth=2.4, label=f"{class_name} prototype")
        ax.set_title("All trajectories with prototypes (PCA2)")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(plots_dir / "trajectory_plot.png")
        plt.close(fig)

        self._plot_class_only("polite", by_class, scaler, pca_mid, pca_2d, plots_dir / "trajectory_plot_polite.png")
        self._plot_class_only("ambiguous", by_class, scaler, pca_mid, pca_2d, plots_dir / "trajectory_plot_ambiguous.png")
        self._plot_class_only(
            "truesmile", by_class, scaler, pca_mid, pca_2d, plots_dir / "trajectory_plot_truesmile.png"
        )

        fig2, ax2 = plt.subplots(figsize=(8, 6), dpi=150)
        for class_name in CLASS_NAMES:
            proto = np.load(self.cfg.output_root / "prototypes" / f"prototype_{class_name}.npy")
            pxy = self._project_track(proto, scaler, pca_mid, pca_2d)
            ax2.plot(pxy[:, 0], pxy[:, 1], color=COLORS[class_name], linewidth=2.6, label=class_name)
            ax2.scatter(pxy[0, 0], pxy[0, 1], color=COLORS[class_name], s=18)
            ax2.scatter(pxy[-1, 0], pxy[-1, 1], color=COLORS[class_name], s=18, marker="x")
        ax2.set_title("Prototype trajectories only (PCA2)")
        ax2.set_xlabel("PC1")
        ax2.set_ylabel("PC2")
        ax2.legend(loc="best")
        fig2.tight_layout()
        fig2.savefig(plots_dir / "trajectory_plot_cross.png")
        plt.close(fig2)

        print(f"[STEP10] Saved PCA trajectory plots (two-stage PCA: {fit_points.shape[1]} -> {mid_dim} -> 2).")


def main() -> None:
    parser = SequenceTaskBase.build_common_arg_parser("Step 10: PCA trajectory projection.")
    parser.add_argument("--pca_mid_dim", type=int, default=50)
    args = parser.parse_args()
    task = ProjectionPCATask(PipelineConfig.from_args(args), pca_mid_dim=args.pca_mid_dim)
    task.run()


if __name__ == "__main__":
    main()
