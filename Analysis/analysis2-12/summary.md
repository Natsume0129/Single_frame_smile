# analysis2-12 Summary

## 1. 做了哪些事？

本目录围绕「笑容开始帧（vs）与峰值帧（vp）」完成了从数据整理到特征分析的完整流程：

1. 从区间数据提取窗口帧（按 `start-end` 建子文件夹）。
2. 将 `key_frames` 两帧一组的数据整理成结构化清单（manifest）。
3. 复用已有 VGGFace fc7 提取器，提取每对的 `vs` 和 `vp` 特征，并计算差分向量 `diff = vp - vs`。
4. 做基础聚类分析（标准化 + PCA + KMeans/GMM/层次聚类）。
5. 做方向聚类分析（过滤低幅度样本 + 单位向量化 + 方向聚类）。
6. 导出聚类结果、代表样本和可视化图，用于人工检查与报告展示。

---

## 2. 所有代码的功能说明

### `extract_window.py`
- 作用：从 `.dat` 文件读取帧区间（前两列 `start end`），在原始单帧目录中按帧号匹配图片，并按区间复制到输出目录。
- 关键函数：
  - `parse_dat_ranges`：解析区间。
  - `build_frame_index`：建立 `frame_id -> 文件路径` 映射。
  - `extract_windows`：执行窗口提取与统计。
- 输出：`out/<start-end>/` 子目录，包含该区间可匹配到的帧图像。

### `build_keyframe_pairs_csv.py`
- 作用：扫描 `key_frames` 目录下的每个片段子目录（命名 `YYYYMMDD_start-end`），读取其中 2 张图，按时间戳排序得到开始/结束帧。
- 输出：`key_frames_manifest.csv`，包含：
  - `date`
  - `segment_folder`
  - `start_image` / `end_image`
  - `start_ts` / `end_ts`
  - `segment_start` / `segment_end`
  - `pair_span` / `segment_span`
  - `folder_path`

### `extract_fc7_pair_diff.py`
- 作用：读取 manifest，复用 `Analysis/feature_extractor/feature_extractor_fc7.py` 中的模型定义与预处理流程，批量提取 `vs/vp` fc7 特征并计算 `diff = vp - vs`。
- 关键点：
  - `vs`、`vp` 在提取后做了 L2 归一化（`F.normalize`）。
  - `diff` 本身不再归一化。
- 输出到 `feature_vectors`：
  - `fc7_pair_diff.pt`（核心特征数据）
  - `fc7_pair_diff_manifest.csv`
  - `fc7_pair_diff_meta.json`

### `analyze_fc7_pair_diff.py`
- 作用：对 `diff` 做基础聚类分析。
- 方法流程：
  - `StandardScaler`
  - PCA 降维（聚类使用 `pca_dim`，默认 50）
  - 聚类：`KMeans`、`GaussianMixture`、`Agglomerative(Ward)`
  - 指标：`silhouette`、`Calinski-Harabasz`、`Davies-Bouldin`
  - 可视化：PCA2 与 t-SNE 散点图
- 输出到 `analysis_result`：
  - `summary.json`
  - `cluster_assignments.csv`
  - `cluster_counts_by_date.csv`
  - `kmeans_representatives.csv`
  - `pca_explained_variance.csv`
  - `pca_variance_curve.png`
  - 多张 `scatter_*.png`

### `analyze_fc7_direction.py`
- 作用：针对“真笑可能方向一致”的假设，做方向型聚类分析。
- 方法流程：
  - 计算每个 `diff` 的范数，按分位点过滤低幅度样本（默认去掉底部 20%）
  - 对保留样本做单位向量化（只分析方向）
  - 聚类：
    - `kmeans_unit`（单位向量上的 KMeans，近似 spherical kmeans）
    - `agg_cosine`（余弦距离 + average linkage）
  - 指标：
    - `silhouette_cosine` / `silhouette_euclidean`
    - 簇内方向一致性（mean resultant length）
- 输出到 `analysis_result/directional`：
  - `summary.json`
  - `directional_cluster_assignments.csv`
  - `all_samples_norms_and_filter.csv`
  - `directional_cluster_counts_by_date.csv`
  - `kmeans_unit_representatives.csv`
  - `agg_cosine_representatives.csv`
  - `diff_norm_histogram.png`
  - `scatter_pca2_*.png`
  - `scatter_tsne_*.png`

### `analyze_fc7_direction.py` 之外的聚类导出（执行过程产物）
- 已按 `kmeans_unit` 导出两个簇的清单与图片分组（用于人工检查）：
  - `.../directional/kmeans_unit_exports/kmeans_unit_cluster_0.csv`
  - `.../directional/kmeans_unit_exports/kmeans_unit_cluster_1.csv`
  - `.../directional/kmeans_unit_exports/report_images_split/cluster_0`
  - `.../directional/kmeans_unit_exports/report_images_split/cluster_1`

### `log.dat`
- 作用：实验过程的日志/中间记录文件（非可执行脚本）。

---

## 3. 结论分析

### 3.1 基础聚类（`analyze_fc7_pair_diff.py`）
- 标准化后，`KMeans/GMM/Agglomerative` 在 `k=2` 下都接近「一个主簇 + 极少离群点」。
- 这说明在当前 `diff` 特征空间中，数据没有形成非常强的天然二分结构。

### 3.2 方向聚类（`analyze_fc7_direction.py`）
- 在过滤低幅度样本并单位化后，`kmeans_unit` 得到相对可用的二分结果（例如 59/37），可用于人工筛查。
- `agg_cosine` 仍更偏向「主簇 + 小簇异常」。
- PCA2 图像出现“中间像被切开”而不是“两团明显分离”是正常现象：
  - 聚类在高维方向空间完成；
  - 2D 图只是投影，视觉上会呈现线性切分而非紧凑簇。

### 3.3 当前阶段可落地结论
- 就无监督结果而言，**目前最实用的是 `kmeans_unit` 作为分组工具**，用于把样本先拆成两组做人工复核。
- 现阶段还不能仅凭聚类结果直接命名为“真笑/非真笑”；更稳妥做法是对每簇抽样进行人工标注，再验证映射关系。

