# analysis_sequence 代码说明（中文）

## 1. 设计原则
- 采用“一个脚本一个子任务”的结构，不用单脚本完成全部流程。
- 把共同逻辑抽到基类，所有 step 脚本继承同一套能力。
- 输入目录默认：`E:\Matsuda_data\2-18meeting`
- 输出目录默认：`E:\Matsuda_data\2-27meeting`

---

## 2. 基类与公共能力

### `common/base.py`
主要提供：
- `PipelineConfig`：统一配置（输入/输出目录、权重路径、fps、归一化长度、设备等）。
- `SequenceTaskBase`：所有 step 的基类，包含：
  - 自动创建输出主目录：`prototypes/`、`metrics/`、`plots/`、`csv/`、`report/`
  - 统一发现序列：只扫描 `polite/truesmile/ambiguous`，自动忽略 `videos/video`
  - 统一帧排序：从文件名末尾整数提取帧号并按数值排序
  - 统一 I/O：`npy/json/csv` 的读写
  - 统一每序列输出路径管理

---

## 3. 分步脚本说明

### `01_extract_features.py`
- 功能：对每个片段提取 VGG-Face `fc7`（4096d）特征。
- 复用：`Analysis/feature_extractor/feature_extractor_fc7.py` 的模型和预处理。
- 输出：
  - `metrics/sequence_features/<class>/<seq>/sequence_features.npy`（`[T, D]`）
  - `metrics/sequence_features/<class>/<seq>/frame_names.json`

### `02_baseline_align.py`
- 功能：做 baseline 对齐，`f0 = mean(f[0:5])`，`f_rel = f - f0`。
- 输出：
  - `metrics/sequence_features_rel/<class>/<seq>/sequence_features_rel.npy`
  - `metrics/sequence_features_rel/<class>/<seq>/baseline_f0.npy`

### `03_compute_magnitude.py`
- 功能：计算 `d(t)=||f_rel(t)||`，并统计峰值/均值/标准差。
- 输出：
  - `metrics/distance/<class>/<seq>/distance_curve.npy`
  - `metrics/distance/<class>/<seq>/metrics.json`

### `04_compute_velocity.py`
- 功能：计算 `v(0)=0, v(t)=||f_rel(t)-f_rel(t-1)||`，并统计速度指标。
- 输出：
  - `metrics/velocity/<class>/<seq>/velocity_curve.npy`
  - `metrics/velocity/<class>/<seq>/metrics.json`

### `05_compute_duration_stats.py`
- 功能：按 `duration_sec = frames / fps` 计算时长统计（非 onset）。
- 输出：
  - `csv/duration_stats.csv`（每类平均时长）
  - `csv/duration_per_sequence.csv`（每条序列时长明细）

### `06_time_normalize.py`
- 功能：线性插值重采样到固定长度 `N=20`，并复制重命名帧图。
- 输出：
  - `metrics/normalized/<class>/<seq>/normalized_sequence.npy`（`[20,D]`）
  - `metrics/normalized/<class>/<seq>/distance_norm.npy`（`[20]`）
  - `metrics/normalized/<class>/<seq>/velocity_norm.npy`（`[20]`）
  - `metrics/normalized/<class>/<seq>/sampled_frames.json`（采样映射）
  - `metrics/normalized_frames/<class>/<seq>/000.png ... 019.png`

### `07_build_prototypes.py`
- 功能：
  - 必选：按类计算 median prototype（逐维中位数）
  - 补充：用 Frobenius 距离计算 medoid prototype
- 输出：
  - `prototypes/prototype_polite.npy`
  - `prototypes/prototype_truesmile.npy`
  - `prototypes/prototype_ambiguous.npy`
  - `prototypes/prototype_<class>_medoid.npy`
  - `prototypes/prototype_meta.json`

### `08_class_difference_vectors.py`
- 功能：计算三组类间差分向量（`[20,D]`）。
- 输出：
  - `prototypes/class_difference_vectors.npy`（dict，3 个键）
    - `polite_vs_truesmile`
    - `polite_vs_ambiguous`
    - `truesmile_vs_ambiguous`

### `09_segment_vectors.py`
- 功能：仅对每类 prototype 计算 segment vector：`f(t+1)-f(t)`。
- 输出：
  - `prototypes/segment_vectors.npy`（dict，键为三类名）

### `10_projection_pca.py`
- 功能：使用“两段式 PCA”将轨迹按“每个时间点投影后连线”的方式映射到 2D。
- 方法：
  - `StandardScaler` 标准化
  - 第一段 PCA：`4096 -> mid_dim`（默认 `50`，可通过 `--pca_mid_dim` 调整）
  - 第二段 PCA：`mid_dim -> 2` 用于可视化
- 输出：
  - `plots/trajectory_plot.png`
  - `plots/trajectory_plot_polite.png`
  - `plots/trajectory_plot_ambiguous.png`
  - `plots/trajectory_plot_truesmile.png`
  - `plots/trajectory_plot_cross.png`
  - `prototypes/pca_model_2d.npz`（标准化器 + 两段 PCA 参数）

### `11_class_distance_curve.py`
- 功能：计算三组 prototype 的时间距离曲线 `||A(t)-B(t)||` 并标记峰值时刻。
- 输出：
  - `csv/class_distance_curve.csv`

### `12_projection_scores.py`
- 功能：对每条样本、每个时间点、每个类对轴计算投影分数 `dot(f_norm(t), Δ_pair(t))`。
- 输出：
  - `csv/projection_scores.csv`

### `13_generate_visualizations.py`
- 功能：生成需求中规定的主要图表。
- 输出：
  - `plots/mean_magnitude_curve.png`
  - `plots/mean_velocity_curve.png`
  - `plots/class_distance_over_time.png`
  - `plots/duration_distribution.png`
  - （轨迹图由 `10_projection_pca.py` 生成）
  - `csv/mean_magnitude_curve.csv`
  - `csv/mean_velocity_curve.csv`

### `14_generate_dataset_report.py`
- 功能：生成最终数据报告。
- 输出：
  - `csv/dataset_report.csv`
  - 字段：`class, sequence_id, frames, duration_sec, peak_magnitude, mean_velocity`

---

## 4. 串行执行方式

### `run_pipeline.ps1`
- 功能：按步骤顺序调用 `01` 到 `14`。
- 用法：在 `analysis_sequence` 目录执行

```powershell
powershell -ExecutionPolicy Bypass -File .\run_pipeline.ps1
```

---

## 5. 最终产出在哪里
- 根目录：`E:\Matsuda_data\2-27meeting`
- 子目录：
  - `prototypes/`
  - `metrics/`
  - `plots/`
  - `csv/`
  - `report/`

> 说明：中间过程文件主要在 `metrics/`，最终分析表格在 `csv/`，图在 `plots/`，原型与差分向量在 `prototypes/`。
