# DTW_resample TODO

## 1. 目标

本任务的目标是：

- 先在类内使用 DTW 找到最中心的真实序列（DTW medoid）
- 再把该类别内所有其他序列通过 DTW 对齐到这个中心序列
- 最后把对齐后的结果统一重采样到 20 个点

这样得到：

- polite 类内，对齐到 polite representative sequence 的 20 点序列
- truesmile 类内，对齐到 truesmile representative sequence 的 20 点序列
- ambiguous 类内，对齐到 ambiguous representative sequence 的 20 点序列

这个方案对应我们讨论中的“问题 3 的方案 B”。

---

## 2. 使用的数据文件与目录

### 2.1 主要输入数据

#### 原始序列特征

使用：

```text
E:\Matsuda_data\2-27meeting\metrics\sequence_features_rel\<class>\<seq>\sequence_features_rel.npy
```

说明：

- 这是 baseline 对齐之后的原始长度序列
- 不使用已经重采样到 20 点的 `normalized_sequence.npy`
- 因为 DTW 本身就是用来处理不同长度时序对齐的

#### 原始帧顺序信息

使用：

```text
E:\Matsuda_data\2-27meeting\metrics\sequence_features\<class>\<seq>\frame_names.json
```

说明：

- 用来知道原始每个时间点对应的是哪一帧

#### 类别列表

当前类别：

- `polite`
- `truesmile`
- `ambiguous`

### 2.2 可复用的现有结果

可以参考和复用：

#### analysis_sequence

```text
E:\Single_frame_smile\Analysis\analysis_sequence
```

尤其是：

- `02_baseline_align.py`
- `06_time_normalize.py`
- `07_build_prototypes.py`

#### DTW

```text
E:\Single_frame_smile\Analysis\DTW
```

尤其是：

- `dtw_common.py`
- `run_dtw_pipeline.py`

因为这里已经实现了：

- pairwise DTW distance
- class-wise DTW medoid
- Sakoe-Chiba band

---

## 3. 需要做的步骤

## Step 1. 读取类内所有原始序列

### 1）怎么做

对每个类别：

- 读取该类别下所有 `sequence_features_rel.npy`
- 每条序列记为：

```text
S_i = [f_i(0), f_i(1), ..., f_i(T_i-1)]
```

其中：

- `T_i` 可以不同
- 每个 `f_i(t)` 是高维向量

### 2）公式

记类别 `c` 中的第 `i` 条序列为：

```text
S_i^c = { f_i^c(t) } ,  t = 0, 1, ..., T_i^c - 1
```

### 3）输出结果和储存目录

这一步不一定单独输出文件，但应在中间缓存里保留：

- `class`
- `sequence_id`
- `sequence length`
- `frame_names`

---

## Step 2. 在类内构建 DTW 距离矩阵

### 1）怎么做

对每个类别内部的所有序列，两两计算 DTW distance。

要求：

- 使用 DTW
- 使用 Sakoe-Chiba band
- band 宽度设为序列长度的 `20%`

### 2）公式

对同一类别中的两条序列：

```text
D(i, j) = DTW(S_i, S_j)
```

这里 `D` 是类内 DTW 距离矩阵。

### 3）输出结果和储存目录

建议输出：

```text
E:\Matsuda_data\DTW_resample_output\csv\intra_class_dtw_matrix_<class>.csv
```

字段或矩阵内容：

- 行：sequence id
- 列：sequence id
- 值：DTW distance

---

## Step 3. 选择类内 representative sequence

### 1）怎么做

在类内 DTW 距离矩阵中，找出总 DTW cost 最小的真实序列。

这条序列作为该类别的中心序列。

### 2）公式

对于类别 `c` 中的每条序列：

```text
cost_i = Σ_j D(i, j)
```

代表序列定义为：

```text
i* = argmin_i cost_i
```

### 3）输出结果和储存目录

建议输出：

```text
E:\Matsuda_data\DTW_resample_output\csv\representative_sequences.csv
```

字段建议：

```text
class
representative_sequence_id
centrality_score
second_best_centrality_score
```

---

## Step 4. 将类内所有序列对齐到 representative sequence

### 1）怎么做

对每个类别：

- 固定该类别的 representative sequence 为 reference
- 对类内每条序列与 reference 做 DTW
- 得到 warping path

然后利用 warping path，把原始序列映射到 reference 的时间轴上。

如果多个 source 点映射到同一个 reference 时间点，先取平均。

### 2）公式

对类别 `c` 中的一条序列 `S_i^c` 和代表序列 `S_ref^c`：

```text
P_i = DTW_path(S_i^c, S_ref^c)
```

其中 `P_i` 是 warping path。

然后把 `S_i^c` 通过 `P_i` 映射到 reference 时间轴，得到：

```text
\tilde{S}_i^c
```

这是一条“已经对齐到代表序列时间轴”的序列。

### 3）输出结果和储存目录

建议输出：

```text
E:\Matsuda_data\DTW_resample_output\csv\dtw_alignment_paths_<class>.csv
```

字段建议：

```text
class
sequence_id
reference_sequence_id
reference_time_index
source_time_index
```

以及：

```text
E:\Matsuda_data\DTW_resample_output\metrics\aligned_to_representative\<class>\<seq>\aligned_sequence.npy
```

---

## Step 5. 将对齐后的序列统一重采样到 20 个点

### 1）怎么做

对 `\tilde{S}_i^c` 再做一次统一重采样，长度固定为 20。

注意：

- 这里不是直接对原始序列重采样
- 而是对“已经对齐到类内代表序列时间轴”的序列再重采样

### 2）公式

设对齐后的序列为：

```text
\tilde{S}_i^c = { \tilde{f}_i^c(t) }
```

再重采样得到：

```text
\hat{S}_i^c = { \hat{f}_i^c(k) },  k = 0, 1, ..., 19
```

### 3）输出结果和储存目录

建议输出：

```text
E:\Matsuda_data\DTW_resample_output\metrics\resampled20_aligned\<class>\<seq>\aligned_resampled20.npy
```

同时建议输出 mapping：

```text
E:\Matsuda_data\DTW_resample_output\metrics\resampled20_aligned\<class>\<seq>\alignment_mapping.json
```

字段建议：

```text
resampled_index
reference_time_index
source_time_indices
```

---

## Step 6. 对新的 20 点序列做后续扩展分析

### 1）怎么做

对新的 `aligned_resampled20.npy`，可以继续做：

- prototype 构建
- 类内 / 类间距离分析
- 主轴分析
- minimum distance 分析
- 其他后续扩展

### 2）公式

新的分析对象是：

```text
\hat{S}_i^c = { \hat{f}_i^c(k) },  k = 0, 1, ..., 19
```

后续所有与 20 点标准序列相关的分析，都可以在这个新空间中进行。

### 3）输出结果和储存目录

建议把后续分析统一放在：

```text
E:\Matsuda_data\DTW_resample_output\analysis\
```

例如：

```text
csv\
plots\
report\
```

---

## 4. 关键实现细节

### 4.1 DTW 设置

- 使用 `DTW`
- 使用 `Sakoe-Chiba band`
- band 宽度 = `20%`

### 4.2 representative sequence 的性质

- 必须是真实序列
- 不是合成轨迹

### 4.3 对齐后的聚合方式

如果多个 source 点对应同一个 reference 时间点：

- 先用 `mean`

作为第一版实现。

### 4.4 参考时间轴

类内所有序列都对齐到该类别的 representative sequence 的时间轴。

### 4.5 最终统一长度

对齐完成之后，再统一重采样到 20 点。

---

## 5. 输出目录

统一输出到：

```text
E:\Matsuda_data\DTW_resample_output
```

建议结构：

```text
csv\
metrics\
plots\
report\
```

---

## 6. 最终会得到什么

最终我们会得到：

1. 每个类别的 DTW representative sequence
2. 每条原始序列相对类内中心序列的 DTW 对齐结果
3. 每条序列对齐后再重采样到 20 点的标准表示
4. 一个新的、基于“类内 DTW 对齐后重采样”的时序分析基础数据集

这会比直接线性重采样更自然，因为：

- 它先尊重了各序列的动态节奏差异
- 再在类内时间轴上做统一表达

