# DTW 分析需求文档（中文）

## 1. 目标

在现有 smile trajectory 分析的基础上，新增一套基于 DTW（Dynamic Time Warping）的时间序列相似度分析。

本分析的主要目标是：

1. 计算不同笑容类别的**类内相似度**
2. 计算不同笑容类别的**类间相似度**
3. 比较不同类别在动态模式上的接近程度
4. 为后续判断：
   - `ambiguous` 更像 `polite` 还是更像 `truesmile`
   - intentional polite smile 是否更接近 polite smile 的动态模式

提供新的分析依据。

---

## 2. 基本思路

与此前的点对距离、主轴投影和偏离分析不同，DTW 的重点在于：

- 允许两条时间序列在时间上存在错位
- 通过动态对齐来衡量整体时序模式是否相似

因此，DTW 更适合回答：

- 两条 smile trajectory 的动态演化模式是否相似

而不仅仅是：

- 在同一个时间点上，它们的点位有多接近

---

## 3. 输入数据

### 3.1 数据来源

继续复用已有分析结果：

- `E:\Matsuda_data\2-27meeting`

重点使用：

- `metrics/sequence_features_rel/<class>/<seq>/sequence_features_rel.npy`

也就是：

- 已做 baseline 对齐
- 但还没有做时间重采样的原始时序特征

### 3.2 为什么不优先使用重采样后的 20 点序列

原因是：

1. DTW 本身就是为不同长度序列对齐设计的
2. 如果先重采样到固定长度，DTW 的意义会明显减弱
3. 原始长度序列保留了更多真实动态信息

因此，本分析默认输入应为：

- 原始长度的 `sequence_features_rel.npy`

---

## 4. 分析路线

本 DTW 分析分为三条路线，建议全部执行。

### 4.1 路线 A：基于 1 维 magnitude 曲线的 DTW

定义：

对于每条序列：

```text
d(t) = || f_rel(t) ||_2
```

这是一条 1 维时间序列。

目的：

- 比较不同笑容在“强度变化模式”上的相似性

优点：

- 解释最直观
- 计算量较小
- 可以作为基础 baseline 分析

局限：

- 只保留强度信息
- 丢失高维轨迹方向信息

### 4.2 路线 B：基于 1 维 velocity 曲线的 DTW

定义：

对于每条序列：

```text
v(0) = 0
v(t) = || f_rel(t) - f_rel(t-1) ||_2
```

这同样是一条 1 维时间序列。

目的：

- 比较不同笑容在“变化速率模式”上的相似性

优点：

- 反映笑容展开速度和变化节奏

局限：

- 仍然只保留单一维度信息

### 4.3 路线 C：基于降维后多维轨迹的 DTW

这是主分析路线。

思路：

1. 把所有 `f_rel(t)` 收集起来
2. 在高维特征空间上做 PCA
3. 将每条序列映射到低维轨迹空间
4. 对低维多维序列做 multivariate DTW

降维维度要求：

- PCA 到 `10` 维
- PCA 到 `20` 维
- PCA 到 `30` 维

这三种都要尝试。

目的：

- 在保留更多动态结构的同时，降低计算复杂度
- 观察 DTW 相似度结果是否对 PCA 维度稳定

优点：

- 比 1 维曲线保留更多信息
- 比直接 4096 维 DTW 更稳健

---

## 5. DTW 具体要求

### 5.1 使用库

当前计划使用：

- `tslearn`

核心函数可使用：

- `tslearn.metrics.dtw`

### 5.2 建议使用约束

建议同时使用约束 DTW，例如：

- Sakoe-Chiba band

原因：

1. 防止 warping 过于自由
2. 避免不合理的远距离时间点硬对齐
3. 提高结果解释性

建议尝试：

- 不加约束
- 加一个适中的 Sakoe-Chiba band

作为对比。

---

## 6. 相似度定义

对于任意两条序列 `Si` 和 `Sj`，定义：

```text
DTW(Si, Sj)
```

说明：

- 这里得到的是 **distance**
- distance 越小，表示两条序列越相似
- distance 越大，表示两条序列越不相似

因此文档中应明确区分：

- “DTW distance”
- “similarity”

不能混用。

---

## 7. 需要计算的类别关系

### 7.1 类内相似度

必须计算：

- `polite vs polite`
- `truesmile vs truesmile`
- `ambiguous vs ambiguous`

### 7.2 类间相似度

必须计算：

- `polite vs truesmile`
- `ambiguous vs truesmile`
- `ambiguous vs polite`

### 7.3 类内代表序列（Representative Sequence）的定义

对于某一类别内的所有序列，先构建两两 DTW 距离矩阵：

```text
D(i, j) = DTW(S_i, S_j)
```

其中：

- `S_i` 表示该类别中的第 `i` 条序列
- `D(i, j)` 表示第 `i` 条和第 `j` 条序列之间的 DTW distance

然后定义该类别在该实验分支下的代表序列为：

```text
i* = argmin_i Σ_j D(i, j)
```

也就是说：

- 选出到该类别内所有其他序列的总 DTW 距离最小的那一条真实序列
- 该序列就是该类别在该 branch 下的 **DTW medoid**

说明：

1. 这是一个**真实存在的序列**，不是合成序列。
2. 不同实验分支下，代表序列可能不同。
3. 建议同时输出每条候选序列的 centrality score：

```text
centrality_score(i) = Σ_j D(i, j)
```

这样可以看到：

- 哪条序列最中心
- 第一名和第二名差多少

### 7.4 Representative Pair 的建议定义

对于每个类别对、每个实验分支，建议同时定义两种 pair：

1. **best-match pair**
   - 定义为 DTW distance 最小的序列对

2. **representative pair**
   - 定义为 DTW distance 最接近该类别对分布中位数的序列对

说明：

- best-match pair 用来展示“最相似的时候是什么样”
- representative pair 用来展示“这一类关系通常是什么样”

如果后续做可视化，建议展示的不是两张单独图片，而是：

- 两段图片序列
- 以及对应的时间序列曲线
- 如有可能，再补充 warping path

---

## 8. 需要输出的结果

本分析统一输出到：

```text
E:\Matsuda_data\DTW_analysis\
```

建议子目录：

```text
csv\
plots\
report\
models\
```

---

## 9. CSV 输出要求

### 9.1 所有序列对的 DTW 距离

建议文件：

- `csv/dtw_all_pairs_magnitude.csv`
- `csv/dtw_all_pairs_velocity.csv`
- `csv/dtw_all_pairs_pca10.csv`
- `csv/dtw_all_pairs_pca20.csv`
- `csv/dtw_all_pairs_pca30.csv`

字段建议：

```text
feature_type
sequence1_class
sequence1_id
sequence2_class
sequence2_id
relation_type
dtw_distance
```

其中：

- `feature_type` 取值：
  - `magnitude`
  - `velocity`
  - `pca10`
  - `pca20`
  - `pca30`

### 9.2 类内 / 类间统计汇总

建议文件：

- `csv/dtw_statistics_magnitude.csv`
- `csv/dtw_statistics_velocity.csv`
- `csv/dtw_statistics_pca10.csv`
- `csv/dtw_statistics_pca20.csv`
- `csv/dtw_statistics_pca30.csv`

字段建议：

```text
feature_type
pair
relation_type
count
mean
std
median
q1
q3
```

---

## 10. 图表要求

### 图 1. 类内 / 类间 DTW 分布图

文件建议：

- `plots/dtw_distribution_magnitude.png`
- `plots/dtw_distribution_velocity.png`
- `plots/dtw_distribution_pca10.png`
- `plots/dtw_distribution_pca20.png`
- `plots/dtw_distribution_pca30.png`

内容：

- 横轴：类别对
- 纵轴：DTW distance
- 建议用 boxplot 或 violin plot

用途：

- 最直接比较类内和类间相似度

### 图 2. PCA 维度稳定性比较图

文件建议：

- `plots/dtw_pca_dimension_comparison.png`

内容：

- 比较 `pca10 / pca20 / pca30` 下的结果

用途：

- 观察结论是否对降维维度稳定

### 图 3. Representative pair 可视化

文件建议：

- `plots/dtw_examples_magnitude.png`
- `plots/dtw_examples_velocity.png`
- `plots/dtw_examples_pca10.png`
- `plots/dtw_examples_pca20.png`
- `plots/dtw_examples_pca30.png`

内容：

- 选取 DTW 最小的代表性序列对
- 展示两条时间序列曲线
- 如果可行，补充 warping path 可视化

用途：

- 让 DTW 结果更直观

---

## 11. 报告要求

建议文件：

- `report/dtw_summary.md`

内容应包括：

1. 各条路线的整体结果
2. 哪个类别类内最稳定
3. 哪两个类别类间最接近
4. `ambiguous` 更接近 `polite` 还是 `truesmile`
5. PCA 维度对结果是否稳定
6. magnitude / velocity / PCA-DTW 的结论是否一致

---

## 12. 当前推荐实现顺序

建议按下面顺序实现：

1. 先做 `magnitude DTW`
2. 再做 `velocity DTW`
3. 再做 `PCA + multivariate DTW`
   - `10` 维
   - `20` 维
   - `30` 维

原因：

- 这样更容易逐步验证结果
- 如果 PCA 版本结果异常，前面的 1 维版本可以作为对照

---

## 13. 当前阶段的期望

通过这套 DTW 分析，我们希望回答：

1. 哪个笑容类别内部最一致
2. 哪两个笑容类别动态模式最相似
3. `ambiguous` 是否在动态模式上更接近 `polite`
4. 不同特征表示（`d(t)`、`v(t)`、PCA 低维轨迹）下，结论是否一致

如果多条路线给出相似趋势，那么结果会更有说服力。
