# minimum distance analysis debuged

## 1. 这次 debug 的目的

这次 minimum distance 重新计算的目的，是纠正之前分析定义上的错误。

之前的问题是：

- 我们直接在 `f_rel(t)` 上定义 minimum distance

也就是：

```text
min || f_rel1(t1) - f_rel2(t2) ||
```

但 `f_rel(t) = f(t) - f0` 本质上是相对特征。

它适合用于：

- projection
- deviation
- relative displacement

但不适合直接用于描述两条序列在空间中的**绝对位置关系**。

因此，这次 debug 的核心修正是：

- minimum distance 改为直接在原始 `f(t)` 上计算

---

## 2. 这次使用的数据

### 2.1 输入数据

使用的是原始特征序列：

```text
E:\Matsuda_data\2-27meeting\metrics\sequence_features\<class>\<seq>\sequence_features.npy
```

以及对应帧名：

```text
E:\Matsuda_data\2-27meeting\metrics\sequence_features\<class>\<seq>\frame_names.json
```

这里的 `sequence_features.npy` 表示：

- 每一帧原始图片对应的 fc7 特征向量
- 即原始 `f(t)` 轨迹

### 2.2 不再使用的数据

这次明确不使用：

- `f_rel`
- `normalized_sequence`

因为这次的目标是：

- 看两条原始曲线在特征空间里的最小绝对距离

---

## 3. 这次 minimum distance 的正式定义

对于两条原始序列：

```text
S1 = {f1(t1)}
S2 = {f2(t2)}
```

定义 minimum distance 为：

```text
d_min(S1, S2) = min_{t1,t2} || f1(t1) - f2(t2) ||_2
```

也就是说：

- 在两条曲线上的任意两个点之间
- 找欧氏距离的最小值

同时记录这个最小值发生的位置：

- `t1*`
- `t2*`

并进一步转成百分比位置：

```text
x% = 100 * t1* / (T1 - 1)
y% = 100 * t2* / (T2 - 1)
```

所以每一个 minimum distance 都对应一个二维位置：

```text
(x%, y%)
```

这个位置表示：

- 最小距离发生时，序列 1 处在自己的哪个相对阶段
- 最小距离发生时，序列 2 处在自己的哪个相对阶段

---

## 4. 这次输出的内容

输出目录：

- [minimum_distace_debug](/e:/Matsuda_data/minimum_distace_debug)

### 4.1 CSV

- [raw_minimum_distance_all_pairs.csv](/e:/Matsuda_data/minimum_distace_debug/csv/raw_minimum_distance_all_pairs.csv)
- [raw_minimum_distance_statistics.csv](/e:/Matsuda_data/minimum_distace_debug/csv/raw_minimum_distance_statistics.csv)

### 4.2 图表

- [raw_minimum_distance_all_pairs_scatter.png](/e:/Matsuda_data/minimum_distace_debug/plots/raw_minimum_distance_all_pairs_scatter.png)
- [raw_minimum_distance_scatter_polite_vs_polite.png](/e:/Matsuda_data/minimum_distace_debug/plots/raw_minimum_distance_scatter_polite_vs_polite.png)
- [raw_minimum_distance_scatter_truesmile_vs_truesmile.png](/e:/Matsuda_data/minimum_distace_debug/plots/raw_minimum_distance_scatter_truesmile_vs_truesmile.png)
- [raw_minimum_distance_scatter_ambiguous_vs_ambiguous.png](/e:/Matsuda_data/minimum_distace_debug/plots/raw_minimum_distance_scatter_ambiguous_vs_ambiguous.png)
- [raw_minimum_distance_scatter_ambiguous_vs_polite.png](/e:/Matsuda_data/minimum_distace_debug/plots/raw_minimum_distance_scatter_ambiguous_vs_polite.png)
- [raw_minimum_distance_scatter_polite_vs_truesmile.png](/e:/Matsuda_data/minimum_distace_debug/plots/raw_minimum_distance_scatter_polite_vs_truesmile.png)
- [raw_minimum_distance_scatter_ambiguous_vs_truesmile.png](/e:/Matsuda_data/minimum_distace_debug/plots/raw_minimum_distance_scatter_ambiguous_vs_truesmile.png)

### 4.3 汇总

- [raw_minimum_distance_summary.md](/e:/Matsuda_data/minimum_distace_debug/report/raw_minimum_distance_summary.md)

---

## 5. 图表的含义

### 5.1 总图

`raw_minimum_distance_all_pairs_scatter.png`

含义：

- 每个点表示一个 sequence pair
- 横轴是序列 1 的相对位置百分比
- 纵轴是序列 2 的相对位置百分比

也就是说，点 `(x%, y%)` 表示：

- 这两条序列的最小距离发生在：
  - sequence 1 的 `x%`
  - sequence 2 的 `y%`

### 5.2 每个类别对单独的图

每张图只显示一个类别对的所有 pair。

这样更容易看：

- minimum distance 是不是集中在某些阶段
- 是否接近对角线
- 是否存在明显时间错位

如果点靠近对角线：

- 说明两条序列在相似阶段最接近

如果点偏离对角线很远：

- 说明两条序列最接近时，可能处在不同阶段

---

## 6. 这次结果的数值摘要

根据 [raw_minimum_distance_statistics.csv](/e:/Matsuda_data/minimum_distace_debug/csv/raw_minimum_distance_statistics.csv)：

### 类内

- `polite_vs_polite`
  - mean = `0.5569`
  - median = `0.5816`

- `ambiguous_vs_ambiguous`
  - mean = `0.5844`
  - median = `0.5879`

- `truesmile_vs_truesmile`
  - mean = `0.5156`
  - median = `0.5193`

### 类间

- `ambiguous_vs_polite`
  - mean = `0.5791`
  - median = `0.5945`

- `polite_vs_truesmile`
  - mean = `0.5973`
  - median = `0.6170`

- `ambiguous_vs_truesmile`
  - mean = `0.5698`
  - median = `0.5828`

---

## 7. 当前可以确认的事情

### 7.1 这次 minimum distance 已经和之前不同

这次结果使用的是：

- 原始 `f(t)`

不再是：

- `f_rel(t)`

所以这次 minimum distance 反映的是：

- 原始特征空间中的绝对最小接近

### 7.2 这次更适合拿来讨论“绝对位置关系”

因为我们不再把每条序列都平移到自己的 baseline 原点。

所以这次 minimum distance 比之前更适合用于回答：

- 两条原始 smile trajectory 在空间里最近的时候有多近
- 最近发生在各自哪个相对阶段

### 7.3 这次结果仍然需要进一步解读

虽然这次已经修正了 `f_rel` 的问题，但后续仍然需要继续看：

1. 类间和类内的分布是否真正区分清楚
2. 2D 百分比位置图是否集中在某些阶段
3. 最小距离是否更多发生在相似阶段，还是经常发生在错位阶段

---

## 8. 结论

这次 debug 已经完成了 minimum distance 定义上的关键修正：

- 不再使用 `f_rel`
- 改为直接使用原始 `f(t)` 轨迹

因此，这一版结果更适合作为后续 minimum distance 分析的基础版本。

接下来真正需要讨论的，不再是“定义是不是错了”，而是：

- 这些 minimum distance 在时序上到底意味着什么
- 它们的二维位置分布是否有稳定结构
- 类内与类间能否在这个定义下被更清晰地区分出来
