# synchronized minimum distance 分析需求文档（中文）

## 1. 目标

本分析的目标是重新定义并计算 smile trajectory 之间的 minimum distance。

这里的 minimum distance **不再**使用此前的：

- 任意时间点对任意时间点的最小距离

也就是不再使用：

```text
min_{t1,t2} || C1(t1) - C2(t2) ||_2
```

而是改为：

- 在**同一时间点**上的最小距离

也就是说，给定两条已经时间重采样并对齐好的曲线 `C1` 和 `C2`，我们只比较：

```text
||C1(t) - C2(t)||_2
```

然后找到最小值出现的时间点。

---

## 2. 新的 minimum distance 定义

对于两条已经重采样并对齐的曲线：

```text
C1(t), t = 0, 1, ..., T-1
C2(t), t = 0, 1, ..., T-1
```

定义同步时间点上的最小距离为：

```text
d_sync_min(C1, C2) = min_t || C1(t) - C2(t) ||_2
```

对应的最小距离发生时间点为：

```text
t* = argmin_t || C1(t) - C2(t) ||_2
```

解释：

- 这里的 `t` 在两条曲线上是相同的时间索引
- 这意味着我们比较的是“同一个归一化阶段”上的距离
- 这与此前的 cross-time minimum distance 不同

---

## 3. 为什么要这样定义

我们已经做了：

1. baseline alignment
2. time normalization / resampling

因此不同长度的 smile sequence 已经被放到了同一个时间坐标系中。

在这种前提下，教授希望比较的是：

- 两条曲线在**同一个时间阶段**最接近的时刻

而不是：

- 任意时间点之间的最近接近

换句话说，我们现在真正关心的是：

- 在已经时间对齐之后，哪个共同阶段最相似

所以新的定义比旧的 `min_{t1,t2}` 更符合当前研究逻辑。

---

## 4. 输入数据

### 4.1 数据来源

继续复用：

- `E:\Matsuda_data\2-27meeting`

重点使用：

- `metrics/normalized/<class>/<seq>/normalized_sequence.npy`
- `metrics/normalized/<class>/<seq>/sampled_frames.json`
- `metrics/normalized_frames/<class>/<seq>/`

也就是说，本分析使用：

- **重采样之后**
- **时间已对齐**
- **长度统一为 20**

的序列数据。

### 4.2 为什么这里要使用重采样之后的数据

因为这个新定义要求：

- `C1(t)` 和 `C2(t)` 表示同一个归一化时间阶段

只有在已经做了重采样和时间对齐之后，这个比较才是合理的。

因此，这个分析与之前的 cross-time minimum distance 正好相反：

- 之前那个更适合原始未重采样序列
- 现在这个定义必须建立在**时间对齐之后**

---

## 5. 需要计算的对象

### 5.1 prototype 级

需要在 prototype trajectory 上计算：

- Method A prototype
- Method B prototype

对于每种 method，计算：

- 类内 synchronized minimum distance
- 类间 synchronized minimum distance

### 5.2 sequence 级

还需要在每一条 normalized sequence 上计算同类指标。

这样可以得到：

- prototype 的结果
- 样本级的结果
- 以及它们的统计分布

---

## 6. 需要计算的类别关系

### 6.1 类内

- `polite vs polite`
- `truesmile vs truesmile`
- `ambiguous vs ambiguous`

### 6.2 类间

- `polite vs truesmile`
- `ambiguous vs truesmile`
- `ambiguous vs polite`

---

## 7. prototype 级输出

### 7.1 输出内容

对于每一对曲线，至少输出：

- `minimum_distance`
- `argmin_time_index`
- `frame_name_at_t_for_curve1`
- `frame_name_at_t_for_curve2`

注意：

- 这里的两个 frame name 对应的是**同一个归一化时间点**
- 但它们在原始图片中仍是两个真实图片

### 7.2 建议输出文件

按 method 分开：

- `methodA/csv/sync_min_distance_methodA.csv`
- `methodB/csv/sync_min_distance_methodB.csv`

字段建议：

```text
method
relation_type
curve1_class
curve1_sequence_id
curve2_class
curve2_sequence_id
argmin_time_index
curve1_frame_name
curve2_frame_name
minimum_distance
```

对于 prototype 级，`sequence_id` 可写：

- `prototype`
- 或者保留为 method 对应的 prototype 标识

---

## 8. sequence 级输出

### 8.1 输出内容

对于每一对 sequence，计算：

```text
d_sync_min(C1, C2) = min_t || C1(t) - C2(t) ||_2
```

以及：

```text
t* = argmin_t || C1(t) - C2(t) ||_2
```

### 8.2 建议输出文件

按 method 分开：

- `methodA/csv/sync_min_distance_all_pairs_methodA.csv`
- `methodB/csv/sync_min_distance_all_pairs_methodB.csv`

字段建议：

```text
method
relation_type
sequence1_class
sequence1_id
sequence2_class
sequence2_id
argmin_time_index
sequence1_frame_name
sequence2_frame_name
minimum_distance
```

---

## 9. 统计输出

### 9.1 统计内容

对于类内 / 类间的各类别对，至少输出：

- count
- mean
- median
- q1
- q3

### 9.2 建议输出文件

按 method 分开：

- `methodA/csv/sync_min_distance_statistics_methodA.csv`
- `methodB/csv/sync_min_distance_statistics_methodB.csv`

字段建议：

```text
method
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

### 图 1. 类内 / 类间 synchronized minimum distance 分布图

文件建议：

- `methodA/plots/sync_min_distance_distribution_methodA.png`
- `methodB/plots/sync_min_distance_distribution_methodB.png`

内容：

- 横轴：类别对
- 纵轴：minimum distance
- 建议使用 boxplot

用途：

- 对比各类别对在“同一时间点最接近”的程度

### 图 2. representative example 图

文件建议：

- `methodA/plots/sync_min_distance_examples_methodA.png`
- `methodB/plots/sync_min_distance_examples_methodB.png`

内容：

- 对每个类别对选出若干代表例子
- 展示最小距离出现时对应的两张图片
- 标注：
  - sequence id
  - class
  - argmin time index
  - minimum distance

用途：

- 把“最小距离发生在同一时间点”这个概念可视化

### 图 3. minimum distance 出现时间点分布图

文件建议：

- `methodA/plots/sync_min_distance_time_distribution_methodA.png`
- `methodB/plots/sync_min_distance_time_distribution_methodB.png`

内容：

- 横轴：`argmin_time_index`
- 纵轴：count

用途：

- 看最小距离通常发生在 smile 的哪个阶段

### 图 4. similarity 出现位置热力图

文件建议：

- `methodA/plots/sync_min_distance_time_heatmap_methodA.png`
- `methodB/plots/sync_min_distance_time_heatmap_methodB.png`

内容：

- 如果是新的 synchronized minimum distance 定义，由于两条曲线使用相同的时间索引：
  - 横轴：`argmin_time_index`
  - 纵轴：类别对
  - 颜色：该时间点出现最小距离的频次

用途：

- 更直观地看“相似出现在哪个阶段”
- 比单纯柱状图更容易比较不同类别对之间的时间分布差异

### 图 5. 累积相似度分布图

文件建议：

- `methodA/plots/sync_min_distance_time_cdf_methodA.png`
- `methodB/plots/sync_min_distance_time_cdf_methodB.png`

内容：

- 横轴：`argmin_time_index`
- 纵轴：累积比例
- 不同类别对画成不同曲线

用途：

- 看某一类别对的最小距离是否更倾向于出现在前段、中段还是后段
- 适合快速判断“相似性是否过度集中在最开始几个时间点”

### 图 6. top-k similarity examples 按时间点排序图

文件建议：

- `methodA/plots/sync_min_distance_examples_sorted_methodA.png`
- `methodB/plots/sync_min_distance_examples_sorted_methodB.png`

内容：

- 从所有 sequence pair 中选出最相似的若干对
- 按 `argmin_time_index` 从小到大排序
- 每一行展示：
  - 两条序列在最相似时间点的图片
  - 对应的类别
  - 对应的 sequence id
  - minimum distance
  - `argmin_time_index`

用途：

- 直接把“相似出现的位置分布”落实到真实图像上
- 便于观察早期相似、中期相似、后期相似分别长什么样

---

## 11. 报告要求

建议输出：

- `report/sync_min_distance_summary.md`

内容至少包括：

1. 新 minimum distance 的定义
2. 为什么它和之前的 cross-time minimum distance 不同
3. 类内 / 类间结果概览
4. 最小距离通常出现在 smile 的哪个阶段
5. `ambiguous` 是否仍然更接近 `polite`
6. Method A / Method B 的结果是否一致

---

## 12. 与旧 minimum distance 的区别

需要在文档中明确：

### 旧定义

```text
min_{t1,t2} || C1(t1) - C2(t2) ||_2
```

解释：

- 任意两个时间点之间的最近接近
- 更适合未重采样的原始序列

### 新定义

```text
min_t || C1(t) - C2(t) ||_2
```

解释：

- 只比较同一个归一化时间点
- 更适合已经重采样并时间对齐的序列

因此，这两种 minimum distance 不应混为一谈。

---

## 13. 当前建议

在实现时，建议：

1. 不再继续推进法平面 / 法超平面分析
2. 先集中完成新的 synchronized minimum distance 分析
3. 输出中明确区分：
   - prototype 级
   - sequence 级
4. 输出路径统一写到：

```text
E:\Matsuda_data\analysis_minimum_output
```
