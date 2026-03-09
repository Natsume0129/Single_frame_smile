# Projection Analysis 需求文档（中文）

## 1. 目标

在现有 `analysis_sequence` 流水线结果的基础上，新增一套以真笑为参考的投影分析与差异分析。

所有输出必须基于不同的 prototype trajectory 分开组织，至少分为：

- Method A 输出
- Method B 输出

本分析用于回答三条主线问题：

1. 直接差异主线  
   各个类别的 prototype 在每个时间点与基准类别 prototype 的差异有多大，差异如何随时间演化。

2. 主轴推进主线  
   各个类别在时间演化过程中，沿着真笑主轴前进了多少。

3. 主轴偏离主线  
   各个类别在时间演化过程中，偏离真笑主轴多少。

本分析需要同时支持两种 prototype：

- Method A: median trajectory
- Method B: medoid trajectory

两种方法作为并行主分析分别输出。对于 Method B，由于 prototype 对应真实序列，图表中应尽可能保留真实 `sequence_id`，并在可视化中支持对应图片引用。

---

## 2. 当前已有内容

当前已在 `E:\Single_frame_smile\Analysis\analysis_sequence` 中定义并基本完成如下内容：

1. 输入数据结构  
   数据目录为：

   ```text
   E:\Matsuda_data\2-18meeting\
   ├── polite\
   ├── truesmile\
   └── ambiguous\
   ```

   每个子目录下是一段笑容序列的逐帧图片。

2. 特征提取  
   对每一帧提取 VGG-Face `fc7` 特征：

   ```text
   f(t) ∈ R^D, D = 4096
   ```

3. Baseline 对齐  
   使用前 5 帧均值作为 baseline：

   ```text
   f0 = mean(f(0), f(1), f(2), f(3), f(4))
   f_rel(t) = f(t) - f0
   ```

4. 时间归一化  
   将每条 `f_rel` 序列线性重采样为固定 20 个时间点：

   ```text
   f_norm(t), t = 0, 1, ..., 19
   ```

5. Prototype 构建  
   已定义两种 prototype：

   - Method A: median trajectory
   - Method B: medoid trajectory

6. 已有输出  
   现有流水线已经定义或输出：

   - `metrics/sequence_features/...`
   - `metrics/sequence_features_rel/...`
   - `metrics/normalized/...`
   - `metrics/normalized_frames/...`
   - `prototypes/prototype_<class>.npy`
   - `prototypes/prototype_<class>_medoid.npy`
   - `plots/...`
   - `csv/...`

---

## 3. Prototype 的正式定义

### 3.1 基本记号

对于类别 `c` 中的第 `i` 条归一化序列，记为：

```text
f_i^c(t), t = 0, 1, ..., 19
```

每条序列的形状为：

```text
[20, D]
```

### 3.2 Method A: Median Trajectory

Method A 不是从真实样本中选一条序列，而是从类别内所有归一化序列中，在每个时间点、每个维度上分别取中位数，构造一条新的典型轨迹。

定义：

```text
p_c^A(t)_d = median_i(f_i^c(t)_d)
```

等价地说：

```text
p_c^A(t) = median_i(f_i^c(t))
```

其中 median 为逐维计算。

说明：

- Method A 综合考虑 `t=0..19` 的全部时间点。
- Method A 输出的是一条新的 prototype trajectory。
- 该轨迹通常不对应真实文件。

### 3.3 Method B: Medoid Trajectory

Method B 从该类别内的真实样本中，选出总体最具代表性的一条序列。

记每条归一化序列矩阵为：

```text
F_i^c ∈ R^{20×D}
```

定义任意两条序列之间的整体距离为：

```text
d(F_i^c, F_j^c) = ||F_i^c - F_j^c||_F
```

其中 `||·||_F` 为 Frobenius norm。

则 medoid 的索引定义为：

```text
i_c^* = argmin_i Σ_j d(F_i^c, F_j^c)
```

对应的 prototype 为：

```text
p_c^B(t) = f_{i_c^*}^c(t)
```

说明：

- Method B 综合考虑整条时间序列的总体 cost。
- Method B 输出的是一条真实存在的序列。
- Method B 必须保留对应的 `sequence_id` 和真实 frame 映射。

---

## 4. 三条主线分析

### 4.1 主线一：直接差异分析

该主线直接比较同一时间点两个 prototype 向量之间的差异。

对于任意两个类别 `a` 与 `b`，定义它们在时间点 `t` 的差异为：

```text
diff_{a,b}(t) = || p_a(t) - p_b(t) ||_2
```

其中：

- `p_a(t) - p_b(t)` 是同一时间点的差分向量
- `||·||_2` 是欧氏模长

解释：

- `diff_{a,b}(t)` 越小，表示两个类别在该时间点越接近。
- `diff_{a,b}(t)` 越大，表示两个类别在该时间点差异越大。
- `t=0` 的差异可以直接解释为起始状态差异。
- 该曲线可以回答：差异是一开始就存在，还是在动态过程中逐渐拉大；最大差异出现在何时；差异是否会再次缩小。

### 4.2 主线二：沿真笑主轴推进了多少

在某一种 prototype 设定下，记真笑类别的 prototype 为：

```text
p_true(t), t = 0, 1, ..., 19
```

定义真笑主轴向量为真笑 prototype 的首尾连线：

```text
g = p_true(19) - p_true(0)
```

其单位方向向量为：

```text
u = g / ||g||
```

说明：

- `g` 不是完整真笑轨迹，只是一个一阶近似的语义主轴。
- 本分析明确接受这种“首尾连线近似”的设定。

对于任意类别 `c` 的 prototype：

```text
p_c(t), t = 0, 1, ..., 19
```

定义该类别相对于自身起点的动态向量：

```text
d_c(t) = p_c(t) - p_c(0)
```

因此有：

```text
d_c(0) = 0
```

定义时间点 `t` 上，类别 `c` 在真笑主轴上的投影长度为：

```text
a_c(t) = < d_c(t), u >
```

为了得到相对比例，定义归一化投影进度为：

```text
ratio_along_c(t) = a_c(t) / ||g||
```

解释：

- `ratio_along_c(t) = 0`：还停留在该类别自身起点。
- `ratio_along_c(t) = 1`：其沿真笑轴前进量等于真笑 prototype 首尾长度。
- `ratio_along_c(t) > 1`：在该轴向上超过真笑终点长度。
- `ratio_along_c(t) < 0`：沿真笑轴反方向移动。

### 4.3 主线三：偏离真笑主轴多少

先定义 `d_c(t)` 在真笑主轴上的投影向量：

```text
proj_c(t) = proj_g(d_c(t))
```

再定义偏离向量：

```text
r_c(t) = d_c(t) - proj_c(t)
```

偏离距离定义为：

```text
dist_off_c(t) = || r_c(t) ||_2
```

为了得到相对比例，定义归一化偏离量为：

```text
ratio_off_c(t) = dist_off_c(t) / ||g||
```

解释：

- `ratio_off_c(t)` 越小，说明该类别在该时间点越贴近真笑主轴。
- `ratio_off_c(t)` 越大，说明该类别虽然可能也在运动，但偏离真笑主轴越明显。

---

## 5. 关于 baseline 与 initial bias 的定位

本需求不再把 `initial bias` 当作主线分析步骤，而是把它作为对主线一的补充解释。

原因：

1. 各类别的 `f0` 本来就不同  
   因为 `f0` 是各自序列前 5 帧的均值，不是同一张图片。

2. 在动态投影分析中，我们使用：

   ```text
   d_c(t) = p_c(t) - p_c(0)
   ```

   这样做的目的是专注分析“从各自起点开始以后”的动态路径，因此 `d_c(0)=0` 是刻意设计。

3. 如果研究重点是“差异是不是从一开始就存在，并且如何随时间变化”，那么更自然的做法不是单独构建一个 baseline prototype，而是直接看主线一中的：

   ```text
   diff_{a,b}(0)
   ```

也就是说：

- `t=0` 的直接差异，就是对起始差异的最直接表达。
- 若有需要，可以额外报告 baseline `f0` 的差异作为补充说明。
- 但 `initial bias` 不再单独作为 Method A / Method B 的主步骤。

---

## 6. Method A / Method B 并行分析要求

### 6.1 Method A

使用 median prototype：

```text
p_c^A(t)
```

并据此分别计算：

- `diff_{a,b}^A(t)`
- `g^A`
- `u^A`
- `ratio_along_c^A(t)`
- `ratio_off_c^A(t)`

### 6.2 Method B

使用 medoid prototype：

```text
p_c^B(t)
```

并据此分别计算：

- `diff_{a,b}^B(t)`
- `g^B`
- `u^B`
- `ratio_along_c^B(t)`
- `ratio_off_c^B(t)`

Method B 额外要求：

- 保存 prototype 对应的真实 `sequence_id`
- 保存该真实序列对应的 `normalized_frames`
- 在图表中允许引用或高亮该真实 prototype 的图片

### 6.3 输出组织要求

所有输出都必须按 prototype trajectory 方法分开组织，不允许把 Method A 和 Method B 的结果混在同一个文件中。

建议方式：

- 文件名后缀区分：
  - `..._methodA.*`
  - `..._methodB.*`
- 或者子目录区分：
  - `methodA/...`
  - `methodB/...`

若两种方式同时使用，也可以接受，但必须保证：

- 一眼能区分是 Method A 还是 Method B
- 所有图表、CSV、报告都能独立对应回各自的 prototype trajectory

---

## 7. 分析步骤

### Step 1. 读取已有分析结果

目的：

- 复用 `analysis_sequence` 已有输出，不重复做特征提取。

输入：

- `sequence_features.npy`
- `sequence_features_rel.npy`
- `normalized_sequence.npy`
- `sampled_frames.json`
- `prototype_*.npy`
- `prototype_*_medoid.npy`

### Step 2. 构建 trajectory prototype

目的：

- 从每个类别的多条归一化时间序列中，得到一条代表该类别整体时序形态的 prototype trajectory。

Method A：

- 在每个时间点、每个维度上对所有样本取中位数
- 得到新的 `[20, D]` prototype

Method B：

- 用整体轨迹距离计算 medoid
- 选出一条真实样本作为 `[20, D]` prototype

输出：

- `p_c^A(t)`
- `p_c^B(t)`
- Method B 对应的 `sequence_id`

### Step 3. 计算直接差异曲线

目的：

- 直接比较类别间在每个时间点的差异大小及其时间演化。

定义：

```text
diff_{a,b}(t) = || p_a(t) - p_b(t) ||_2
```

输出：

- `polite vs truesmile`
- `ambiguous vs truesmile`
- `polite vs ambiguous`

每组在 `t=0..19` 的差异曲线。

### Step 4. 构建真笑主轴

目的：

- 定义后续投影分析统一使用的真笑方向。

定义：

```text
g = p_true(19) - p_true(0)
u = g / ||g||
```

注意：

- Method A 和 Method B 分别构建各自的 `g`
- 若 `||g||` 极小，需要报错或显式标记不可分析

### Step 5. 计算各类别的 along-axis progress

目的：

- 观察每个类别在每个时间点沿真笑方向推进了多少。

定义：

```text
d_c(t) = p_c(t) - p_c(0)
a_c(t) = < d_c(t), u >
ratio_along_c(t) = a_c(t) / ||g||
```

输出：

- 每类 20 个时间点的投影长度
- 每类 20 个时间点的归一化投影比例

### Step 6. 计算各类别的 off-axis deviation

目的：

- 观察每个类别在每个时间点偏离真笑主轴多少。

定义：

```text
proj_c(t) = proj_g(d_c(t))
r_c(t) = d_c(t) - proj_c(t)
dist_off_c(t) = ||r_c(t)||_2
ratio_off_c(t) = dist_off_c(t) / ||g||
```

输出：

- 每类 20 个时间点的偏离绝对值
- 每类 20 个时间点的归一化偏离比例

### Step 7. 对所有样本进行补充分析（必须）

目的：

- 除了 prototype trajectory 之外，也对每一条 normalized sequence 做同样的分析。

方法：

- 对每条样本归一化序列 `f_norm_i(t)` 使用相同公式计算：
  - 逐时间点直接差异
  - `ratio_along_i(t)`
  - `ratio_off_i(t)`

用途：

- 计算类均值与标准差
- 观察类内离散程度
- 检查 prototype 是否代表该类总体趋势
- 进行类内统计分析和类间统计比较

### Step 8. 生成图表与汇总表

目的：

- 输出便于解释和汇报的结果。

---

## 8. 输出文件要求

建议输出根目录：

```text
E:\Matsuda_data\projection_analysis\
```

推荐组织方式：

```text
E:\Matsuda_data\projection_analysis\
├── methodA\
│   ├── csv\
│   ├── plots\
│   ├── prototypes\
│   └── report\
└── methodB\
    ├── csv\
    ├── plots\
    ├── prototypes\
    └── report\
```

这种方式优先于把所有结果混放在一个目录。

### 8.1 原型与元信息

建议输出：

- `methodA\prototypes\projection_meta_methodA.json`
- `methodB\prototypes\projection_meta_methodB.json`

内容包括：

- prototype 方法
- 使用的类别列表
- 时间长度 `T=20`
- 特征维度 `D=4096`
- 真笑主轴长度 `||g||`
- Method B 对应的真实 `sequence_id`

### 8.2 直接差异结果

建议输出：

- `methodA\csv\direct_distance_methodA.csv`
- `methodB\csv\direct_distance_methodB.csv`

字段建议：

```text
method
anchor_class
target_class
time_index
difference_norm
```

说明：

- `anchor_class` 表示基准类别
- `target_class` 表示与该基准类别比较的另外一个类别

### 8.3 动态投影结果

建议输出：

- `methodA\csv\projection_along_methodA.csv`
- `methodB\csv\projection_along_methodB.csv`
- `methodA\csv\projection_off_methodA.csv`
- `methodB\csv\projection_off_methodB.csv`

字段建议：

```text
method
class
time_index
projection_length
projection_ratio
off_axis_distance
off_axis_ratio
```

### 8.4 baseline 补充结果（可选）

建议输出：

- `methodA\csv\baseline_offset_supplement_methodA.csv`
- `methodB\csv\baseline_offset_supplement_methodB.csv`

字段建议：

```text
method
class
baseline_offset_to_truesmile
sequence_id   # Method B 使用，Method A 可为空
```

说明：

- 这部分是补充说明，不是主线输出。

### 8.5 样本级补充结果

建议输出：

- `methodA\csv\projection_per_sequence_methodA.csv`
- `methodB\csv\projection_per_sequence_methodB.csv`
- `methodA\csv\per_sequence_direct_distance_methodA.csv`
- `methodB\csv\per_sequence_direct_distance_methodB.csv`

字段建议一：

```text
method
class
sequence_id
time_index
projection_ratio
off_axis_ratio
```

字段建议二：

```text
method
anchor_class
target_class
sequence_id
time_index
difference_norm
```

说明：

- `anchor_class` 表示作为参照 prototype 的类别
- `target_class` 表示该 sequence 所属类别

### 8.6 样本级统计结果

建议输出：

- `methodA\csv\projection_statistics_methodA.csv`
- `methodB\csv\projection_statistics_methodB.csv`
- `methodA\csv\direct_distance_statistics_methodA.csv`
- `methodB\csv\direct_distance_statistics_methodB.csv`

字段建议：

```text
method
metric_type
class
anchor_class
time_index
mean
std
median
q1
q3
```

### 8.7 汇总报告

建议输出：

- `methodA\report\projection_summary_methodA.md`
- `methodB\report\projection_summary_methodB.md`

内容建议：

- 直接差异曲线的主要结论
- 各类别沿真笑轴推进的总体趋势
- 各类别偏离真笑主轴的总体趋势
- 类内统计与类间统计的主要结论
- Method A / Method B 结果异同

---

## 9. 图表要求

### 图 1. Anchor-based 直接差异曲线图（必选）

文件建议：

- `methodA\plots\direct_distance_anchor_polite_methodA.png`
- `methodA\plots\direct_distance_anchor_truesmile_methodA.png`
- `methodA\plots\direct_distance_anchor_ambiguous_methodA.png`
- `methodB\plots\direct_distance_anchor_polite_methodB.png`
- `methodB\plots\direct_distance_anchor_truesmile_methodB.png`
- `methodB\plots\direct_distance_anchor_ambiguous_methodB.png`

内容：

- 横轴：`time_index = 0..19`
- 纵轴：`difference_norm`
- 每一张图固定一个 anchor class 作为基准类别
- 图中有两条曲线，分别表示另外两个类别到该 anchor class 的直接差异曲线

用途：

- 观察类别差异是从一开始就存在，还是在动态过程中拉开
- 判断差异何时最大、是否回落

说明：

- 一共 3 张图对应 3 个 anchor class
- Method A / Method B 各自都要输出一套

### 图 2. Projection along true-smile axis 曲线图（必选）

文件建议：

- `methodA\plots\projection_along_methodA.png`
- `methodB\plots\projection_along_methodB.png`

内容：

- 横轴：`time_index = 0..19`
- 纵轴：百分比或比例
- 三条曲线分别对应：
  - polite prototype trajectory
  - truesmile prototype trajectory
  - ambiguous prototype trajectory

用途：

- 观察各类别沿真笑主轴推进的速度和终点水平

补充要求：

- 推荐显示范围覆盖 `0` 到 `1`
- 若存在超过 `1` 的情况，不截断，需真实显示

### 图 3. Deviation from true-smile axis 曲线图（必选）

文件建议：

- `methodA\plots\projection_off_methodA.png`
- `methodB\plots\projection_off_methodB.png`

内容：

- 横轴：`time_index = 0..19`
- 纵轴：百分比或比例
- 三条曲线分别对应：
  - polite prototype trajectory
  - truesmile prototype trajectory
  - ambiguous prototype trajectory

用途：

- 观察各类别何时开始明显偏离真笑主轴

补充要求：

- 纵轴必须与 `ratio_off_c(t)` 保持一致

### 图 4. along vs off 二维关系图（建议）

文件建议：

- `methodA\plots\projection_phase_methodA.png`
- `methodB\plots\projection_phase_methodB.png`

内容：

- 横轴：`ratio_along_c(t)`
- 纵轴：`ratio_off_c(t)`
- 每类形成一条时间轨迹

用途：

- 同时观察“前进”与“偏离”两种信息

### 图 5. 样本级统计带图（强烈建议）

文件建议：

- `methodA\plots\projection_along_band_methodA.png`
- `methodA\plots\projection_off_band_methodA.png`
- `methodB\plots\projection_along_band_methodB.png`
- `methodB\plots\projection_off_band_methodB.png`

内容：

- prototype 曲线
- 样本均值曲线
- 样本标准差或分位带

用途：

- 观察 prototype 与类内分布是否一致
- 观察类内方差是否随时间变化

### 图 6. 样本级直接差异分布图（建议）

文件建议：

- `methodA\plots\direct_distance_band_anchor_polite_methodA.png`
- `methodA\plots\direct_distance_band_anchor_truesmile_methodA.png`
- `methodA\plots\direct_distance_band_anchor_ambiguous_methodA.png`
- `methodB\plots\direct_distance_band_anchor_polite_methodB.png`
- `methodB\plots\direct_distance_band_anchor_truesmile_methodB.png`
- `methodB\plots\direct_distance_band_anchor_ambiguous_methodB.png`

内容：

- 横轴：`time_index = 0..19`
- 纵轴：样本级 `difference_norm`
- 展示某一 anchor class 下，另外两类样本相对于 anchor prototype 的均值曲线和方差带

用途：

- 观察类间差异不仅在 prototype 层面如何变化，也在样本分布层面如何变化
- 判断类别差异是否稳定，还是受少量样本影响

### 图 7. Method B prototype 对应图片展示（重要）

文件建议：

- `methodB\plots\prototype_frames_methodB_<class>.png`

内容：

- 展示 Method B prototype 对应真实序列的 20 个 normalized frame
- 在图标题中标注 `class` 和 `sequence_id`

用途：

- 强化 Method B“对应真实文件”的可解释性

### 图 8. baseline 差异补充图（可选）

文件建议：

- `methodA\plots\baseline_offset_supplement_methodA.png`
- `methodB\plots\baseline_offset_supplement_methodB.png`

用途：

- 仅作为起始差异的补充说明，不作为主线图。

---

## 10. 样本级统计分析要求

除了 prototype trajectory 之外，必须对每一条 normalized sequence 进行同样的分析。

对于每一条样本级序列，需要至少计算：

1. 相对于 anchor prototype 的逐时间点直接差异
2. 相对于真笑主轴的 `ratio_along`
3. 相对于真笑主轴的 `ratio_off`

基于样本级结果，至少需要做以下统计：

1. 每个类别、每个时间点的均值
2. 每个类别、每个时间点的标准差
3. 每个类别、每个时间点的中位数
4. 每个类别、每个时间点的四分位区间或置信带

若条件允许，建议进一步做：

1. 峰值时间点比较
2. 面积下面积（AUC）比较
3. 最终时间点比较
4. 类间统计检验

样本级统计分析的目标是：

- 判断 prototype 结论是否代表类整体趋势
- 判断类内分散程度是否很大
- 判断不同类别差异是否具有统计稳定性

---

## 11. 解释原则

1. 直接差异 `diff_{a,b}(t)` 回答“两个类别在该时间点相差多少”。

2. `ratio_along` 回答“该类别有没有沿真笑方向走，以及走了多少”。

3. `ratio_off` 回答“该类别是否走偏了真笑主轴，以及偏离了多少”。

4. `ratio_along` 大，不代表一定更像真笑  
   它只说明沿真笑主轴推进得更多。

5. `ratio_off` 小，代表更贴近真笑主轴  
   但仍需要结合 `ratio_along` 一起解释。

6. 最理想的“接近真笑”状态通常是：

- `ratio_along` 较大
- `ratio_off` 较小

7. `t=0` 的直接差异就是起始差异的最直接表达。

8. 真笑主轴是首尾连线近似，不代表真笑轨迹本身一定是直线。

---

## 12. 本文档的结论性要求

本分析必须：

1. 以现有 `analysis_sequence` 的输出为输入基础。
2. 明确把 Method A 定义为中位数时序法，把 Method B 定义为距离最小 medoid 法。
3. 使用前 5 帧平均定义 `f0`。
4. 使用 20 个时间点，索引统一为 `0..19`。
5. 使用真笑 prototype 的首尾连线定义主轴。
6. 把以下三条主线作为正式输出：

- 逐时间点直接差异
- 沿真笑主轴的投影推进
- 偏离真笑主轴的距离

7. 使用以下归一化定义：

   ```text
   ratio_along_c(t) = projection_length / ||g||
   ratio_off_c(t)   = off_axis_distance / ||g||
   ```

8. Method A / Method B 作为并行主分析分别输出。
9. 核心图表至少包括：

- 3 张 anchor-based 直接差异曲线图（每种方法各一套）
- 1 张 along-axis progress 曲线图（每种方法各一套）
- 1 张 off-axis deviation 曲线图（每种方法各一套）

10. Method B 输出中必须保留真实 `sequence_id`，并支持与真实图片对应。
11. 除了 prototype trajectory 之外，必须对每一条 normalized sequence 做同类分析，并输出样本级统计结果。
