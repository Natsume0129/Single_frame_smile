# Projection Analysis 需求文档（中文）

## 1. 目标

在现有 `analysis_sequence` 流水线结果的基础上，新增一套以“真笑主轴”为核心的投影分析，用于回答两个问题：

1. 某一类笑容在时间演化过程中，沿着真笑主轴前进了多少。
2. 某一类笑容在时间演化过程中，偏离真笑主轴多少。

本分析需要同时支持两种 prototype：

- Method A: median prototype
- Method B: medoid prototype

两种方法作为并行主分析分别输出。对于 Method B，由于 prototype 对应真实序列，图表中应尽可能保留真实序列 ID，并在可视化中支持对应图片引用。

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

## 3. 本次新增分析的核心思想

### 3.1 真笑主轴

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

### 3.2 初始偏置

由于不同类别各自使用自己的 `f0` 做 baseline 对齐，因此在 `f_rel` 空间中不能把不同类别的起点直接解释为共享同一个原点。

因此需要把“初始偏置”单独定义为原始 baseline 之间的差异，而不混入动态投影分析。

对于类别 `c`，定义其 baseline prototype 为该类所有样本 baseline 的代表值：

- Method A: baseline median
- Method B: baseline medoid 对应样本的 baseline

记为：

```text
b_c ∈ R^D
```

则相对于真笑的初始偏置为：

```text
offset_c = || b_c - b_true ||_2
```

该量用于描述各类别在笑容开始前或开始时的静态差异，不参与 Question A / Question B 的动态投影计算。

---

## 4. 动态投影分析定义

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

这不是问题，而是刻意设计，用于只分析“从该类别自身起点开始”的动态变化。

### 4.1 Question A: 沿真笑主轴前进了多少

定义时间点 `t` 上，类别 `c` 在真笑主轴上的投影长度为：

```text
a_c(t) = < d_c(t), u >
```

其中 `<·,·>` 表示内积。

为了得到相对比例，定义归一化投影进度为：

```text
ratio_along_c(t) = a_c(t) / ||g||
```

解释：

- `ratio_along_c(t) = 0`：还停留在该类别自身起点。
- `ratio_along_c(t) = 1`：其沿真笑轴前进量等于真笑 prototype 首尾长度。
- `ratio_along_c(t) > 1`：在该轴向上超过真笑终点长度。
- `ratio_along_c(t) < 0`：沿真笑轴反方向移动。

### 4.2 Question B: 偏离真笑主轴多少

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

## 5. 为什么要把初始偏置和动态偏离分开

本需求明确把分析拆成两层：

1. 初始偏置  
   比较各类别 baseline `f0` 与真笑 baseline 的差异。

2. 动态投影  
   比较各类别从自己的起点出发，沿真笑主轴推进多少，以及偏离多少。

这样做的原因是：

- 如果直接在 `f_rel` 空间里比较不同类别的绝对点位置，会混淆不同 baseline 坐标系。
- 使用 `d_c(t) = p_c(t) - p_c(0)` 后，`t=0` 一定为 0，这有利于专注分析动态形态。
- 初始差异是有意义的，但应单独报告，而不应和动态偏离混在一个指标里。

---

## 6. Method A / Method B 并行分析要求

### 6.1 Method A

使用 median prototype：

```text
p_c^A(t)
```

并据此计算：

- `g^A`
- `u^A`
- `offset_c^A`
- `ratio_along_c^A(t)`
- `ratio_off_c^A(t)`

### 6.2 Method B

使用 medoid prototype：

```text
p_c^B(t)
```

并据此计算：

- `g^B`
- `u^B`
- `offset_c^B`
- `ratio_along_c^B(t)`
- `ratio_off_c^B(t)`

Method B 额外要求：

- 保存 prototype 对应的真实 `sequence_id`
- 保存该真实序列对应的 `normalized_frames`
- 在图表中允许引用或高亮该真实 prototype 的图片

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

### Step 2. 构建 baseline prototype

目的：

- 为初始偏置分析建立每类的 baseline 代表向量。

定义：

- Method A: 每类所有样本 `f0` 做 element-wise median
- Method B: 直接取 medoid 序列自己的 `f0`

输出：

- 每类 baseline prototype
- 每类相对真笑的 baseline offset

### Step 3. 构建真笑主轴

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

### Step 4. 计算各类别的 along-axis progress

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

### Step 5. 计算各类别的 off-axis deviation

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

### Step 6. 对所有样本进行补充分析（建议）

目的：

- 不仅比较 prototype，还比较类内样本分布。

方法：

- 对每条样本归一化序列 `f_norm_i(t)` 使用相同公式计算：
  - `ratio_along_i(t)`
  - `ratio_off_i(t)`

用途：

- 计算类均值与标准差
- 观察类内离散程度
- 检查 prototype 是否代表该类总体趋势

### Step 7. 生成图表与汇总表

目的：

- 输出便于解释和汇报的结果。

---

## 8. 输出文件要求

建议输出根目录：

```text
E:\Matsuda_data\projection_analysis\
```

建议子目录：

```text
prototypes\
csv\
plots\
report\
```

### 8.1 原型与元信息

建议输出：

- `prototypes\projection_meta_methodA.json`
- `prototypes\projection_meta_methodB.json`

内容包括：

- prototype 方法
- 使用的类别列表
- 时间长度 `T=20`
- 特征维度 `D=4096`
- 真笑主轴长度 `||g||`
- Method B 对应的真实 `sequence_id`

### 8.2 初始偏置结果

建议输出：

- `csv\baseline_offsets_methodA.csv`
- `csv\baseline_offsets_methodB.csv`

字段建议：

```text
method
class
offset_to_truesmile
baseline_norm
sequence_id   # Method B 使用，Method A 可为空
```

### 8.3 动态投影结果

建议输出：

- `csv\projection_along_methodA.csv`
- `csv\projection_along_methodB.csv`
- `csv\projection_off_methodA.csv`
- `csv\projection_off_methodB.csv`

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

### 8.4 样本级补充结果

建议输出：

- `csv\projection_per_sequence_methodA.csv`
- `csv\projection_per_sequence_methodB.csv`

字段建议：

```text
method
class
sequence_id
time_index
projection_ratio
off_axis_ratio
```

### 8.5 汇总报告

建议输出：

- `report\projection_summary_methodA.md`
- `report\projection_summary_methodB.md`

内容建议：

- 初始偏置排序
- 各类别沿真笑轴推进的总体趋势
- 各类别偏离真笑主轴的总体趋势
- Method A / Method B 结果异同

---

## 9. 图表要求

### 图 1. baseline 初始偏置柱状图

文件建议：

- `plots\baseline_offsets_methodA.png`
- `plots\baseline_offsets_methodB.png`

内容：

- 横轴：类别
- 纵轴：`offset_c`

用途：

- 单独说明不同类别在 baseline 上的初始差异

### 图 2. prototype along-axis progress 曲线

文件建议：

- `plots\projection_along_methodA.png`
- `plots\projection_along_methodB.png`

内容：

- 横轴：`time_index = 0..19`
- 纵轴：`ratio_along_c(t)`
- 三条类别曲线同图展示

用途：

- 观察各类别沿真笑主轴推进的速度和终点水平

### 图 3. prototype off-axis deviation 曲线

文件建议：

- `plots\projection_off_methodA.png`
- `plots\projection_off_methodB.png`

内容：

- 横轴：`time_index = 0..19`
- 纵轴：`ratio_off_c(t)`
- 三条类别曲线同图展示

用途：

- 观察各类别何时开始明显偏离真笑主轴

### 图 4. along vs off 二维关系图

文件建议：

- `plots\projection_phase_methodA.png`
- `plots\projection_phase_methodB.png`

内容：

- 横轴：`ratio_along_c(t)`
- 纵轴：`ratio_off_c(t)`
- 每类形成一条时间轨迹

用途：

- 同时观察“前进”与“偏离”两种信息

### 图 5. 样本级置信带图（建议）

文件建议：

- `plots\projection_along_band_methodA.png`
- `plots\projection_off_band_methodA.png`
- `plots\projection_along_band_methodB.png`
- `plots\projection_off_band_methodB.png`

内容：

- prototype 曲线
- 样本均值曲线
- 样本标准差或分位带

用途：

- 观察 prototype 与类内分布是否一致

### 图 6. Method B prototype 对应图片展示（重要）

文件建议：

- `plots\prototype_frames_methodB_<class>.png`

内容：

- 展示 Method B prototype 对应真实序列的 20 个 normalized frame
- 在图标题中标注 `class` 和 `sequence_id`

用途：

- 强化 Method B“对应真实文件”的可解释性

---

## 10. 解释原则

1. `ratio_along` 大，不代表一定更像真笑  
   它只说明沿真笑主轴推进得更多。

2. `ratio_off` 小，代表更贴近真笑主轴  
   但仍需要结合 `ratio_along` 一起解释。

3. 最理想的“接近真笑”状态通常是：

- `ratio_along` 较大
- `ratio_off` 较小

4. 初始偏置与动态偏离是两个层面：

- 初始偏置回答“起点像不像”
- 动态偏离回答“变化路径像不像”

5. 真笑主轴是首尾连线近似，不代表真笑轨迹本身一定是直线。

---

## 11. 本文档的结论性要求

本分析必须：

1. 以现有 `analysis_sequence` 的输出为输入基础。
2. 明确区分 baseline 初始偏置 与 动态投影偏离。
3. 使用前 5 帧平均定义 `f0`。
4. 使用 20 个时间点，索引统一为 `0..19`。
5. 使用真笑 prototype 的首尾连线定义主轴。
6. 使用以下归一化定义：

   ```text
   ratio_along_c(t) = projection_length / ||g||
   ratio_off_c(t)   = off_axis_distance / ||g||
   ```

7. Method A / Method B 作为并行主分析分别输出。
8. Method B 输出中必须保留真实 `sequence_id`，并支持与真实图片对应。

