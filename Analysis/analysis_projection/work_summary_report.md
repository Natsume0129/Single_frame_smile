# analysis_projection 工作总结报告

## 1. 工作背景

这阶段的工作，主要围绕笑容序列的动态分析展开。核心目标不是只看单帧，而是把每一段笑容看成一条随时间变化的轨迹，然后比较不同类型笑容在高维特征空间中的演化方式。

我们复用了前期 `analysis_sequence` 的结果，把每条笑容序列先做特征提取、baseline 对齐和时间归一化，再在此基础上构建 prototype trajectory，并进一步分析不同类别之间的距离关系、沿真笑主轴的推进情况，以及偏离真笑主轴的程度。

---

## 2. 已完成的主要工作

### 2.1 构建了新的分析模块

在目录 [analysis_projection](/e:/Single_frame_smile/Analysis/analysis_projection) 下，新建并完成了一套独立的分析代码，主要包括：

- [common.py](/e:/Single_frame_smile/Analysis/analysis_projection/common.py)
- [01_build_projection_prototypes.py](/e:/Single_frame_smile/Analysis/analysis_projection/01_build_projection_prototypes.py)
- [02_compute_direct_distance.py](/e:/Single_frame_smile/Analysis/analysis_projection/02_compute_direct_distance.py)
- [03_compute_projection_metrics.py](/e:/Single_frame_smile/Analysis/analysis_projection/03_compute_projection_metrics.py)
- [04_compute_per_sequence_metrics.py](/e:/Single_frame_smile/Analysis/analysis_projection/04_compute_per_sequence_metrics.py)
- [05_compute_statistics.py](/e:/Single_frame_smile/Analysis/analysis_projection/05_compute_statistics.py)
- [06_generate_plots.py](/e:/Single_frame_smile/Analysis/analysis_projection/06_generate_plots.py)
- [07_generate_report.py](/e:/Single_frame_smile/Analysis/analysis_projection/07_generate_report.py)
- [08_polite_axis_deviation_analysis.py](/e:/Single_frame_smile/Analysis/analysis_projection/08_polite_axis_deviation_analysis.py)
- [run_projection_pipeline.ps1](/e:/Single_frame_smile/Analysis/analysis_projection/run_projection_pipeline.ps1)

这套脚本复用了 [analysis_sequence](/e:/Single_frame_smile/Analysis/analysis_sequence) 的输出，不重复做原始特征提取，而是直接基于已经整理好的 `normalized_sequence.npy`、prototype 和 normalized frames 继续分析。

### 2.2 完成了中英文需求文档

为了明确研究目标、数学定义和输出要求，我们整理了两份需求文档：

- [projection_analysis_cn.md](/e:/Single_frame_smile/Analysis/analysis_projection/projection_analysis_cn.md)
- [projection_analysis_en.md](/e:/Single_frame_smile/Analysis/analysis_projection/projection_analysis_en.md)

文档中明确了：

- Method A 与 Method B 的定义
- 三条主线分析
- 输出文件和图表要求
- 样本级统计分析要求

### 2.3 完成了汇报材料准备

为了周汇报，我们还整理并翻译了发言稿：

- [slide.md](/e:/Single_frame_smile/Analysis/analysis_projection/slide.md)
- [slide_en.md](/e:/Single_frame_smile/Analysis/analysis_projection/slide_en.md)

这部分工作主要是把技术内容转成更适合口头汇报的表达，同时尽量保持术语准确。

---

## 3. 分析方法总结

### 3.1 数据预处理流程

当前方法建立在以下步骤之上：

1. 对每一帧提取 VGG-Face fc7 特征
2. 用前 5 帧均值作为 baseline `f0`
3. 计算相对特征 `f_rel(t) = f(t) - f0`
4. 把每条序列重采样为 20 个时间点

因此，每一条笑容序列最终表示为一条 `[20, 4096]` 的时序轨迹。

### 3.2 Prototype 构建

我们并行使用了两种 prototype 方法：

- Method A：Median Trajectory  
  在每个时间点、每个维度上，对所有样本取中位数，构造一条新的典型轨迹。

- Method B：Medoid Trajectory  
  用 Frobenius norm 计算序列之间的整体距离，从真实样本里选出总距离最小的一条，作为代表序列。

Method A 更偏统计学上的中心轨迹，Method B 更偏真实存在的代表样本。

### 3.3 三条主线分析

#### 主线一：直接差异

对于任意两个类别，在同一时间点计算 prototype 向量之间的欧氏距离：

`diff_{a,b}(t) = ||p_a(t) - p_b(t)||_2`

这个量用来回答：

- 起始阶段是否已经存在差异
- 随着笑容展开，差异是否增大
- 哪些类别彼此更接近

#### 主线二：沿真笑主轴推进

以 true smile prototype 的首尾连线定义真笑主轴：

`g = p_true(19) - p_true(0)`

然后看其他类别从自身起点出发后，在这条轴上推进了多少。

#### 主线三：偏离真笑主轴

对每个时间点的动态向量，先做投影，再看剩余残差的长度：

- 投影反映沿主轴推进的分量
- 残差反映偏离真笑主轴的程度

### 3.4 样本级统计

除了 prototype trajectory，我们还对每一条 normalized sequence 做了同样的计算，并汇总：

- mean
- std
- median
- q1
- q3

这样可以同时观察：

- prototype 的行为
- 样本整体分布
- 类内波动大小

---

## 4. 已得到的主要结果

### 4.1 直接差异结果

从 [projection_summary_methodA.md](/e:/Matsuda_data/3-10meeting/methodA/report/projection_summary_methodA.md) 和 [projection_summary_methodB.md](/e:/Matsuda_data/3-10meeting/methodB/report/projection_summary_methodB.md) 可以看到：

- 各类别在 `t=0` 的距离通常较小
- 随着时间推进，类别间距离普遍增大
- `polite` 与 `ambiguous` 一般比它们和 `truesmile` 更接近

这说明：

- neutral 阶段更相似
- 笑容展开后，动态差异被放大

### 4.2 真笑主轴分析结果

沿 true-smile axis 的分析显示：

- true smile 的 prototype 的确沿这条轴推进
- 但 polite 和 ambiguous 沿这条轴推进得较少
- 尤其是 Method B 下，ambiguous 的推进甚至非常有限

这提示：

- 另外两类并没有明显沿着 true smile 的主方向展开

### 4.3 偏离真笑主轴结果

偏离分析显示：

- polite 和 ambiguous 的 off-axis deviation 较大
- 但 true smile 自己在中间阶段也并不总是贴近真笑主轴

这说明：

- true smile 的真实轨迹本身并不是一条直线
- true-smile 首尾连线只是一个粗略参考方向

### 4.4 polite 轴补充分析

根据老师在汇报中的建议，我们又额外做了一次“以 polite smile 为基轴”的偏离分析，使用独立脚本：

- [08_polite_axis_deviation_analysis.py](/e:/Single_frame_smile/Analysis/analysis_projection/08_polite_axis_deviation_analysis.py)

输出包括：

- [polite_axis_summary_methodA.md](/e:/Matsuda_data/3-10meeting/methodA/report/polite_axis_summary_methodA.md)
- [polite_axis_summary_methodB.md](/e:/Matsuda_data/3-10meeting/methodB/report/polite_axis_summary_methodB.md)

从这个补充分析中可以看到：

- polite smile 相对自身主轴的偏离较小
- true smile 相对 polite 主轴的偏离明显更大
- 在 Method A 下，true smile prototype 在后期相对 polite 轴的偏离甚至可以达到 polite 轴长度的两倍左右

这说明：

- polite smile 也可能具有自己的动态方向
- polite 与 true smile 的差异不只是“是否沿 true-smile 方向推进”，也可能是“各自走向了不同的方向”

---

## 5. 当前阶段的理解与结论

基于目前的结果，我们可以得到几条相对稳定的阶段性结论：

1. 不同笑容类别在特征空间中的动态轨迹确实不同。
2. neutral 阶段更相似，而随着笑容增强，类别间差异会被逐步放大。
3. polite 与 ambiguous 整体上更接近彼此，而都明显不同于 true smile。
4. polite 与 ambiguous 并没有明显沿 true-smile 的全局主轴展开。
5. polite 也可能具有自己的主方向，因此 polite smile 不能只被理解为“偏离真笑”。
6. true smile 的轨迹本身也不是线性的，所以“首尾连线主轴”只能看作粗粒度参考，不能代表完整动态路径。

---

## 6. 已发现的问题与方法局限

在代码实现和结果解释过程中，我们也明确发现了一些局限：

### 6.1 单一主轴模型过于粗糙

用 true smile 或 polite smile 的首尾连线定义一条主轴，是一种很直观的方法，但它本质上只是一根直线。

问题是：

- 实际笑容轨迹可能是弯曲的
- 中间阶段可能有转向
- 因此“沿主轴推进多少”和“偏离主轴多少”只能反映一部分信息

### 6.2 Prototype 曲线与样本均值曲线不能直接视为同类中心

在 band plot 里我们发现：

- prototype 曲线经常整体低于样本均值和四分位区间

这并不一定说明 prototype 异常，而是因为：

- prototype 是在高维特征空间中先构建出来的
- 均值和分位区间是对样本逐条算完指标以后再汇总得到的

由于 projection 和 deviation 是非线性的，这两种中心量本来就不能简单等价。

### 6.3 当前指标混合了方向差异与长度差异

当前使用的是未额外做单位化的 fc7 差向量，因此：

- distance
- projection
- off-axis deviation

都同时受到方向和长度的影响。

这意味着当前结果更适合解释为“综合空间差异”，而不是纯粹的夹角关系或纯方向相似性。

---

## 7. 与老师讨论后新增的思考

老师在汇报后提出了很重要的问题：

- 如果把 polite smile 而不是 true smile 当作基轴，会出现什么结果？
- 对于 Matsuda-kun 有意做出的 polite smile，它到 polite 主轴的距离是否更小？

这个问题推动我们把分析从“只以真笑为中心”扩展到“比较不同笑容是否各自具有稳定主方向”。

这一步很重要，因为它把问题从：

- “谁更像真笑”

扩展成：

- “不同笑容是否分别走向了特征空间中的不同区域”

这让当前研究问题变得更丰富，也更接近真实社交情境中的笑容差异。

---

## 8. 后续可以继续推进的方向

基于现在的工作，后续比较自然的几个方向是：

1. 不只用一根轴，而是考虑一条轨迹或一个子空间
2. 区分“笑容方向”与“笑容强度”
3. 更明确地处理 sample mean、sample median 和 prototype 之间的关系
4. 针对 Matsuda-kun 的 intentional polite smile 单独做更细的样本级分析
5. 继续比较 true-smile axis 与 polite-smile axis 下的结果异同

---

## 9. 本阶段工作产出清单

### 文档

- [projection_analysis_cn.md](/e:/Single_frame_smile/Analysis/analysis_projection/projection_analysis_cn.md)
- [projection_analysis_en.md](/e:/Single_frame_smile/Analysis/analysis_projection/projection_analysis_en.md)
- [slide.md](/e:/Single_frame_smile/Analysis/analysis_projection/slide.md)
- [slide_en.md](/e:/Single_frame_smile/Analysis/analysis_projection/slide_en.md)

### 核心脚本

- [common.py](/e:/Single_frame_smile/Analysis/analysis_projection/common.py)
- [01_build_projection_prototypes.py](/e:/Single_frame_smile/Analysis/analysis_projection/01_build_projection_prototypes.py)
- [02_compute_direct_distance.py](/e:/Single_frame_smile/Analysis/analysis_projection/02_compute_direct_distance.py)
- [03_compute_projection_metrics.py](/e:/Single_frame_smile/Analysis/analysis_projection/03_compute_projection_metrics.py)
- [04_compute_per_sequence_metrics.py](/e:/Single_frame_smile/Analysis/analysis_projection/04_compute_per_sequence_metrics.py)
- [05_compute_statistics.py](/e:/Single_frame_smile/Analysis/analysis_projection/05_compute_statistics.py)
- [06_generate_plots.py](/e:/Single_frame_smile/Analysis/analysis_projection/06_generate_plots.py)
- [07_generate_report.py](/e:/Single_frame_smile/Analysis/analysis_projection/07_generate_report.py)
- [08_polite_axis_deviation_analysis.py](/e:/Single_frame_smile/Analysis/analysis_projection/08_polite_axis_deviation_analysis.py)
- [run_projection_pipeline.ps1](/e:/Single_frame_smile/Analysis/analysis_projection/run_projection_pipeline.ps1)

### 输出结果

主输出目录：

- [3-10meeting](/e:/Matsuda_data/3-10meeting)

其中包括：

- `methodA/csv`
- `methodA/plots`
- `methodA/prototypes`
- `methodA/report`
- `methodB/csv`
- `methodB/plots`
- `methodB/prototypes`
- `methodB/report`

---

## 10. 总结

这一阶段我们已经从“静态看笑容差异”推进到“动态看笑容轨迹差异”，并且初步建立了：

- prototype 层面的时序比较
- 样本级统计比较
- 以 true smile 为基轴的分析
- 以 polite smile 为基轴的补充分析

现阶段最重要的认识是：

- 不同笑容类别确实在特征空间中走向不同区域
- polite 与 ambiguous 更接近彼此
- true smile 与另外两类存在明显差异
- 用单一首尾连线主轴分析是一个有用的第一步，但还不足以完整描述 smile dynamics

这说明我们的方法已经能支持阶段性研究判断，同时也明确暴露出了下一阶段需要优化的方向。
