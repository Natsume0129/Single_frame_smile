# Heatmap 方案补充需求文档

## 1. 文档目的

本文件用于明确当前 heatmap 研究的正式实现方案，以及如果后续一定要使用 Grad-CAM，该如何理解和使用。

当前结论已经比较明确：

1. **主方案**
   - 使用 start frame 和 current frame 的 feature difference，直接可视化哪些区域的高层特征变化更明显

2. **补充方案**
   - 如果后续一定要使用 Grad-CAM，它更适合作为 comparison-based explanation
   - 而不是替代主方案

本文件会明确：

- 数据输入
- 模型与预处理
- 计算层
- 聚合算法
- 数据链路
- 输出目录组织
- Grad-CAM 的定位

---

## 2. 当前研究问题

当前我们真正要回答的问题是：

**在一条笑容序列中，从起始帧到当前帧，脸上哪些区域的高层特征变化最明显。**

这里关注的是：

- feature change
- spatial change pattern
- temporal evolution of change

而不是：

- 分类决策依据
- 比较模型为什么输出某一类

因此，这一版不把 Grad-CAM 作为主算法。

---

## 3. 主方案：Feature Difference Heatmap

## 3.1 核心思想

对一条序列：

- 固定起始帧 `frame_0`
- 对任意当前帧 `frame_t`
- 分别提取它们在某一卷积层的 feature map
- 计算两者差异
- 再把差异聚合成二维热力图

这样得到的热力图表示的是：

**从起始帧到当前帧，哪些区域的高层特征变化最大。**

---

## 3.2 数据输入

### 3.2.1 图像来源

来自：

- `E:\Matsuda_data\2-18meeting\polite\13`
- `E:\Matsuda_data\2-18meeting\truesmile\3`
- `E:\Matsuda_data\2-18meeting\ambiguous\27`

这些路径已经记录在：

- [source.dat](/e:/Single_frame_smile/Analysis/Heatmap/source.dat)

要求：

- 目录中的每一张 PNG 都要参与分析
- 不是只分析关键帧
- 而是看完整序列随时间的变化

### 3.2.2 序列逻辑

对于每条序列：

- 第 0 帧作为起始帧
- 第 `t` 帧作为当前帧

因此一条序列会产生：

- `frame_0 vs frame_1`
- `frame_0 vs frame_2`
- `frame_0 vs frame_3`
- ...

的一系列特征变化热力图。

---

## 3.3 模型输入预处理

要求和原始 VGGFace 特征提取流程保持一致。

### 3.3.1 预处理步骤

1. 读取图像并转换为 `RGB`
2. resize 到 `224 x 224`
3. 转成 tensor
4. 使用 VGGFace 的均值做 normalize

即：

```python
transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        (129.1863 / 255, 104.7624 / 255, 93.5940 / 255),
        (1.0, 1.0, 1.0)
    )
])
```

### 3.3.2 不做最后一步特征归一化

这里明确：

- 不对卷积特征再做 L2 normalize

原因：

- 当前目标是看高层特征的空间变化
- 不是做 embedding similarity 分析

---

## 3.4 使用的模型

### 3.4.1 模型结构

使用：

- `VGGFace_conv`

模型代码基础来自：

- [face_comp_torch.py](/e:/Single_frame_smile/Analysis/Heatmap/codes/face_comp_torch.py)

### 3.4.2 模型权重

权重路径：

- `E:\Single_frame_smile\data\models\vggface.pth`

### 3.4.3 说明

虽然当前 heatmap 不使用 `fc7`，但：

- 输入图像预处理
- backbone 结构

都与之前 VGGFace 特征提取过程保持一致。

---

## 4. 使用的层（多层并行分析）

本次实现不只做一层，而是同时做多个层的分析。

需要同时分析的层为：

1. `maxp_5_3`
2. `relu_5_3`
3. `relu_4_3`

### 4.1 每层的意义

#### `maxp_5_3`

- 与现有项目中已有 `conv5_3` 提取流程保持一致
- 空间分辨率最低
- 结果可能较粗，但和既有分析最一致

#### `relu_5_3`

- 比 `maxp_5_3` 少一次 pooling
- 保留更细的空间结构
- 是最值得重点观察的一层

#### `relu_4_3`

- 空间分辨率更高
- 细节更丰富
- 但语义层次相对更低一些

### 4.2 输出要求

每一层都需要独立输出，不能混放。

建议输出结构：

```text
output/
  maxp_5_3/
    polite/
    truesmile/
    ambiguous/
  relu_5_3/
    polite/
    truesmile/
    ambiguous/
  relu_4_3/
    polite/
    truesmile/
    ambiguous/
```

---

## 5. 核心算法

设：

- `F_0(c,h,w)` = 起始帧的 feature map
- `F_t(c,h,w)` = 当前帧的 feature map

定义特征差：

```text
ΔF_t(c,h,w) = F_t(c,h,w) - F_0(c,h,w)
```

接下来，需要对 `ΔF_t` 在通道维上做聚合，生成二维热力图。

---

## 6. 聚合算法（A / B / C / D 全部都做）

这次不是只做一种聚合方式，而是同时计算四种。

### 算法 A：绝对差的通道平均

定义：

```text
H_t^A(h,w) = mean_c |ΔF_t(c,h,w)|
```

含义：

- 在这个空间位置上，所有通道平均变化了多少

特点：

- 最直观
- 最稳定
- 易解释

### 算法 B：通道差向量的 L2 norm

定义：

```text
H_t^B(h,w) = ||ΔF_t(:,h,w)||_2
```

含义：

- 在这个空间位置上，整个通道向量整体变化了多少

特点：

- 更保留整体变化量
- 比 A 更敏感

### 算法 C：绝对差的通道和

定义：

```text
H_t^C(h,w) = Σ_c |ΔF_t(c,h,w)|
```

含义：

- 所有通道变化量直接累加

特点：

- 和 A 相近，但尺度更大

### 算法 D：相对变化量

定义：

```text
H_t^D(h,w) = ||ΔF_t(:,h,w)||_2 / (||F_0(:,h,w)||_2 + eps)
```

含义：

- 相对于起始帧，这个位置变化了多少比例

特点：

- 适合看“相对变化”
- 但数值可能更敏感

---

## 7. 热力图处理流程

对每一个 `H_t`：

1. 做 min-max normalize

```text
H'_t = (H_t - min(H_t)) / (max(H_t) - min(H_t))
```

2. 上采样到原图大小

当前参数：

- `interpolation_method = bilinear`

3. 映射为伪彩色

当前参数：

- `colormap = turbo`

4. 叠加到原图

当前参数：

- `alpha = 0.4`

---

## 8. 输出结果

### 8.1 输出目录

当前根目录：

- `E:\Single_frame_smile\Analysis\Heatmap\output`

本次按“层 / 聚合方式 / 类别”分目录：

```text
output/
  maxp_5_3/
    agg_A/
      polite/
      truesmile/
      ambiguous/
    agg_B/
    agg_C/
    agg_D/
  relu_5_3/
    agg_A/
    agg_B/
    agg_C/
    agg_D/
  relu_4_3/
    agg_A/
    agg_B/
    agg_C/
    agg_D/
```

### 8.2 每一帧输出

每一张输入图像，至少输出：

- `*_original.png`
- `*_heatmap.png`
- `*_overlay.png`
- `*_heatmap.npy`

### 8.3 序列级输出

对每个类别、每个层、每个聚合算法，还应输出：

- heatmap still image
- overlay still image
- 阈值面积随时间变化图
- 阈值热值总和随时间变化图
- 点级热值随时间变化的可视化

### 8.4 点级热值随时间变化（新增）

这一部分用于观察：

- 在同一个 feature map 上，不同空间位置的热值如何随时间变化

### 8.4.1 重要要求

这里必须明确：

- **不能直接使用每一帧各自做 min-max normalize 之后的 heatmap 值**

原因是：

- 每一帧单独归一化后，数值尺度已经被重新拉伸到 `0~1`
- 不同时间点之间的数值不再处于同一个可比尺度

因此，如果要分析“每个点的热值随时间变化”，应该使用：

- **归一化之前的原始聚合值**

也就是说，对 `H_t(h,w)` 直接分析，而不是对逐帧归一化后的 `H'_t(h,w)` 分析。

### 8.4.2 推荐展示方式

对于某一层、某一种聚合算法，设其空间尺寸为：

```text
H x W
```

把所有空间点展开后，总共有：

```text
H * W
```

个位置。

推荐输出方式为：

#### 方案 A：时间-空间热图（主推荐）

- 横轴：时间 `t`
- 纵轴：空间位置 index（共 `H*W` 个）
- 颜色：该点在该时间的原始聚合热值

如果是 `7 x 7`，则一共是 `49` 个点。

这个图的优点是：

- 最直观地展示“哪些空间位置在什么时间变强”
- 比把 49 条曲线叠在一张图上更容易读

### 8.4.3 输出文件建议

对于每个层、每个聚合算法、每个类别，输出：

- `pointwise_timeseries_heatmap.png`

建议路径结构：

```text
output/
  <layer_name>/
    <agg_name>/
      <class_name>/
        pointwise_timeseries_heatmap.png
```

### 8.4.4 图的含义

这张图表示的是：

- 某个空间位置的特征变化热值，在整条序列中如何随时间演化

它适合回答：

1. 哪些位置从一开始就活跃
2. 哪些位置是在中期才明显增强
3. 哪些位置在后期才出现明显变化
4. 不同 smile representative sequence 是否在空间-时间模式上不同

---

## 9. 这个方案回答什么问题

这个主方案回答的是：

1. 从 start frame 到 current frame，哪些区域的高层特征变化最大
2. 这些变化如何随时间展开
3. 不同 smile representative sequence 的变化模式是否不同
4. 不同层 / 不同聚合算法下，结论是否稳定

它不直接回答：

1. 模型为什么判成某个类别
2. 哪些区域是分类决策依据
3. 哪些区域对某个比较输出最关键

---

## 10. 如果一定要用 Grad-CAM

## 10.1 结论

可以做，但它应被明确看作：

- **comparison-based explanation**

而不是：

- **direct feature-change visualization**

### 10.2 输入

Grad-CAM 需要：

- 两张图同时输入 Siamese comparison model

合理输入可以是：

1. 同一序列中的起始帧 vs 当前帧
2. DTW 对齐后两条曲线在同一时间点的原始帧对

### 10.3 输出

输出的是：

- 针对某个比较结果的梯度解释热力图

也就是说，它解释的是：

- 模型比较判断的依据

而不是：

- 单纯的特征变化本身

### 10.4 当前定位

因此，Grad-CAM 在当前项目中的定位是：

- 可做
- 但不作为主结果
- 更适合作为后续补充分析

---

## 11. 当前正式建议

### 主方案

使用：

- 多层 feature difference heatmap
- 多种聚合方式对照

其中：

- 层：`maxp_5_3`、`relu_5_3`、`relu_4_3`
- 聚合：`A`、`B`、`C`、`D`

### 不变的内容

- 输入图像预处理和原 VGGFace 特征提取流程一致
- 不做最后一步特征向量归一化
- `turbo` colormap
- `alpha=0.4`
- `bilinear` interpolation

### Grad-CAM 的定位

- 不替代主方案
- 只作为后续补充分析保留
