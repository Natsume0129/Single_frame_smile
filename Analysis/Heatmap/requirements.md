# 方案 A 需求文档

## 项目名称

基于 VGGFace 最后卷积层激活的面部区域热力图可视化

---

## 1. 背景与目的

当前我们已经在笑容分析流程中使用 VGGFace 提取高维特征，并主要使用 `fc7` 的 4096 维特征做后续分析。

现在希望进一步回答一个更直观的问题：

**模型在提取高层视觉特征时，主要对脸部的哪些区域响应更强？**

由于 `fc7` 本身不保留空间结构，因此不能直接从 `fc7` 反推出模型在图像空间中的关注区域。

因此，本方案拟使用：

- VGGFace 的最后卷积层 `conv5_3`
- 对其输出的 feature map 做通道聚合
- 得到二维热力图
- 再将热力图上采样并叠加回原始图片

这样可以直观看到：

**模型在高层卷积表征阶段，对面部哪些区域整体响应更强。**

需要明确的是，这里的热力图含义是：

- 高层卷积响应的空间分布

而不是：

- 人眼注视点
- attention 权重
- gradient-based saliency
- 分类决策的严格证据

---

## 2. 分析对象

本方案分析的是我们已经挑选好的三个代表性序列。

路径来自：

- [source.dat](/e:/Single_frame_smile/Analysis/Heatmap/source.dat)

当前对象为：

- `polite_source_dir=E:\Matsuda_data\2-18meeting\polite\13`
- `truesmile_source_dir=E:\Matsuda_data\2-18meeting\truesmile\3`
- `ambiguous_source_dir=E:\Matsuda_data\2-18meeting\ambiguous\27`

这些目录中的 **每一张 PNG** 都需要生成热力图。

也就是说：

- 不是只看关键帧
- 不是只看单张图片
- 而是看完整序列的热力图变化

---

## 3. 模型与输入预处理

### 3.1 模型

模型路径来自：

- `vggface_model_path=E:\Single_frame_smile\data\models\vggface.pth`

使用模型：

- VGGFace

目标层为：

- `conv5_3`

对应输出形状为：

```text
512 x 7 x 7
```

### 3.2 输入图像预处理

本方案要求与原来的 VGGFace 特征提取流程保持一致。

输入图像预处理使用以下步骤：

1. 读取图像并转换为 `RGB`
2. resize 到 `224 x 224`
3. 转为 tensor
4. 使用 VGGFace 的均值做 normalize

具体为：

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

### 3.3 不做最后一步特征归一化

这里明确：

- **不需要**对最后得到的卷积特征再做 L2 normalize

原因是：

- 我们现在要观察的是 `conv5_3` 的空间响应强度分布
- 不是做 `fc7` embedding 相似度分析

因此，原来 `feature_extractor_fc7.py` 中最后那种：

```python
F.normalize(feats, dim=1)
```

在本方案中不使用。

---

## 4. 核心思路

对于输入图像，经过 VGGFace 前向传播后，提取 `conv5_3` 的输出：

```text
F ∈ R^(C x H x W)
```

在这里：

```text
C = 512
H = 7
W = 7
```

然后对通道维做聚合。

第一版固定使用：

- **channel-wise mean**

即：

```text
Hmap(i,j) = mean_c F(c,i,j)
```

得到一个二维热力图：

```text
Hmap ∈ R^(7 x 7)
```

之后：

1. 对热力图做最小-最大归一化
2. 上采样回原图大小
3. 用伪彩色显示
4. 叠加回原始图片

最终形成 heatmap 和 overlay 结果。

---

## 5. 参数设置

参数同样从 [source.dat](/e:/Single_frame_smile/Analysis/Heatmap/source.dat) 中读取。

当前已确定的参数为：

- `output_dir=E:\Single_frame_smile\Analysis\Heatmap\output`
- `heatmap_alpha=0.4`
- `interpolation_method=bilinear`

另外补充一个默认约定：

- colormap 使用 `turbo`

---

## 6. 输出要求

### 6.1 输出目录组织

由于这次每个类别只分析一个已经选定好的代表序列，因此：

- **不需要再额外加 sequence_id 子目录**

建议输出结构为：

```text
E:\Single_frame_smile\Analysis\Heatmap\output\
├── polite\
├── truesmile\
└── ambiguous\
```

在每个类别目录下，按原始图片文件名输出对应结果。

例如：

```text
output/polite/
  20251029_23-36-23-40_0_0_80_original.png
  20251029_23-36-23-40_0_0_80_heatmap.png
  20251029_23-36-23-40_0_0_80_overlay.png
  20251029_23-36-23-40_0_0_80_heatmap.npy
```

### 6.2 输出内容

对每一张输入图片，至少输出：

1. 原图副本
2. 归一化后的 heatmap 图
3. heatmap 叠加图
4. heatmap 数组文件

建议命名：

- `*_original.png`
- `*_heatmap.png`
- `*_overlay.png`
- `*_heatmap.npy`

---

## 7. 功能需求

### 7.1 基础功能

系统应能完成：

1. 加载 VGGFace 模型权重
2. 对输入 PNG 批量做预处理
3. 提取 `conv5_3` 特征图
4. 做 channel-wise mean 聚合
5. 归一化热力图
6. 上采样到原图尺寸
7. 生成 heatmap 和 overlay
8. 保存结果

### 7.2 序列处理

系统应支持：

1. 读取目录下全部 PNG
2. 按文件名顺序处理
3. 形成一整段热力图序列

第一版不要求：

- 自动挑选关键帧
- 只分析起点 / 中点 / 峰值帧

因为当前目标就是先看完整序列。

### 7.3 可扩展但暂不实现的内容

以下内容先不做：

- channel-wise max
- channel selection
- CAM / Grad-CAM
- gradient attribution
- 不同层比较
- 自动关键帧筛选

---

## 8. 结果解释原则

输出热力图表示的是：

- 模型在 `conv5_3` 高层卷积表征阶段，对图像中不同空间区域的整体响应强度

它适合用于：

1. 观察模型是否主要响应在面部关键区域
2. 观察不同表情阶段的响应变化
3. 检查模型是否错误地对背景、头发、边缘区域响应过强

它不适合直接用于：

1. 解释模型分类决策
2. 证明某一区域就是决定表情类别的因果依据

---

## 9. 当前第一版实现边界

第一版明确只做下面这些内容：

1. 分析三个已选定的 representative sequence
2. 每张 PNG 都生成 heatmap
3. 使用 `conv5_3`
4. 使用 channel-wise mean
5. 使用 `turbo` colormap
6. 使用 `alpha = 0.4`
7. 使用 `bilinear` 上采样
8. 输入预处理和原 VGGFace 特征提取流程一致
9. 不做最后一步特征归一化

这版完成后，再决定是否需要扩展到更复杂的方法。
