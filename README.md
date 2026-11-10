
# 笑容强度排序与时序分类

这个项目研究同一个人表现 polite、bitter 和 true smile 时，笑容从 onset 到 peak 的变化。先用参与者自己的排序模型给每一张有效人脸帧评分，再用同一批片段比较单帧分类、曲线摘要分类和完整时序分类。当前分类实验覆盖五名已知参与者，使用 1,215 段人工确认的片段。

| 从哪里开始 | 内容 |
|---|---|
| [时序实验原始输入包（ZIP，3.09 MB）](datasets/smile_temporal_inputs_20261110.zip) | 26,680 帧的 18 维排名分数、时间与标签、1,215 段 clip 的冻结划分、177 段测试 clip 的基准预测，以及包内中文使用说明 |
| [实验结果汇总（英文）](reports/temporal_smile/README.md) | 数据划分、测试集 ACC 和 macro-F1、固定长度与变长张量对比，以及结果适用范围 |
| [标注与建模方法（中文）](Temporal_analysis_report.md) | 从人工选帧、片段内配对、个人排序模型到 18 条区域曲线的方法，以及原工作区的数据索引 |
| [标注、训练与报告源码](experiments/person_smile/) | 人脸追踪调用、标注页模板、分组划分、训练、推理和报告生成 |
| [单帧分阶段实验](experiments/person_smile/single_frame_phase_experiment_20261105.py) | 比较 3 维和 18 维输入，以及统一和个人分类器 |
| [完整时序实验](experiments/person_smile/tensor_sequence_experiment_20261109.py) | 用相同分组比较固定 20 点和原始变长序列，训练轨迹核 SVM、双向 GRU 和 TCN |

**在新电脑上使用数据包。** 克隆仓库后，在仓库根目录执行下面的 PowerShell 命令。解压后 `output/` 与 `experiments/` 并列，`README_使用说明.md` 列出了 Python 环境、5 个输入文件的用途和校验值。清单里的原视频路径用于追溯，时序分类脚本直接读取包内特征。

```powershell
Expand-Archive -LiteralPath .\datasets\smile_temporal_inputs_20261110.zip -DestinationPath .
```

**从视频重新提取分数。** 这一步使用研究机器上的原视频、追踪人脸帧、个人排序权重和参考量表。原视频位于 `new_dataset/`。追踪程序、SmileComp 与 VGGFace 权重的位置见[配置示例](experiments/person_smile/config.example.json)。

**从已有曲线重跑时序实验。** 在具备冻结输入的独立工作副本中，从项目根目录执行下列命令。训练脚本读取单帧实验的 `dataset.npz`、`frame_metadata.json` 和 `protocol.json`、原三分类器的 `feature_manifest.json`，以及区域分类器的基准预测。结果写到 `output/tensor_sequence_classification_20261109/`。训练使用 NumPy、scikit-learn、joblib 和 PyTorch，报告绘图使用 Matplotlib。

```powershell
python experiments/person_smile/tensor_sequence_experiment_20261109.py
python experiments/person_smile/render_tensor_sequence_report_20261109.py
```

## 早期工作流：单帧数据准备与标注

本项目用于构建**基于单帧正脸图像的笑容状态分类数据集**，包含从视频中抽取数据、人工标注、以及后续数据筛选与采样的完整流程。

目标任务为单帧三分类：

- `true_smile`
- `not_smile`
- `other_smile`

项目整体以**单帧视角**为核心，不依赖 segment 作为训练输入结构，但在数据准备阶段会利用笑容片段（segment）信息进行筛选。

---

## 1. 数据准备（整体流程）

### 1.1 原始数据假设

- 输入数据来自**连续视频**
- 视频中并非所有时间段都包含笑容
- 已通过 FaceTracking 提取出：
  - 仅包含**正脸**的单帧图像（224×224）
- 同时，人工或半自动方式标注了**笑容片段区间**（`segment.dat`）

因此，数据准备阶段包含两类核心数据：
1. 单帧正脸图像文件夹
2. 对应视频中的笑容片段区间定义文件（`segment.dat`）

---

## 2. 标注工具（annotation.html）

### 2.1 数据输入

- 在网页中点击“选择文件夹”
- 浏览器读取该文件夹下的所有图片文件
- 页面一次只显示一张图片（当前索引 `i`）

**说明**  
由于浏览器安全限制，纯前端 HTML 无法直接访问系统真实路径。  
通常使用：

```html
<input type="file" webkitdirectory>
````

浏览器会提供：

* `File` 对象
* `webkitRelativePath`（相对路径）

导出标注结果时，使用该相对路径作为“目录 + 文件名”的表示。

---

### 2.2 标注标签与交互设计

固定三类标签：

* `true_smile`
* `not_smile`
* `other_smile`

交互方式：

| 操作方式        | 行为            |
| ----------- | ------------- |
| 鼠标点击        | 对应标签按钮        |
| `q`         | `true_smile`  |
| `p`         | `not_smile`   |
| `Space`     | `other_smile` |
| `Backspace` | 返回上一张图片       |

* 完成当前图片标注后，**自动跳转下一张**
* 返回上一张时，可修改已有标注

---

### 2.3 标注状态管理

* 为每张图片维护一个 `path -> label` 的映射
* 当切换图片时：

  * 若该图片已被标注
  * 页面会显示当前已有的 label 状态

---

### 2.4 数据导出机制

点击 **Export .dat** 时，流程如下：

1. 将所有标注结果拼接为一段纯文本
2. 使用 `Blob` 在浏览器内存中生成临时文件
3. 触发浏览器下载行为

本质上与下载 PDF / ZIP 文件一致。

导出格式：

```
relative_path label
```

---

### 2.5 更新：
Press D → discard current image (removed from the annotation queue, not exported)

A Discard button

UI stats updated to show Loaded / Remaining / Labeled / Discarded

Hint text updated to include D



## 3. 数据准备脚本

### 3.1 data_prepare.py

**基于区间的单帧人脸图像抽取脚本**

#### 功能说明

从 FaceTracking 输出的单帧正脸图像目录中，根据记录笑容片段区间的 `.dat` 文件，批量抽取（移动或复制）区间内对应帧号的所有图像，并汇总到新的目标文件夹中。

* 以**单帧视角**工作
* 不使用 segment 作为训练结构
* 仅在数据筛选阶段利用区间信息

---

#### 输入

1. **源图像文件夹（SRC_DIR）**

   * FaceTracking 输出的正脸图像
   * 文件名格式：

     ```
     <PREFIX><frame_number><EXT>
     ```
   * 示例：

     ```
     20250926_0_0_155.png
     ```

2. **区间定义文件（DAT_PATH）**

   * 文本格式
   * 每行表示一个区间
   * 仅解析前两列（起始帧号、结束帧号）

   示例：

   ```
   # start_frame end_frame label remark
   155 299 false_smile
   783 863 false_smile
   5264 5377 false_smile
   ```

---

#### 输出

* 一个新的目标文件夹（`DST_DIR`），包含：

  * 所有区间内且实际存在的帧图像
  * 可选生成 `missing_frames.txt`，记录区间内缺失的帧（例如因非正脸被 FaceTracking 过滤）

---

#### 核心功能

1. **区间解析与去重**

   * 展开所有 `[start_frame, end_frame]`
   * 合并为唯一帧号集合
   * 即使区间重叠，同一帧只处理一次

2. **自动容错**

   * 缺失帧自动跳过
   * 不中断执行
   * 可记录到 `missing_frames.txt`

3. **文件操作模式**

   * 支持 `move / copy`
   * 支持：

     * Dry-run
     * 是否覆盖已有文件

---

### 3.2 select_data.py

**多目录随机抽样单帧图像脚本**

#### 功能说明

从任意数量的图像文件夹中汇总所有单帧图像，并随机抽取指定数量的样本，统一输出到一个新的文件夹中。

主要用于：

* 构建**预标注数据集**
* 构建正式数据集前的随机子集
* 在不引入时间 / segment 结构的情况下进行单帧采样

---

#### 输入

1. **源目录列表（SOURCE_DIRS）**

   * 支持任意数量目录
   * 每个目录包含 FaceTracking 输出的正脸单帧图像
   * 可配置是否递归遍历

2. **采样数量（SAMPLE_N）**

   * 如：`1000` 用于预标注
   * 后续可调整为更大数值

---

#### 输出

* 一个输出目录（`OUTPUT_DIR`），包含：

  * 随机抽取的 `SAMPLE_N` 张图像
  * 若发生文件名冲突，自动追加后缀（`__2`, `__3`, …）

---

#### 核心功能

1. **多目录汇总**

   * 自动收集所有符合扩展名的图像
   * 支持：

     ```
     .png / .jpg / .jpeg / .bmp / .webp
     ```

2. **随机抽样**

   * 使用 `random.sample`（无放回）
   * 支持固定随机种子（可复现）
   * 当采样数大于可用样本数时直接报错

3. **文件输出控制**

   * 支持 copy / move
   * 是否覆盖已有文件可配置
   * Dry-run 模式支持

---

## 4. 设计原则总结

* **单帧视角优先**
* segment 仅用于数据筛选，不作为模型输入结构
* 所有工具均支持：

  * 可回滚
  * 可复现
  * 可扩展

该流程适用于后续：

* 单帧笑容分类实验
* 预标注 / 正式标注阶段切换
* 不同采样策略的对比实验
---
12-22
我想要分别一张图是真笑和假笑。
但是我发现有很多的照片我是没有办法分辨的。
所以针对这个，我在想。
先识别出这个人在笑再说。



12-23:
我标注的图片的来源是我之前的笑容片段。
我发现，即使数据源是笑容片段，还是有非常多的单帧数据被我标注为没有在笑。
不同人其实对同一张图片的认知其实是不同的
所以在打标的时候，是不是用投票机制会好一点？
我自己用了voting

增加了voting脚本

不要在全量数据上先删再切分。正确做法是：

先切分

只对 train 做下采样/增强

val/test 保持原分布（否则评估失真）

所以是数据清洗之后拆分，然后做下采样和增强

0 448条 
1 254条

路径需要对齐


cd E:\Single_frame_smile
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
