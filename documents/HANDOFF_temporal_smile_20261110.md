# 换机交接：笑容强度曲线与三分类

更新于 2026-11-10。仓库为 [Natsume0129/Single_frame_smile](https://github.com/Natsume0129/Single_frame_smile)。数据包随提交 `51907f8` 进入 `main`。

**现在可以从逐帧分数直接接着分析。** 我们已经把五位参与者的 1,215 段笑容整理成从 onset 到 peak 的时间序列。每个有效追踪帧有 18 个分数。克隆仓库并解压随附数据包，就能画曲线、按人或类别统计、训练单帧或完整序列分类器。要回看人脸、修改视频边界或给新视频打分，则需从原研究机器迁移相应视频、追踪帧、模型权重和参考量表。

## 1. 新电脑从这里开始

下面使用 Windows PowerShell。首次使用时执行以下命令；已有仓库副本可在仓库内执行 `git pull`，从解压步骤继续。

```powershell
git clone https://github.com/Natsume0129/Single_frame_smile.git
cd Single_frame_smile
Expand-Archive -LiteralPath .\datasets\smile_temporal_inputs_20261110.zip -DestinationPath .
notepad .\README_使用说明.md
```

解压后，`output/` 和 `experiments/` 应位于同一层。包内中文说明给出了创建虚拟环境、安装指定版本依赖的命令，以及五个文件的 SHA-256。原实验使用 Python 3.10.11，时序脚本支持 CPU 和 CUDA。[下载或查看数据包](../datasets/smile_temporal_inputs_20261110.zip)

## 2. 我们做了什么

| 阶段 | 已完成的工作 |
|---|---|
| 视频分类与人脸追踪 | 整理每人的 polite、bitter、true 三类原视频，对视频运行 face tracking，保留成功追踪的人脸帧、源帧号和时间。当前队列为 JIANG、JIE、TAKEDA、XU、YAMAMOTO。REN 按研究约定排除。|
| 笑容强度排序标注 | 早期版本每段均匀抽约 10 帧，人工比较同人、同类别、不同视频的图片对。本轮改为人工接受上升区间并选节点，将间隔至少 2 个源帧的节点两两配对，以较晚节点为较强。这个假设只用于已接受区间的选中节点，允许节点之间短暂回退。最终 323 段、6,832 对用于排序任务。|
| 个人模型与量表 | 每人每类训练一个基础全脸排序模型，共 15 个。每个模型仅用其训练集建立 10 级参考量表，通过与参考帧比较得到 0–9 的连续强度分数。各模型有自己的量表，分数表示相对强度。|
| 完整视频评分与边界 | 先对 1,235 段原视频的 120,364 个有效追踪帧评分，再由脚本提出 onset 和 peak，人工复核、调整或 discard。最终接受 1,215 段，舍弃 20 段。当前分析采用人工确认的 **onset→peak**；旧的 1,029 段 start→peak 清单属于更早版本。|
| 为什么比较多条曲线 | 在同一段片上运行该参与者自己的三种模型。基础全脸模型中，true 模型对 polite、bitter、true 三类视频的增幅中位数依次为 2.37、2.21、6.69。这个观察促使我们把多个模型的响应组合起来预测类别。|
| 区域模型与下游分类 | 再训练 75 个保留局部区域的排序模型和 15 个同训练协议的全脸对照。对同一人的每帧同时使用 6 个区域 × 3 个类别模型，得到 18 维向量。先用曲线摘要分类，再比较单帧、固定长度张量和原始变长张量。|

眼周、鼻部及鼻翼、口周及嘴角、上半脸、下半脸各构成一个局部区域。包内的 3 维全脸分数取自同协议对照，是 18 维的前三列；早期基础全脸三模型另有训练记录。详细标注规则和评分公式见[研究方法报告](../Temporal_analysis_report.md)。

## 3. 数据包里的五个文件怎么用

以下路径都相对于仓库根目录。

| 文件 | 内容与关联键 |
|---|---|
| `output/single_frame_phase_classification_20261105/dataset.npz` | 数组 `X` 形状为 `(26680, 18)`，每行是一个实际追踪帧的原始分数。|
| 同目录 `frame_metadata.json` | 用 `row_id` 取 `X` 的行；按 `clip_id` 聚合，按 `clip_seconds` 排序。`source_frame`、`source_seconds` 对应原视频，`clip_seconds` 从 onset 的 0 秒开始。`phase` 取 0–9，表示相对时长的十等分，peak 归入最后一段。|
| 同目录 `protocol.json` | `feature_space_definitions["18d"]` 给出确切列序。区域依次为全脸、眼周、鼻部、口周、上半脸、下半脸；各区域内部依次为 `true`、`bitter`、`polite` 模型。|
| `output/three_class_curve_classifier_20261102/feature_manifest.json` | 每段一条记录。用 `id` 对应帧表的 `clip_id`，从 `label` 取人工类别，`features.duration` 取时长（秒），`split` 取冻结划分，`component` 标识同源或精确重复帧连接而成的分组。|
| `output/regional_ranking_extension_20261103/classifier/test_predictions.json` | 177 段测试片的历史预测。用 `id` 对齐，用 **`prediction`** 取平滑 73 维 SVM 的预测。`baseline_prediction` 对应更早的模型。在最终评估时读取。|

数据包提供了平滑 73 维基准的测试预测，可直接用于比较。若要逐项复核当时的平滑特征或加载当时拟合的 SVM，原研究机器上的目录为 `output/regional_ranking_extension_20261103/classifier/`。包里的 `X` 保存未平滑的 18 个通道。

**先取出一条曲线。** 以下 Python 示例可以直接在仓库根目录运行。它取第一条训练 clip，读出真实时间轴和 T × 18 矩阵。换成指定 `clip_id` 即可分析另一段。时长分析使用 `clip_seconds`；分析原始变化速度时，应保留相邻帧的实际时间差。

```python
import json
from pathlib import Path
import numpy as np

out = Path("output")
frames_dir = out / "single_frame_phase_classification_20261105"
frames = json.loads((frames_dir / "frame_metadata.json").read_bytes())
clips = json.loads((out / "three_class_curve_classifier_20261102/feature_manifest.json").read_bytes())
columns = json.loads((frames_dir / "protocol.json").read_bytes())["feature_space_definitions"]["18d"]
clip = next(c for c in clips if c["split"] == "train")
rows = sorted((r for r in frames if r["clip_id"] == clip["id"]), key=lambda r: r["clip_seconds"])
with np.load(frames_dir / "dataset.npz") as data:
    scores = data["X"][[r["row_id"] for r in rows]]
seconds = np.array([r["clip_seconds"] for r in rows])
print(clip["person"], clip["label"], scores.shape, seconds[-1])
true_full_face_curve = scores[:, columns.index("full__true")]
```

## 4. 已得到的结果

训练、验证、测试固定为 **879 / 159 / 177 段**，合计 471 段 polite、395 段 bitter、349 段 true。每段及其全部帧都在同一集合，同一 `component` 也只出现于一个集合。上游排序模型训练和验证使用过的来源组已从下游测试来源组中排除。

**每段输出一个类别**，下表的分母均为同一批 177 段测试片，每段等权。

| 输入与模型 | 测试正确数 | ACC | 这个结果回答什么 |
|---|---:|---:|---|
| 基础全脸三条曲线的增幅＋时长，逻辑回归 | 156 | 88.14% | 简单的整段摘要已有分类信息 |
| 平滑曲线的 73 维摘要，RBF-SVM | 158 | 89.27% | 主要比较基准。每条曲线取起点、终点、增幅、时间加权平均，再加一项时长 |
| 原始曲线的 73 维摘要，RBF-SVM | 157 | 88.70% | 使用未平滑分数的对照 |
| 变长序列，余弦系数＋RBF-SVM | 154 | 87.01% | 完整时序实验中由验证集选出的主模型 |
| 固定 20 点 TCN / 变长 TCN | 161 / 159 | 90.96% / 89.83% | 固定 20 点 TCN 的观察值在这次家族比较中最高 |
| 固定 20 点 / 变长双向 GRU | 159 / 159 | 89.83% / 89.83% | 两种输入长度都可以训练 |

每条原始序列有 5–97 个有效帧。固定方案在 onset→peak 之间按真实时间等距取 20 点，对分数插值；跨越超过 0.15 秒追踪缺口的插值位置被屏蔽。变长方案使用全部有效观测和时间差。主时序模型相对平滑 73 维基准少判对 4 段，差值 −2.26 个百分点，来源组配对 bootstrap 的 95% 区间为 −6.08 至 +1.65。**目前尚不能确认完整时序输入提高了准确率。**

**每帧输出一个类别。** 将 onset→peak 的时间十等分，每个阶段单独训练。下表按每个有观测的“clip × 阶段”等权汇总，长片不会仅因帧更多而占更大权重。

| 分类器 | 3 维全脸 | 18 维全脸＋区域 |
|---|---:|---:|
| 五人合并训练 | 68.55% | 83.81% |
| 每人分别训练 | 74.60% | 85.31% |

帧标签继承所属 clip 的类别，包括 onset 附近强度很弱的帧。阶段计算使用人工确认的 peak，因此结果描述已知片段边界下的离线识别。统一分类器合并了五个人，而输入分数仍来自每个人自己的排序模型。跨新人表现需要另行评估。[完整结果、macro-F1 和混淆矩阵](../reports/temporal_smile/README.md)

## 5. 从哪个脚本继续

以下命令假设已按包内说明安装依赖，在仓库根目录运行。**两组实验可以独立运行。**

**完整时序分类。** 比较固定 20 点与原始变长序列，输出训练好的模型、逐段预测和报告。

```powershell
.\.venv\Scripts\python.exe experiments/person_smile/tensor_sequence_experiment_20261109.py
.\.venv\Scripts\python.exe experiments/person_smile/render_tensor_sequence_report_20261109.py
```

结果入口为 `output/tensor_sequence_classification_20261109/index.html`。同目录的 `selection_before_test.json` 记录验证集选择，`test_predictions.json` 给出主方案与基准的逐片预测。已有完整 `report_data.json` 时训练脚本会停止。比较新模型时请使用独立的实验输出目录，便于按 clip 对照既有预测。

**单帧分阶段分类。** 数据包已经包含准备好的特征，直接从训练开始即可。

```powershell
.\.venv\Scripts\python.exe experiments/person_smile/single_frame_phase_experiment_20261105.py --train --report
.\.venv\Scripts\python.exe experiments/person_smile/render_single_frame_phase_reports_20261105.py
```

结果入口为 `output/single_frame_phase_classification_20261105/index.html`。其中四份报告对应五人合并／按人训练 × 3 维／18 维输入，可以查看类别、参与者与时间阶段的交叉结果。

**继续比较模型时要保留的条件。** 沿用清单中的 `split` 和 `component`；只在训练集拟合标准化，在验证集选择模型。保留每段的预测，再与 73 维基准按同一 `id` 做配对比较。模型输入使用排名分数和预先定义的时间特征，真实类别和文件路径仅用于标注或索引。现有 177 段测试片已用于多轮研究，且边界审阅参考过真实类别对应的曲线。若要确认固定 20 点 TCN 的增益，下一项独立检验是先冻结方案，再评估一批新的、独立审阅的片段。
