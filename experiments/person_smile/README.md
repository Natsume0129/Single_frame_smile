# 每人一组 Polite / Bitter Smile Ranking 实验

这是两阶段入口：`prepare` 提取图片并生成标注任务；`run` 读取人工标签，
训练每人两套模型并分析。没有新数据时只运行 `check`。不访问旧实验的标签、
模型成品或封存测试集，不执行 RVM 或绿幕合成，不修改外部 FaceTracking / SmileComp 工程。

## 输入与启动

独立的 XU true-smile 强度标注使用 `config.XU_true.json`，输出到
`E:\Dataset\person_smile_XU_true_20261023\XU\ranking\true`。
执行 `pipeline.py prepare --config experiments/person_smile/config.XU_true.json --person XU`。
配置中的 `labels: ["true"]` 仅扩展准备阶段；`run` 对非 polite/bitter 类别会停止，
避免误启动原有两类分析。默认配置的行为不变。原75段视频保留，候选脸图沿用
正脸预处理规则；这里的 true 是原文件夹类别，人工比较的是笑容强弱，不是判断真假。
文件名 expression_001 等暂作为片段级来源组，不能据此声称跨原始录制来源独立。

原视频已经切分，且自带绿幕：

```text
input_root/
  person_001/
    polite/clip01.mp4 ...
    bitter/clip01.mp4 ...
  person_002/
    polite/...
    bitter/...
```

每个片段应只有目标人物。路径可以包含空格或中文。每类至少需要 6 个合格片段，
使 train/val/test 各至少 2 个片段；数据越少，独立评价越不稳定。
本流程不会把一个人的片段借给另一个人凑数。

复制 `config.example.json` 为自己的配置，修改 `input_root`、`output_root`。
所有路径使用绝对路径。已启动的输出目录冻结配置；换参数或数据使用新的输出目录。

在仓库根目录的 PowerShell 中：

```powershell
# 当前可运行：只检查程序与权重路径，不需要新数据
.\experiments\person_smile\run.ps1 -Stage check

# 数据准备完后执行，结束于等待人工标注
.\experiments\person_smile\run.ps1 -Stage prepare -Config "E:\path\experiment.json"

# 人工完成并导出 annotations.csv 后，才执行
.\experiments\person_smile\run.ps1 -Stage run -Config "E:\path\experiment.json"

# 可只处理一个人
.\experiments\person_smile\run.ps1 -Stage run -Config "E:\path\experiment.json" -Person person_001
```

程序不会在 `prepare` 后自动训练。

## 数据清洗和人工标注

- 调用指定 FaceTracking 的 `CUI-pyplot/face_detection.py`，沿用检测阈值 0.7、
  方向阈值 15°、224×224 裁剪，只收集方向目录 `0` 的正脸图片。
- 每条序列保留源视频、track ID 和原始帧号；不复制前帧来填补缺失帧。
- 默认少于 20 张正脸图的片段排除，原因记录在 `preparation_qc.json`。
- 多个 track 时停止，不自动猜目标人物。在人物输出目录 `track_selection.json`
  写入 `{"polite_<hash>": "0"}` 指定 track 后重跑。没有 track 的片段可人工排除。
- 手工排除片段可在同目录 `exclude_clips.json` 写入 `{"polite_<hash>": "遮挡严重"}`。
  这些决定必须在标注集冻结前完成；已经 prepared 的数据如需更改，使用新输出目录。
- 每条序列生成 `review/<clip_id>.png` 20 帧拼图。遮挡、笑容类别、片段起止并非自动判定，
  仍需人工检查；正脸方向合格不代表没有遮挡。
- 按源片段预先分配 train/val/test，并检查跨 split 的精确解码图像重复。
  不按 pair 随机切分，不自动将时间较晚的图片标为更强。
- 每序列默认均匀抽取 10 张候选图，在各 split 内生成跨序列人工比较对。
  默认 train/val/test 为 200/40/40 对；候选不足时使用实际可用数，写入 task_summary.json。

每人两个标注入口：

```text
output_root/person_001/ranking/polite/annotate.html
output_root/person_001/ranking/bitter/annotate.html
```

浏览器直接打开。`←` 表示 A 更强，`→` 表示 B 更强，`=` 表示相近。
支持上一对、下一对、本地暂存、CSV 导出及导入继续。请定期导出；浏览器暂存不是备份。
将下载的 `annotations.csv` 放回对应页面旁边。

所有 pair 必须标注为 `1/-1/0/discard`。其中 `0` 和 `discard` 保留审计、
不进入原 SmileComp 二分类损失，不伪造成有方向的标签。任一 split 没有有效方向标签就停止。
训练前验证两类完整标签，且不接受修改 pair 图片或未知 pair ID。

## 训练定义

复用原 SmileComp PyTorch 架构和现有训练/scale 构建函数：

- 初始化为原工程 `models/vggface_conv.pth`，不是旧人物已经训练的比较模型。
- 冻结 VGG-Face 卷积提取器，学习比较头；同一人物训练 polite 和 bitter 两套。
- 默认 100 epochs、Adam、lr=5e-5、batch=32；使用旧流程的交换一致性损失。
- val loss 选 checkpoint；test pair 仅评价选定模型。
- scale10 仅使用 train 序列候选，不能使用 test 图构建 anchors。
- 保留原始人工标签、划分、训练历史和 holdout 指标。

训练沿用原脚本的 CUDA 自动检测；分析设备由配置 `device` 指定。
原脚本新增的 `--use_existing_splits` 仅供读取预分配 split，旧调用默认行为不变。

## 分析定义

只分析用户指定的前四项。每人两个主模型都分析两类 **test 序列**，按序列汇总。
当前实现为描述性跨人物复现，不自动宣布差异显著。输出版本为 `four_conclusions_v2`；
旧版结果留在原目录，但新汇总不会读取它们。

1. **空间方向**：使用历史冻结 VGG-Face fc7，逐帧 L2 归一化，前 5 帧均值作 baseline。
   最大位移帧作为可复现的参考，不称为人工笑容峰值。报告幅度、类内/类间方向余弦、
   投影与离轴轨迹。起始帧未必是中性；若片段含完整 onset/offset，
   不使用最后一帧冒充峰值。此表征不受 ranking 比较头重训影响。
2. **区域相对评分变化分布**：复用原面积匹配遮罩与 local-mean 填充。
   眼周、嘴周、鼻部、脸颊分别重训、重建各自 native scale，再与未遮罩主模型比较。
   因此每人增加 8 套辅助模型，主模型仍为 2 套。使用相同 split、随机种子和训练超参数。
   面积比例只从 train 候选图冻结；保留遮罩候选图片、QC、模型和量表以便复查。
   先计算逐帧绝对分数变化，再按序列平均、两类输入等权，将四区域归一化为 100%。
   报告 polite 脸颊占比及其与其他最大占比之差，以及两模型偏离均匀分布的 total variation。
   Polite−Bitter 不均衡度为正表示 bitter 更均衡。总变化为零时占比不可定义，不报告成均衡。
   这复现原相对变化指标的两类版本，不是固定模型区域删除效应，也不是直接的器官权重。
3. **Keep-ROI**：复用旧 68 点自然区域、3px feather、11px blur 和 local-mean。
   两模型共享 10 张 train-only anchors（polite/bitter 各 5 张）。
   目标与 anchors 做相同处理；统计未截断 signed response，展示 0–9 分。
   默认每序列均匀抽 20 帧，输出序列内排序保持、偏差和填充敏感性。
   区域为上下半脸、眼周、嘴周、脸颊、鼻部；本次不沿用 true-model ROI discovery。
   上下半脸差在同一序列内配对计算，blur 和 local-mean 都报告，不只挑一种处理方式。
4. **Grad-CAM 整体相似与局部差异**：复用旧标准 ReLU Grad-CAM，两固定类别、
   A/B 两分支均保存。B 为输入类别 scale 的 level04，同一 A/B 交给两个模型。
   输出模型间整体 CAM 余弦及各区域 density 的配对模型差；先按序列聚合，再对输入类别等权。
   保留不认同 A>B 的结果和空 CAM（空图的余弦不可定义，不能记成完全相似）。
   这是两类适配协议，不是旧的逐帧选择弱 anchor 协议。

不再计算第5项 CAM 与 Keep-ROI 的相关、对应表或对应图，也不在跨人物汇总中包含它。
第2项包含遮罩重训，其他分析使用固定训练后的主模型。二者不能混称为器官必要性或因果贡献。
两类共同 anchors 替代旧 true anchors 后，数值不与旧实验逐项等同；
比较的是待复现的关系与效应方向。非显著/不稳定不能写成器官无信息。
统计推断如需 bootstrap/permutation，须另行冻结具体跨人假设；此版本不制造帧级 p 值。

## 输出与中断

```text
output_root/
  experiment_config.json
  person_001/
    tracking/                  # FaceTracking 原始输出
    preparation_qc.json
    review/                    # 序列拼图
    sequences.csv
    sequences/{polite,bitter}/
    ranking/{polite,bitter}/
      images/ ranking_items.csv pairs.csv annotate.html annotations.csv
      train.dat val.dat test.dat annotation_audit.json
      model/ scale10/
    regional_systems/
      protocol.json relative_effect_profiles.csv sequence_effects.csv frame_effects.csv
      {polite,bitter}/{periocular,mouth_perioral,nose,cheeks}/
        images/ mask_qc.csv model/ scale10/ complete.json
    logs/
    analysis/four_conclusions_v2/
      report.md summary.json summary.png
      frame_roi_scores.csv sequence_roi_summary.csv roi_cells.csv
      frame_cam_regions.csv sequence_cam_summary.csv cams/
      fill_sensitivity.csv upper_lower_paired_sequences.csv upper_lower_cells.csv
      cam_sequence_similarity.csv cam_region_model_differences.csv
      direction_sequences.csv direction_pairs.csv direction_trajectories.csv
  person_comparison.csv
  report.md
```

FaceTracking 每片段成功后写完成标志；半成品输出会停止，避免混合重跑残留。
prepared 的标注集不会重建覆盖。训练和量表完整完成后可跳过，
如果标签或配置改变则要求新输出目录。区域辅助模型完成后也会复用；
分析可重跑，会更新该人物 analysis/four_conclusions_v2。不同版本不会混入同一张跨人汇总表。
任何子程序失败都会返回非零退出码；详情在人物 logs 中。

## 当前验证范围

此封装尚无新人物数据，不能声称完成真实视频到正式报告的端到端验证。
仓库提供小型合成准备流程测试（模拟 FaceTracking 文件输出，不模拟科学结论），
覆盖按片段划分、无自动标签、人工标注阻断、标签导入和训练命令衔接。
正式数据到达后先对少量片段做实际 FaceTracking 和训练/分析冒烟运行。
