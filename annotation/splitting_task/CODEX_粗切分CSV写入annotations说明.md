# 给 Codex 使用：粗切分 CSV 写入 annotations.csv 说明

本文件用于指导 Codex 把 Gemini 生成的单视频粗切分 CSV 转换并写入现有标注系统使用的 `annotations.csv`。本方案采用方案 B：不改现有标注工具、不新增临时 `smile` 标签，粗 smile 在项目 CSV 中暂时写为 `smiling_but_ambiguous`，并通过 `note` 标明它是待人工精标的粗切分结果。

## 1. 目标

把 Gemini 输出的中间 CSV：

```csv
start_frame,end_frame,label,confidence,note
```

转换为现有文件：

```text
E:\Single_frame_smile\annotation\dataset-annotation\annotations.csv
```

转换完成后，用户打开对应视频时，标注工具右下角应能看到已经预填的 episode 列表。用户随后逐条检查边界，把粗 smile 改成具体笑容类型，并补充 peak frame。

## 2. 不要修改现有系统

本任务不修改现有标注工具代码，不修改 `annotation_store.py` 的标签体系，不要求 UI 支持新标签。

现有可用标签为：

```text
truesmile
polite_smile
bitter_smile
smiling_but_ambiguous
neutral
discard
```

因此 Gemini 的粗标签需要按规则映射到这些标签。

## 3. 输入

Codex 执行导入时，需要用户提供：

- 视频文件目录，或一个明确的视频文件列表。
- Gemini 为每个视频生成的粗切分 CSV 文件。
- 目标 `annotations.csv` 路径，默认是：
  ```text
  E:\Single_frame_smile\annotation\dataset-annotation\annotations.csv
  ```
- 导入策略：追加新 rows，还是替换某些视频已有 rows。

推荐目录结构：

```text
E:\Single_frame_smile\annotation\splitting_task
  GEMINI_视频粗切分任务说明.md
  CODEX_粗切分CSV写入annotations说明.md
  gemini_outputs
    <video_stem>.coarse.csv
```

其中 `<video_stem>` 必须能和视频文件 stem 对上。例如：

```text
E:\Matsuda_data\split_videos\GX010001_part001.mp4
E:\Single_frame_smile\annotation\splitting_task\gemini_outputs\GX010001_part001.coarse.csv
```

## 4. 多文件对齐规则

每个 Gemini CSV 只能对应一个视频文件。

优先使用显式 manifest 对齐。如果没有 manifest，则使用文件名 stem 对齐：

```text
video:      GX010001_part001.mp4
gemini csv: GX010001_part001.coarse.csv
```

对齐时必须检查：

- 一个视频最多对应一个 Gemini CSV。
- 一个 Gemini CSV 最多对应一个视频。
- 如果不同目录下有相同 stem 的视频，不能仅凭 stem 自动匹配，必须让用户提供显式映射。
- `annotations.csv` 中写入的 `clip_path` 必须是视频文件的绝对路径。
- `video_id` 使用视频文件 stem。

如果视频来自重叠切分后的文件，`start_frame` 和 `end_frame` 都使用切分后 clip 的局部帧号，不换算成原始大视频帧号。

默认行为是不做跨 clip 去重。因为当前标注工具是打开一个 clip 后显示该 clip 的 episode；同一原始视频内容如果出现在两个重叠 clip 中，会作为两个 clip-local episode 存在。只有当用户明确要求原始视频级去重时，才使用 `split_mapping.csv` 设计额外去重策略。

## 5. Gemini CSV 校验

写入前必须逐文件校验。

表头必须是：

```csv
start_frame,end_frame,label,confidence,note
```

每一行必须满足：

- `start_frame` 和 `end_frame` 是整数。
- `0 <= start_frame < end_frame <= total_frames - 1`。
- 标签只能是 `smile`、`neutral`、`discard`。
- 如果 Gemini 旧输出中出现 `occlusion`，可以转换为 `discard`，但必须在导入报告或终端输出中说明。
- `confidence` 是 0 到 1 之间的小数。
- 行按 `start_frame` 升序排列。
- 片段不能重叠。
- 建议连续满足：
  ```text
  next.start_frame = previous.end_frame + 1
  ```

如果发现大段缺口、重叠、越界、无法解析的标签，停止导入该视频并报告问题，不要静默写入错误 rows。

## 6. 帧边界语义

Gemini 输出使用闭区间：

```text
start_frame 和 end_frame 都属于该片段。
```

Codex 写入 `annotations.csv` 时保留 Gemini 的帧号，不自动改成半开区间。

注意：现有标注工具人工保存时可能会把下一段 start 设为上一段 end。导入阶段不因此改写 Gemini 输出；导入目标是让用户能看到可调整的初始 episode。

## 7. 标签映射

写入 `annotations.csv` 时使用以下映射：

```text
Gemini smile   -> annotations.csv main_label = smiling_but_ambiguous
Gemini neutral -> annotations.csv main_label = neutral
Gemini discard -> annotations.csv main_label = discard
Gemini occlusion -> annotations.csv main_label = discard
```

`smile` 被写成 `smiling_but_ambiguous` 只是方案 B 的兼容做法，不代表最终分类。必须在 `note` 里保留原始标签。

## 8. annotations.csv 字段写入规则

每条 Gemini row 转换为一条 episode row。

字段建议：

```text
episode_id: 全局递增，格式 E000001
video_id: 视频文件 stem
clip_path: 视频绝对路径
person_id: 留空，除非用户提供
start_frame: Gemini start_frame
peak_frame: 留空
end_frame: Gemini end_frame
start_time: start_frame / fps
peak_time: 留空
end_time: end_frame / fps
main_label: 按标签映射规则写入
confidence: 把 Gemini 0-1 confidence 转换为 1-5 整数
intensity: 3
eye_involvement: 3
mouth_movement: 3
cheek_raise: 3
symmetry: unknown
visible_quality: 根据 Gemini confidence 粗略映射
usable_for_training: no
note: 记录这是自动粗切分，以及 Gemini 原标签和原 confidence
occlusion_type: none
occlusion_start_frame: 留空
occlusion_end_frame: 留空
occlusion_severity: none
occlusion_note: 留空
occlusion_segments: []
```

confidence 转换建议：

```text
project_confidence = round(gemini_confidence * 4 + 1)
```

然后 clamp 到 1 到 5。

visible_quality 转换建议：

```text
gemini_confidence >= 0.85 -> good
gemini_confidence >= 0.65 -> medium
otherwise -> poor
```

`usable_for_training` 必须写 `no`。这些 rows 只是预标注，人工修正前不能直接进入训练。

## 9. note 格式

为了避免把粗 smile 和真实 `smiling_but_ambiguous` 混淆，`note` 必须保留机器粗切分信息。

建议格式：

```text
auto_coarse_gemini; raw_label=smile; raw_confidence=0.95; needs_fine_label_and_peak; 嘴角上扬，出现明显笑容
```

neutral 示例：

```text
auto_coarse_gemini; raw_label=neutral; raw_confidence=0.90; 自然表情，没有明显笑意
```

discard 示例：

```text
auto_coarse_gemini; raw_label=discard; raw_confidence=0.85; 手遮挡嘴部和下半张脸
```

如果 Gemini 输出了旧标签 `occlusion`，建议写：

```text
auto_coarse_gemini; raw_label=occlusion; normalized_label=discard; raw_confidence=0.85; 手遮挡嘴部和下半张脸
```

## 10. 写入策略

写入前必须备份当前 `annotations.csv`。备份应放在现有备份目录：

```text
E:\Single_frame_smile\annotation\backups
```

默认不要盲目重复导入同一个视频。导入时应检查目标 CSV 中是否已经存在相同 `clip_path` 或相同 `video_id` 的 rows。

建议策略：

- 如果目标视频没有已有 rows，可以直接追加。
- 如果目标视频已有 rows，默认跳过该视频并报告，不自动覆盖。
- 如果用户明确说“替换这些视频的粗切分结果”，也只能替换未人工精标的自动粗切分 rows。
- 不要删除其他视频的 rows。

多文件导入时，应先在内存中完成所有校验，再一次性写入，避免只导入一半。

## 11. 增量导入规则

本流程必须支持增量导入。增量导入的核心原则是：新数据可以追加，已经人工精标的数据永远不自动改动。

### 11.1 新视频追加

如果导入的视频在现有 `annotations.csv` 中没有相同 `clip_path` 或相同 `video_id` 的 rows：

- 直接追加 Gemini 粗切分转换后的 rows。
- `episode_id` 从当前最大编号继续递增。
- 不影响已有视频的 rows。

这是默认且最安全的增量模式。

### 11.2 已有视频默认跳过

如果导入的视频在现有 `annotations.csv` 中已经存在 rows：

- 默认跳过该视频。
- 在导入报告中列出该视频已有多少 rows。
- 不删除、不覆盖、不合并这些已有 rows。

这样可以避免新一轮 Gemini 粗切分覆盖用户已经完成的精细标注。

### 11.3 只允许替换未精标的自动粗切分 rows

只有当用户明确要求“替换这些视频的粗切分结果”时，才允许删除旧 rows 后重新导入。

即使用户要求替换，也只能删除同时满足以下条件的 rows：

```text
note 包含 auto_coarse_gemini
usable_for_training = no
peak_frame 为空
main_label 是 neutral、discard 或 smiling_but_ambiguous
```

这些 rows 可以认为仍然是未完成的自动粗切分结果。

### 11.4 永远保护人工精标 rows

任何情况下都不能自动删除或覆盖满足以下任一条件的 rows：

```text
usable_for_training = yes
peak_frame 非空
main_label 是 truesmile、polite_smile 或 bitter_smile
note 不包含 auto_coarse_gemini
```

如果同一个视频中同时存在可替换的自动粗切分 rows 和需要保护的人工精标 rows，默认停止该视频的替换并报告冲突。除非用户给出更具体的人工确认，否则不要混合替换。

### 11.5 增量导入报告

每次导入完成后，应报告：

- 新增了哪些视频、多少 rows。
- 跳过了哪些已有视频、原因是什么。
- 替换了哪些视频的未精标自动粗切分 rows。
- 发现了哪些受保护的人工精标 rows。
- `annotations.csv` 的备份文件路径。

## 12. 导入后验证

导入完成后至少验证：

- `annotations.csv` 能被 `AnnotationStore.read_rows()` 读取。
- 新增 rows 的 `clip_path` 都是存在的视频文件。
- 每个导入视频都能通过 `episodes_for_video(video_path)` 查到 rows。
- 新增 rows 的 `episode_id` 没有重复。
- `main_label` 只包含现有系统支持的标签。
- `smile` 粗标签都已经转为 `smiling_but_ambiguous`，并且 note 中含有 `raw_label=smile`。
- 所有新增 rows 的 `usable_for_training` 都是 `no`。

如果环境缺少 PySide6，不影响 CSV 导入验证；只需要说明没有启动 GUI。

## 13. 用户后续人工处理方式

导入后，用户在现有标注工具中：

1. 打开对应视频文件。
2. 在右下角 episode 列表中选择自动生成的片段。
3. 检查并调整 `start_frame`、`end_frame`。
4. 对 `main_label=smiling_but_ambiguous` 且 note 含 `raw_label=smile` 的行，判断真实类型：
   ```text
   truesmile
   polite_smile
   bitter_smile
   smiling_but_ambiguous
   ```
5. 为最终 smile 类型补充 `peak_frame`。
6. 确认可用于训练后，把 `usable_for_training` 改为 `yes`。

neutral 和 discard rows 通常只需要检查边界和大类是否正确。

## 14. Codex 执行请求模板

用户可以这样要求 Codex：

```text
请把 Gemini 粗切分结果写入 annotations.csv。

视频目录：
E:\Matsuda_data\split_videos

Gemini 输出目录：
E:\Single_frame_smile\annotation\splitting_task\gemini_outputs

目标 CSV：
E:\Single_frame_smile\annotation\dataset-annotation\annotations.csv

导入策略：
如果某个视频已有 rows，先停下来告诉我，不要自动覆盖。
```

如果用户明确要替换某些视频：

```text
请把这些 Gemini 粗切分结果写入 annotations.csv。
对于同名 video_id 或相同 clip_path 的旧 rows，先备份 annotations.csv，然后只替换这些视频对应的 rows。
不要影响其他视频。
```
