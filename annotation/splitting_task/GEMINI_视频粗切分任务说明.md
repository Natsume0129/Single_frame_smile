# 给 Gemini 使用：视频表情粗切分任务说明

本文件只用于指导 Gemini 对单个视频进行视觉粗切分。Gemini 的任务是判断视频中每一段的粗表情状态，并输出一个小型中间 CSV。Gemini 不需要理解本项目的完整 `annotations.csv` schema，也不要输出 `episode_id`、`clip_path`、`video_id`、`peak_frame` 等项目内部字段。

## 1. 任务目标

对给定的单个视频文件，按照帧号顺序切分成连续片段。每个片段只标注三类粗状态之一：

```text
smile
neutral
discard
```

本阶段只是粗切分，不做精细笑容分类，不标注 peak frame。后续人工会在本地标注工具中检查这些片段，把粗 `smile` 片段进一步改成具体笑容类型，并补充 peak frame。

## 2. 每次输入

每次只处理一个视频文件。用户应在 Gemini 对话中提供：

```text
video_filename: <当前视频文件名>
total_frames: <视频总帧数>
fps: <视频帧率>
frame_index_base: 0
```

说明：

- 帧号从 0 开始。
- 最后一帧的帧号是 `total_frames - 1`。
- 如果用户处理的是切分后的视频片段，Gemini 只需要使用该片段自己的局部帧号，不要换算回原始大视频帧号。
- 如果多个视频需要处理，必须一个视频一个输出，不能把不同视频混在同一个 CSV 里。

## 3. 输出格式

只输出 CSV 文本。不要输出 Markdown 表格，不要输出 JSON，不要输出代码块，不要输出解释。

表头必须完全一致：

```csv
start_frame,end_frame,label,confidence,note
```

示例：

```csv
start_frame,end_frame,label,confidence,note
0,480,neutral,0.90,自然表情，没有明显笑意
481,810,discard,0.85,手遮挡嘴部和下半张脸
811,930,neutral,0.88,手放下，恢复自然表情
5401,5520,smile,0.95,嘴角上扬，出现明显笑容
```

字段要求：

- `start_frame`：片段开始帧，整数。
- `end_frame`：片段结束帧，整数。
- `label`：只能是 `smile`、`neutral`、`discard`。
- `confidence`：0 到 1 之间的小数。
- `note`：简短中文，说明判断依据。

## 4. 帧边界规则

使用闭区间帧号：

```text
start_frame 和 end_frame 都包含在该片段内。
```

必须满足：

- 第一段的 `start_frame` 应为 `0`。
- 最后一段的 `end_frame` 应为 `total_frames - 1`。
- 相邻片段应连续：
  ```text
  next.start_frame = previous.end_frame + 1
  ```
- 片段之间不能重叠。
- 尽量不要有空缺帧。
- 不要输出视频范围之外的帧号。

## 5. 标签定义

### smile

当被试脸上出现明显或较明显笑意时，标注为 `smile`。

包括：

- 嘴角上扬。
- 露齿笑。
- 闭嘴微笑。
- 笑容正在形成。
- 笑容正在保持。
- 笑容正在消退，但仍然能看出笑意。
- 苦笑、礼貌笑、真实笑、模糊笑等所有笑容状态。

注意：

- 本阶段不区分具体笑容类型。
- 不要标注 peak frame。
- 只要仍能看出笑意，就继续归入同一个 `smile` 片段，不要过早切回 `neutral`。

### neutral

当被试脸部处于自然、非笑状态时，标注为 `neutral`。

包括：

- 自然表情。
- 无明显笑意。
- 普通说话状态。
- 嘴部有说话动作，但不是笑。
- 表情变化较小，无法判断为笑。

注意：

- 说话时嘴巴运动不等于笑。
- 如果无法确定是否在笑，优先标注为 `neutral`，并在 `note` 中写 `uncertain_smile`。

### discard

当画面无法可靠判断表情，或该片段不适合后续表情标注时，标注为 `discard`。

包括：

- 手、水杯、纸张、麦克风、头发等遮挡关键面部区域。
- 脸转得太偏，无法判断表情。
- 低头、出画、脸部检测困难。
- 画面模糊、运动过快，无法判断表情。
- 其他人或物体挡住被试面部。

注意：

- 不要输出 `occlusion` 标签。遮挡、看不清、无法判断都统一输出为 `discard`。
- 如果遮挡很轻微，但仍能明确判断表情，可以继续标注为 `smile` 或 `neutral`，并在 `note` 中写 `partial_occlusion`。

## 6. 优先级规则

当多个标签可能同时成立时，按以下优先级判断：

```text
严重遮挡或无法可靠判断 -> discard
可以判断为笑 -> smile
没有明显笑意 -> neutral
```

具体规则：

1. 如果脸部被严重遮挡，无法判断是否在笑，标注为 `discard`。
2. 如果有轻微遮挡，但仍能明确判断在笑，标注为 `smile`。
3. 如果没有遮挡，但笑意不明显，标注为 `neutral`。
4. 如果不确定是否为笑，优先标注为 `neutral`，并在 `note` 中写 `uncertain_smile`。

## 7. 切分粒度

这是粗切分任务，优先保证大类正确，不要过度切碎。

- 少于 15 帧的短暂变化，一般不要单独切分。
- 如果视频不是 30fps，可把 0.5 秒以内的短暂变化理解为一般不单独切分。
- 明显影响判断的严重遮挡，即使较短，也可以标注为 `discard`。
- 一个完整笑容的形成、保持、消退阶段应尽量放在同一个 `smile` 片段中。
- 如果两个相邻片段标签相同，应合并成一个片段。

## 8. 多文件处理规则

Gemini 每次只处理一个视频。处理多个视频时，用户会重复发起多个任务。

每个视频的输出应单独保存，建议文件名使用：

```text
<video_stem>.coarse.csv
```

例如：

```text
GX010001_part001.coarse.csv
GX010001_part002.coarse.csv
```

如果视频来自大文件切分后的重叠片段，Gemini 不需要处理跨文件去重。每个切分后视频都按自己的局部帧号独立输出。

## 9. 最终自检

输出前检查：

- 表头是否为 `start_frame,end_frame,label,confidence,note`。
- 是否只输出 CSV 文本。
- 标签是否只包含 `smile`、`neutral`、`discard`。
- 第一段是否从 0 开始。
- 最后一段是否到 `total_frames - 1`。
- 相邻片段是否连续且不重叠。
- 是否没有输出 `occlusion`、timestamp、JSON 或额外解释。

## 10. 用户可复制的请求模板

```text
请按照“视频表情粗切分任务说明”处理这个视频。

video_filename: <填写视频文件名>
total_frames: <填写总帧数>
fps: <填写 fps>
frame_index_base: 0

请只输出 CSV 文本，表头必须是：
start_frame,end_frame,label,confidence,note

标签只能使用 smile、neutral、discard。
不要输出 occlusion。
不要输出 Markdown、JSON、timestamp 或额外解释。
```
