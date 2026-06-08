# 任务：将现有 smile episode 标注工具改造为“按时间顺序切分整段视频的状态标注工具”

## 0. 当前背景

请先阅读当前交接文档：

```text
E:\Single_frame_smile\annotation\dataset-annotation\incremental_multi_occlusion_segments.md
```

该文档中原本计划将单段遮挡升级为多段遮挡，并使用 `occlusion_segments` 记录多个遮挡片段。

但是现在需求发生变化：

**不再实现多段遮挡标注。**

新的设计是：

1. 不再细分遮挡类型。
2. 所有被遮挡、看不清、追踪失败、无法判断的区间，统一标为 `discard`。
3. 将整个视频按时间顺序切分为多个连续区间。
4. 每个区间只需要一个主状态标签。
5. 已经完成的旧标注区间需要尽量保留区间边界，但允许重新标注 label。

也就是说，本次任务是从：

```text
只标注 smile episode
```

改为：

```text
按时间顺序切分整个视频，并给每个区间赋予一个状态标签
```

---

# 1. 新的标注目标

当前工具的核心标注单位仍然是一段时间区间。

但是语义从：

```text
smile episode
```

扩展为：

```text
video segment / state interval
```

每一行 CSV 表示一个时间连续区间：

```text
start_frame → end_frame
```

每个区间都有一个状态标签：

```text
truesmile
polite_smile
bitter_smile
smiling_but_ambiguous
neutral
discard
```

---

# 2. 新标签集合

请将主标签集合改为以下固定值：

```text
truesmile
polite_smile
bitter_smile
smiling_but_ambiguous
neutral
discard
```

不要使用自由输入。

UI 中 label 必须是下拉框。

---

## 2.1 truesmile

表示真实、自然、积极的笑容。

使用条件：

- 表情整体看起来比较自然
- 嘴角上扬明显
- 可能伴随脸颊提升、眼周变化
- 更接近真实愉快、轻松、认同的笑

该标签属于 smile 状态。

必须设置 peak。

---

## 2.2 polite_smile

表示礼貌性、社交性、回应性的笑容。

使用条件：

- 笑容主要用于回应、配合、维持互动
- 表情可能较浅、较短、较受控制
- 嘴角上扬存在，但情绪强度不一定高
- 更像是社交场合中的礼貌反应

该标签属于 smile 状态。

必须设置 peak。

---

## 2.3 bitter_smile

表示苦笑、尴尬笑、自嘲笑、无奈笑等偏复杂或带有负性语境的笑。

使用条件：

- 看起来不像单纯积极的笑
- 可能伴随尴尬、回避、低头、视线移开、摸脸等行为
- 表情或语境带有苦笑、无奈、自嘲、为难等成分

该标签属于 smile 状态。

必须设置 peak。

---

## 2.4 smiling_but_ambiguous

表示正在笑，但难以判断属于 true / polite / bitter 中哪一种。

使用条件：

- 可以确定存在笑容
- 但是笑容类型难以明确归类
- 可能混合真实、礼貌、苦笑等成分
- 表情或上下文具有多义性

该标签属于 smile 状态。

必须设置 peak。

---

## 2.5 neutral

表示非笑容的自然状态。

使用条件：

- 没有明显笑容
- 表情接近自然、中性、听讲、思考、普通说话状态
- 画面质量可以正常观察
- 人脸可见，不属于 discard

该标签不需要 peak。

保存时：

```text
peak_frame = empty
peak_time = empty
```

---

## 2.6 discard

表示该区间不适合用于训练或分析。

所有被其他因素干扰、遮挡、看不清、追踪失败的区间都统一标为 `discard`。

使用条件包括但不限于：

- 手遮挡嘴部
- 手遮挡下半脸
- 杯子、物体、口罩遮挡关键区域
- 人脸大面积不可见
- 低头导致关键区域不可见
- 脸转开，无法判断表情
- 画面严重模糊
- tracking crop 错误
- 多个人脸混淆
- 视频质量太差
- 表情状态无法可靠判断
- 其他任何会让该区间不适合训练的情况

`discard` 不需要 peak。

保存时：

```text
peak_frame = empty
peak_time = empty
```

注意：

- 不再需要精细区分 `mouth_partial`, `mouth_severe`, `lower_face_occluded` 等遮挡类型。
- 只要该区间因为遮挡或其他因素不适合判断，就标为 `discard`。
- `discard` 默认不进入后续训练集。

---

# 3. peak 规则

## 3.1 必须有 peak 的标签

以下 smile 状态必须设置 peak：

```text
truesmile
polite_smile
bitter_smile
smiling_but_ambiguous
```

保存时必须满足：

```text
start_frame < peak_frame < end_frame
```

如果没有设置 peak，应阻止保存并提示用户。

---

## 3.2 不需要 peak 的标签

以下非 smile / 不可用状态不需要 peak：

```text
neutral
discard
```

保存时应自动清空：

```text
peak_frame
peak_time
```

即使 UI 中之前设置过 peak，只要 label 是 `neutral` 或 `discard`，最终保存时也应写空值。

---

# 4. 新的标注方式：按时间顺序切分整个视频

当前目标不是只找 smile episode，而是将整个视频按时间顺序切成若干段。

例如：

```text
0 - 120      neutral
120 - 180    polite_smile
180 - 220    neutral
220 - 260    discard
260 - 330    bitter_smile
330 - 400    neutral
```

每一段都是一条 CSV 记录。

---

## 4.1 区间连续性

理想情况下，同一个视频中的所有区间应按时间顺序排列：

```text
segment 1: start_frame = 0,   end_frame = 120
segment 2: start_frame = 120, end_frame = 180
segment 3: start_frame = 180, end_frame = 220
```

也就是说，推荐相邻区间满足：

```text
next.start_frame = previous.end_frame
```

但是不要在存储层强制禁止 gap，因为用户可能需要手动调整。

---

## 4.2 保存后自动准备下一段

当前工具已有“新增 episode 保存后，自动把下一条 episode 的 start frame 设置为刚保存 episode 的 end frame”的逻辑。

请保留并强化这个逻辑。

保存一个新 segment 后：

```text
next_start_frame = current_end_frame
```

同时清空：

```text
peak_frame
end_frame
note
```

这样用户可以按时间顺序快速切分整段视频。

---

## 4.3 新增区间不再只代表 smile

原来的 UI 里可能叫：

```text
Smile Episode
Episode
```

可以继续使用 `episode` 作为内部变量名，避免大规模重构。

但 UI 文案建议改为：

```text
Segment
State Segment
Video Segment
```

例如：

```text
Save Episode
```

可以改成：

```text
Save Segment
```

如果改动较大，也可以暂时保留按钮名，但代码注释中需要说明：

```text
episode row is now used as a general temporal segment, not only a smile episode
```

---

# 5. 已有标注数据的处理：保留区间，重新标注 label

当前已经有一部分完成标注的区间。

新需求不是删除这些区间，而是：

```text
保留已有 start_frame / peak_frame / end_frame 区间，允许用户重新标注 label
```

也就是说：

- 旧 CSV 中已有的区间边界应保留
- `episode_id` 应保留
- `start_frame` / `end_frame` 应保留
- 如果旧 label 是 smile 类，并且用户保留为新的 smile 类，则 peak 可以保留
- 如果用户把 label 改为 `neutral` 或 `discard`，则 peak 自动清空
- 如果用户把 label 改为 smile 类，但原来没有 peak，则必须补 peak 后才能保存

---

## 5.1 旧标签映射建议

如果旧 CSV 中存在旧标签，可以在读取时或迁移时进行映射。

建议映射如下：

```text
genuine_like_smile -> truesmile
polite_like_smile -> polite_smile
bitter_awkward_like_smile -> bitter_smile
ambiguous_smile -> smiling_but_ambiguous
neutral_or_no_smile -> neutral
unclear -> discard
```

如果旧项目中还有其他标签，无法确定时映射为：

```text
discard
```

更推荐：

```text
未知旧标签 -> discard
```

因为新的规则中，无法可靠判断的状态统一不进入训练。

---

## 5.2 不要自动破坏旧 CSV

读取旧 CSV 时可以在内存中转换 label，但不要在读取时直接覆盖原文件。

保存前必须沿用已有 backup 机制。

如果当前工具还没有 backup 机制，则必须在覆盖保存前增加自动备份：

```text
annotations_backup_YYYYMMDD_HHMMSS.csv
```

---

# 6. occlusion 字段的处理

由于新规则决定：

```text
所有遮挡或不可用情况统一标为 discard
```

因此不再需要继续开发多段遮挡字段。

---

## 6.1 不再实现 occlusion_segments

请不要继续实现以下功能：

```text
occlusion_segments
多段遮挡 JSON
Add Occlusion Segment
Update Selected Segment
Delete Selected Segment
```

这些功能不是当前需求。

---

## 6.2 已存在的 occlusion 字段如何处理

如果上一阶段代码中已经存在：

```csv
occlusion_type
occlusion_start_frame
occlusion_end_frame
occlusion_severity
occlusion_note
```

不要强行删除字段，避免破坏兼容性。

但是 GUI 中不再需要把它们作为主要功能展示。

建议处理方式：

1. 保留 CSV 字段，避免旧数据读取失败。
2. 保存新数据时，可以统一写默认值：

```text
occlusion_type = none
occlusion_start_frame = empty
occlusion_end_frame = empty
occlusion_severity = none
occlusion_note = empty
```

3. 如果 label = `discard`，也不需要填写 occlusion 字段。
4. `discard` 本身就代表该区间不可用，包括遮挡情况。
5. 后续训练时只根据 `main_label == discard` 排除即可。

---

## 6.3 如果旧行有 occlusion 标注

如果旧行中存在 occlusion 字段，并且表示有遮挡，例如：

```text
occlusion_type != none
```

建议在迁移时将该行的 `main_label` 设为：

```text
discard
```

因为新规则是：

```text
所有被遮挡的情况统一 discard
```

注意：

- 不要在读取时直接覆盖文件
- 只在内存中显示为 discard
- 用户保存时才写入 CSV
- 保存前需要 backup

---

# 7. annotation_store.py 修改要求

请在现有 `annotation_store.py` 基础上增量修改。

不要重写整个模块。

---

## 7.1 更新 label 常量

将 `MAIN_LABELS` 改为：

```python
MAIN_LABELS = [
    "truesmile",
    "polite_smile",
    "bitter_smile",
    "smiling_but_ambiguous",
    "neutral",
    "discard",
]
```

---

## 7.2 更新需要 peak 的标签集合

```python
PEAK_REQUIRED_LABELS = {
    "truesmile",
    "polite_smile",
    "bitter_smile",
    "smiling_but_ambiguous",
}
```

---

## 7.3 更新不需要 peak 的标签

以下标签不需要 peak：

```python
NO_PEAK_LABELS = {
    "neutral",
    "discard",
}
```

如果当前代码没有 `NO_PEAK_LABELS`，可以不新增，只要 `label_requires_peak()` 返回正确即可。

---

## 7.4 更新 label_requires_peak()

```python
def label_requires_peak(main_label: str) -> bool:
    return main_label in PEAK_REQUIRED_LABELS
```

---

## 7.5 保存时 normalize peak

在构造 row 或 validate 时处理：

如果：

```python
main_label in {"neutral", "discard"}
```

则强制：

```python
peak_frame = None
peak_time = ""
```

这样可以避免旧 UI 状态中残留 peak。

---

## 7.6 label migration 函数

新增或修改一个旧标签迁移函数：

```python
OLD_TO_NEW_LABEL = {
    "genuine_like_smile": "truesmile",
    "polite_like_smile": "polite_smile",
    "bitter_awkward_like_smile": "bitter_smile",
    "ambiguous_smile": "smiling_but_ambiguous",
    "neutral_or_no_smile": "neutral",
    "unclear": "discard",
}

def migrate_main_label(old_label: str) -> str:
    if old_label in MAIN_LABELS:
        return old_label
    return OLD_TO_NEW_LABEL.get(old_label, "discard")
```

读取 rows 时可以对 `main_label` 做内存迁移。

注意：

- 不要在 read 阶段直接覆盖 CSV
- 保存时才写入新标签
- 如果做了迁移，建议在 GUI 或日志中提示用户检查

---

## 7.7 occlusion 到 discard 的迁移逻辑

如果旧 row 中存在：

```python
row["occlusion_type"] != "" and row["occlusion_type"] != "none"
```

则建议将该 row 的 `main_label` 在内存中设为：

```python
discard
```

原因：

新规则是：

```text
所有遮挡情况统一 discard
```

注意：

- 只在内存中转换
- 不要读取时直接覆盖 CSV
- 用户保存或 update 时才写入文件
- 保存前必须有 backup

---

## 7.8 保留现有 CSV 字段

不要删除现有 CSV 字段。

如果现有 CSV_COLUMNS 中已经有：

```python
occlusion_type
occlusion_start_frame
occlusion_end_frame
occlusion_severity
occlusion_note
```

可以继续保留。

但是不要新增：

```python
occlusion_segments
```

除非当前代码已经生成了该字段。  
当前新需求不需要它。

---

# 8. GUI 修改要求

请修改 `smile_episode_annotation_tool.py`。

---

## 8.1 更新 label 下拉框

将 label 下拉框改为固定选项：

```text
truesmile
polite_smile
bitter_smile
smiling_but_ambiguous
neutral
discard
```

不要继续显示旧标签：

```text
genuine_like_smile
polite_like_smile
bitter_awkward_like_smile
ambiguous_smile
neutral_or_no_smile
unclear
```

如果加载旧 CSV 时出现旧标签，应通过 migration 显示为新标签。

---

## 8.2 更新 peak 交互逻辑

当 label 是 smile 类：

```text
truesmile
polite_smile
bitter_smile
smiling_but_ambiguous
```

必须允许并要求设置 peak。

当 label 是：

```text
neutral
discard
```

应自动清空 peak，并且 UI 可以禁用或忽略 peak。

建议行为：

- 用户选择 `neutral` 或 `discard` 时：
  - 清空 peak marker
  - peak 显示为空
  - 保存时不要求 peak
- 用户选择 smile 类时：
  - 恢复 peak 设置需求
  - 保存时如果没有 peak，则弹出 warning 并阻止保存

---

## 8.3 移除或隐藏遮挡标注 UI

由于新规则不再需要精细遮挡标注：

- 不再需要遮挡类型下拉框
- 不再需要 occlusion start/end
- 不再需要 occlusion segment list
- 不再需要 Add Occlusion Segment 等按钮

如果删除 UI 改动太大，可以先隐藏或禁用。

推荐文案：

```text
Occluded / invalid intervals should be labeled as discard.
```

---

## 8.4 按时间顺序切分整个视频

当前保存新 episode 后会自动将下一条 start 设置为上一个 end。

请保留该逻辑，并将其作为核心工作流。

用户操作流程应变为：

1. 设置当前 segment start
2. 设置 peak，如果是 smile 类
3. 设置当前 segment end
4. 选择 label
5. 保存
6. 工具自动把下一段 start 设置为当前 end
7. 用户继续设置下一段 end 和 label

---

## 8.5 episode list 显示

当前右下方 episode list 可以继续显示：

```text
episode_id
start
peak
end
label
conf
usable
```

建议把 UI 文案中的 episode 尽量改成 segment，但内部变量可以不改。

如果有空间，可以显示：

```text
segment_id
start
peak
end
label
usable
```

---

## 8.6 编辑已有区间

用户已经有一部分完成标注的区间。

点击已有行时：

- 加载原 start_frame
- 加载原 end_frame
- 如果新 label 是 smile 类，则加载 peak
- 如果新 label 是 neutral/discard，则 peak 显示为空
- 用户可以重新选择 label
- 点击保存后更新原 row，不新增 row，不改变 episode_id

---

# 9. 训练数据筛选规则

## 9.1 smile type classification

只使用：

```text
truesmile
polite_smile
bitter_smile
smiling_but_ambiguous
```

排除：

```text
neutral
discard
```

---

## 9.2 smile detection

正类：

```text
truesmile
polite_smile
bitter_smile
smiling_but_ambiguous
```

负类：

```text
neutral
```

排除：

```text
discard
```

---

## 9.3 discard

`discard` 永远不进入主训练集。

用途：

- 保留时间切分完整性
- 标记不可用区间
- 避免反复检查同一段坏数据

---

# 10. 测试要求

请更新 `test_annotation_store.py`。

至少增加或修改以下测试。

---

## Test 1：新标签合法性

以下标签合法：

```python
[
    "truesmile",
    "polite_smile",
    "bitter_smile",
    "smiling_but_ambiguous",
    "neutral",
    "discard",
]
```

旧标签不应作为最终保存标签。

---

## Test 2：smile 类必须有 peak

以下标签没有 peak 时应报错：

```python
truesmile
polite_smile
bitter_smile
smiling_but_ambiguous
```

---

## Test 3：neutral 不需要 peak

`neutral` 没有 peak 可以保存。

保存后：

```text
peak_frame = empty
peak_time = empty
```

---

## Test 4：discard 不需要 peak

`discard` 没有 peak 可以保存。

保存后：

```text
peak_frame = empty
peak_time = empty
```

---

## Test 5：neutral/discard 会清空旧 peak

如果 draft 或旧 row 中有 peak，但 label 是：

```text
neutral
discard
```

保存后 peak 应为空。

---

## Test 6：旧标签迁移

验证：

```python
genuine_like_smile -> truesmile
polite_like_smile -> polite_smile
bitter_awkward_like_smile -> bitter_smile
ambiguous_smile -> smiling_but_ambiguous
neutral_or_no_smile -> neutral
unclear -> discard
```

---

## Test 7：旧 occlusion 行迁移为 discard

如果旧 row 中：

```text
occlusion_type = mouth_severe
```

读取或迁移后应显示为：

```text
main_label = discard
```

---

## Test 8：更新已有 row 不改变 episode_id

编辑已有区间 label 时：

- episode_id 不变
- start/end 保留，除非用户手动修改
- 只更新当前 row
- 不新增 row

---

## Test 9：连续 segment 保存逻辑

新增一个 segment 后，下一条 segment 的 start 自动设置为上一条的 end。

---

# 11. 验证命令

修改后至少运行：

```powershell
cd E:\Single_frame_smile\annotation\dataset-annotation
python -m py_compile annotation_store.py smile_episode_annotation_tool.py test_annotation_store.py
python test_annotation_store.py
```

如果环境支持 PySide6，请运行：

```powershell
python smile_episode_annotation_tool.py
```

手动测试：

1. 打开视频
2. 标注一段 `neutral`，不设置 peak，保存成功
3. 标注一段 `truesmile`，不设置 peak，保存应失败
4. 给 `truesmile` 设置 peak 后保存成功
5. 标注一段 `discard`，不设置 peak，保存成功
6. 点击已有旧区间，确认旧 label 被映射成新 label
7. 将已有 smile 区间改成 `discard`，保存后 peak 清空
8. 将已有区间改成 smile 类，如果没有 peak，应要求补 peak
9. 保存后下一段 start 自动等于上一段 end
10. 重新打开工具，确认 CSV 可正常读取

---

# 12. 不要做的事情

不要做以下事情：

- 不要继续开发 `occlusion_segments`
- 不要继续开发多段遮挡标注 UI
- 不要精细区分 mouth_partial / mouth_severe / lower_face_occluded
- 不要把遮挡作为独立字段要求用户填写
- 不要删除已有已标注区间
- 不要重建 annotations.csv
- 不要改变 frame index 规则
- 不要改变 update 逻辑：编辑已有 row 时必须保留 episode_id
- 不要把 `neutral` 强制要求 peak
- 不要把 `discard` 强制要求 peak
- 不要让遮挡区间进入训练标签
- 不要把工具改成 web app

---

# 13. 最终目标

完成后，工具应从：

```text
smile episode annotation tool
```

转变为：

```text
temporal video segment state annotation tool
```

但可以保留原有 episode 数据结构和 CSV 结构，以减少改动。

最终用户可以按时间顺序切分整个视频，并为每个区间选择：

```text
truesmile
polite_smile
bitter_smile
smiling_but_ambiguous
neutral
discard
```

其中：

- 四个 smile 类必须有 peak
- `neutral` 不需要 peak
- `discard` 不需要 peak
- 所有遮挡、看不清、不可用情况都标为 `discard`
- 已有标注区间尽量保留，只重新标注 label
