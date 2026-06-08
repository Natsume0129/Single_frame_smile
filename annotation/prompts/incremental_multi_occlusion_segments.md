# 增量任务：将单段遮挡标注升级为同一 episode 内的多段遮挡标注

## 0. 当前前提

当前标注工具已经完成了基础遮挡标注功能。

也就是说，现在每个 smile episode 已经可以标注一段遮挡信息，现有字段大致包括：

```csv
occlusion_type,occlusion_start_frame,occlusion_end_frame,occlusion_severity,occlusion_note
```

现有逻辑可以支持：

```text
一个 episode 中的一段遮挡
```

例如：

```text
episode: 100 - 240
occlusion: 170 - 190, mouth_severe
```

但是现在需要进一步支持：

```text
一个 episode 中的多段遮挡
```

例如：

```text
episode: 100 - 240

occlusion segment 1: 120 - 135, mouth_partial
occlusion segment 2: 170 - 190, mouth_severe
occlusion segment 3: 210 - 220, hand_near_face_but_not_occluding
```

本任务只实现这个增量需求。  
不要重新设计整个标注工具。  
不要改动 smile episode 的主标注逻辑。  
不要把一个 episode 拆成多个 episode。

---

# 1. 核心需求

## 1.1 一个 episode 可以包含多个 occlusion segment

当前的遮挡标注逻辑只能记录一个区间：

```text
occlusion_start_frame
occlusion_end_frame
```

现在需要升级为：

```text
一个 episode 内可以有 0 个、1 个或多个 occlusion segment
```

每个 occlusion segment 包含：

```text
start
end
type
severity
note
```

示例：

```json
[
  {
    "start": 120,
    "end": 135,
    "type": "mouth_partial",
    "severity": "mild",
    "note": "finger near mouth"
  },
  {
    "start": 170,
    "end": 190,
    "type": "mouth_severe",
    "severity": "severe",
    "note": "hand covers mouth near peak"
  }
]
```

---

## 1.2 episode 仍然是主标注单位

不要因为一个 episode 中有多个遮挡片段，就把该 episode 拆开。

错误做法：

```text
原 episode: 100 - 240

拆成：
100 - 119
136 - 169
191 - 240
```

正确做法：

```text
episode: 100 - 240
occlusion_segments:
  - 120 - 135
  - 170 - 190
  - 210 - 220
```

原因：

- smile episode 是后续时序模型的训练样本单位
- 遮挡只是 episode 内部的辅助标注
- 拆分 episode 会破坏 onset → peak → offset 的完整时序结构

---

# 2. 新增字段：occlusion_segments

在 CSV 中新增一个字段：

```csv
occlusion_segments
```

该字段用于保存当前 episode 内的所有遮挡片段。

格式使用 JSON 字符串。

---

## 2.1 无遮挡

如果一个 episode 没有任何遮挡，保存为：

```json
[]
```

CSV 中对应：

```csv
[]
```

---

## 2.2 单段遮挡

如果只有一段遮挡，保存为：

```json
[
  {
    "start": 170,
    "end": 190,
    "type": "mouth_severe",
    "severity": "severe",
    "note": "hand covers mouth near peak"
  }
]
```

---

## 2.3 多段遮挡

如果有多段遮挡，保存为：

```json
[
  {
    "start": 120,
    "end": 135,
    "type": "mouth_partial",
    "severity": "mild",
    "note": "finger near mouth"
  },
  {
    "start": 170,
    "end": 190,
    "type": "mouth_severe",
    "severity": "severe",
    "note": "hand covers mouth near peak"
  },
  {
    "start": 210,
    "end": 220,
    "type": "hand_near_face_but_not_occluding",
    "severity": "mild",
    "note": "hand near chin"
  }
]
```

---

# 3. 保留现有单段遮挡字段作为摘要字段

当前已有的字段不要删除：

```csv
occlusion_type
occlusion_start_frame
occlusion_end_frame
occlusion_severity
occlusion_note
```

这些字段继续保留，但它们的角色从“完整遮挡信息”变为：

```text
episode-level occlusion summary
```

也就是说：

- `occlusion_segments` 保存完整多段遮挡信息
- 原来的 `occlusion_type` 等字段保存摘要信息，方便快速查看和兼容旧逻辑

---

## 3.1 无遮挡时的摘要字段

如果：

```json
occlusion_segments = []
```

则摘要字段为：

```text
occlusion_type = none
occlusion_start_frame = empty
occlusion_end_frame = empty
occlusion_severity = none
occlusion_note = empty
```

---

## 3.2 单段遮挡时的摘要字段

如果：

```json
[
  {
    "start": 170,
    "end": 190,
    "type": "mouth_severe",
    "severity": "severe",
    "note": "hand covers mouth near peak"
  }
]
```

则摘要字段为：

```text
occlusion_type = mouth_severe
occlusion_start_frame = 170
occlusion_end_frame = 190
occlusion_severity = severe
occlusion_note = hand covers mouth near peak
```

---

## 3.3 多段遮挡时的摘要字段

如果：

```json
[
  {
    "start": 120,
    "end": 135,
    "type": "mouth_partial",
    "severity": "mild",
    "note": "finger near mouth"
  },
  {
    "start": 170,
    "end": 190,
    "type": "mouth_severe",
    "severity": "severe",
    "note": "hand covers mouth near peak"
  }
]
```

则摘要字段为：

```text
occlusion_type = mouth_severe
occlusion_start_frame = 120
occlusion_end_frame = 190
occlusion_severity = severe
occlusion_note = multiple occlusion segments
```

规则：

- `occlusion_start_frame` = 所有 segment 中最早的 start
- `occlusion_end_frame` = 所有 segment 中最晚的 end
- `occlusion_type` = 最严重的遮挡类型
- `occlusion_severity` = 最高严重程度
- `occlusion_note` = 如果有多个 segment，则写 `multiple occlusion segments`

---

# 4. 遮挡类型与严重程度沿用现有定义

不要重新定义遮挡类型。  
继续使用当前已经实现的遮挡类型：

```text
none
mouth_partial
mouth_severe
lower_face_occluded
hand_near_face_but_not_occluding
```

但是在 `occlusion_segments` 里通常不需要添加：

```text
type = none
```

如果没有遮挡，直接使用：

```json
[]
```

segment 中允许的 type 建议为：

```text
mouth_partial
mouth_severe
lower_face_occluded
hand_near_face_but_not_occluding
```

severity 继续使用当前已有定义：

```text
none
mild
moderate
severe
```

---

# 5. 自动摘要规则

为了从多个 segment 自动生成摘要字段，请增加排序规则。

## 5.1 severity rank

```python
SEVERITY_RANK = {
    "none": 0,
    "mild": 1,
    "moderate": 2,
    "severe": 3,
}
```

## 5.2 occlusion type rank

```python
OCCLUSION_TYPE_RANK = {
    "none": 0,
    "hand_near_face_but_not_occluding": 1,
    "mouth_partial": 2,
    "mouth_severe": 3,
    "lower_face_occluded": 4,
}
```

摘要字段中的：

```text
occlusion_type
```

应选择 rank 最高的 segment type。

摘要字段中的：

```text
occlusion_severity
```

应选择 rank 最高的 segment severity。

---

# 6. annotation_store.py 增量修改要求

请在现有 `annotation_store.py` 基础上增量修改。

不要重写整个存储模块。

---

## 6.1 新增 OcclusionSegment 数据结构

建议增加：

```python
@dataclass(frozen=True)
class OcclusionSegment:
    start: int
    end: int
    type: str
    severity: str
    note: str = ""
```

要求：

- `start` 和 `end` 使用 0-based frame index
- `start <= end`
- `type` 必须在已有遮挡类型集合中
- `severity` 必须在已有 severity 集合中

---

## 6.2 扩展 EpisodeDraft

在现有 `EpisodeDraft` 中新增：

```python
occlusion_segments: list[OcclusionSegment] = field(default_factory=list)
```

如果当前项目为了兼容 Python 版本或代码风格，不方便使用 `OcclusionSegment`，也可以使用：

```python
occlusion_segments: list[dict] = field(default_factory=list)
```

但更推荐 dataclass。

---

## 6.3 新增 JSON 序列化函数

增加：

```python
def serialize_occlusion_segments(segments: list[OcclusionSegment]) -> str:
    ...
```

要求：

- 输入为空 list 时返回 `"[]"`
- 返回合法 JSON 字符串
- 使用 `json.dumps(..., ensure_ascii=False)`
- 每个 segment 写出字段：
  - `start`
  - `end`
  - `type`
  - `severity`
  - `note`

---

## 6.4 新增 JSON 解析函数

增加：

```python
def parse_occlusion_segments(value: str) -> list[OcclusionSegment]:
    ...
```

要求：

- `""`、`None`、缺失值都解析为 `[]`
- `"[]"` 解析为 `[]`
- 合法 JSON list 解析为 `OcclusionSegment` list
- 非法 JSON 应抛出 `ValueError`
- 每个 segment 解析后都要做校验

---

## 6.5 新增 segment 校验函数

增加：

```python
def validate_occlusion_segment(segment: OcclusionSegment) -> None:
    ...
```

校验规则：

- `start` 必须是 int
- `end` 必须是 int
- `start <= end`
- `type` 必须合法
- `severity` 必须合法

不要强制要求 segment 必须位于 episode 的 start/end 内。

原因：

- 遮挡可能从 episode 前已经开始
- 遮挡可能持续到 episode 后
- 这种情况只需要 GUI warning，不应由存储层禁止

---

## 6.6 新增摘要生成函数

增加：

```python
def summarize_occlusion_segments(segments: list[OcclusionSegment]) -> dict[str, str]:
    ...
```

无 segment 时返回：

```python
{
    "occlusion_type": "none",
    "occlusion_start_frame": "",
    "occlusion_end_frame": "",
    "occlusion_severity": "none",
    "occlusion_note": "",
}
```

有 segment 时：

```python
{
    "occlusion_type": "<最严重 type>",
    "occlusion_start_frame": "<最早 start>",
    "occlusion_end_frame": "<最晚 end>",
    "occlusion_severity": "<最高 severity>",
    "occlusion_note": "<单段 note 或 multiple occlusion segments>",
}
```

---

## 6.7 扩展 CSV_COLUMNS

在已有 CSV 字段末尾追加：

```python
"occlusion_segments"
```

如果当前已有：

```python
"occlusion_type",
"occlusion_start_frame",
"occlusion_end_frame",
"occlusion_severity",
"occlusion_note",
```

不要重复添加，只追加 `occlusion_segments`。

最终遮挡相关字段应为：

```python
"occlusion_type",
"occlusion_start_frame",
"occlusion_end_frame",
"occlusion_severity",
"occlusion_note",
"occlusion_segments",
```

---

## 6.8 append/update 时写入 occlusion_segments

在 `append_episode()` 和 `update_episode()` 中：

1. 从 `draft.occlusion_segments` 生成 JSON 字符串
2. 写入 `occlusion_segments`
3. 调用 `summarize_occlusion_segments()` 自动生成摘要字段
4. 写入原有摘要字段

也就是说，保存时应以：

```python
draft.occlusion_segments
```

作为遮挡信息的唯一可靠来源。

摘要字段不要由 GUI 手工维护，避免不一致。

---

# 7. 旧数据兼容要求

当前已经存在上一个版本的 CSV，其中可能只有单段遮挡字段，但没有 `occlusion_segments`。

因此读取旧 CSV 时需要兼容。

---

## 7.1 情况 A：没有 occlusion_segments 字段

如果 CSV 中没有：

```csv
occlusion_segments
```

但有已有单段字段：

```csv
occlusion_type,occlusion_start_frame,occlusion_end_frame,occlusion_severity,occlusion_note
```

则读取时自动补出 `occlusion_segments`。

---

## 7.2 单段旧数据迁移为 segments

如果旧数据中：

```text
occlusion_type = none
```

则：

```json
occlusion_segments = []
```

如果旧数据中：

```text
occlusion_type != none
```

并且：

```text
occlusion_start_frame 和 occlusion_end_frame 都存在
```

则迁移为：

```json
[
  {
    "start": occlusion_start_frame,
    "end": occlusion_end_frame,
    "type": occlusion_type,
    "severity": occlusion_severity,
    "note": occlusion_note
  }
]
```

---

## 7.3 已经有 occlusion_segments 的新版数据

如果 CSV 已经有：

```csv
occlusion_segments
```

则：

- 直接读取并解析
- 不要覆盖已有 JSON
- 不要重复生成
- 保存时可以根据 `occlusion_segments` 重新生成摘要字段

---

# 8. GUI 增量修改要求

当前 GUI 已经有单段遮挡标注控件。  
现在需要把它升级为多段遮挡标注控件。

---

## 8.1 增加 Occlusion Segments 表格

在当前遮挡标注区域增加一个小表格：

```text
start | end | type | severity | note
```

用于展示当前 episode 内已经添加的所有遮挡片段。

示例：

```text
120 | 135 | mouth_partial | mild | finger near mouth
170 | 190 | mouth_severe  | severe | hand covers mouth near peak
210 | 220 | hand_near_face_but_not_occluding | mild | hand near chin
```

---

## 8.2 当前单段遮挡控件改为 segment draft 控件

现有的：

```text
occlusion_start_frame
occlusion_end_frame
occlusion_type
occlusion_severity
occlusion_note
```

不要再理解为整个 episode 的唯一遮挡信息。

请把它们改造成：

```text
当前准备添加或编辑的 occlusion segment draft
```

也就是说：

- Set Occlusion Start = 设置当前 segment draft 的 start
- Set Occlusion End = 设置当前 segment draft 的 end
- occlusion_type = 当前 segment draft 的 type
- occlusion_severity = 当前 segment draft 的 severity
- occlusion_note = 当前 segment draft 的 note

---

## 8.3 新增 segment 操作按钮

增加以下按钮：

```text
Add Segment
Update Selected Segment
Delete Selected Segment
Clear Segment Draft
Clear All Segments
```

### Add Segment

使用当前 draft 创建一个新的 occlusion segment，并加入当前 episode 的 segment list。

添加后：

- 刷新 segment table
- 清空 segment draft
- 自动更新摘要显示

### Update Selected Segment

当用户在 segment table 中选中某个 segment 后：

- 将该 segment 载入 draft 控件
- 用户修改后点击 Update
- 更新当前选中的 segment
- 刷新 segment table
- 自动更新摘要显示

### Delete Selected Segment

删除当前选中的 segment。

删除后：

- 刷新 segment table
- 自动更新摘要显示

### Clear Segment Draft

只清空当前正在编辑的 draft。

不要删除已经添加的 segment。

### Clear All Segments

清空当前 episode 的所有 segments。

建议弹出确认。

---

## 8.4 GUI 内部状态

在主窗口中新增：

```python
self.current_occlusion_segments = []
```

这个 list 表示当前 episode 内已经添加的所有遮挡片段。

同时保留一个 draft 状态，例如：

```python
self.current_occlusion_segment_draft = {
    "start": None,
    "end": None,
}
```

或者沿用现有 start/end 控件状态。

重点是：

```text
current_occlusion_segments 才是最终保存的数据
```

---

## 8.5 加载已有 episode 时

当用户点击 episode list 中已有 episode 时：

1. 从 row 中读取 `occlusion_segments`
2. 解析为 segment list
3. 写入 `self.current_occlusion_segments`
4. 刷新 segment table
5. 清空 segment draft
6. 显示摘要字段

---

## 8.6 保存 episode 时

保存 episode 时：

- 将 `self.current_occlusion_segments` 写入 `EpisodeDraft.occlusion_segments`
- 不要只保存当前 draft
- 不要只保存当前选中的 segment
- 不要只保存摘要字段

---

## 8.7 Clear Current Episode 时

清空当前 episode 表单时：

- 清空 `self.current_occlusion_segments`
- 清空 segment table
- 清空 segment draft
- 摘要回到 none

---

# 9. 校验规则

## 9.1 添加 segment 前校验

点击 `Add Segment` 时必须检查：

- start 不为空
- end 不为空
- start <= end
- type 合法
- severity 合法

如果不满足，弹出 warning，不添加。

---

## 9.2 更新 segment 前校验

点击 `Update Selected Segment` 时必须检查：

- 已经选中了一个 segment
- draft 合法
- start <= end
- type 合法
- severity 合法

如果不满足，弹出 warning，不更新。

---

## 9.3 segment 超出 episode 范围

如果 segment 的 start/end 超出当前 episode 的 start/end：

```text
segment.start < episode.start_frame
或
segment.end > episode.end_frame
```

可以 warning，但不要强制禁止。

---

## 9.4 segment 重叠

如果新添加或更新的 segment 与已有 segment 重叠，可以 warning。

本次可以不强制禁止重叠。

---

# 10. 测试要求

请更新 `test_annotation_store.py`，增加以下测试。

---

## Test 1：serialize_occlusion_segments

输入一个 segment list，确认输出是合法 JSON，并且字段正确。

---

## Test 2：parse_occlusion_segments

输入合法 JSON 字符串，确认可以解析回 segment list。

---

## Test 3：空 segments

以下输入都应解析为 `[]`：

```python
""
None
"[]"
```

---

## Test 4：非法 type

如果 segment type 非法，应抛出 `ValueError`。

---

## Test 5：start > end

如果 segment start 大于 end，应抛出 `ValueError`。

---

## Test 6：summary 无遮挡

输入：

```python
[]
```

摘要应为：

```python
{
    "occlusion_type": "none",
    "occlusion_start_frame": "",
    "occlusion_end_frame": "",
    "occlusion_severity": "none",
    "occlusion_note": "",
}
```

---

## Test 7：summary 多段遮挡

输入：

```python
[
    OcclusionSegment(120, 135, "mouth_partial", "mild", "finger near mouth"),
    OcclusionSegment(170, 190, "mouth_severe", "severe", "hand covers mouth near peak"),
    OcclusionSegment(210, 220, "hand_near_face_but_not_occluding", "mild", "hand near chin"),
]
```

摘要应为：

```python
{
    "occlusion_type": "mouth_severe",
    "occlusion_start_frame": "120",
    "occlusion_end_frame": "220",
    "occlusion_severity": "severe",
    "occlusion_note": "multiple occlusion segments",
}
```

---

## Test 8：旧单段遮挡数据迁移

旧 row：

```text
occlusion_type = mouth_partial
occlusion_start_frame = 120
occlusion_end_frame = 135
occlusion_severity = moderate
occlusion_note = hand partially covers mouth
```

读取后应迁移为：

```json
[
  {
    "start": 120,
    "end": 135,
    "type": "mouth_partial",
    "severity": "moderate",
    "note": "hand partially covers mouth"
  }
]
```

---

## Test 9：append episode 保存多个 segments

新增 episode 时传入多个 segments。

预期：

- CSV 中 `occlusion_segments` 是合法 JSON
- 摘要字段正确
- 再次读取后 segments 仍然正确

---

## Test 10：update episode 修改 segments

编辑已有 episode 的 segments。

预期：

- `episode_id` 不变
- `occlusion_segments` 更新
- 摘要字段同步更新
- 不新增重复 episode

---

# 11. 验证命令

修改后运行：

```powershell
cd E:\Single_frame_smile\annotation\dataset-annotation
python -m py_compile annotation_store.py smile_episode_annotation_tool.py test_annotation_store.py
python test_annotation_store.py
```

如果环境支持 PySide6，请手动运行：

```powershell
python smile_episode_annotation_tool.py
```

手动测试：

1. 打开一个视频
2. 新增 episode，不添加任何 occlusion segment，保存后应为 `[]`
3. 新增 episode，添加一个 segment，保存后重新加载确认存在
4. 新增 episode，添加多个 segment，保存后重新加载确认全部存在
5. 选中一个已有 segment，修改后 update，保存后重新加载确认更新成功
6. 删除一个 segment，保存后重新加载确认删除成功
7. Clear Segment Draft 不应删除已有 segments
8. Clear All Segments 应清空所有 segments
9. 保存已有 episode 时不要丢失 episode_id
10. 旧单段遮挡 CSV 能自动迁移为单个 segment

---

# 12. 不要做的事情

不要做以下事情：

- 不要重新设计整个项目
- 不要修改 smile episode 的主标注逻辑
- 不要把 episode 拆成多个 episode
- 不要删除现有单段遮挡字段
- 不要删除已有 CSV 字段
- 不要改变已有 label 规则
- 不要改变 frame index 规则
- 不要自动把 severe occlusion 改成 unusable
- 不要自动改写 main_label
- 不要只保存当前 draft 而忘记保存 segment list
- 不要在读取旧 CSV 时直接覆盖原文件

---

# 13. 最终目标

完成后，工具应支持：

```text
一个 smile episode 内包含 0 个、1 个或多个 occlusion segment
```

CSV 中：

```csv
occlusion_segments
```

保存完整多段遮挡信息。

现有摘要字段：

```csv
occlusion_type,occlusion_start_frame,occlusion_end_frame,occlusion_severity,occlusion_note
```

继续保留，用于快速查看。

最终结果应满足：

- 单段遮挡旧数据可以自动迁移为一个 segment
- 新数据可以保存多个 segment
- GUI 可以添加、编辑、删除多个 segment
- 重新打开 episode 后，多个 segment 可以正确回填
- episode 本身的 start / peak / end 不受影响
