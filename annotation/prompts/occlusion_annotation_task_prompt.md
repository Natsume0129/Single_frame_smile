# 任务：在现有 smile episode 标注工具中增加“遮挡状态标注”，并安全兼容已有 annotations.csv

请先阅读项目交接文档：

```text
E:\Single_frame_smile\annotation\codex_project_handoff.md
```

不要重新设计整个项目。  
当前项目是 `Single_frame_smile` 仓库中的本地桌面视频标注工具，核心目录是：

```text
E:\Single_frame_smile\annotation\dataset-annotation
```

核心文件：

```text
smile_episode_annotation_tool.py
annotation_store.py
test_annotation_store.py
annotations.csv
```

当前架构是：

- `smile_episode_annotation_tool.py`：PySide6 + OpenCV GUI
- `annotation_store.py`：CSV 存储、校验、新增、编辑、删除
- `test_annotation_store.py`：存储逻辑测试
- `annotations.csv`：用户已经在使用的真实标注数据

重要约束：

- 不要把工具改成 web app
- 不要重构整个项目
- 不要清空、重建或随意覆盖 `annotations.csv`
- 不要破坏已有 episode 标注
- 不要改变已有字段含义
- frame index 继续保持 0-based
- `episode_id` 继续全局递增
- 更新已有 episode 时不要生成新 `episode_id`
- 新增 episode 时继续使用 `next_episode_id()`
- 继续保持一个统一的 `annotations.csv` 管理多个视频
- 修改后必须运行现有测试

---

## 一、当前需求

现在需要在现有 smile episode 标注工具中增加一个新功能：

### 为每个 episode 增加“面部遮挡状态标注”

原因：

目前标注的是 smile episode，用于后续训练时序模型。  
但是真实会议视频中有些 episode 会出现手部、杯子、物体、口罩等遮挡，尤其是嘴部或下半脸遮挡。

这些遮挡会影响复杂笑容识别，因此需要在 CSV 中为每个 episode 记录遮挡状态，方便后续训练时：

- 筛选 clean 数据
- 排除 severe occlusion 数据
- 比较 clean / occluded 条件下的模型表现
- 分析手势遮挡对复杂笑容识别的影响
- 将 occlusion 信息作为后续训练或分析的辅助字段

---

## 二、当前问题

项目里已经存在一部分真实标注数据：

```text
annotation/dataset-annotation/annotations.csv
```

这个文件不能随意覆盖、删除或重写。

当前 `annotations.csv` 已经有固定字段顺序：

```csv
episode_id,video_id,clip_path,person_id,start_frame,peak_frame,end_frame,start_time,peak_time,end_time,main_label,confidence,intensity,eye_involvement,mouth_movement,cheek_raise,symmetry,visible_quality,usable_for_training,note
```

如果直接修改 CSV schema，可能导致：

- 旧 CSV 无法读取
- 旧标注数据丢失
- update/delete 重写 CSV 时破坏原文件
- 旧测试失败
- GUI 加载已有 episode 时字段缺失
- 下游训练脚本读取异常

因此，这次修改必须同时解决：

```text
新增遮挡标注字段
+
旧 annotations.csv 的安全迁移与兼容
```

---

## 三、解决方案总原则

请采用“向后兼容 schema migration”的方案。

也就是说：

1. 旧 CSV 可以正常读取
2. 读取旧 CSV 时自动补充新字段的默认值
3. 不要在读取时直接覆盖原 CSV
4. 保存、更新、删除之前必须有备份机制
5. 新版 CSV 再次读取时不要重复添加字段
6. 原有字段和未知字段都不能丢失
7. `annotations.csv` 的已有真实数据必须安全保留

---

## 四、新增 CSV 字段

在现有 CSV 字段基础上，新增以下字段：

```csv
occlusion_type,occlusion_start_frame,occlusion_end_frame,occlusion_severity,occlusion_note
```

字段含义：

| 字段名 | 含义 |
|---|---|
| occlusion_type | 遮挡类型 |
| occlusion_start_frame | 遮挡开始帧 |
| occlusion_end_frame | 遮挡结束帧 |
| occlusion_severity | 遮挡严重程度 |
| occlusion_note | 遮挡相关备注 |

推荐新版 CSV 字段顺序是在现有字段后追加新字段：

```csv
episode_id,video_id,clip_path,person_id,start_frame,peak_frame,end_frame,start_time,peak_time,end_time,main_label,confidence,intensity,eye_involvement,mouth_movement,cheek_raise,symmetry,visible_quality,usable_for_training,note,occlusion_type,occlusion_start_frame,occlusion_end_frame,occlusion_severity,occlusion_note
```

注意：

- 不要插入到旧字段中间
- 不要改变旧字段顺序
- 不要改变旧字段含义
- 新字段追加在末尾即可

---

## 五、遮挡类型定义

`occlusion_type` 必须使用固定选项，不允许自由输入。

固定值：

```text
none
mouth_partial
mouth_severe
lower_face_occluded
hand_near_face_but_not_occluding
```

### 1. none

表示没有明显面部遮挡。

使用条件：

- 嘴部、嘴角、脸颊、下半脸都清晰可见
- 没有手、物体、头发、口罩等遮挡关键表情区域
- 可以正常判断笑容形态
- 手没有进入面部附近，或者即使出现也完全不影响表情区域

默认值：

```text
occlusion_type = none
occlusion_start_frame = empty
occlusion_end_frame = empty
occlusion_severity = none
occlusion_note = empty
```

### 2. mouth_partial

表示嘴部被部分遮挡，但仍然可以看到部分嘴角或嘴部运动。

使用条件：

- 手、杯子、笔、衣物等遮挡了一部分嘴部
- 至少一侧嘴角仍然可见
- 仍然可以大致判断是否在笑
- 但对嘴角幅度、左右对称性、唇部形态的判断受到影响

典型例子：

- 手指挡住嘴唇中间，但嘴角还能看到
- 杯子短暂遮住部分嘴部
- 一侧嘴角被挡，另一侧嘴角可见
- 手靠近嘴边并遮住部分唇部

建议 severity：

```text
mild
moderate
```

### 3. mouth_severe

表示嘴部严重遮挡，嘴角和嘴唇的大部分信息不可见。

使用条件：

- 双侧嘴角基本不可见
- 嘴唇大部分被手或物体遮住
- 无法可靠判断嘴部形态
- smile peak 附近嘴部被严重遮挡
- 只能依靠眼周、脸颊、上下文推测是否在笑

典型例子：

- 手掌覆盖嘴部
- 拳头挡住整个嘴部区域
- 杯子完全挡住嘴部
- 口罩遮挡嘴部
- peak frame 附近嘴部完全不可见

建议 severity：

```text
severe
```

注意：

- `mouth_severe` 不一定自动等于 `usable_for_training = no`
- `mouth_severe` 不一定自动等于 `unclear`
- 只记录遮挡事实，不要自动改写 `main_label`
- 是否用于训练由后续训练脚本决定

### 4. lower_face_occluded

表示不只是嘴部，而是下半脸大面积被遮挡。

使用条件：

- 嘴部、下巴、部分脸颊同时被遮挡
- 下半脸整体形态不可见
- 不仅影响嘴部判断，也影响 cheek / nasolabial fold 等区域分析
- 无法可靠分析下半脸的表情变化

典型例子：

- 手掌遮住嘴部和下巴
- 文件、衣领、物体遮住下半张脸
- 人低头导致下半脸不可见
- 口罩加手部遮挡
- 画面边缘裁掉下半脸

建议 severity：

```text
severe
```

### 5. hand_near_face_but_not_occluding

表示手靠近脸，但没有真正遮挡关键表情区域。

使用条件：

- 手出现在脸附近
- 手没有遮挡嘴部、嘴角、脸颊等关键区域
- 嘴部和下半脸仍然清晰可见
- 不影响人工判断笑容类型
- 但需要记录，因为手势可能影响模型学习上下文信息

典型例子：

- 手托腮，但嘴角清晰可见
- 手在嘴边附近移动，但没有挡住嘴
- 手靠近下巴，但不遮挡嘴部
- 手在脸旁边，但没有覆盖表情区域

建议 severity：

```text
none
mild
```

---

## 六、遮挡严重程度定义

新增固定值集合：

```python
OCCLUSION_SEVERITY_VALUES = [
    "none",
    "mild",
    "moderate",
    "severe",
]
```

推荐对应关系：

| occlusion_type | 推荐 occlusion_severity |
|---|---|
| none | none |
| hand_near_face_but_not_occluding | none 或 mild |
| mouth_partial | mild 或 moderate |
| mouth_severe | severe |
| lower_face_occluded | severe |

不需要强制阻止不推荐组合，但应在保存前进行基本校验或 warning。

---

## 七、annotation_store.py 修改要求

请优先修改 `annotation_store.py`，因为 CSV schema、数据类、读写、校验都集中在这里。

### 1. 扩展常量

新增：

```python
OCCLUSION_TYPES = [
    "none",
    "mouth_partial",
    "mouth_severe",
    "lower_face_occluded",
    "hand_near_face_but_not_occluding",
]

OCCLUSION_SEVERITY_VALUES = [
    "none",
    "mild",
    "moderate",
    "severe",
]
```

### 2. 扩展 CSV_COLUMNS

在现有 `CSV_COLUMNS` 末尾追加：

```python
"occlusion_type",
"occlusion_start_frame",
"occlusion_end_frame",
"occlusion_severity",
"occlusion_note",
```

不要插入到中间。

### 3. 扩展 EpisodeDraft

在 `EpisodeDraft` 中新增字段，建议放在 `note` 后面：

```python
occlusion_type: str = "none"
occlusion_start_frame: int | None = None
occlusion_end_frame: int | None = None
occlusion_severity: str = "none"
occlusion_note: str = ""
```

注意：

- 默认值必须保证旧逻辑创建 `EpisodeDraft` 时不会大量报错
- 如果现有代码构造 `EpisodeDraft` 使用位置参数，请小心避免破坏
- 更推荐在 GUI 里使用关键字参数构造

### 4. 增加旧 CSV 迁移函数

实现一个明确的 migration 函数：

```python
OCCLUSION_COLUMNS_WITH_DEFAULTS = {
    "occlusion_type": "none",
    "occlusion_start_frame": "",
    "occlusion_end_frame": "",
    "occlusion_severity": "none",
    "occlusion_note": "",
}

def ensure_csv_schema_columns(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    ...
```

或者如果当前代码是按 DataFrame 实现，可以用 DataFrame 版本。

要求：

- 读取旧 CSV 后，如果缺少新字段，自动补默认值
- 不覆盖已有新字段值
- 不删除未知字段
- 不改变已有旧字段值
- 不在 read 阶段写回文件

### 5. 修改 read_rows()

当前 `read_rows()` 可能会验证 header 是否严格等于 `CSV_COLUMNS`。

请将它改成“向后兼容读取”：

- 如果 CSV 不存在，返回空列表
- 如果 CSV 是旧 schema，允许读取
- 如果缺少旧的核心字段，才报错
- 如果缺少 occlusion 字段，自动在内存中补默认值
- 如果 CSV 已有额外未知字段，不要删除它们
- 读取完成后返回的 row 必须包含 occlusion 相关字段

核心字段至少包括：

```python
REQUIRED_BASE_COLUMNS = [
    "episode_id",
    "video_id",
    "clip_path",
    "person_id",
    "start_frame",
    "peak_frame",
    "end_frame",
    "start_time",
    "peak_time",
    "end_time",
    "main_label",
    "confidence",
    "intensity",
    "eye_involvement",
    "mouth_movement",
    "cheek_raise",
    "symmetry",
    "visible_quality",
    "usable_for_training",
    "note",
]
```

### 6. 写 CSV 时保持稳定字段顺序

写出 CSV 时，字段顺序应为：

1. `CSV_COLUMNS` 中定义的字段
2. 原 CSV 中可能存在的未知字段，追加在后面

如果当前实现只支持 `CSV_COLUMNS`，至少要保证：

- 不破坏当前已有字段
- 新增 occlusion 字段稳定出现在末尾
- 不重复写字段

### 7. 修改 row 构造逻辑

在 `append_episode()` 和 `update_episode()` 中，将 `EpisodeDraft` 的 occlusion 字段写入 row：

```python
"occlusion_type": draft.occlusion_type
"occlusion_start_frame": "" if draft.occlusion_start_frame is None else str(draft.occlusion_start_frame)
"occlusion_end_frame": "" if draft.occlusion_end_frame is None else str(draft.occlusion_end_frame)
"occlusion_severity": draft.occlusion_severity
"occlusion_note": draft.occlusion_note
```

### 8. 修改校验逻辑

在 `validate_episode_draft(draft)` 中增加：

- `draft.occlusion_type` 必须在 `OCCLUSION_TYPES`
- `draft.occlusion_severity` 必须在 `OCCLUSION_SEVERITY_VALUES`
- 如果 `occlusion_type == "none"`：
  - `occlusion_start_frame` 和 `occlusion_end_frame` 应该为 None
  - `occlusion_severity` 应该为 `"none"`
- 如果 `occlusion_start_frame` 和 `occlusion_end_frame` 都存在：
  - 必须满足 `occlusion_start_frame <= occlusion_end_frame`
- 不要强制要求遮挡范围必须在 `start_frame` 到 `end_frame` 之间
  - 因为遮挡可能从 episode 前开始，或持续到 episode 后
  - 可以在 GUI 层 warning，但存储层不要过度禁止

---

## 八、CSV 备份与旧数据保护

交接文档已经明确指出：`annotations.csv` 当前是真实标注数据，不应随意重写或删除。  
而当前 `update_episode()` 和 `delete_episode()` 会读取所有行后重写 CSV。

因此这次必须增加备份机制。

### 1. 自动备份函数

在 `annotation_store.py` 中增加函数，例如：

```python
def _backup_csv_before_rewrite(self) -> Path | None:
    ...
```

备份文件名建议：

```text
annotations.backup.YYYYMMDD-HHMMSS.csv
```

或者：

```text
annotations_backup_YYYYMMDD_HHMMSS.csv
```

要求：

- 只有当原 CSV 存在时才备份
- backup 内容必须是修改前的原始 CSV
- backup 创建成功后才允许执行 update/delete/rewrite
- backup 失败时应停止写入并抛出异常
- append 操作可以不备份，因为 append 风险较低
- update/delete 必须备份，因为它们会重写整个 CSV
- 如果 schema migration 后第一次保存会重写 CSV，也必须备份

### 2. 旧 schema 检测

增加一个方法，例如：

```python
def is_old_schema(self) -> bool:
    ...
```

或者在 `read_rows()` 中返回状态。

目的：

- GUI 可以提示用户当前 CSV 是旧格式
- 第一次写出新 schema 前可以确保已备份

如果实现复杂，可以先在 `AnnotationStore` 内部保存一个状态变量，例如：

```python
self.last_read_was_old_schema = False
```

读取时发现缺少 occlusion 字段，则设为 True。

---

## 九、smile_episode_annotation_tool.py 修改要求

请在 GUI 中增加一个新的 group：

```text
Occlusion
```

建议放在现有 Episode Label group 下方，Action row 上方。

### 1. 新增控件

需要增加：

#### occlusion_type 下拉框

选项：

```text
none
mouth_partial
mouth_severe
lower_face_occluded
hand_near_face_but_not_occluding
```

默认：

```text
none
```

#### occlusion_severity 下拉框

选项：

```text
none
mild
moderate
severe
```

默认：

```text
none
```

#### occlusion_start_frame 显示行

需要显示当前遮挡开始帧。

按钮：

```text
Set Occ Start
Go
Clear
```

#### occlusion_end_frame 显示行

需要显示当前遮挡结束帧。

按钮：

```text
Set Occ End
Go
Clear
```

#### occlusion_note 文本框

用于记录遮挡备注。

### 2. 新增状态变量

在 `SmileEpisodeAnnotationWindow` 中新增：

```python
self.current_occlusion_marks = {
    "start": None,
    "end": None,
}
```

或者直接用两个成员变量：

```python
self.occlusion_start_frame = None
self.occlusion_end_frame = None
```

要求：

- 不要和 smile episode 的 start/peak/end 混淆
- 遮挡 range 是 episode 内部的辅助标注，不是 episode 本身边界

### 3. 设置遮挡开始/结束

新增函数：

```python
set_occlusion_start()
set_occlusion_end()
clear_occlusion_start()
clear_occlusion_end()
jump_to_occlusion_start()
jump_to_occlusion_end()
```

行为：

- `Set Occ Start`：将当前帧写入 occlusion_start_frame
- `Set Occ End`：将当前帧写入 occlusion_end_frame
- `Go`：跳转到对应遮挡帧
- `Clear`：清空对应遮挡帧

### 4. occlusion_type = none 的特殊行为

当用户选择：

```text
none
```

时，建议自动：

```text
occlusion_start_frame = None
occlusion_end_frame = None
occlusion_severity = none
```

不要自动清空 `occlusion_note`，避免误删人工备注。

### 5. 保存 episode 时构造 EpisodeDraft

在 `_append_new_episode()` 和 `_update_loaded_episode()` 中构造 `EpisodeDraft` 时加入：

```python
occlusion_type=self.occlusion_type_combo.currentText(),
occlusion_start_frame=self.current_occlusion_marks["start"],
occlusion_end_frame=self.current_occlusion_marks["end"],
occlusion_severity=self.occlusion_severity_combo.currentText(),
occlusion_note=self.occlusion_note_edit.toPlainText().strip(),
```

具体变量名按现有代码风格调整。

### 6. 加载已有 episode 时回填 occlusion 字段

在 `_load_episode_into_form(row)` 中新增：

- 读取 `row["occlusion_type"]`
- 读取 `row["occlusion_start_frame"]`
- 读取 `row["occlusion_end_frame"]`
- 读取 `row["occlusion_severity"]`
- 读取 `row["occlusion_note"]`

并正确写回 UI。

如果旧 CSV 中没有这些字段，`read_rows()` 应该已经补默认值，所以这里可以假设字段存在。

### 7. Clear Current Episode 时清空 occlusion

`Clear Current Episode` 应该同时清空：

```text
occlusion_type = none
occlusion_start_frame = None
occlusion_end_frame = None
occlusion_severity = none
occlusion_note = ""
```

### 8. episode list 表格可以轻量扩展

当前 episode list 显示：

```text
episode_id, start, peak, end, label, conf, usable
```

建议新增一列：

```text
occ
```

显示 `occlusion_type` 的简短值。

例如：

```text
none
mouth_partial
mouth_severe
lower_face_occluded
hand_near_face
```

如果 UI 空间不够，可以先不加，但推荐加，便于检查哪些 episode 已经补充遮挡标注。

### 9. slider 可视化先不做

本次不要优先实现遮挡区间在 slider 上的可视化，避免改动过大。

如果简单，可以显示文本：

```text
Occlusion Range: 120 - 150
```

如果没有遮挡范围：

```text
Occlusion Range: None
```

---

## 十、保存前 warning

GUI 层保存前增加轻量 warning，不要过度阻止保存。

建议规则：

### 1. occlusion_type = none 但有遮挡 range

自动清空 range，并把 severity 设为 none。

### 2. occlusion_type != none 但 severity = none

弹出 warning 或状态栏提示，但允许用户继续保存。

### 3. occlusion_start_frame > occlusion_end_frame

弹出 warning，建议阻止保存，因为这是明显错误。

### 4. 遮挡 range 超出 episode start/end

只 warning，不阻止保存。

原因：

- 遮挡可能从 episode 前开始
- 遮挡可能持续到 episode 后
- 这里不是严重错误

---

## 十一、测试要求

必须修改或新增 `test_annotation_store.py`。

当前已有 12 个测试，必须保证全部继续通过。

新增测试至少包括：

### Test 1：旧 CSV 自动补 occlusion 字段

构造一个旧 schema CSV，只包含原来的字段。

读取后应满足：

```python
row["occlusion_type"] == "none"
row["occlusion_start_frame"] == ""
row["occlusion_end_frame"] == ""
row["occlusion_severity"] == "none"
row["occlusion_note"] == ""
```

### Test 2：append episode 写入 occlusion 字段

新增一个带 occlusion 的 episode。

检查 CSV row 中包含：

```python
"occlusion_type": "mouth_partial"
"occlusion_start_frame": "100"
"occlusion_end_frame": "120"
"occlusion_severity": "moderate"
"occlusion_note": "hand partially covers mouth"
```

### Test 3：update episode 保留 episode_id 并更新 occlusion 字段

编辑已有 episode 时：

- `episode_id` 不变
- occlusion 字段被正确更新
- 不新增重复行

### Test 4：新版 CSV 再次读取不重复添加字段

读取已经包含 occlusion 字段的 CSV。

要求：

- 不重复添加 header
- 不覆盖已有 occlusion 值

### Test 5：update/delete 前自动备份

调用 `update_episode()` 或 `delete_episode()` 后，应生成 backup 文件。

要求：

- backup 文件存在
- backup 内容是修改前的 CSV
- 原 CSV 修改成功

### Test 6：occlusion_type 非法值会报错

例如：

```python
occlusion_type="wrong_value"
```

应触发 ValueError。

### Test 7：occlusion_start_frame > occlusion_end_frame 会报错

例如：

```python
occlusion_start_frame=200
occlusion_end_frame=100
```

应触发 ValueError。

---

## 十二、验证命令

修改后至少运行：

```powershell
cd E:\Single_frame_smile\annotation\dataset-annotation
python -m py_compile annotation_store.py smile_episode_annotation_tool.py test_annotation_store.py
python test_annotation_store.py
```

如果当前环境没有 PySide6，无法启动 GUI，请明确说明：

```text
GUI was not fully launched because PySide6 is not available in this environment.
Storage logic and syntax checks were completed.
```

如果环境有 PySide6，请手动启动：

```powershell
python smile_episode_annotation_tool.py
```

并至少点测：

- 打开视频
- 新增 episode，occlusion_type = none
- 新增 episode，occlusion_type = mouth_partial，并设置 occlusion start/end
- 点击已有 episode，确认 occlusion 字段能回填
- 编辑已有 episode 的 occlusion 字段并保存
- 删除 episode，确认 backup 生成
- 重新打开工具，确认新版 CSV 能正常读取

---

## 十三、验收标准

完成后应满足：

1. 旧版 `annotations.csv` 可以正常打开
2. 旧数据不会被清空、覆盖或破坏
3. 读取旧 CSV 时会自动补充 occlusion 字段默认值
4. 保存后新版 CSV 包含 occlusion 字段
5. 新增 episode 可以保存 occlusion 信息
6. 编辑已有 episode 可以更新 occlusion 信息
7. 删除和更新前会自动备份 CSV
8. `episode_id` 逻辑不变
9. `main_label`、peak 规则、frame order 校验不被破坏
10. 现有测试继续通过
11. 新增测试覆盖 schema migration、occlusion 字段读写、backup 机制

---

## 十四、不要做的事情

不要做以下事情：

- 不要重构整个项目
- 不要把项目改成 web app
- 不要修改旧 HTML 标注工具
- 不要改变 frame index 规则
- 不要改变已有 label 集合
- 不要把 `mouth_severe` 自动改成 `unclear`
- 不要自动把 `usable_for_training` 改成 `no`
- 不要在读取旧 CSV 时直接覆盖原文件
- 不要删除未知字段
- 不要让 update/delete 在没有 backup 的情况下重写真实 `annotations.csv`
- 不要生成新的 `episode_id` 来替代被编辑的 episode

---

## 十五、最终目标

本次任务完成后，现有标注工具应该在保持原有功能稳定的基础上，支持为每个 smile episode 补充遮挡信息。

最终每一行 episode 不仅包含：

```text
start_frame / peak_frame / end_frame / main_label / confidence / visual attributes
```

还应包含：

```text
occlusion_type / occlusion_start_frame / occlusion_end_frame / occlusion_severity / occlusion_note
```

这样后续训练时序模型时，可以根据遮挡状态筛选样本，避免嘴部严重遮挡样本污染复杂笑容分类实验。
