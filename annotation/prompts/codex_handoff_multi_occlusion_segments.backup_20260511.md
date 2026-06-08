# 项目交接文档

## 1. 项目概要

- 当前项目是 `Single_frame_smile` 仓库中的本地视频标注工具，核心目录是 `E:\Single_frame_smile\annotation\dataset-annotation`。
- 项目目标是构建一个用于会议视频片段的 smile episode 标注工具。标注单位是一个有起止帧的 smile episode，不是单帧，也不是整段视频。
- 当前技术栈是 Python + PySide6 + OpenCV。GUI 负责加载视频、播放/暂停、逐帧或按秒跳转、标记 episode 的 `start/peak/end`、选择 smile label 和视觉属性、管理遮挡标注，并保存到统一 CSV。
- 当前正在开发的功能是 episode 内多段面部遮挡标注。每个 episode 可以有 0 个、1 个或多个 occlusion segment，每个 segment 包含 `start/end/type/severity/note`。
- 用户最终想要的是稳定生成 episode-level 的 `annotations.csv`，后续训练脚本可以按每一行读取 `clip_path`、`start_frame`、`end_frame` 和 `occlusion_segments`，从视频中抽取对应帧段，构建时序模型训练样本，并按遮挡状态筛选或分析数据。

## 2. 当前开发状态

已经完成的功能：

- 加载本地视频文件，支持 `.mp4`, `.avi`, `.mov`, `.mkv` 和任意文件。
- 显示视频文件名、绝对路径、FPS、总帧数、当前帧和当前时间。
- 视频播放/暂停、slider seek、逐帧跳转、5 帧跳转、1 秒跳转。
- 支持 `1.0x` 和 `0.5x` 播放速度。
- 支持快捷键：
  - `Space`：播放/暂停
  - `Left Arrow` / `Right Arrow`：后退/前进 1 帧
  - `A` / `D`：后退/前进 5 帧
  - `J` / `L`：后退/前进 1 秒
  - `S`：设置 episode start
  - `P`：设置 episode peak
  - `E`：设置 episode end
  - `Ctrl+S`：保存
  - `F11`：全屏
  - `Esc`：退出全屏
- Slider 上可显示 start/peak/end marker。
- Start/Peak/End 行都有 `Go` 按钮，Peak 行有 `Clear` 按钮。
- 支持主标签、confidence、intensity、eye involvement、mouth movement、cheek raise、symmetry、visible quality、usable for training、note。
- 所有视频共用一个 `annotations.csv`，一行对应一个 episode。
- `episode_id` 全局递增，不按视频重置。
- 当前视频的 episode list 会从统一 CSV 中筛选当前视频相关行。
- 点击右下方 episode list 的行，会加载该 episode 全部字段并跳到 start frame。
- 未加载已有 episode 时，`Save Episode` 会 append 新行。
- 加载已有 episode 后，`Save Episode` 会 update 原 `episode_id` 对应行，不生成新 ID。
- 保存新 episode 后，会自动把下一条 episode 的 start 设置为刚保存 episode 的 end，并清空 peak/end/note/occlusion segments。
- 保存已编辑 episode 后，也会自动进入下一条新 episode：`start = 刚保存 episode 的 end`，`loaded_episode_id = None`，下一次保存会新增而不是继续覆盖。
- 可删除选中 episode；删除前有确认弹窗。
- 可回放选中 episode，从 start 播放到 end 并自动停止。
- `neutral_or_no_smile` 和 `unclear` 不使用 peak，保存时 `peak_frame` 和 `peak_time` 为空。
- 四类 smile label 必须有 peak：`genuine_like_smile`, `polite_like_smile`, `bitter_awkward_like_smile`, `ambiguous_smile`。
- 已实现遮挡标注：
  - 单 episode 支持多个 occlusion segment。
  - GUI 中 `Occlusion` 区域包含 segment draft 和 segment table。
  - 支持 `Add Segment`, `Update Selected Segment`, `Delete Selected Segment`, `Clear Segment Draft`, `Clear All Segments`。
  - 保存时以 `current_occlusion_segments` 作为完整遮挡来源。
  - CSV 中 `occlusion_segments` 保存 JSON 字符串。
  - 原来的 `occlusion_type`, `occlusion_start_frame`, `occlusion_end_frame`, `occlusion_severity`, `occlusion_note` 保留为 summary fields，由 `occlusion_segments` 自动生成。
- 存储层支持旧 CSV 向后兼容：
  - 旧 CSV 缺少 occlusion 字段时，读取阶段只在内存补默认值，不写回文件。
  - 旧单段 occlusion 字段可迁移为一个 JSON segment。
  - 已有 `occlusion_segments` 时不重复生成，不覆盖已有 JSON。
  - 未知额外字段会在 update/delete 写回时保留。
- update/delete 以及旧 schema 第一次 append 触发 schema 写回前，会自动生成备份文件，格式类似 `annotations.backup.YYYYMMDD-HHMMSS-ffffff.csv`。

已经创建或修改的关键文件：

- `annotation/dataset-annotation/annotation_store.py`
- `annotation/dataset-annotation/smile_episode_annotation_tool.py`
- `annotation/dataset-annotation/test_annotation_store.py`
- `annotation/dataset-annotation/smile_episode_annotation_tool_review.md`
- `annotation/dataset-annotation/annotations.csv`
- `annotation/dataset-annotation/annotations.backup.*.csv`
- `annotation/prompts/occlusion_annotation_task_prompt.md`
- `annotation/prompts/incremental_multi_occlusion_segments.md`
- `annotation/prompts/codex_handoff_multi_occlusion_segments.md`

已经验证的模块：

- `python -m py_compile annotation_store.py smile_episode_annotation_tool.py test_annotation_store.py` 通过。
- `python test_annotation_store.py` 通过，当前为 `31 tests OK`。
- 当前检查时 `annotations.csv` 已是新版 header，包含 `occlusion_segments`，共有 62 行 episode 数据。

尚未完成或未充分验证：

- 当前 Codex 环境没有安装 PySide6，因此 GUI 未能实际启动点测。
- 没有自动化 GUI 测试。
- 没有打包成 exe。
- 没有真正的视频队列管理，`Next Video` 当前等价于重新打开视频选择器。
- 没有多 annotator 支持。
- 没有导出 preview clip。
- 没有 raw video 和 processed video side-by-side 显示。
- 没有 undo 删除或 undo save。
- GUI 中尚未实现明显的 `Mode: New Episode / Editing E000xxx` 状态标签。

当前已知风险：

- `annotations.csv` 是真实标注数据，不应随意删除、清空、重建或手工覆盖。
- `annotations.backup.*.csv` 是自动备份文件，不要误删，除非用户明确要求清理。
- update/delete 会重写整个 CSV，虽然已经有自动备份，但仍要谨慎。
- `clip_path` 保存绝对路径，跨机器移动数据时需要额外处理路径。
- `__pycache__` 中可能有被测试命令更新的 `.pyc` 文件，不应作为功能改动提交或关注。

## 3. 项目目录与关键文件

关键目录：

- `E:\Single_frame_smile\annotation`
  - 标注相关目录，包含旧 HTML 标注工具、prompt 文档、当前 dataset episode 标注工具。

- `E:\Single_frame_smile\annotation\dataset-annotation`
  - 当前 smile episode 视频标注工具核心目录。

- `E:\Single_frame_smile\annotation\prompts`
  - 当前和未来交接、需求、增量任务文档目录。

关键文件：

- `annotation/dataset-annotation/smile_episode_annotation_tool.py`
  - PySide6 + OpenCV GUI 主程序。
  - 主要类是 `SmileEpisodeAnnotationWindow(QMainWindow)`。
  - 负责视频加载、播放、快捷键、start/peak/end 标记、episode 表单、episode list、遮挡 segment 表格、保存/编辑/删除/回放。

- `annotation/dataset-annotation/annotation_store.py`
  - CSV 存储和校验模块。
  - 定义 CSV 列顺序、标签集合、`OcclusionSegment`、`EpisodeDraft`、`AnnotationStore`。
  - 负责 `append_episode()`, `update_episode()`, `delete_episode()`, `episodes_for_video()`, `next_episode_id()`。
  - 负责 occlusion JSON 解析/序列化、旧 schema 兼容、summary 生成、备份和重复检测。

- `annotation/dataset-annotation/test_annotation_store.py`
  - 不依赖 GUI 的单元测试。
  - 当前 31 个测试，覆盖保存、更新、删除、ID 递增、重复检测、peak 规则、旧 schema 迁移、occlusion segments JSON、summary、备份、未知字段保留等。

- `annotation/dataset-annotation/requirements.txt`
  - GUI 运行依赖：
    - `PySide6>=6.5`
    - `opencv-python>=4.8`

- `annotation/dataset-annotation/annotations.csv`
  - 实际标注输出文件。
  - 所有视频共用一个 CSV。
  - 当前已经包含多段遮挡字段 `occlusion_segments`。
  - 这是用户真实数据，不能随意覆盖或清空。

- `annotation/dataset-annotation/annotations.backup.*.csv`
  - 自动备份文件。
  - update/delete 或旧 schema 写回前生成。
  - 内容应为写入前原 CSV。

- `annotation/dataset-annotation/smile_episode_annotation_tool_review.md`
  - 开发和验证记录，包含功能迭代说明、验证命令和已知限制。

- `annotation/dataset-annotation/smile_episode_annotation_tool_task.md`
  - 原始任务需求文档，定义工具目标、推荐技术、CSV 格式、标签集合、交互要求、验收标准。

- `annotation/prompts/occlusion_annotation_task_prompt.md`
  - 单段遮挡功能的任务文档。

- `annotation/prompts/incremental_multi_occlusion_segments.md`
  - 多段遮挡增量任务文档。

- `annotation/annotation_tool.html` 和 `annotation/annotation_tool_isSmile.html`
  - 旧的图片标注 HTML 工具，不是当前视频 episode 标注工具核心。不要为当前需求改动它们，除非用户明确要求。

文件调用关系：

- `smile_episode_annotation_tool.py` 从 `annotation_store.py` 导入：
  - `AnnotationStore`
  - `EpisodeDraft`
  - `OcclusionSegment`
  - `MAIN_LABELS`
  - `OCCLUSION_TYPES`
  - `OCCLUSION_SEVERITY_VALUES`
  - `SYMMETRY_VALUES`
  - `USABLE_VALUES`
  - `VISIBLE_QUALITY_VALUES`
  - `default_usable_for_training`
  - `label_requires_peak`
  - `parse_occlusion_segments`
  - `summarize_occlusion_segments`
- GUI 运行时创建 `AnnotationStore(DEFAULT_CSV_PATH)`，其中 `DEFAULT_CSV_PATH = APP_DIR / "annotations.csv"`。
- 保存新 episode 时，GUI 构造 `EpisodeDraft` 并调用 `store.append_episode(draft)`。
- 编辑已有 episode 时，GUI 使用 `loaded_episode_id` 调用 `store.update_episode(episode_id, draft)`。
- 删除 episode 时，GUI 调用 `store.delete_episode(episode_id)`。
- 刷新右下方 episode list 时，GUI 调用 `store.episodes_for_video(self.video_path)`。

## 4. 当前核心设计

### 数据结构

核心 CSV 基础字段：

```csv
episode_id,video_id,clip_path,person_id,start_frame,peak_frame,end_frame,start_time,peak_time,end_time,main_label,confidence,intensity,eye_involvement,mouth_movement,cheek_raise,symmetry,visible_quality,usable_for_training,note
```

遮挡 summary 字段：

```csv
occlusion_type,occlusion_start_frame,occlusion_end_frame,occlusion_severity,occlusion_note
```

多段遮挡完整字段：

```csv
occlusion_segments
```

当前完整 CSV header：

```csv
episode_id,video_id,clip_path,person_id,start_frame,peak_frame,end_frame,start_time,peak_time,end_time,main_label,confidence,intensity,eye_involvement,mouth_movement,cheek_raise,symmetry,visible_quality,usable_for_training,note,occlusion_type,occlusion_start_frame,occlusion_end_frame,occlusion_severity,occlusion_note,occlusion_segments
```

`OcclusionSegment`：

```python
@dataclass(frozen=True)
class OcclusionSegment:
    start: int
    end: int
    type: str
    severity: str
    note: str = ""
```

`EpisodeDraft`：

```python
@dataclass(frozen=True)
class EpisodeDraft:
    video_path: str
    person_id: str
    start_frame: int
    peak_frame: int | None
    end_frame: int
    fps: float
    main_label: str
    confidence: int
    intensity: int
    eye_involvement: int
    mouth_movement: int
    cheek_raise: int
    symmetry: str
    visible_quality: str
    usable_for_training: str
    note: str = ""
    occlusion_type: str = "none"
    occlusion_start_frame: int | None = None
    occlusion_end_frame: int | None = None
    occlusion_severity: str = "none"
    occlusion_note: str = ""
    occlusion_segments: list[OcclusionSegment] = field(default_factory=list)
```

注意：`occlusion_type/start/end/severity/note` 保留是为了旧逻辑兼容。新版保存时应优先使用 `occlusion_segments`，summary 字段由存储层自动生成。

### UI 结构

主窗口是左右分栏：

- 顶部：
  - `Open Video`
  - `Next Video`
  - CSV 路径显示

- 左侧：
  - 大视频显示区 `VideoLabel`
  - 带 start/peak/end marker 的 `FrameSlider`
  - 播放控制区：
    - `Play`
    - `Speed` 下拉，支持 `1.0x` 和 `0.5x`
    - `Prev Frame`
    - `Next Frame`
    - `Back 5`
    - `Forward 5`
    - `Back 1s`
    - `Forward 1s`

- 右侧：
  - Video 信息 group
  - Episode Frames group：
    - Start frame：显示值、`Set Start`、`Go`
    - Peak frame：显示值、`Set Peak`、`Go`、`Clear`
    - End frame：显示值、`Set End`、`Go`
    - Person ID
  - Episode Label group：
    - Label
    - Confidence
    - Intensity
    - Eye involvement
    - Mouth movement
    - Cheek raise
    - Symmetry
    - Visible quality
    - Usable for training
    - Note
  - Occlusion group：
    - Summary
    - Draft type
    - Draft severity
    - Draft start：`Set Occ Start`、`Go`、`Clear`
    - Draft end：`Set Occ End`、`Go`、`Clear`
    - Draft note
    - `Add Segment`
    - `Update Selected Segment`
    - `Delete Selected Segment`
    - `Clear Segment Draft`
    - `Clear All Segments`
    - Segment table：`start | end | type | severity | note`
  - Action row：
    - `Save Episode`
    - `Clear Current Episode`
    - `Play Selected Episode`
    - `Delete Selected Episode`
  - Episode list table：
    - `episode_id`
    - `start`
    - `peak`
    - `end`
    - `label`
    - `conf`
    - `usable`
    - `occ`

### 状态管理方式

`SmileEpisodeAnnotationWindow` 的主要状态：

- `self.capture`：当前 OpenCV `cv2.VideoCapture`。
- `self.video_path`：当前视频绝对路径。
- `self.fps`：当前视频 FPS，OpenCV 读不到时 fallback 为 30.0。
- `self.total_frames`：总帧数。
- `self.current_frame`：当前显示帧，0-based。
- `self.current_marks`：episode 边界标记，结构为 `{"start": int|None, "peak": int|None, "end": int|None}`。
- `self.current_occlusion_marks`：当前 occlusion segment draft 的 start/end，结构为 `{"start": int|None, "end": int|None}`。
- `self.current_occlusion_segments`：当前 episode 已添加的完整 occlusion segment list。保存时使用这个 list。
- `self.current_episode_rows`：右下方 episode table 对应的完整 CSV 行列表。
- `self.loaded_episode_id`：当前表单是否来自已有 episode。非 None 表示 update 模式，None 表示 append 模式。
- `self.playback_rate`：播放速度。
- `self.playback_stop_frame`：选中 episode 回放时的停止帧。

### 主要函数/类职责

`annotation_store.py`：

- `CSV_COLUMNS`
  - 当前完整 CSV 字段顺序，旧字段在前，遮挡 summary 字段和 `occlusion_segments` 在末尾。
- `OcclusionSegment`
  - 表示 episode 内一个遮挡片段。
- `EpisodeDraft`
  - GUI 保存时传给存储层的草稿对象。
- `serialize_occlusion_segments(segments)`
  - 将 segment list 转为 JSON 字符串，空 list 返回 `"[]"`。
- `parse_occlusion_segments(value)`
  - 将 JSON 字符串解析为 `list[OcclusionSegment]`，`""`, `None`, `"[]"` 都解析为 `[]`。
- `validate_occlusion_segment(segment)`
  - 校验 start/end/type/severity。
- `summarize_occlusion_segments(segments)`
  - 根据完整 segment list 自动生成 summary 字段。
- `ensure_csv_schema_columns(rows)`
  - 读取旧 CSV 后在内存中补齐新字段，并将旧单段字段迁移为 `occlusion_segments`。
- `AnnotationStore.read_rows()`
  - 兼容读取 CSV，缺少 occlusion 字段时不直接写回文件。
- `AnnotationStore.append_episode(draft)`
  - 新增 episode。若遇到旧 schema CSV，写回前会先备份并迁移 header。
- `AnnotationStore.update_episode(episode_id, draft)`
  - 覆盖已有 episode，保留原 ID，写回前备份。
- `AnnotationStore.delete_episode(episode_id)`
  - 删除已有 episode，写回前备份。
- `AnnotationStore.has_duplicate(draft, ignore_episode_id=None)`
  - 判断同一视频下是否已有完全相同 start/peak/end 的 episode。

`smile_episode_annotation_tool.py`：

- `FrameSlider`
  - 自定义 QSlider，绘制 start/peak/end marker。
- `VideoLabel`
  - 显示当前视频帧，保持宽高比缩放。
- `SmileEpisodeAnnotationWindow`
  - 主窗口和业务逻辑。
  - `_build_ui()` 构建主界面。
  - `_build_occlusion_group()` 构建多段遮挡 UI。
  - `set_mark()`, `clear_peak_mark()`, `jump_to_mark()` 管理 episode start/peak/end。
  - `add_occlusion_segment()`, `update_selected_occlusion_segment()`, `delete_selected_occlusion_segment()`, `clear_occlusion_segment_draft()`, `clear_all_occlusion_segments()` 管理 segment list。
  - `save_episode()` 根据 `loaded_episode_id` 决定 append 或 update。
  - `_append_new_episode()` 保存新 episode，并自动进入下一段。
  - `_update_loaded_episode()` 更新已有 episode，并自动进入下一段。
  - `_prepare_next_episode_start(frame_index)` 设置下一段 start，清空 peak/end/note/occlusion segments。
  - `_load_episode_into_form(row)` 从 CSV 行加载已有标注和 occlusion segments。

### 文件读写逻辑

- CSV 路径固定为 `annotation/dataset-annotation/annotations.csv`。
- `append_episode()` 正常情况下追加写入。
- 如果 append 时发现 CSV 是旧 schema，会先备份，再迁移整表 header 和旧行字段，然后 append。
- `update_episode()` 和 `delete_episode()` 会读取所有行后重写整个 CSV，写前必须备份。
- 写 CSV 时字段顺序为：
  1. `CSV_COLUMNS`
  2. 旧 CSV 中可能存在的未知字段，追加在后面
- `clip_path` 写为绝对路径。
- 路径比较使用 `_normal_path()`。

### 输入输出格式

`occlusion_segments` 是 CSV 单元格中的 JSON 字符串。例如：

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

无遮挡时：

```json
[]
```

Summary 规则：

- 无 segment：
  - `occlusion_type = none`
  - `occlusion_start_frame = ""`
  - `occlusion_end_frame = ""`
  - `occlusion_severity = none`
  - `occlusion_note = ""`
- 单 segment：
  - summary 等于该 segment。
- 多 segment：
  - `occlusion_start_frame` = 最早 start
  - `occlusion_end_frame` = 最晚 end
  - `occlusion_type` = rank 最高的 type
  - `occlusion_severity` = rank 最高的 severity
  - `occlusion_note = multiple occlusion segments`

Type rank：

```python
OCCLUSION_TYPE_RANK = {
    "none": 0,
    "hand_near_face_but_not_occluding": 1,
    "mouth_partial": 2,
    "mouth_severe": 3,
    "lower_face_occluded": 4,
}
```

Severity rank：

```python
SEVERITY_RANK = {
    "none": 0,
    "mild": 1,
    "moderate": 2,
    "severe": 3,
}
```

### 用户操作流程

新标注流程：

1. 运行：
   ```powershell
   cd E:\Single_frame_smile\annotation\dataset-annotation
   pip install -r requirements.txt
   python smile_episode_annotation_tool.py
   ```
2. 点击 `Open Video` 加载视频。
3. 设置 episode `start_frame`、必要时设置 `peak_frame`、设置 `end_frame`。
4. 选择 smile label、confidence、视觉属性、是否用于训练、note。
5. 如有遮挡：
   - 设置 draft type/severity/note。
   - 在视频帧上点击 `Set Occ Start` 和 `Set Occ End`。
   - 点击 `Add Segment`。
   - 多段遮挡重复上述步骤。
6. 点击 `Save Episode`。
7. 工具写入 CSV，并自动把下一段 episode 的 start 设置为刚保存 episode 的 end。

编辑已有 episode 流程：

1. 加载对应视频。
2. 在右下方 episode list 点击某一行。
3. 工具加载该 episode 的字段和 occlusion segments，并跳到 start frame。
4. 修改 episode 字段或 segment list。
5. 点击 `Save Episode`。
6. 工具覆盖原 `episode_id` 对应行。
7. 保存成功后自动进入下一段新 episode，下一次保存会新增。

删除流程：

1. 在右下方 episode list 选中一行。
2. 点击 `Delete Selected Episode`。
3. 确认弹窗。
4. 工具备份并删除对应 CSV 行。

回放流程：

1. 在右下方 episode list 选中一行。
2. 点击 `Play Selected Episode`。
3. 工具从 start 播到 end 并自动停止。

### 下游训练数据处理流程

当前训练脚本未在本工具中实现，但 CSV 是为下游准备的。预期流程：

1. 读取 `annotations.csv`。
2. 过滤：
   - `usable_for_training == "yes"`
   - `confidence >= 4`
   - 可选：按 `occlusion_type` 或 `occlusion_segments` 过滤 severe occlusion。
3. 对每一行读取 `clip_path`。
4. 从 `start_frame` 到 `end_frame` 抽帧。
5. 将帧段重采样为固定长度 T，例如 20 或 32。
6. 提取视觉特征。
7. 保存为 `.npy` 或 `.pt`，每个 CSV row 对应一个训练样本。
8. 可基于 `occlusion_segments` 分析遮挡对模型表现的影响。

## 5. 重要约定与设计决策

- 标注单位是 episode，不是 frame，也不是整段 clip。
  - 原因：最终训练 temporal model，每个训练样本需要完整 onset → peak → offset 结构。

- 一个 episode 内可以有多个 occlusion segment，但不能因为遮挡把 episode 拆成多个 episode。
  - 原因：遮挡是 episode 内辅助信息，不应破坏 smile event 的时序完整性。

- 所有视频共用一个 `annotations.csv`。
  - 原因：下游训练更容易统一读取和过滤。

- `episode_id` 全局唯一，不按视频重置。
  - 格式：`E000001`, `E000002`, ...
  - 原因：方便跨视频追踪单个 episode。

- frame index 使用 0-based。
  - 原因：与 OpenCV `CAP_PROP_POS_FRAMES` 保持一致。
  - 不要改成 1-based，否则已有 CSV 和抽帧会错位。

- `clip_path` 保存绝对路径。
  - 原因：本机训练流水线可直接定位视频。
  - 风险：跨机器迁移数据时路径可能失效。

- smile 类标签必须有 peak：
  - `genuine_like_smile`
  - `polite_like_smile`
  - `bitter_awkward_like_smile`
  - `ambiguous_smile`
  - 原因：smile episode 需要峰值帧描述 temporal dynamics。

- 非 smile 或不可判断状态不使用 peak：
  - `neutral_or_no_smile`
  - `unclear`
  - 保存时 `peak_frame` 和 `peak_time` 为空。

- 遮挡类型固定为：
  - `none`
  - `mouth_partial`
  - `mouth_severe`
  - `lower_face_occluded`
  - `hand_near_face_but_not_occluding`

- 遮挡严重程度固定为：
  - `none`
  - `mild`
  - `moderate`
  - `severe`

- `occlusion_segments` 是完整遮挡信息来源。
  - 原有 5 个 occlusion 字段只是 summary，不要让 GUI 手工维护 summary。

- segment 校验：
  - `start` 和 `end` 必须是 int。
  - `start <= end`。
  - type/severity 必须合法。
  - 存储层不强制 segment 必须在 episode start/end 内。
  - 原因：遮挡可能从 episode 前开始或持续到 episode 后。

- segment 重叠目前只在 GUI 状态栏 warning，不强制禁止。
  - 原因：真实视频中遮挡片段边界可能模糊，暂不做硬限制。

- gap 当前没有单独建模。
  - 保存后自动设置 next start=end 只是连续标注便利功能。
  - 如果 episode 间有 gap，用户需要手动调整下一段 start。

- 不要自动分类视频。
  - 所有标签由人工标注。
  - 模型训练单独实现。

- 不要随意改 CSV 列顺序。
  - 原因：已有数据和未来训练脚本依赖固定 header。

- 不要删除旧 occlusion summary 字段。
  - 原因：兼容旧逻辑和便于快速查看。

- 不要在读取旧 CSV 时直接覆盖原文件。
  - 只有保存、更新、删除或旧 schema 第一次 append 时才允许写回，并且写回前必须备份。

## 6. 当前用户偏好与开发要求

- 用户希望直接实现功能，不只是给方案。
- 用户偏好本地桌面工具，不希望第一版做 web app。
- 技术栈应保持 Python + PySide6 + OpenCV。
- 用户通过实际 GUI 测试持续反馈交互需求，交互效率很重要。
- UI 应优先保证人脸观察清楚，视频显示区域要大。
- 快捷键应高效，尤其是播放/暂停、逐帧、按秒跳、设置 start/peak/end。
- 文件格式稳定非常重要，尤其是 `annotations.csv`。
- 修改代码时应最小化对已有 CSV 和已有功能的破坏。
- 不要改旧 HTML 工具，除非用户明确要求。
- 不要把 severe occlusion 自动改成 `unclear`。
- 不要自动把 severe occlusion 改成 `usable_for_training = no`。
- 保存 episode 后必须能继续自动开下一段新 episode：
  - 新增保存后如此。
  - 编辑保存后也如此。
- 如果 PySide6 不可用，必须明确说明 GUI 没有实际启动验证。

## 7. 已讨论但尚未实现的功能

### 高优先级

- GUI 人工回归测试清单：
  - 加载视频。
  - 新增无 occlusion episode。
  - 新增单 segment episode。
  - 新增多 segment episode。
  - 选中已有 segment，修改后 update。
  - 删除一个 segment。
  - Clear Segment Draft 不删除已有 segments。
  - Clear All Segments 清空所有 segments。
  - 编辑已有 episode 后保存，确认自动进入下一段新 episode。
  - 旧单段遮挡 CSV 自动迁移为单 segment。
- 在 GUI 中增加明显模式显示：
  - `Mode: New Episode`
  - `Mode: Editing E000123`
  - 当前内部已有 `loaded_episode_id` 判断，但 UI 没有醒目标识。
- 确认真实 `annotations.csv` 中已有行的 `occlusion_segments` 是否需要批量回填或仅让后续保存自然迁移。

### 中优先级

- Undo last delete / undo last save。
- 批量视频队列和真正的 `Next Video`。
- CSV 相对路径保存选项。
- episode list 增加更多列或筛选功能。
- 对已有 episode 双击播放或快捷键播放。
- 删除、更新、保存后的更明确 toast 或状态显示。
- 打包成 Windows 可运行脚本或 exe。
- 对 occlusion segments 增加可视化，例如 slider 上显示区间。
- 对 segment overlap 增加可选硬限制或更明显的 warning。

### 低优先级

- 多 annotator 支持。
- inter-annotator agreement 计算。
- raw video 和 processed video side-by-side。
- 自动导出 episode preview clip。
- frame thumbnail preview。
- 项目文件 import/export。
- keyboard-only 标注模式进一步增强。

## 8. 当前最推荐的下一步任务

- [ ] 在有 PySide6 的本机环境启动 GUI：
  ```powershell
  cd E:\Single_frame_smile\annotation\dataset-annotation
  python smile_episode_annotation_tool.py
  ```
- [ ] 手动测试保存新 episode 后是否自动进入下一段：
  - 设置 start/peak/end。
  - 添加 0 个或多个 occlusion segments。
  - 点击 `Save Episode`。
  - 检查 start 是否自动等于上一段 end，peak/end/note/segments 是否清空。
- [ ] 手动测试编辑已有 episode 后保存是否自动进入下一段：
  - 点击已有 episode。
  - 修改 label 或 segment。
  - 点击 `Save Episode`。
  - 检查下一次保存会新增而不是继续覆盖。
- [ ] 手动测试多段遮挡 GUI：
  - Add Segment。
  - Update Selected Segment。
  - Delete Selected Segment。
  - Clear Segment Draft。
  - Clear All Segments。
  - 保存后重新点击该 episode，确认 segments 完整回填。
- [ ] 如果 GUI 交互确认稳定，在 `smile_episode_annotation_tool_review.md` 中追加人工测试结果。
- [ ] 考虑实现 `Mode: New Episode / Editing E000xxx` 状态提示，降低误覆盖风险。
- [ ] 不要改动 `annotations.csv` 前先确认是否需要备份；虽然代码会自动备份，但手动操作也要谨慎。

## 9. 给下一个 Codex 的注意事项

- 不要重新设计整个工具。当前架构是 PySide6 GUI + `AnnotationStore` CSV 存储，继续沿用。
- 不要把项目改成 web app。
- 不要把一个 episode 拆成多个 episode。遮挡是 episode 内部辅助标注。
- 不要删除 `occlusion_type`, `occlusion_start_frame`, `occlusion_end_frame`, `occlusion_severity`, `occlusion_note`。它们是 summary 字段。
- 不要让 GUI 手工维护 summary 字段。保存时应以 `occlusion_segments` 为完整来源，让 `annotation_store.py` 自动生成 summary。
- 不要改 frame index 规则，继续 0-based。
- 不要改 `episode_id` 规则，继续全局递增。
- 不要清空、重建或随意覆盖 `annotations.csv`。
- 不要删除 `annotations.backup.*.csv`，除非用户明确要求清理。
- 修改保存逻辑前先阅读：
  - `annotation_store.py`
  - `smile_episode_annotation_tool.py`
  - `test_annotation_store.py`
- 修改 UI 行为前先理解：
  - `loaded_episode_id`
  - `current_marks`
  - `current_occlusion_marks`
  - `current_occlusion_segments`
- 保存新 episode 后和编辑 episode 后，都应该自动调用 `_prepare_next_episode_start(end_frame)`，让下一次保存变成新增。
- `Play Selected Episode` 使用 `playback_stop_frame`，不要破坏普通播放逻辑。
- 如果上下文不足，优先查看：
  - `annotation/prompts/codex_handoff_multi_occlusion_segments.md`
  - `annotation/prompts/incremental_multi_occlusion_segments.md`
  - `annotation/prompts/occlusion_annotation_task_prompt.md`
  - `annotation/codex_project_handoff.md`
  - `annotation/dataset-annotation/smile_episode_annotation_tool_review.md`
- 修改代码后至少运行：
  ```powershell
  cd E:\Single_frame_smile\annotation\dataset-annotation
  python -m py_compile annotation_store.py smile_episode_annotation_tool.py test_annotation_store.py
  python test_annotation_store.py
  ```
- 如果 PySide6 未安装，不能完整验证 GUI，只能做语法和存储逻辑验证，必须明确告知用户。
- 注意仓库当前可能有其他目录的 unrelated changes，不要回滚用户改动。

## 10. 当前问题与开放问题

- 是否需要批量把已有 `annotations.csv` 中旧行的 `occlusion_segments` 全部显式写入文件，还是保持当前按保存/更新自然迁移的策略？
- 是否需要对 `annotations.backup.*.csv` 设置清理策略，例如保留最近 N 个或按日期归档？
- 是否需要在 GUI 上实现 `Mode: New Episode / Editing E000xxx` 明显提示？
- 保存新 episode 后 start 自动等于上一段 end 是否总是合理？如果 episode 间有 gap，用户目前需要手动调整。
- 是否需要增加 `auto chain start` 开关？
- `clip_path` 是否未来需要改成相对路径以便跨机器训练？
- 是否需要增加 annotator 字段？
- `person_id` 是否需要固定命名规则，例如 `P01`, `P02`？
- 是否需要编辑保存时保留历史版本或 audit log？
- 是否需要对 occlusion segment overlap 做硬性禁止，还是继续只 warning？
- 是否需要把 occlusion segments 可视化到 slider 上？

## 11. 可直接复制给新 Codex 的简短任务说明

```text
这是一个从旧 Codex 对话交接过来的项目。请先阅读交接文档 `E:\Single_frame_smile\annotation\prompts\codex_handoff_multi_occlusion_segments.md`，不要重新设计整个项目。你的任务是基于现有代码继续开发，保持已有设计和文件格式稳定。

项目是 `E:\Single_frame_smile` 中的 smile episode 视频标注工具。核心代码在 `E:\Single_frame_smile\annotation\dataset-annotation`：

- `smile_episode_annotation_tool.py`：PySide6 + OpenCV GUI。
- `annotation_store.py`：CSV 存储、校验、新增、编辑、删除、备份、schema migration、多段遮挡 JSON。
- `test_annotation_store.py`：存储逻辑测试，当前 31 tests OK。
- `annotations.csv`：用户真实标注数据，当前已包含 `occlusion_segments`，不要覆盖或清空。

当前设计：

- 所有视频共用一个 `annotations.csv`。
- 一行是一个 smile episode。
- frame index 是 0-based。
- `episode_id` 全局递增，不按视频重置。
- `clip_path` 保存绝对路径。
- smile 标签必须有 peak；`neutral_or_no_smile` 和 `unclear` 不使用 peak。
- 一个 episode 可以包含 0 个、1 个或多个 occlusion segment。
- `occlusion_segments` 保存完整 JSON list。
- `occlusion_type`, `occlusion_start_frame`, `occlusion_end_frame`, `occlusion_severity`, `occlusion_note` 是 summary 字段，由 `occlusion_segments` 自动生成。
- 点击右下方 episode 行会加载已有标注；保存时覆盖原 `episode_id`。
- 保存新 episode 后，以及编辑保存已有 episode 后，都要自动进入下一段新 episode：start 等于刚保存 episode 的 end，peak/end/note/segments 清空，下一次 Save 是新增。
- Delete Selected Episode 会备份并删除 CSV 中对应 `episode_id` 行。
- Play Selected Episode 会从 start 播到 end 自动停止。

当前最优先的任务是：在有 PySide6 的环境中手动点测多段遮挡 GUI 和“保存后自动开下一段 episode”的交互，然后根据实际反馈修正 UI。不要改旧 HTML 工具，不要把项目改成 web app，不要删除真实 CSV 或 backup 文件。

修改后必须运行：

cd E:\Single_frame_smile\annotation\dataset-annotation
python -m py_compile annotation_store.py smile_episode_annotation_tool.py test_annotation_store.py
python test_annotation_store.py

如果 PySide6 环境不可用，请说明 GUI 未能实际启动验证。
```
