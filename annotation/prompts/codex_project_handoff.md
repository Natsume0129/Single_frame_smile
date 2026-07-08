# 项目交接文档

更新时间：2026-07-08

本文件是当前项目的主交接文档。旧的多段遮挡阶段交接文档已经备份为：

```text
E:\Single_frame_smile\annotation\prompts\codex_handoff_multi_occlusion_segments.backup_20260511.md
```

后续如果用户说“给我交接文档”，应优先更新本文件，而不是新增另一个 handoff。

## 1. 项目概要

- 当前项目是 `Single_frame_smile` 仓库中的本地视频标注与视频预处理工具集合。
- 核心代码目录是：
  ```text
  E:\Single_frame_smile\annotation\dataset-annotation
  ```
- 当前主要工具有两个：
  - `smile_episode_annotation_tool.py`：PySide6 + OpenCV 桌面标注工具。
  - `split_videos_overlap.py`：批量无重编码切分视频，并在相邻片段保留重叠区间。
- 标注工具已经从早期的 “smile episode 标注” 转为 “按时间顺序切分整段视频的 temporal segment / state interval 标注工具”。
- 当前一行 CSV 表示一个连续视频区间，而不再只表示 smile episode。
- 当前主标签集合是：
  ```text
  truesmile
  polite_smile
  bitter_smile
  smiling_but_ambiguous
  neutral
  discard
  ```
- 所有遮挡、看不清、追踪失败、无法可靠判断的区间都统一标为 `discard`，不再继续开发精细 occlusion 标注 UI。

## 2. 当前开发状态

已经完成的标注工具功能：

- 加载本地视频文件，支持 `.mp4`, `.avi`, `.mov`, `.mkv` 和任意文件。
- 显示视频文件名、绝对路径、FPS、总帧数、当前帧和当前时间。
- 视频播放/暂停、slider seek、逐帧跳转、5 帧跳转、1 秒跳转。
- 支持 `1.0x` 和 `0.5x` 播放速度。
- 支持快捷键：
  - `Space`：播放/暂停
  - `Left Arrow` / `Right Arrow`：后退/前进 1 帧
  - `A` / `D`：后退/前进 5 帧
  - `J` / `L`：后退/前进 1 秒
  - `S`：设置 segment start
  - `P`：设置 segment peak，仅 smile 类标签使用
  - `E`：设置 segment end
  - `Ctrl+S`：保存
  - `Delete`：删除当前选中的 segment 记录，复用原删除确认逻辑
  - `F11`：全屏
  - `Esc`：退出全屏
- Slider 上可显示 start/peak/end marker。
- Start/Peak/End 行都有 `Go` 按钮，Peak 行有 `Clear` 按钮。
- GUI 文案已改为 `Temporal Segment State Annotation Tool`、`Save Segment`、`Segment Frames`、`Segment State`。
- `neutral` 和 `discard` 不使用 peak；用户选择这两个标签时 UI 会清空并禁用 peak。
- 四类 smile 标签必须有 peak：
  - `truesmile`
  - `polite_smile`
  - `bitter_smile`
  - `smiling_but_ambiguous`
- 保存时仍使用原有 `EpisodeDraft` 和 `episode_id` 字段，以减少对 CSV 和旧代码的破坏；语义上应理解为 general temporal segment。
- 所有视频共用一个 `annotations.csv`。
- `episode_id` 全局递增，不按视频重置。
- 点击右下方 segment list 的行，会加载该行字段并跳到 start frame。
- 未加载已有行时，`Save Segment` 会 append 新行。
- 加载已有行后，`Save Segment` 会 update 原 `episode_id` 对应行，不生成新 ID。
- 保存新 segment 或编辑已有 segment 后，会自动进入下一段新 segment：`start = 刚保存 segment 的 end`，`peak/end/note` 清空，下一次保存会新增。
- 可删除选中 segment；删除前有确认弹窗。
- 可回放选中 segment，从 start 播放到 end 并自动停止。
- Occlusion 主 UI 已不再展示为精细标注功能，改为提示：
  ```text
  Occluded / invalid intervals should be labeled as discard.
  ```

已经完成的数据层功能：

- `annotation_store.py` 保留旧 CSV 字段以兼容历史数据。
- 旧标签会在读取或保存时映射为新标签：
  ```text
  genuine_like_smile -> truesmile
  polite_like_smile -> polite_smile
  bitter_awkward_like_smile -> bitter_smile
  ambiguous_smile -> smiling_but_ambiguous
  neutral_or_no_smile -> neutral
  unclear -> discard
  unknown old label -> discard
  ```
- 旧 occlusion 字段或旧 `occlusion_segments` 中存在遮挡时，读取到内存后显示为 `discard`。
- 新保存的数据统一写默认 occlusion summary：
  ```text
  occlusion_type = none
  occlusion_start_frame = empty
  occlusion_end_frame = empty
  occlusion_severity = none
  occlusion_note = empty
  occlusion_segments = []
  ```
- update/delete/旧 schema append 前会自动备份 CSV。
- 自动备份文件已经从代码目录移到：
  ```text
  E:\Single_frame_smile\annotation\backups
  ```
- `dataset-annotation` 目录下不再堆放 `annotations.backup.*.csv`。

已经完成的视频切分脚本功能：

- 新增 `split_videos_overlap.py`。
- 自动扫描指定输入目录中的 `.mp4/.mov/.mkv/.avi`，大小写均兼容。
- 对大于 `max_size_gb` 的视频使用 ffmpeg 无重编码切分。
- 相邻切分片段保留 `overlap_time` 秒重叠。
- 小于等于 `max_size_gb` 的视频不切分，但会复制到输出目录，并在 mapping 中标记 `COPIED_UNDER_SIZE`。
- 输出文件直接放在 `output_dir`，不为每个源视频创建子目录。
- 生成：
  ```text
  split_mapping.csv
  split_mapping.json
  ```
- mapping 记录 source、part、理论 start/end、理论 duration、实际 duration、size、status、warning。
- ffmpeg 命令使用 list 参数调用，不使用 shell 字符串。
- 切分命令使用 `-c copy`，不重编码。
- 针对 GoPro `.MP4` 的 data stream 问题，命令增加 `-dn`，跳过 data streams，保留视频/音频等主流，避免 MP4 muxer 报错。
- `--overwrite` 只删除当前源视频对应的旧 part 文件，不删除整个输出目录，不删除原始视频。

当前真实数据状态：

- 当前主 CSV：
  ```text
  E:\Single_frame_smile\annotation\dataset-annotation\annotations.csv
  ```
- 当前 `annotations.csv` 只有 header，数据行数为 0，文件大小约 339 bytes。
- 旧数据备份位于：
  ```text
  E:\Single_frame_smile\annotation\backups
  ```
- 当前 backups 中有 83 个 `annotations.backup.*.csv`。
- 最大旧备份约有 62 行数据。不要删除这些备份，除非用户明确要求清理或确认已恢复。

当前视频切分处理状态：

- 已对以下目录完成切分：
  ```text
  E:\Matsuda_data\rawdata
  ```
- 输出目录：
  ```text
  E:\Matsuda_data\split_videos
  ```
- 结果：
  ```text
  mapping rows: 130
  OK: 125
  COPIED_UNDER_SIZE: 5
  ERROR: 0
  max output size: 1.922721 GB
  ```
- 生成了：
  ```text
  E:\Matsuda_data\split_videos\split_mapping.csv
  E:\Matsuda_data\split_videos\split_mapping.json
  ```

## 3. 项目目录与关键文件

关键目录：

- `E:\Single_frame_smile\annotation`
  - 标注相关目录，包含旧 HTML 工具、prompt 文档、当前 dataset annotation 工具、backups。
- `E:\Single_frame_smile\annotation\dataset-annotation`
  - 当前 PySide6 标注工具、CSV 存储层、视频切分脚本、测试文件所在目录。
- `E:\Single_frame_smile\annotation\prompts`
  - 当前和未来交接、需求、增量任务文档目录。
- `E:\Single_frame_smile\annotation\backups`
  - `annotations.csv` 的自动备份目录。

关键文件：

- `annotation/dataset-annotation/smile_episode_annotation_tool.py`
  - PySide6 + OpenCV GUI 主程序。
  - 主要类是 `SmileEpisodeAnnotationWindow(QMainWindow)`。
  - 负责视频加载、播放、快捷键、start/peak/end 标记、segment 表单、segment list、保存/编辑/删除/回放。

- `annotation/dataset-annotation/annotation_store.py`
  - CSV 存储和校验模块。
  - 定义 CSV 列顺序、标签集合、`OcclusionSegment`、`EpisodeDraft`、`AnnotationStore`。
  - 负责 label migration、occlusion -> discard 内存迁移、append/update/delete、备份、重复检测。

- `annotation/dataset-annotation/test_annotation_store.py`
  - 不依赖 GUI 的单元测试。
  - 当前覆盖新标签、peak 规则、旧标签迁移、旧 occlusion -> discard、更新保留 ID、备份和未知字段保留等。

- `annotation/dataset-annotation/split_videos_overlap.py`
  - 批量视频切分脚本。
  - 使用 ffmpeg/ffprobe，无重编码，支持 overlap、copy under-size、mapping 输出、overwrite、dry_run。

- `annotation/dataset-annotation/test_split_videos_overlap.py`
  - 不依赖真实视频和 ffmpeg 的单元测试。
  - 覆盖区间计划、命名、旧 part 匹配、ffmpeg 命令参数、小文件复制逻辑。

- `annotation/dataset-annotation/requirements.txt`
  - GUI 运行依赖：
    ```text
    PySide6>=6.5
    opencv-python>=4.8
    ```

- `annotation/dataset-annotation/annotations.csv`
  - 实际标注输出文件。
  - 所有视频共用一个 CSV。
  - 当前只有 header，不要手工清空或覆盖。

- `annotation/backups/annotations.backup.*.csv`
  - 自动备份文件。
  - 当前是恢复旧标注数据的重要来源。

- `annotation/prompts/codex_project_handoff.md`
  - 当前主交接文档，即本文档。

- `annotation/prompts/codex_handoff_multi_occlusion_segments.backup_20260511.md`
  - 旧多段遮挡阶段交接文档，仅作历史参考。

- `annotation/prompts/temporal_segment_state_annotation_prompt.md`
  - temporal segment state 标注改造需求文档。

- `annotation/prompts/video_split_overlap_coding_task.md`
  - 视频重叠切分脚本需求文档。

- `annotation/annotation_tool.html` 和 `annotation/annotation_tool_isSmile.html`
  - 旧 HTML 标注工具。不是当前 PySide6 temporal segment 工具的核心，不要为当前需求改动它们，除非用户明确要求。

## 4. 当前核心设计

### 4.1 标注数据结构

当前 CSV 仍保留历史字段，完整 header 包含：

```csv
episode_id,video_id,clip_path,person_id,start_frame,peak_frame,end_frame,start_time,peak_time,end_time,main_label,confidence,intensity,eye_involvement,mouth_movement,cheek_raise,symmetry,visible_quality,usable_for_training,note,occlusion_type,occlusion_start_frame,occlusion_end_frame,occlusion_severity,occlusion_note,occlusion_segments
```

当前实际语义：

- 一行表示一个 temporal segment / state interval。
- `episode_id` 字段仍沿用，实际可理解为 segment id。
- `start_frame` 和 `end_frame` 是区间边界。
- `peak_frame` 只对 smile 类标签使用。
- `main_label` 使用新标签集合。
- occlusion 字段保留为兼容字段，新保存时默认写 none/empty/[]。

`EpisodeDraft` 当前仍保留历史字段以兼容旧逻辑：

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

### 4.2 标签和 peak 规则

必须有 peak 的标签：

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

不需要 peak 的标签：

```text
neutral
discard
```

保存时自动清空：

```text
peak_frame
peak_time
```

### 4.3 GUI 状态管理

`SmileEpisodeAnnotationWindow` 的主要状态：

- `self.capture`：当前 OpenCV `cv2.VideoCapture`。
- `self.video_path`：当前视频绝对路径。
- `self.fps`：当前视频 FPS，OpenCV 读不到时 fallback 为 30.0。
- `self.total_frames`：总帧数。
- `self.current_frame`：当前显示帧，0-based。
- `self.current_marks`：segment 边界标记，结构为 `{"start": int|None, "peak": int|None, "end": int|None}`。
- `self.current_episode_rows`：右下方 segment table 对应的完整 CSV 行列表。
- `self.loaded_episode_id`：当前表单是否来自已有行。非 None 表示 update 模式，None 表示 append 模式。
- `self.playback_rate`：播放速度。
- `self.playback_stop_frame`：选中 segment 回放时的停止帧。

保存逻辑：

- `loaded_episode_id is None`：append 新 segment。
- `loaded_episode_id is not None`：update 原 segment。
- append/update 成功后调用 `_prepare_next_episode_start(end_frame)`，自动进入下一段新 segment。

### 4.4 视频切分逻辑

切分脚本：

```text
annotation/dataset-annotation/split_videos_overlap.py
```

推荐命令：

```powershell
cd E:\Single_frame_smile\annotation\dataset-annotation
python split_videos_overlap.py --input_dir "E:\Matsuda_data\rawdata" --output_dir "E:\Matsuda_data\split_videos" --segment_time 300 --overlap_time 10 --max_size_gb 2
```

overwrite 重新生成：

```powershell
python split_videos_overlap.py --input_dir "E:\Matsuda_data\rawdata" --output_dir "E:\Matsuda_data\split_videos" --segment_time 300 --overlap_time 10 --max_size_gb 2 --overwrite
```

dry-run 预览：

```powershell
python split_videos_overlap.py --input_dir "E:\Matsuda_data\rawdata" --output_dir "E:\Matsuda_data\split_videos" --segment_time 300 --overlap_time 10 --dry_run
```

切分规则：

```text
step_time = segment_time - overlap_time
next.start_time = previous.end_time - overlap_time
```

ffmpeg 命令核心参数：

```text
-ss START -i input -t DURATION -map 0 -c copy -dn -avoid_negative_ts make_zero output
```

说明：

- `-c copy`：不重编码。
- `-dn`：跳过 GoPro metadata/data streams，避免 MP4 muxer 失败。
- `-map 0`：尽量保留源中的主要流；加 `-dn` 后不会输出 data stream。
- 小于等于 `max_size_gb` 的源视频不会切分，会直接复制到 output_dir。

## 5. 重要约定与设计决策

- 标注单位现在是 temporal segment，不是单帧，也不再只限于 smile episode。
- 内部仍大量沿用 `episode` 命名，这是为了降低重构风险。不要为了命名洁癖大规模改变量名。
- 所有视频共用一个 `annotations.csv`。
- `episode_id` 继续全局唯一，不按视频重置。
- frame index 继续使用 0-based。
- `clip_path` 继续保存绝对路径。
- 不要改 CSV 列顺序。
- 不要删除 occlusion summary 字段或 `occlusion_segments` 字段，因为旧数据和旧逻辑需要兼容。
- 不要继续开发多段 occlusion GUI；遮挡/不可用统一标为 `discard`。
- 不要让 severe occlusion 自动变成旧标签 `unclear`，新规则是 `discard`。
- `discard` 永远不进入主训练集。
- `neutral` 可作为 smile detection 的负类，`discard` 应排除。
- update/delete 会重写 CSV，必须保留自动备份机制。
- 备份文件应写入 `annotation/backups`，不要再写到代码目录。
- 不要清空、重建或手工覆盖 `annotations.csv`。
- 如果需要恢复旧标注数据，优先从 `annotation/backups` 选择合适版本，恢复前应再备份当前 `annotations.csv`。
- 旧 HTML 工具不是当前主工具，不要随意修改。
- 视频切分脚本不得删除、移动或修改原始视频文件。
- 视频切分输出目录不得按源视频创建子文件夹，所有输出 part 直接放到一个 `output_dir`。
- 视频切分时如果 part 超过 2GB，应在 mapping 中标记 warning，而不是中断整个任务。

## 6. 当前用户偏好与开发要求

- 用户希望直接实现功能，不只是给方案。
- 用户偏好本地桌面工具，不希望第一版做 web app。
- 技术栈应保持 Python + PySide6 + OpenCV。
- 用户通过实际 GUI 测试持续反馈交互需求，交互效率很重要。
- UI 应优先保证人脸观察清楚，视频显示区域要大。
- 快捷键应高效，尤其是播放/暂停、逐帧、按秒跳、设置 start/peak/end、删除选中记录。
- 文件格式稳定非常重要，尤其是 `annotations.csv`。
- 修改代码时应最小化对已有 CSV 和已有功能的破坏。
- 保存 segment 后必须能继续自动开下一段新 segment。
- 如果 PySide6 不可用，必须明确说明 GUI 没有实际启动验证。

## 7. 已验证内容

最近已验证命令：

```powershell
cd E:\Single_frame_smile\annotation\dataset-annotation
python -m py_compile annotation_store.py smile_episode_annotation_tool.py split_videos_overlap.py test_annotation_store.py test_split_videos_overlap.py
python test_annotation_store.py
python test_split_videos_overlap.py
python split_videos_overlap.py --help
```

当前测试结果：

- `test_annotation_store.py`：`16 tests OK`
- `test_split_videos_overlap.py`：`7 tests OK`
- `split_videos_overlap.py --help` 正常显示参数。
- 当前环境检测结果：`PySide6 unavailable`，因此 GUI 仍未在 Codex 环境实际启动点测。
- 本机 PATH 中可找到 `ffmpeg` 和 `ffprobe`。
- `E:\Matsuda_data\rawdata` 已成功处理，输出 130 个视频，mapping 无 ERROR。

## 8. 尚未完成或未充分验证

- GUI 没有自动化测试。
- Codex 当前环境没有 PySide6，GUI 不能在当前环境实际启动验证。
- 没有打包成 exe。
- 没有真正的视频队列管理，`Next Video` 当前等价于重新打开视频选择器。
- 没有多 annotator 支持。
- 没有导出 preview clip。
- 没有 raw video 和 processed video side-by-side 显示。
- 没有 undo delete 或 undo save。
- GUI 中仍没有明显的 `Mode: New Segment / Editing E000xxx` 状态标签。
- 当前 `annotations.csv` 是空表，是否需要从某个 backup 恢复旧数据尚未由用户确认。
- 视频切分脚本已完成 rawdata 批处理，但如果后续更换输出目录或参数，应重新检查 mapping。

## 9. 2026-07-07 数据工作流与 sequence 数据集状态

本节记录 2026-07-07 前后完成的新数据预处理、序列整合和 ambiguous 重标注工作。它是当前训练数据整理状态，不等同于 `Analysis\documents\analysis_handoff.md` 中旧 74 条分析流水线的历史结论。

### 9.1 新数据 workflow 代码状态

已参数化并接入 workflow 的关键脚本：

- `E:\toolkit\greenbackground\rvm_extract.py`
  - 支持命令行参数：`--input`, `--output-video`, `--output-frames`, `--model`, `--device`, `--downsample-ratio`, `--fourcc`, `--frame-digits`, `--start-index`, `--overwrite`。
  - 空 `--output-frames` 表示只输出视频，不输出逐帧图片。
- `E:\Single_frame_smile\annotation\new_data_workflow.py`
  - 按 dat 文件切 clip、调用 RVM 生成绿幕视频、调用 FaceTracking 提取人脸帧。
  - 默认 FaceTracking 脚本：
    ```text
    E:\SmileAnnotation\FaceTracking-Smile_Detection\FaceTracking\CUI-pyplot\face_detection.py
    ```
  - 默认 RVM 脚本：
    ```text
    E:\toolkit\greenbackground\rvm_extract.py
    ```
  - 支持 `--allow_incomplete_facetracking`：允许 FaceTracking 缺帧并记录为 `facetracking_incomplete`，但 raw/RVM 帧数不一致仍视为错误。
- `E:\SmileAnnotation\FaceTracking-Smile_Detection\FaceTracking\CUI-pyplot\face_detection.py`
  - 修复过 threshold frame 写出遗漏问题。
  - `cv2.imwrite` 返回值已检查，写图失败会抛出 `FaceDetectionError`。

### 9.2 已批处理的新数据

输入目录：

```text
E:\Dataset\rawdata
E:\Dataset\annotation
```

已按 annotation 文件为准批处理，窗口长度为 20 秒。明确跳过未完成文件：

```text
E:\Dataset\annotation\202605162211(5).dat
```

已处理的输出根目录包括：

```text
E:\Dataset\1119
E:\Dataset\202605162211(1)
E:\Dataset\202605162211(3)
E:\Dataset\202605162211(4)
```

当时 workflow 汇总：

- `E:\Dataset\1119`：27 clips，`ok=26`，`facetracking_incomplete=1`，不完整 clip 为 `bitter/011`。
- `E:\Dataset\202605162211(1)`：7 clips，`ok=6`，`facetracking_incomplete=1`，不完整 clip 为 `true/006`。
- `E:\Dataset\202605162211(3)`：6 clips，`ok=5`，`facetracking_incomplete=1`，不完整 clip 为 `polite/005`。
- `E:\Dataset\202605162211(4)`：10 clips，`ok=7`，`facetracking_incomplete=3`，不完整 clips 为 `bitter/005`, `bitter/006`, `bitter/007`。

这些不完整 FaceTracking 输出被用户接受，因为后续还会人工校验和处理。

### 9.3 当前整合后的训练序列目录

当前统一图片序列目录：

```text
E:\Dataset\sequence
```

当前统一视频切片目录：

```text
E:\Dataset\sequence_clips
```

整理规则：

- 从 FaceTracking 输出中只取正面序列，即 `DetectedFaces\...\0\0`。
- 序列复制到 `E:\Dataset\sequence\<label>\<id>`，帧统一重命名为 `000000.png` 等。
- 如果某一帧因朝向或检测问题缺失，用前一帧补齐，并在 `frame_manifest.csv` 中记录。
- 对应 raw clip 当前扁平存放在 `E:\Dataset\sequence_clips\<label>\<label>_<id>.mp4`，例如 `E:\Dataset\sequence_clips\bitter\bitter_0.mp4`。
- 旧结构曾是 `E:\Dataset\sequence_clips\<label>\<id>\clip_raw.mp4`；已在 2026-07-07 flatten。旧编号子目录可能仍保留 `clip_manifest.csv` 元数据，但不再存放视频文件。
- 每个序列和全局都有 manifest，后续训练或抽查应优先读 manifest，不要只靠目录名猜来源。

主要 manifest：

```text
E:\Dataset\sequence\sequence_manifest.csv
E:\Dataset\sequence_clips\sequence_clip_manifest.csv
E:\Dataset\sequence\ambiguous_relabel_manifest.csv
E:\Dataset\sequence_clips\ambiguous_relabel_manifest.csv
E:\Dataset\sequence_clips\flatten_sequence_clips_manifest.csv
```

### 9.4 2-18 数据并入情况

已把 `E:\Matsuda_data\2-18meeting` 的旧数据并入 `E:\Dataset\sequence` 和 `E:\Dataset\sequence_clips`。

原始 2-18 类别和数量：

- `polite`: 41
- `truesmile`: 6
- `ambiguous`: 27

并入时的标签映射：

- `polite -> polite`
- `truesmile -> true`
- `ambiguous -> ambiguous`，随后在 2026-07-07 被用户重新标注并拆入 `bitter/polite/true`。

2-18 导入记录：

```text
E:\Dataset\sequence\2-18_import_manifest.csv
E:\Dataset\sequence_clips\2-18_import_manifest.csv
```

注意：这两个 import manifest 中 `source_label` 保留原始来源标签；例如被拆分后的 ambiguous 序列仍保留 `source_label=ambiguous`，但 `new_label` 和 `new_sequence_id` 已更新为当前训练集中的目标类别和编号。

### 9.5 ambiguous 重标注和最终计数

用户重新标注了整理后 `E:\Dataset\sequence\ambiguous\0..26` 的 27 条序列：

```text
0-4   : bbbpp
5-9   : pbbpt
10-14 : pbbbb
15-19 : pbttp
20-24 : ppppt
25-26 : pp
```

其中 `b=bitter`, `p=polite`, `t=true`。

已执行的结果：

- 原 `ambiguous/0..26` 图片序列已移动进对应类别。
- 原 `ambiguous/0..26` 视频切片已同步移动进对应类别。
- `E:\Dataset\sequence\ambiguous` 和 `E:\Dataset\sequence_clips\ambiguous` 当前没有子序列目录。
- 全局和类别级 manifest 已同步更新。
- 重标注映射保存于：
  ```text
  E:\Dataset\sequence\ambiguous_relabel_manifest.csv
  E:\Dataset\sequence_clips\ambiguous_relabel_manifest.csv
  ```

重标注分布：

- `bitter`: 10
- `polite`: 13
- `true`: 4

最终全局计数：

- 图片序列 manifest：`bitter=24`, `polite=88`, `true=12`
- 视频切片 manifest：`bitter=24`, `polite=88`, `true=12`
- 类别级 manifest 已核对：
  - `E:\Dataset\sequence\bitter\sequence_manifest.csv`: 24
  - `E:\Dataset\sequence\polite\sequence_manifest.csv`: 88
  - `E:\Dataset\sequence\true\sequence_manifest.csv`: 12
  - `E:\Dataset\sequence\ambiguous\sequence_manifest.csv`: 0
  - `E:\Dataset\sequence_clips\bitter\sequence_clip_manifest.csv`: 24
  - `E:\Dataset\sequence_clips\polite\sequence_clip_manifest.csv`: 88
  - `E:\Dataset\sequence_clips\true\sequence_clip_manifest.csv`: 12
  - `E:\Dataset\sequence_clips\ambiguous\sequence_clip_manifest.csv`: 0

2026-07-07 已对 `E:\Dataset\sequence_clips` 视频文件做扁平化整理：

- `bitter`: 24 个根目录 mp4，命名为 `bitter_<id>.mp4`
- `polite`: 88 个根目录 mp4，命名为 `polite_<id>.mp4`
- `true`: 12 个根目录 mp4，命名为 `true_<id>.mp4`
- 嵌套的 `clip_raw.mp4` 当前数量为 0。
- 全局和类别级 `sequence_clip_manifest.csv` 的 `destination_dir` / `clip_raw_path` 已更新为扁平路径。
- 旧编号目录中的 `clip_manifest.csv` 若存在，也已更新为指向扁平 mp4；这些目录只是元数据残留，不再包含视频。
- flatten 操作记录：
  ```text
  E:\Dataset\sequence_clips\flatten_sequence_clips_manifest.csv
  ```
- flatten 前 active manifest 备份：
  ```text
  E:\Dataset\sequence_clips\_manifest_backups\flatten_sequence_clips_20260707_185348
  ```

本次 ambiguous 重标注前的 manifest 备份在：

```text
E:\Dataset\sequence\_manifest_backups\ambiguous_relabel_20260707_155315
E:\Dataset\sequence_clips\_manifest_backups\ambiguous_relabel_20260707_155315
```

### 9.6 当前风险和注意事项

- `E:\Dataset\sequence` 里部分旧 workflow 图片序列目录曾被用户手动删除过；因此不要仅靠目录数量判断历史完整性，应以 manifest 和实际目录共同核对。
- `E:\Dataset\sequence_clips` 的视频切片目录比图片序列更完整，训练图片模型时应先检查对应图片目录存在且帧数符合预期。
- 2-18 的旧分析输出仍然基于 `polite/truesmile/ambiguous` 三类历史设定。现在的 `E:\Dataset\sequence` 是新的训练数据整理结果，不应直接拿旧分析结论替代新标签下的统计。
- 如果后续要训练 smile ranking 或分类模型，优先使用 `E:\Dataset\sequence\sequence_manifest.csv` 和 `E:\Dataset\sequence_clips\sequence_clip_manifest.csv` 作为数据索引。

### 9.7 stillimages、true-smile ranking 数据集和 10 级 scale

本节记录 2026-07-07 完成的 still image 制作、true-smile ranking pair 构建、人工确认、模型训练和 10 级 smiling scale 产物。

#### 9.7.1 sequence stillimages

已根据之前 still image 的制作方式，把当前实际存在的 `E:\Dataset\sequence` 序列制作为 20 张采样帧拼接图：

```text
E:\Dataset\stillimages
```

输出按类别直接分为：

```text
E:\Dataset\stillimages\true
E:\Dataset\stillimages\polite
E:\Dataset\stillimages\bitter
```

当前实际输出数量：

- `true`: 12
- `polite`: 81
- `bitter`: 23

说明：

- 用户曾因面部方向不符合需求手动删除过部分 `E:\Dataset\sequence` 子目录，因此实际目录数少于 manifest 历史计数。
- still image 每条 sequence 统一采样 20 张图，组成 5x4 grid。
- still image manifest：
  ```text
  E:\Dataset\stillimages\stillimages_manifest.csv
  ```
- 生成脚本：
  ```text
  E:\Single_frame_smile\annotation\build_sequence_stillimages.py
  ```

#### 9.7.2 true-smile ranking 数据集

当前 true-smile-only ranking 数据集根目录：

```text
E:\Dataset\smile_ranking_true
```

生成逻辑：

- 数据源只使用 `E:\Dataset\sequence\true`。
- 12 条 true sequence，每条按 onset 到 peak 的时间顺序采样 10 个候选 level。
- 得到 `120` 张 ranking candidate image。
- 每条 sequence 内部的 temporal prior 只作为初始弱监督：后面的帧大概率笑容更强，但不视为绝对跨 sequence 真值。
- 因为当前 true-smile sequence 都来自同一个人，跨 sequence 的 smile intensity ranking 被认为有意义，后续训练优先使用人工确认过的 cross-sequence pair。

关键文件：

```text
E:\Dataset\smile_ranking_true\images
E:\Dataset\smile_ranking_true\ranking_items.csv
E:\Dataset\smile_ranking_true\manual_review_cross_sequence_sample200.dat
E:\Dataset\smile_ranking_true\manual_review_cross_sequence_sample200_meta.csv
E:\Dataset\smile_ranking_true\train_manual_cross_sequence_198.dat
```

人工确认情况：

- `manual_review_cross_sequence_sample200.dat` 是供人工审核的 cross-sequence sample-like pair 文件，共 200 pair。
- 用户已完成该文件的人工检查。
- 人工检查后原文件中有 `2` 个 `0`/ambiguous pair。
- 训练时过滤 `0` 标签，得到 `train_manual_cross_sequence_198.dat`，共 `198` 个可训练 pair。
- label 约定沿用 SmileComp/SiameseNet：
  - `1`: image1 比 image2 笑容更强。
  - `-1`: image1 比 image2 笑容更弱。
  - `0`: ambiguous，不进入当前训练。

原始 SmileComp HTML 人工审核工具路径：

```text
E:\SmileAnnotation\FaceTracking-Smile_Detection\SmileComp_SiameseNet\Pytorch-Shimonishi\tools\annotation_tool
```

该工具已复制到 ranking 数据集根目录，便于直接打开：

```text
E:\Dataset\smile_ranking_true\annotation.html
E:\Dataset\smile_ranking_true\annotation.js
E:\Dataset\smile_ranking_true\annotation.css
```

注意：浏览器工具保存时会下载新文件，不会直接覆盖输入 `.dat` 文件。

相关生成脚本：

```text
E:\Single_frame_smile\annotation\build_true_smile_ranking_dataset.py
E:\Single_frame_smile\annotation\build_true_smile_cross_sequence_review.py
```

#### 9.7.3 已训练的 true-smile ranking 模型

使用原 PyTorch SmileComp SiameseNet 项目训练：

```text
E:\SmileAnnotation\FaceTracking-Smile_Detection\SmileComp_SiameseNet\Pytorch-Shimonishi
```

训练使用的预训练卷积权重：

```text
E:\SmileAnnotation\FaceTracking-Smile_Detection\SmileComp_SiameseNet\Pytorch-Shimonishi\models\vggface_conv.pth
```

训练数据：

```text
E:\Dataset\smile_ranking_true\train_manual_cross_sequence_198.dat
E:\Dataset\smile_ranking_true\images
```

训练设置：

```text
epochs = 100
batch_size = 16
learning_rate = 0.0001
device = cuda:0
train/val split = 178/20
```

训练完成的模型权重：

```text
E:\Dataset\smile_ranking_true\model\smile_rank_true_cross_sequence_100ep.pth
```

训练曲线：

```text
E:\Dataset\smile_ranking_true\model\smile_rank_true_cross_sequence_100ep_history.png
```

训练脚本会保存 validation loss 最低的模型版本。由于 validation set 只有 20 pair，validation accuracy 波动很大，不应把它当作稳定泛化指标。

#### 9.7.4 10 级 true-smile scale

已使用训练后的模型对 120 张 true-smile candidate image 做全组合 pairwise prediction：

- 总 pair 数：`7140`
- 排序分数：每张图强于其他图的 pairwise 概率/投票汇总。
- scale 方向：`0 = weakest smile`, `9 = strongest smile`
- 10 个 reference anchor：从弱到强 ranked list 中等间隔选取，保留最弱和最强端点。
- equal-population level：`scale10_ranked_all.csv` 中每个 level 12 张图。

scale 输出目录：

```text
E:\Dataset\smile_ranking_true\scale10
```

关键输出：

```text
E:\Dataset\smile_ranking_true\scale10\scale10_items.csv
E:\Dataset\smile_ranking_true\scale10\scale10_ranked_all.csv
E:\Dataset\smile_ranking_true\scale10\scale10_montage.png
E:\Dataset\smile_ranking_true\scale10\scale10_montage_grid.png
E:\Dataset\smile_ranking_true\scale10\pairwise_predictions.csv
E:\Dataset\smile_ranking_true\scale10\strong_probability_matrix.csv
E:\Dataset\smile_ranking_true\scale10\stream_consistency_matrix.csv
E:\Dataset\smile_ranking_true\scale10\manual_pair_eval.csv
E:\Dataset\smile_ranking_true\scale10\scale10_summary.json
```

scale 生成脚本：

```text
E:\Single_frame_smile\annotation\build_true_smile_scale_from_model.py
```

生成摘要：

```text
num_images = 120
num_pairwise_predictions = 7140
mean_stream_consistency = 0.9435046315193176
min_stream_consistency = 0.6290798783302307
max_stream_consistency = 0.9999069571495056
manual_pair_rows_used = 198
manual_pair_accuracy_on_training_pairs = 173/198 = 0.8737373737373737
```

注意：`manual_pair_accuracy_on_training_pairs` 是模型对参与训练的人工 pair 的一致率，不是独立测试集准确率。

当前 10 个 scale anchor：

| Level | source image | sequence | provisional level | rank position | vote score |
|---:|---|---:|---:|---:|---:|
| 0 | `true_seq001_level00.png` | 1 | 0 | 0 | 0.02639130 |
| 1 | `true_seq011_level00.png` | 11 | 0 | 13 | 13.13941548 |
| 2 | `true_seq003_level01.png` | 3 | 1 | 26 | 26.23330581 |
| 3 | `true_seq008_level09.png` | 8 | 9 | 40 | 40.35406316 |
| 4 | `true_seq009_level03.png` | 9 | 3 | 53 | 53.45111344 |
| 5 | `true_seq002_level08.png` | 2 | 8 | 66 | 66.57175339 |
| 6 | `true_seq004_level09.png` | 4 | 9 | 79 | 79.67673088 |
| 7 | `true_seq001_level08.png` | 1 | 8 | 93 | 93.75230169 |
| 8 | `true_seq009_level09.png` | 9 | 9 | 106 | 106.81656743 |
| 9 | `true_seq006_level07.png` | 6 | 7 | 119 | 119.92980257 |

解释和风险：

- scale 是 true-smile-only、同一个人数据上的当前学习结果，不是跨人群通用的绝对笑容强度标尺。
- 由于训练 pair 不超过 198 个且没有独立测试集，当前 scale 应视为第一版 working scale。
- 跨 sequence ranking 会打散原始 temporal prior。例如某些 provisional level 9 会被模型排到较弱 level，这可能反映不同 sequence 间 peak 强度差异，也可能反映图像质量、姿态或少量标注冲突。
- 后续不要把 `provisional_level` 当作最终 scale level；最终以 `scale10_ranked_all.csv` 和 `scale10_items.csv` 为准。

### 9.8 典型 case 的 s-d plot 和 t-ranking 交互图

本节记录 2026-07-07 至 2026-07-08 针对用户选出的典型 case 生成的交互式分析图。

选中 case：

```text
bitter: 8, 11, 12, 13, 15, 17, 23
true: 0, 2, 5, 10
polite: 8, 14, 17, 18, 21
```

说明：用户最初写的是 `true/19`，后续更正为 `true/10`。当前所有 16 条选中 sequence 均在 `E:\Dataset\sequence` 中存在，并且对应 still image 均存在于 `E:\Dataset\stillimages\<label>\<id>.png`。

#### 9.8.1 selected s-d plot

s-d plot 使用已经训练好的 SmileComp model extractor 重新计算 feature vector。公共 baseline 使用：

```text
E:\Dataset\sequence\true\5
```

具体定义：

- baseline axis = `feature(true/5 last frame) - feature(true/5 first frame)`。
- 对每条选中 sequence，仍以自身第一帧作为起点：`delta_t = feature_t - feature_sequence_first`。
- `s = dot(delta_t, axis_unit)`。
- `d = norm(delta_t - s * axis_unit)`。
- baseline axis norm：`5.59029483795166`。

脚本：

```text
E:\Single_frame_smile\annotation\plot_selected_sd_cases.py
E:\Single_frame_smile\annotation\build_selected_sd_interactive_html.py
```

输出目录：

```text
E:\Dataset\sd_plot_selected_cases
```

关键输出：

```text
E:\Dataset\sd_plot_selected_cases\selected_cases_sd_plot.png
E:\Dataset\sd_plot_selected_cases\selected_cases_sd_interactive.html
E:\Dataset\sd_plot_selected_cases\selected_cases_sd_coordinates.csv
E:\Dataset\sd_plot_selected_cases\selected_cases_thumbnail_frames.csv
E:\Dataset\sd_plot_selected_cases\selected_cases_manifest.csv
E:\Dataset\sd_plot_selected_cases\selected_cases_sd_summary.json
```

已验证摘要：

- valid sequence count：`16`
- total frame count：`1059`
- static PNG thumbnail stride：`8` frames
- interactive HTML 默认只显示曲线；hover 曲线或图例时高亮该 sequence，并显示该 sequence 每 8 帧一张的曲线帧图。
- interactive HTML 内嵌缩略图，不依赖浏览器读取本地绝对图片路径。
- 使用本机 Chrome 验证过：曲线、图例和 hover 缩略图均正常，无 JavaScript page error。

#### 9.8.2 selected t-ranking plot

t-ranking plot 使用已经生成的 frame-level smile ranking 分数，不重复跑模型：

```text
E:\Dataset\smileranking_plot\frame_smile_ranking_scores.csv
```

该 CSV 的分数来自 SmileComp model 和 10-level true-smile scale：

```text
score_0_9 = clamp(sum_k P(frame stronger than true-smile scale anchor k) - 0.5, 0, 9)
```

脚本：

```text
E:\Single_frame_smile\annotation\build_selected_t_ranking_interactive_html.py
```

输出目录：

```text
E:\Dataset\t_ranking_selected_cases
```

关键输出：

```text
E:\Dataset\t_ranking_selected_cases\selected_cases_t_ranking_interactive.html
E:\Dataset\t_ranking_selected_cases\selected_cases_t_ranking_adjusted_scores.csv
E:\Dataset\t_ranking_selected_cases\selected_cases_t_ranking_manifest.csv
E:\Dataset\t_ranking_selected_cases\selected_cases_t_ranking_summary.json
```

HTML 中包含 4 张图：

- `T_T plot`: 选中的 true sequence。
- `T_B plot`: 选中的 bitter sequence。
- `T_P plot`: 选中的 polite sequence。
- `ALL plot`: 全部选中 sequence；同一 class 使用同一种颜色，true=blue, bitter=red, polite=green。

当前布局和交互：

- `main class="plots"` 当前为单列纵向布局，一行一张图，四张图从上到下排列。
- hover 曲线或右侧图例时，会高亮该 sequence；因为同一 sequence 同时在 class plot 和 ALL plot 中出现，所以通常会同时高亮 2 条曲线。
- 右侧显示该 sequence 的 20-frame still image。
- 点击 plot 空白区域会恢复显示所有曲线，并清空当前高亮、曲线帧图和右侧 still image。
- 右侧有 `Frame thumbnails` checkbox，默认勾选。
  - 勾选时：高亮曲线从第 0 帧开始每 6 帧显示一张帧图；30fps 下约等于每 0.2 秒一张。
  - 取消勾选时：保持曲线高亮和右侧 still image，但不显示曲线帧图。

开头压缩规则：

- FPS：`30.0`
- `max_initial_static_frames = 6`
- `growth_delta = 0.05`
- 第一次增长定义为：某帧 `score_0_9 > frame0_score_0_9 + 0.05`。
- 如果第一次增长发生在第 6 帧之后，则把 `0..growth_frame` 线性压缩到 `0..6`，后续帧整体左移；原始 frame index 和 adjusted frame index 都保存在 CSV 中。

当前触发开头压缩的 sequence：

| label | sequence | growth frame | compression shift |
|---|---:|---:|---:|
| bitter | 11 | 15 | 9 frames |
| polite | 18 | 20 | 14 frames |

已验证摘要：

- sequence count：`16`
- frame count：`1059`
- curve thumbnail count：`180`
- compressed sequence count：`2`
- HTML 内嵌图片数：`196`，包括 16 张 still image 和 180 张曲线帧图。
- 使用本机 Chrome 验证过：
  - 4 张图为纵向 1x4 布局。
  - 勾选 `Frame thumbnails` 后，hover `polite/18` 显示 26 张曲线帧图（`T_P` 和 `ALL` 各 13 张）。
  - 取消勾选后曲线帧图为 0，曲线高亮和右侧 still image 保持。
  - 点击空白区域后 active/muted 状态和曲线帧图全部清空。
  - JavaScript page error：`0`。

注意和风险：

- t-ranking HTML 依赖现有 `E:\Dataset\smileranking_plot\frame_smile_ranking_scores.csv`。如果后续重新训练 scale 或重算 frame-level ranking，需要重新运行 `build_selected_t_ranking_interactive_html.py`。
- 开头压缩阈值 `growth_delta=0.05` 是当前工作参数，不是经过统计优化的固定标准；后续可根据人工观察调整。
- s-d plot 和 t-ranking plot 都是针对当前同一个人、当前 10 级 true-smile working scale 的分析图，不应解释为跨人群通用强度结论。

## 10. 当前最推荐的下一步任务

- [ ] 确认是否需要从 `annotation/backups` 恢复旧标注数据到 `annotations.csv`。
- [ ] 在有 PySide6 的环境启动 GUI：
  ```powershell
  cd E:\Single_frame_smile\annotation\dataset-annotation
  python smile_episode_annotation_tool.py
  ```
- [ ] 手动测试 temporal segment 标注流程：
  - 新增 `neutral`，不设置 peak，保存成功。
  - 新增 `discard`，不设置 peak，保存成功。
  - 新增 `truesmile`，不设置 peak，保存应失败。
  - 给 smile 类设置 peak 后保存成功。
  - 点击已有行，修改 label 后保存，确认 update 原 `episode_id`。
  - 选中行按 `Delete`，确认弹窗后删除。
  - 保存后下一段 start 自动等于上一段 end。
- [ ] 在 GUI 上增加明显模式显示：
  ```text
  Mode: New Segment
  Mode: Editing E000123
  ```
- [ ] 考虑给备份目录增加清理策略，例如保留最近 N 个或按日期归档，但不要默认删除。
- [ ] 如果后续需要用 `split_videos_overlap.py` 处理新数据目录，先 dry-run，再正式运行。
- [ ] 在训练前对 `E:\Dataset\sequence` 做一次抽样人工校验，重点检查补帧序列、`facetracking_incomplete` 来源序列和 2-18 ambiguous 重标注后的边界案例。
- [ ] 人工查看 `E:\Dataset\smile_ranking_true\scale10\scale10_montage_grid.png`，确认 0-9 anchor 是否符合肉眼强度递增。
- [ ] 优先抽查 `scale10_ranked_all.csv` 中 provisional level 与最终 rank 冲突较大的样本，判断是合理跨 sequence 差异、姿态/质量问题，还是 pair 标注需要修正。
- [ ] 如果当前 10 级 scale 不够稳定，下一轮优先补标模型低 consistency 或人工/模型不一致的 cross-sequence pair，而不是盲目扩大同 sequence pair。
- [ ] 人工查看 `E:\Dataset\sd_plot_selected_cases\selected_cases_sd_interactive.html` 和 `E:\Dataset\t_ranking_selected_cases\selected_cases_t_ranking_interactive.html`，确认典型 case 的 curve pattern、ranking trajectory 和 still image 对照是否符合预期。
- [ ] 如 t-ranking 开头压缩效果不符合肉眼观察，优先调整 `build_selected_t_ranking_interactive_html.py` 中的 `--growth_delta` 或 `--max_initial_static_frames` 后重新生成，不要手工改 HTML。

## 11. 给下一个 Codex 的注意事项

- 先读：
  ```text
  E:\Single_frame_smile\annotation\prompts\agents.md
  E:\Single_frame_smile\annotation\prompts\codex_project_handoff.md
  ```
- 不要再把 `codex_handoff_multi_occlusion_segments.backup_20260511.md` 当作当前状态，它只是历史备份。
- 不要重新设计整个工具。当前架构是 PySide6 GUI + `AnnotationStore` CSV 存储 + 独立 ffmpeg split 脚本。
- 不要把项目改成 web app。
- 不要清空、重建或随意覆盖 `annotations.csv`。
- 不要删除 `annotation/backups` 中的旧备份。
- 不要继续开发多段 occlusion GUI。
- 不要删除 occlusion 字段。
- 不要改 frame index 规则，继续 0-based。
- 不要改 `episode_id` 规则，继续全局递增。
- 修改保存逻辑前先阅读：
  - `annotation_store.py`
  - `smile_episode_annotation_tool.py`
  - `test_annotation_store.py`
- 修改视频切分逻辑前先阅读：
  - `split_videos_overlap.py`
  - `test_split_videos_overlap.py`
  - `video_split_overlap_coding_task.md`
- 修改或重排训练数据目录前，先读：
  - `E:\Dataset\sequence\sequence_manifest.csv`
  - `E:\Dataset\sequence_clips\sequence_clip_manifest.csv`
  - `E:\Dataset\sequence_clips\flatten_sequence_clips_manifest.csv`
  - `E:\Dataset\sequence\ambiguous_relabel_manifest.csv`
  - `E:\Dataset\sequence_clips\ambiguous_relabel_manifest.csv`
- 继续 true-smile ranking/scale 工作前，先读：
  - `E:\Dataset\smile_ranking_true\ranking_items.csv`
  - `E:\Dataset\smile_ranking_true\train_manual_cross_sequence_198.dat`
  - `E:\Dataset\smile_ranking_true\scale10\scale10_summary.json`
  - `E:\Dataset\smile_ranking_true\scale10\scale10_items.csv`
  - `E:\Single_frame_smile\annotation\build_true_smile_scale_from_model.py`
- 继续典型 case s-d / t-ranking 可视化工作前，先读：
  - `E:\Dataset\sd_plot_selected_cases\selected_cases_sd_summary.json`
  - `E:\Dataset\t_ranking_selected_cases\selected_cases_t_ranking_summary.json`
  - `E:\Single_frame_smile\annotation\plot_selected_sd_cases.py`
  - `E:\Single_frame_smile\annotation\build_selected_sd_interactive_html.py`
  - `E:\Single_frame_smile\annotation\build_selected_t_ranking_interactive_html.py`
- 修改代码后至少运行：
  ```powershell
  cd E:\Single_frame_smile\annotation\dataset-annotation
  python -m py_compile annotation_store.py smile_episode_annotation_tool.py split_videos_overlap.py test_annotation_store.py test_split_videos_overlap.py
  python test_annotation_store.py
  python test_split_videos_overlap.py
  ```
- 如果 PySide6 未安装，不能完整验证 GUI，只能做语法和存储逻辑验证，必须明确告知用户。
- 注意仓库可能有 unrelated changes，不要回滚用户改动。

## 12. 当前问题与开放问题

- 是否需要从某个旧 backup 恢复 `annotations.csv`？
- 当前 `annotations.csv` 空表是否是用户有意清空，还是误删后的待恢复状态？
- 是否需要给 GUI 增加 `Mode: New Segment / Editing E000xxx` 明显提示？
- 保存后 `next_start = previous_end` 是否需要可选开关？
- `clip_path` 是否未来需要改成相对路径以便跨机器训练？
- 是否需要增加 annotator 字段？
- `person_id` 是否需要固定命名规则，例如 `P01`, `P02`？
- 是否需要编辑保存时保留 audit log？
- 是否需要对 `annotation/backups` 增加清理或归档策略？
- 视频切分脚本是否需要增加参数控制是否保留 data streams？当前默认 `-dn` 是为 GoPro MP4 兼容性做出的实用选择。
- 已切分视频输出目录 `E:\Matsuda_data\split_videos` 是否作为后续标注视频来源？如果是，后续 `clip_path` 应指向该目录中的 part 文件。
- `E:\Dataset\sequence` 中被用户手动删除过的旧 workflow 图片序列是否需要恢复，还是保持当前人工筛选后的状态？
- 新标签体系下是否还需要保留 `ambiguous` 作为训练类别？当前训练序列中 ambiguous 已清空。
- 当前 10 级 true-smile scale 是否需要第二轮人工确认，尤其是 provisional level 与最终 rank 冲突较大的 anchor 和样本？
- 是否需要建立独立 holdout pair 或第二批人工 pair，给 ranking scale 一个真正独立的评估指标？

## 13. 可直接复制给新 Codex 的简短任务说明

```text
这是 `Single_frame_smile` 项目的当前主交接说明。请先阅读：

E:\Single_frame_smile\annotation\prompts\agents.md
E:\Single_frame_smile\annotation\prompts\codex_project_handoff.md

不要使用旧的 `codex_handoff_multi_occlusion_segments.backup_20260511.md` 作为当前状态，它只是历史备份。

项目核心目录：
E:\Single_frame_smile\annotation\dataset-annotation

当前主要文件：
- smile_episode_annotation_tool.py：PySide6 + OpenCV temporal segment/state 标注 GUI。
- annotation_store.py：CSV 存储、label migration、occlusion->discard、备份、append/update/delete。
- test_annotation_store.py：存储层测试，当前 16 tests OK。
- split_videos_overlap.py：ffmpeg 无重编码视频重叠切分脚本。
- test_split_videos_overlap.py：切分脚本测试，当前 7 tests OK。
- annotations.csv：当前主标注 CSV，目前只有 header，旧数据在 annotation/backups。

当前标注标签：
truesmile, polite_smile, bitter_smile, smiling_but_ambiguous, neutral, discard

四个 smile 类必须有 peak；neutral/discard 不需要 peak 且保存时清空 peak。
所有遮挡、看不清、追踪失败、不可用区间统一标为 discard。
不要继续开发多段 occlusion GUI，但保留 occlusion 字段兼容旧数据。

当前训练序列数据集：
E:\Dataset\sequence
E:\Dataset\sequence_clips

当前视频切片命名约定：
E:\Dataset\sequence_clips\<label>\<label>_<id>.mp4

当前训练序列全局计数：
bitter=24, polite=88, true=12, ambiguous=0

ambiguous 已在 2026-07-07 按用户新标注拆入 bitter/polite/true。重标注映射见：
E:\Dataset\sequence\ambiguous_relabel_manifest.csv
E:\Dataset\sequence_clips\ambiguous_relabel_manifest.csv

当前 true-smile-only smile ranking 数据集和模型：
E:\Dataset\smile_ranking_true
E:\Dataset\smile_ranking_true\train_manual_cross_sequence_198.dat
E:\Dataset\smile_ranking_true\model\smile_rank_true_cross_sequence_100ep.pth

当前 10 级 true-smile scale：
E:\Dataset\smile_ranking_true\scale10
E:\Dataset\smile_ranking_true\scale10\scale10_items.csv
E:\Dataset\smile_ranking_true\scale10\scale10_ranked_all.csv
E:\Dataset\smile_ranking_true\scale10\scale10_montage_grid.png

scale 方向是 0=weakest smile，9=strongest smile。它是基于 198 个已确认 cross-sequence pair 训练得到的第一版 working scale；`manual_pair_accuracy=173/198` 是训练 pair 一致率，不是独立测试集准确率。

当前典型 case 可视化输出：
E:\Dataset\sd_plot_selected_cases\selected_cases_sd_interactive.html
E:\Dataset\t_ranking_selected_cases\selected_cases_t_ranking_interactive.html

典型 case:
bitter: 8, 11, 12, 13, 15, 17, 23
true: 0, 2, 5, 10
polite: 8, 14, 17, 18, 21

s-d plot 使用 true/5 第一帧到最后一帧的 SmileComp feature 向量作为公共 baseline axis。t-ranking plot 读取 `E:\Dataset\smileranking_plot\frame_smile_ranking_scores.csv`，基于 true-smile 10-level scale 的 `score_0_9` 绘图。t-ranking HTML 当前是 1x4 纵向布局，带 `Frame thumbnails` 勾选栏；勾选时高亮曲线每 6 帧显示一张曲线帧图，取消勾选时只保留曲线高亮和右侧 still image。

相关脚本：
E:\Single_frame_smile\annotation\plot_selected_sd_cases.py
E:\Single_frame_smile\annotation\build_selected_sd_interactive_html.py
E:\Single_frame_smile\annotation\build_selected_t_ranking_interactive_html.py

修改代码后至少运行：
cd E:\Single_frame_smile\annotation\dataset-annotation
python -m py_compile annotation_store.py smile_episode_annotation_tool.py split_videos_overlap.py test_annotation_store.py test_split_videos_overlap.py
python test_annotation_store.py
python test_split_videos_overlap.py

如果用户说“给我交接文档”，请更新 `codex_project_handoff.md`，保持当前结构，补充最新状态、验证结果、风险、下一步任务和开放问题；不要新增多个 handoff 文档。
```
