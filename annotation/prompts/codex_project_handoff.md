# 项目交接文档

更新时间：2026-05-11

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

## 9. 当前最推荐的下一步任务

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

## 10. 给下一个 Codex 的注意事项

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
- 修改代码后至少运行：
  ```powershell
  cd E:\Single_frame_smile\annotation\dataset-annotation
  python -m py_compile annotation_store.py smile_episode_annotation_tool.py split_videos_overlap.py test_annotation_store.py test_split_videos_overlap.py
  python test_annotation_store.py
  python test_split_videos_overlap.py
  ```
- 如果 PySide6 未安装，不能完整验证 GUI，只能做语法和存储逻辑验证，必须明确告知用户。
- 注意仓库可能有 unrelated changes，不要回滚用户改动。

## 11. 当前问题与开放问题

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

## 12. 可直接复制给新 Codex 的简短任务说明

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

修改代码后至少运行：
cd E:\Single_frame_smile\annotation\dataset-annotation
python -m py_compile annotation_store.py smile_episode_annotation_tool.py split_videos_overlap.py test_annotation_store.py test_split_videos_overlap.py
python test_annotation_store.py
python test_split_videos_overlap.py

如果用户说“给我交接文档”，请更新 `codex_project_handoff.md`，保持当前结构，补充最新状态、验证结果、风险、下一步任务和开放问题；不要新增多个 handoff 文档。
```
