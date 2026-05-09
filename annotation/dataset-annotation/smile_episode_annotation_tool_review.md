# Smile Episode Annotation Tool - Coding & Review

## 交付文件

- `smile_episode_annotation_tool.py`: PySide6 + OpenCV 桌面标注工具入口。
- `annotation_store.py`: CSV 列定义、episode ID 生成、帧顺序校验、重复 episode 检测、CSV 追加写入。
- `test_annotation_store.py`: 不依赖 GUI 的核心逻辑单元测试。
- `requirements.txt`: 运行 GUI 所需依赖。
- `annotations.csv`: 工具保存时自动创建，默认位于本目录。

## 运行方式

```powershell
cd e:\Single_frame_smile\annotation\dataset-annotation
pip install -r requirements.txt
python smile_episode_annotation_tool.py
```

当前实现使用 0-based frame index，与 OpenCV `CAP_PROP_POS_FRAMES` 一致。

## 已实现功能

- 加载本地视频：支持 `.mp4`, `.avi`, `.mov`, `.mkv` 和任意文件选择。
- 视频播放控制：播放/暂停、进度条 seek、上一帧/下一帧、前后 5 帧、前后 1 秒。
- 视频信息展示：文件名、绝对路径、FPS、总帧数、当前帧、当前秒数。
- episode 标记：`start_frame`, `peak_frame`, `end_frame` 三个按钮和快捷键。
- 进度条标记：start/peak/end 在 slider 上分别以绿/红/蓝竖线显示。
- 标签与属性：主标签、confidence、intensity、eye/mouth/cheek、symmetry、visible_quality、usable_for_training、note。
- CSV 保存：一行对应一个 smile episode，列顺序与任务文档一致。
- ID 规则：`E000001` 起步，已有 CSV 会继续递增，不按视频重置。
- episode 列表：显示当前视频已保存 episode，点击列表行会加载该 episode 并跳到 start frame。
- 快捷键：Space 播放/暂停，左右箭头逐帧，A/D 前后 5 帧，J/L 前后 1 秒，S/P/E 设置 start/peak/end，Ctrl+S 保存。

## Review 结论

核心 v0.1-v0.6 已完成，包括新增、回放、修改和删除 episode。

2026-05-07 复测反馈后已调整两点：

- 视频区域改为左右分栏布局，左侧保留大尺寸视频显示、进度条和播放按钮，右侧放标注表单和 episode 列表；默认窗口改为 `1680x980`，视频控件最小尺寸改为 `960x540`。可用 `F11` 进入全屏，`Esc` 退出全屏。
- 播放逻辑改为按真实经过时间同步到原始 FPS。连续播放时优先顺序读取下一帧，避免每帧 OpenCV seek；如果解码或渲染跟不上原始 FPS，会跳到目标帧追上真实时间，避免播放整体变慢。

2026-05-07 追加需求后已调整两点：

- 播放控制区新增 `Speed` 下拉框，可在 `1.0x` 和 `0.5x` 之间切换；0.5 倍速会按 `fps * 0.5` 推进播放时钟。
- 快捷键改为全窗口级别处理：`Space` 播放/暂停。文本输入框和展开中的下拉菜单不会被这些快捷键打断。

2026-05-07 追加交互修正：

- `peak_frame` 改为按标签条件要求：`genuine_like_smile`, `polite_like_smile`, `bitter_awkward_like_smile`, `ambiguous_smile` 必须设置 peak；`neutral_or_no_smile` 和 `unclear` 不使用 peak，CSV 的 `peak_frame` 和 `peak_time` 写空值。
- Start/Peak/End 三行都新增 `Go` 按钮；对应帧已标定时可快速跳转回该帧。

2026-05-07 追加操作修正：

- 点击右下方 episode 列表中的任意一行，会把该 episode 的 start/peak/end、label、confidence、可见质量、note 等字段加载回当前表单，并自动跳到该 episode 的 start frame。
- `Left Arrow` / `Right Arrow` 改为后退/前进 1 帧；1 秒跳转继续由 `J` / `L` 和界面按钮提供。

2026-05-07 追加删除和连续标注修正：

- 新增 `Delete Selected Episode` 按钮；选中 episode 表格行后可按 `episode_id` 从 `annotations.csv` 删除该行。
- 保存 episode 后，工具会自动把下一条 episode 的 `start_frame` 设置为刚保存 episode 的 `end_frame`，并清空 peak/end/note，便于连续标注。

2026-05-07 追加回放和 peak 清理修正：

- 新增 `Play Selected Episode` 按钮；选中 episode 表格行后，可从该 episode 的 `start_frame` 播放到 `end_frame` 并自动停止。
- Peak 行新增 `Clear` 按钮；误标 peak 时可只清除 `peak_frame`，保留 start/end。

2026-05-09 追加编辑修正：

- 选中右下方 episode 列表中的已有标注后，表单会进入编辑状态；修改区间、标签或属性后点击 `Save Episode` 会覆盖原 `episode_id` 对应的 CSV 行，而不是追加新行。
- 如果没有加载已有 episode，`Save Episode` 仍按新 episode 追加，并继续自动设置下一条 start frame。

保存逻辑的风险点主要在路径和重复检测。实现中已把 `clip_path` 保存为绝对路径，并在重复检测时使用解析后的规范路径，避免 Windows 短路径/长路径导致同一个视频重复标注。重复 episode 的判定规则是同一 `clip_path` 且 `start_frame/peak_frame/end_frame` 完全相同。

`usable_for_training` 会随 label、confidence、visible_quality 自动按默认规则更新：confidence >= 4、可见质量为 good/medium、且 label 不是 unclear 时默认 yes，否则 no。用户仍可在保存前手动改为 yes/no。

## 验证结果

- `python -m py_compile annotation_store.py smile_episode_annotation_tool.py test_annotation_store.py`: passed
- `python test_annotation_store.py`: 12 tests passed

当前 Python 环境已安装 OpenCV，但未安装 PySide6；因此本次未在本机启动 GUI 窗口。安装 `requirements.txt` 后即可运行桌面工具。

## 已知限制

- 不自动分类视频，只做人类 episode 标注。
- `Next Video` 当前等价于重新打开视频文件选择器，未实现目录批量队列。
- CSV 使用绝对 `clip_path`，更适合本机训练流水线；如果后续需要跨机器共享，可增加相对路径保存选项。
