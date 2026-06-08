# Coding Task: 批量无损切分视频文件夹，并在切分点前后保留重叠视频

## 任务背景

我有一个文件夹，里面包含多个较大的视频文件。现在需要自动将这些视频文件无损切分成若干个较小的视频片段，每个片段尽量控制在 2GB 以下。

这个任务的核心需求是：

因为视频中包含面部表情 episode，我不希望一个表情刚好在切分点附近被完全断开。因此，相邻切分片段之间需要保留一段重叠视频。

例如：

```text
segment_time = 300 秒
overlap_time = 10 秒
```

那么切分结果应该类似：

```text
part000: 0s   ~ 300s
part001: 290s ~ 590s
part002: 580s ~ 880s
part003: 870s ~ 1170s
```

也就是说：

```text
下一段的 start_time = 上一段的 end_time - overlap_time
```

这样即使表情 episode 出现在切分点附近，也能在前后片段中保留下来。

---

## 当前代码位置

请在以下项目位置中实现或集成该功能：

```text
E:\Single_frame_smile\annotation\dataset-annotation\annotation_store.py
```

如果你认为这个功能不适合直接写入 `annotation_store.py`，也可以新建独立脚本，例如：

```text
E:\Single_frame_smile\annotation\dataset-annotation\split_videos_overlap.py
```

但需要保证该脚本可以在当前项目中直接运行，并且不要破坏现有标注系统代码。

---

## 核心目标

实现一个 Python 脚本或功能模块，用于：

1. 自动扫描指定输入文件夹中的所有视频文件。
2. 对每个视频进行无损切分。
3. 相邻切分片段之间保留 `overlap_time` 秒重叠。
4. 保留原始视频文件，不删除、不移动、不覆盖。
5. 将所有切分后的视频直接保存到指定输出文件夹中。
6. 不要为每个源视频创建单独子文件夹。
7. 对切分文件进行规范命名。
8. 生成 `split_mapping.csv` 和 `split_mapping.json`，记录每个切分片段和源文件之间的映射关系。
9. 映射文件中必须记录理论开始时间、理论结束时间、实际时长、文件大小、是否超过 2GB 等信息。

---

## 使用方式示例

推荐命令形式：

```bash
python split_videos_overlap.py --input_dir "E:/Matsuda_data/raw_videos" --output_dir "E:/Matsuda_data/split_videos" --segment_time 300 --overlap_time 10 --max_size_gb 2
```

如果切出来仍然超过 2GB，可以把 `segment_time` 调小，例如：

```bash
python split_videos_overlap.py --input_dir "E:/Matsuda_data/raw_videos" --output_dir "E:/Matsuda_data/split_videos" --segment_time 240 --overlap_time 10 --max_size_gb 2 --overwrite
```

只预览，不实际切分：

```bash
python split_videos_overlap.py --input_dir "E:/Matsuda_data/raw_videos" --output_dir "E:/Matsuda_data/split_videos" --segment_time 300 --overlap_time 10 --dry_run
```

---

## 重要原则

必须使用 ffmpeg 无重编码切分：

```bash
ffmpeg -y -ss START_TIME -i input.mp4 -t DURATION -map 0 -c copy -avoid_negative_ts make_zero output_part000.mp4
```

要求：

- 必须使用 `-c copy`
- 不要重新编码
- 不要使用 `libx264`
- 不要按字节硬切
- 不要删除原始视频
- 不要移动原始视频
- 不要覆盖原始视频
- 路径中有空格、中文、日文时也要正常运行

---

## 为什么不用普通 ffmpeg segment 模式

普通 segment 模式类似：

```bash
ffmpeg -i input.mp4 -map 0 -c copy -f segment -segment_time 300 output_%03d.mp4
```

它可以快速切分，但是不方便实现相邻片段之间的固定重叠。

本任务需要手动计算每个片段的：

```text
start_time
duration
end_time
```

然后对每个片段单独调用一次 ffmpeg。

---

## 切分逻辑

假设：

```text
video_duration = 1800 秒
segment_time = 300 秒
overlap_time = 10 秒
```

那么：

```text
step_time = segment_time - overlap_time = 290 秒
```

切分区间为：

```text
part000: start=0,    end=300
part001: start=290,  end=590
part002: start=580,  end=880
part003: start=870,  end=1170
part004: start=1160, end=1460
part005: start=1450, end=1750
part006: start=1740, end=1800
```

最后一个片段的 `end_time` 不应超过视频总时长。

伪代码：

```python
start = 0
part_index = 0

while start < video_duration:
    end = min(start + segment_time, video_duration)
    duration = end - start

    export_part(start, duration, part_index)

    if end >= video_duration:
        break

    start += segment_time - overlap_time
    part_index += 1
```

需要检查：

```text
overlap_time < segment_time
```

否则直接报错并退出。

---

## 推荐默认参数

```text
segment_time = 300
overlap_time = 10
max_size_gb = 2
```

说明：

- `segment_time = 300` 表示每段约 5 分钟。
- `overlap_time = 10` 表示相邻片段之间重叠 10 秒。
- 因为有重叠区间，输出总大小会略微增加。
- 如果单个片段仍然超过 2GB，可以继续减小 `segment_time`。

---

## 支持的视频格式

至少支持以下格式：

```text
.mp4
.mov
.mkv
.avi
.MP4
.MOV
.MKV
.AVI
```

实现时使用小写后缀判断即可：

```python
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi"}
```

---

## 输出目录结构

不要为每个源视频创建单独子文件夹。

所有切分后的视频文件都直接保存到 `output_dir` 中。

示例：

```text
output_dir/
  meeting_2026_05_10_part000_start000000s_end000300s.mp4
  meeting_2026_05_10_part001_start000290s_end000590s.mp4
  meeting_2026_05_10_part002_start000580s_end000880s.mp4
  another_video_part000_start000000s_end000300s.mp4
  another_video_part001_start000290s_end000590s.mp4
  split_mapping.csv
  split_mapping.json
```

---

## 输出命名规则

假设源视频为：

```text
meeting_2026_05_10.mp4
```

输出文件直接保存为：

```text
meeting_2026_05_10_part000_start000000s_end000300s.mp4
meeting_2026_05_10_part001_start000290s_end000590s.mp4
meeting_2026_05_10_part002_start000580s_end000880s.mp4
```

命名格式：

```text
{source_stem}_part{part_index:03d}_start{start_sec:06d}s_end{end_sec:06d}s{suffix}
```

例如：

```text
meeting_2026_05_10_part001_start000290s_end000590s.mp4
```

这样即使所有视频片段都放在同一个文件夹中，也能通过文件名区分来源和时间范围。

---

## 映射文档要求

生成：

```text
split_mapping.csv
split_mapping.json
```

这两个文件都保存在 `output_dir` 下面。

CSV 至少包含以下字段：

```csv
source_file,source_path,part_file,part_path,part_index,start_time_sec,end_time_sec,theoretical_duration_sec,actual_duration_sec,overlap_time_sec,size_bytes,size_mb,size_gb,status,warning
```

字段说明：

| 字段 | 含义 |
|---|---|
| source_file | 源视频文件名 |
| source_path | 源视频完整路径 |
| part_file | 切分后片段文件名 |
| part_path | 切分后片段完整路径 |
| part_index | part 编号，从 0 开始 |
| start_time_sec | 该片段在源视频中的理论开始时间 |
| end_time_sec | 该片段在源视频中的理论结束时间 |
| theoretical_duration_sec | 理论切分时长 |
| actual_duration_sec | 实际文件时长，使用 ffprobe 获取 |
| overlap_time_sec | 与前后片段的重叠秒数 |
| size_bytes | 片段大小，单位 bytes |
| size_mb | 片段大小，单位 MB |
| size_gb | 片段大小，单位 GB |
| status | OK / WARNING_OVER_SIZE / ERROR |
| warning | 具体警告信息 |

---

## 关键帧问题说明

因为要求无重编码，所以 `-c copy` 切分时，ffmpeg 可能会把实际切点移动到附近关键帧。

这意味着：

```text
理论 start_time / end_time 可能和实际视频内容有轻微偏差。
```

这是可以接受的。

要求：

- mapping 中记录理论 start / end。
- actual_duration_sec 用 ffprobe 获取。
- 如果实际时长和理论时长差异较大，比如超过 2 秒，在 `warning` 字段中记录：

```text
ACTUAL_DURATION_DIFF
```

---

## ffmpeg 命令实现方式

推荐使用：

```bash
ffmpeg -y -ss START_TIME -i input.mp4 -t DURATION -map 0 -c copy -avoid_negative_ts make_zero output.mp4
```

说明：

```text
-y                         允许覆盖当前目标输出文件，仅在 overwrite 模式下安全使用
-ss START_TIME             从源视频指定时间附近开始
-i input.mp4               输入视频
-t DURATION                输出片段时长
-map 0                     保留全部流
-c copy                    无重编码
-avoid_negative_ts make_zero 避免负时间戳
output.mp4                 输出片段
```

在 Python 中必须使用 list 形式调用：

```python
subprocess.run([
    "ffmpeg",
    "-y",
    "-ss", str(start_time),
    "-i", str(input_path),
    "-t", str(duration),
    "-map", "0",
    "-c", "copy",
    "-avoid_negative_ts", "make_zero",
    str(output_path)
], check=True)
```

不要使用：

```python
subprocess.run("ffmpeg ...", shell=True)
```

因为路径中可能包含空格、中文、日文。

---

## 参数设计

脚本至少支持：

```bash
--input_dir
--output_dir
--segment_time
--overlap_time
--max_size_gb
--overwrite
--dry_run
```

说明：

| 参数 | 含义 |
|---|---|
| `--input_dir` | 输入视频文件夹 |
| `--output_dir` | 输出文件夹 |
| `--segment_time` | 每个片段的目标时长，单位秒，默认 300 |
| `--overlap_time` | 相邻片段之间的重叠时长，单位秒，默认 10 |
| `--max_size_gb` | 单个片段最大目标大小，默认 2 |
| `--overwrite` | 如果输出目录中已经存在该源视频对应的旧 part，是否允许覆盖 |
| `--dry_run` | 只打印计划执行的 ffmpeg 命令，不实际执行 |

参数检查：

```text
segment_time > 0
overlap_time >= 0
overlap_time < segment_time
max_size_gb > 0
```

---

## overwrite 行为

因为所有切分后的视频都直接放在同一个 `output_dir` 中，所以不能删除整个输出目录。

如果没有指定 `--overwrite`：

- 如果 `output_dir` 中已经存在该源视频对应的 part 文件，则跳过该源视频。
- 判断方式可以使用：

```text
{source_stem}_part*.mp4
```

或更通用地匹配：

```text
{source_stem}_part*
```

- 不要覆盖已有结果。
- 终端输出：

```text
[SKIP] Output already exists for meeting_2026_05_10. Use --overwrite to regenerate.
```

如果指定 `--overwrite`：

- 只删除 `output_dir` 中该源视频对应的旧 part 文件。
- 不要删除整个 `output_dir`。
- 不要删除其他源视频的切分结果。
- 绝对不能删除原始视频文件。

---

## dry_run 行为

如果指定：

```bash
--dry_run
```

则：

- 不实际执行 ffmpeg。
- 打印将要处理的视频文件。
- 打印每个视频将要生成的切分区间。
- 打印对应 ffmpeg 命令。
- 不生成真实视频文件。
- 可以不生成 mapping 文件；如果生成，也需要明确标记为 dry run。

---

## ffmpeg / ffprobe 检查

脚本开始时检查：

```text
ffmpeg
ffprobe
```

是否可用。

如果不可用，给出清楚报错：

```text
[ERROR] ffmpeg not found. Please install ffmpeg and add it to PATH.
[ERROR] ffprobe not found. Please install ffmpeg and add it to PATH.
```

可以使用：

```python
shutil.which("ffmpeg")
shutil.which("ffprobe")
```

---

## 获取视频时长

使用 ffprobe：

```bash
ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 input.mp4
```

Python 解析为 float。

如果获取失败：

- 不要崩溃整个程序。
- 在 mapping 中记录该源视频状态为 ERROR。
- 跳过该视频。
- 继续处理其他视频。

---

## 文件大小检查

每个 part 输出完成后检查大小。

如果：

```text
size_gb > max_size_gb
```

则 status 设为：

```text
WARNING_OVER_SIZE
```

warning 写入：

```text
File size exceeds max_size_gb. Consider reducing --segment_time.
```

终端打印：

```text
[WARNING] meeting_2026_05_10_part003_start000870s_end001170s.mp4 is 2.13 GB, larger than 2.0 GB.
Consider reducing --segment_time, e.g. 240 seconds.
```

---

## 实际时长检查

每个 part 输出完成后，用 ffprobe 获取 `actual_duration_sec`。

如果获取失败：

- `actual_duration_sec` 为空或 `null`
- warning 加上：

```text
FFPROBE_DURATION_FAILED
```

如果：

```text
abs(actual_duration_sec - theoretical_duration_sec) > 2.0
```

warning 加上：

```text
ACTUAL_DURATION_DIFF
```

这可能是关键帧对齐导致的，可以接受，但要记录。

---

## 错误处理

需要处理：

1. 输入目录不存在。
2. 输出目录不存在时自动创建。
3. 文件夹内没有视频文件。
4. 某个视频切分失败时，不要中断整个批处理。
5. 某个 part 切分失败时，在 mapping 中记录 ERROR。
6. 路径中包含空格、中文、日文时不要出错。
7. ffprobe 获取时长失败时不要崩溃。
8. `overlap_time >= segment_time` 时直接报错并退出。
9. `--overwrite` 只能删除当前源视频对应的旧 part 文件，不能删除整个输出目录。
10. 不得删除、移动或修改原始视频文件。

---

## 日志输出示例

```text
[INFO] Found 12 video files.
[INFO] Processing 1/12: meeting_2026_05_10.mp4
[INFO] Video duration: 1832.44 sec
[INFO] segment_time=300, overlap_time=10, step_time=290
[INFO] Exporting part000: 0.00s -> 300.00s
[INFO] Exporting part001: 290.00s -> 590.00s
[INFO] Exporting part002: 580.00s -> 880.00s
[INFO] Exporting part003: 870.00s -> 1170.00s
[OK] Finished: meeting_2026_05_10.mp4

[WARNING] 1 part is larger than 2.0 GB.
[INFO] Mapping saved to: E:/Matsuda_data/split_videos/split_mapping.csv
[INFO] Mapping saved to: E:/Matsuda_data/split_videos/split_mapping.json
```

---

## 推荐实现方式

请优先使用 Python 标准库：

```text
argparse
pathlib
subprocess
csv
json
shutil
datetime
re
math
```

不需要复杂第三方库。

---

## 最终交付物

请生成或修改以下代码：

```text
E:\Single_frame_smile\annotation\dataset-annotation\annotation_store.py
```

如果新建独立脚本，则生成：

```text
E:\Single_frame_smile\annotation\dataset-annotation\split_videos_overlap.py
```

并附带简短使用说明。

---

## 验收标准

完成后应满足：

1. 可以批量处理整个文件夹的视频。
2. 原始视频文件完整保留。
3. 所有切分片段直接进入指定 `output_dir`。
4. 不为每个源视频创建单独子文件夹。
5. 相邻片段之间有 `overlap_time` 秒重叠。
6. 切分文件名包含源文件名、part 编号、start 秒数、end 秒数。
7. 生成 `split_mapping.csv`。
8. 生成 `split_mapping.json`。
9. 映射表可以追溯每个 part 对应的源视频和原始时间区间。
10. 使用 `-c copy`，不发生重编码。
11. 如果某些 part 超过 2GB，清楚标记 warning。
12. 路径中有空格、中文、日文时可以正常运行。
13. 单个视频失败不影响其他视频继续处理。
14. `--overwrite` 只删除当前源视频对应的旧 part 文件，不删除整个输出目录。
15. 不删除、不移动、不修改原始视频文件。
