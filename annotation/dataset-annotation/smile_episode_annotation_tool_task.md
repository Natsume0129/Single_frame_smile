# Coding Task: Smile Episode Video Annotation Tool

## 1. Project Goal

Build a local video annotation tool for labeling smile episodes in meeting video clips.

The tool should allow the user to:

- Load a video clip.
- Play, pause, and seek through the video.
- Mark the `start_frame`, `peak_frame`, and `end_frame` of each smile episode.
- Assign a visual smile label to each episode.
- Save all annotations into a unified CSV file.
- Allow one video clip to contain multiple smile episodes.

The annotation unit is **one smile episode**, not one frame and not one full video clip.

---

## 2. Background

The dataset consists of meeting video clips.  
A single clip may contain:

- zero smile episodes
- one smile episode
- multiple smile episodes

Each smile episode should be annotated independently.

The final goal is to train a temporal video/feature model to distinguish visually genuine-like smiles from other types of smiles.

The annotation tool should therefore produce episode-level annotations that can later be converted into model training samples.

---

## 3. Recommended Technology

Use:

- Python
- PySide6 or PyQt6
- OpenCV for video reading
- pandas or Python csv module for CSV writing

Preferred: **PySide6 + OpenCV**

Do not build a web app for the first version.

---

## 4. Core Concept

### Relationship between clip and episode

```text
clip = a video file
episode = one smile event inside the clip
```

One clip can contain multiple episodes:

```text
clip_001.mp4
├── E0001: frame 120 - 210, genuine_like_smile
├── E0002: frame 430 - 500, polite_like_smile
└── E0003: frame 800 - 900, ambiguous_smile
```

The CSV should use:

```text
one row = one smile episode
```

Not:

```text
one row = one video clip
```

Not:

```text
one row = one frame
```

---

## 5. Required Features

### 5.1 Video Loading

The tool must support loading local video files, for example:

- `.mp4`
- `.avi`
- `.mov`

After loading a video, the tool should display:

- video filename
- video path
- FPS
- total frame count
- current frame index
- current time in seconds

---

### 5.2 Video Playback

The tool must support:

- play
- pause
- seek by progress bar
- jump to previous frame
- jump to next frame
- jump backward by 5 frames
- jump forward by 5 frames
- jump backward by 1 second
- jump forward by 1 second

Recommended keyboard shortcuts:

| Key | Action |
|---|---|
| Space | Play / Pause |
| Left Arrow | Previous frame |
| Right Arrow | Next frame |
| A | Backward 5 frames |
| D | Forward 5 frames |
| J | Backward 1 second |
| L | Forward 1 second |
| S | Set start frame |
| P | Set peak frame |
| E | Set end frame |
| Ctrl + S | Save current episode |

---

### 5.3 Progress Bar

The progress bar should allow frame-level seeking or near-frame-level seeking.

It should display markers for:

- start frame
- peak frame
- end frame

Recommended marker colors:

```text
start = green
peak = red
end = blue
```

If custom colored markers are difficult in v0.1, display the selected frame numbers clearly in text fields.

---

### 5.4 Episode Marking

The user should be able to set:

- `start_frame`
- `peak_frame`
- `end_frame`

By pressing buttons:

- `Set Start`
- `Set Peak`
- `Set End`

Each button should record the current frame index.

The tool must validate:

```text
start_frame < peak_frame < end_frame
```

If this condition is not satisfied, the tool should not save the episode.

---

### 5.5 Label Selection

The tool should provide the following labels:

```text
genuine_like_smile
polite_like_smile
bitter_awkward_like_smile
ambiguous_smile
neutral_or_no_smile
unclear
```

Definitions:

| Label | Meaning |
|---|---|
| genuine_like_smile | Visually genuine-like smile with clear upper-face and lower-face coordination |
| polite_like_smile | Social/polite-like smile, often mainly mouth-based with weak eye involvement |
| bitter_awkward_like_smile | Bitter, awkward, tense, or asymmetrical smile-like expression |
| ambiguous_smile | Smile type is visually unclear or unstable |
| neutral_or_no_smile | No clear smile in the selected interval |
| unclear | Bad visibility, severe occlusion, tracking failure, or impossible to judge |

---

### 5.6 Confidence Score

The user should select a confidence score:

```text
1, 2, 3, 4, 5
```

Meaning:

| Score | Meaning |
|---|---|
| 1 | very uncertain |
| 2 | uncertain |
| 3 | moderately uncertain |
| 4 | fairly confident |
| 5 | very confident |

Training scripts will later use only high-confidence samples, for example:

```text
confidence >= 4
```

---

### 5.7 Additional Visual Attributes

The tool should allow optional annotation of:

```text
intensity
eye_involvement
mouth_movement
cheek_raise
symmetry
visible_quality
usable_for_training
note
```

Suggested values:

#### intensity

```text
1, 2, 3, 4, 5
```

#### eye_involvement

```text
1, 2, 3, 4, 5
```

#### mouth_movement

```text
1, 2, 3, 4, 5
```

#### cheek_raise

```text
1, 2, 3, 4, 5
```

#### symmetry

```text
symmetric
slightly_asymmetric
asymmetric
unknown
```

#### visible_quality

```text
good
medium
poor
```

#### usable_for_training

```text
yes
no
```

Default rule:

- `yes` if visibility is acceptable and confidence is high.
- `no` if the sample is unclear, low confidence, or visually problematic.

#### note

Free text field.

---

## 6. CSV Output Format

The main output file should be:

```text
annotations.csv
```

The tool should append new episodes to this file.

Required columns:

```csv
episode_id,video_id,clip_path,person_id,start_frame,peak_frame,end_frame,start_time,peak_time,end_time,main_label,confidence,intensity,eye_involvement,mouth_movement,cheek_raise,symmetry,visible_quality,usable_for_training,note
```

Example:

```csv
episode_id,video_id,clip_path,person_id,start_frame,peak_frame,end_frame,start_time,peak_time,end_time,main_label,confidence,intensity,eye_involvement,mouth_movement,cheek_raise,symmetry,visible_quality,usable_for_training,note
E0001,clip_001,dataset/raw_videos/clip_001.mp4,P01,120,155,210,4.000,5.167,7.000,genuine_like_smile,5,4,4,4,4,symmetric,good,yes,"clear eye and cheek involvement"
E0002,clip_001,dataset/raw_videos/clip_001.mp4,P01,430,455,500,14.333,15.167,16.667,polite_like_smile,4,2,1,3,1,symmetric,good,yes,"mouth-dominant smile"
```

---

## 7. ID Rules

### 7.1 video_id

The tool may generate `video_id` from the filename.

Example:

```text
clip_001.mp4 → clip_001
```

### 7.2 episode_id

The tool should generate globally unique episode IDs.

Recommended format:

```text
E000001
E000002
E000003
```

The episode ID should not reset for each video.

If `annotations.csv` already exists, the tool should read the existing file and continue from the latest episode ID.

---

## 8. User Interface Layout

Recommended layout:

```text
-------------------------------------------------
|                                               |
|               Video Display Area              |
|                                               |
-------------------------------------------------

Current frame: 155 / 1800
Current time: 5.167 sec
FPS: 30.0

[Progress Bar]

[Play/Pause] [Prev Frame] [Next Frame] [Back 5] [Forward 5] [Back 1s] [Forward 1s]

Start frame: 120    [Set Start]
Peak frame: 155     [Set Peak]
End frame: 210      [Set End]

Label:              [Dropdown]
Confidence:         [Dropdown / Spinbox]
Intensity:          [Dropdown / Spinbox]
Eye involvement:    [Dropdown / Spinbox]
Mouth movement:     [Dropdown / Spinbox]
Cheek raise:        [Dropdown / Spinbox]
Symmetry:           [Dropdown]
Visible quality:    [Dropdown]
Usable for training:[Checkbox or Dropdown]
Note:               [Text field]

[Save Episode] [Clear Current Episode] [Next Video]

Episode List for Current Video:
-------------------------------------------------
| episode_id | start | peak | end | label | conf |
-------------------------------------------------
```

---

## 9. Episode List

The tool should show saved episodes for the current video.

For each saved episode, display:

```text
episode_id
start_frame
peak_frame
end_frame
main_label
confidence
usable_for_training
```

Optional but useful:

- click an existing episode to jump to its start frame
- edit an existing episode
- delete an existing episode

For v0.1, editing and deletion are optional.  
For v0.2, editing and deletion are recommended.

---

## 10. Validation Rules

Before saving an episode, check:

1. video is loaded
2. start_frame is set
3. peak_frame is set
4. end_frame is set
5. `start_frame < peak_frame < end_frame`
6. label is selected
7. confidence is selected
8. visible_quality is selected
9. episode_id is unique
10. the same episode is not accidentally duplicated

If validation fails, show a clear error message.

---

## 11. Save Behavior

When the user clicks `Save Episode`:

1. validate fields
2. generate `episode_id`
3. compute:
   - `start_time = start_frame / fps`
   - `peak_time = peak_frame / fps`
   - `end_time = end_frame / fps`
4. append one row to `annotations.csv`
5. refresh the episode list
6. clear current start/peak/end selection or keep it depending on user preference

Recommended default: clear current episode selection after save.

---

## 12. Data Compatibility for Training

The annotation output should support a later dataset-building script.

Expected downstream process:

```text
annotations.csv
↓
filter usable_for_training == yes
↓
filter confidence >= 4
↓
read clip_path
↓
extract frames from start_frame to end_frame
↓
resample each episode to fixed length T, e.g. 20 or 32 frames
↓
extract visual features
↓
save feature sequence as .npy or .pt
```

Each training sample will correspond to one row in `annotations.csv`.

Example:

```text
E0001 → frames 120:210 → resampled to 20 frames → feature shape [20, 512]
E0002 → frames 430:500 → resampled to 20 frames → feature shape [20, 512]
```

---

## 13. Optional Future Features

Do not implement these in v0.1 unless core features are already stable.

Possible future features:

- side-by-side raw video and processed green-screen video
- automatic preview clip export for each episode
- loading processed video path together with raw video path
- multiple annotator support
- inter-annotator agreement calculation
- import/export project file
- thumbnail display of start/peak/end frames
- frame extraction preview
- auto-save backup
- undo last save
- filter episode list by label
- keyboard-only annotation mode

---

## 14. Recommended Development Milestones

### v0.1: Basic Video Viewer

Required:

- load video
- display video
- play/pause
- seek with progress bar
- show current frame and time

### v0.2: Episode Marking

Required:

- set start/peak/end
- display selected frame numbers
- validate frame order

### v0.3: CSV Saving

Required:

- label selection
- confidence selection
- save episode as one row in `annotations.csv`
- auto-generate episode ID

### v0.4: Episode List

Required:

- show saved episodes for current video
- jump to saved episode

### v0.5: Keyboard Shortcuts

Required:

- frame stepping
- play/pause
- set start/peak/end
- save episode

### v0.6: Editing Support

Recommended:

- edit saved episode
- delete saved episode
- prevent duplicate annotations

---

## 15. Acceptance Criteria

The implementation is acceptable if the following are true:

1. The user can load a local video.
2. The video can be played, paused, and seeked.
3. The current frame index is displayed correctly.
4. The user can set start, peak, and end frames.
5. The tool prevents invalid episode ranges.
6. The user can choose a label and confidence score.
7. Saving creates exactly one CSV row per smile episode.
8. One video can have multiple saved episodes.
9. Existing `annotations.csv` can be reopened and appended without overwriting previous data.
10. Episode IDs are unique across all videos.
11. The saved CSV can be used later to extract training samples.

---

## 16. Important Design Principle

The tool should not classify the video automatically.

The tool is only for human annotation.

The model training pipeline will be implemented separately.

The most important output is a clean and stable episode-level `annotations.csv`.

