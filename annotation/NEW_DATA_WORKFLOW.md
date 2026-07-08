# New Data Workflow

This workflow turns a source video plus a simple event dat file into event clips, RVM green-background videos, and FaceTracking outputs.

## Input

Dat format:

```text
time class
0:11 polite
0:20 polite
8:24 polite
```

Supported time formats:

- seconds: `11`
- minute:second: `0:11`
- hour:minute:second: `1:02:03`

By default, each event time is expanded to a 10 second centered window: 5 seconds before and 5 seconds after. Near video boundaries the window is shifted inward when possible.

## Recommended Order

Use this order:

```text
source video -> cut event windows -> RVM green-background video -> FaceTracking
```

Reason:

- RVM is a temporal video matting model, so it should see continuous video rather than isolated face crops.
- Running FaceTracking after RVM keeps the final face-frame domain consistent with the old green-background analysis data.

Risk:

- If the green-background output creates artifacts around the face, FaceTracking may fail on some frames. Check the FaceTracking output counts and inspect failed clips.

## Script

Path:

```text
E:\Single_frame_smile\annotation\new_data_workflow.py
```

Example:

```powershell
python E:\Single_frame_smile\annotation\new_data_workflow.py `
  --video_dir "E:\Matsuda_data\new_raw_video" `
  --dat "E:\Matsuda_data\new_raw_video\events.dat" `
  --output_root "E:\Matsuda_data\new_data_workflow" `
  --rvm_cmd 'python "E:\path\to\rvm_script.py" --input "{input}" --output "{output}" --green' `
  --facetracking_cmd '"E:\path\to\FaceTracking.exe" "{input}" "{output_dir}"'
```

Use `--dry_run` first to verify the cut windows and external commands without running them.

If RVM or FaceTracking is not available yet, test only the clip-cutting stage:

```powershell
--stop_after clips
```

If you intended 10 seconds before and 10 seconds after the event time, pass:

```powershell
--pre_sec 10 --post_sec 10
```

## Output

```text
output_root\
  workflow_config.json
  manifest.csv
  clips_raw\<class>\<seq_id>\*.mp4
  rvm_greenbg\<class>\<seq_id>\*.mp4
  facetracking\<class>\<seq_id>\...
```

`manifest.csv` records the source video, event time, cut window, raw clip, RVM output, and FaceTracking output directory for every sequence.

## Current External Dependency Gap

This repository has scripts for video splitting and downstream analysis, but it does not contain a verified runnable RVM command or FaceTracking executable. The workflow script therefore requires explicit `--rvm_cmd` and `--facetracking_cmd` templates.
