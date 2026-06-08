# Analysis Handoff

Last updated: 2026-06-02

This is the canonical handoff document for `E:\Single_frame_smile\Analysis`.

## 0. Current State

Verified locally on 2026-05-25:

- `Analysis/documents` existed and was empty before this file was added.
- No previous canonical handoff file was found under `Analysis`.
- The repository worktree already had unrelated local edits and untracked files before this document was created. Those were not changed.
- Current additions include a non-DTW linear-aligned s-d comparison script and generated plots. No full end-to-end pipeline rerun was performed.

Main working assumption:

- The current analysis foundation is the existing output under `E:\Matsuda_data`, not a fresh rerun of all pipelines.

Additional verified locally on 2026-06-02:

- Added nearest-baseline curve analysis for the current s-d follow-up question.
- Added a Pro-model explanation document for the nearest-baseline method and current results.
- Created a portable Pro-review zip package with explanation, HTML report, plots, CSVs, and code.
- Added a standalone interactive nearest-6 HTML report with hover highlighting and class filters.
- Existing unrelated dirty/untracked worktree files were present and were not changed.

Important output roots verified locally:

| Output root | Files | Size | Purpose |
|---|---:|---:|---|
| `E:\Matsuda_data\2-27meeting` | 2394 | 282.70 MB | Main `analysis_sequence` output |
| `E:\Matsuda_data\3-10meeting` | 144 | 64.62 MB | Main `analysis_projection` output |
| `E:\Matsuda_data\3-10meeting\linear_axis_extension` | 28 | 3.15 MB | Non-DTW linear-aligned axis-extension s-d plots |
| `E:\Matsuda_data\3-10meeting\nearest_baseline_curve` | 33 | 3.56 MB | Nearest-baseline progress-distance static CSV/plots/report |
| `E:\Matsuda_data\3-10meeting\nearest_baseline_curve_interactive` | 1 | 0.25 MB | Standalone interactive nearest-6 nearest-baseline HTML |
| `E:\Matsuda_data\analysis_minimum_output` | 12 | 1.93 MB | Synchronized minimum-distance output |
| `E:\Matsuda_data\DTW_analysis` | 58 | 4.11 MB | DTW similarity output |
| `E:\Matsuda_data\DTW_resample_output` | 476 | 98.40 MB | DTW representative alignment and resampling output |
| `E:\Matsuda_data\DTW_resample_output\axis_extension` | 13 | 0.72 MB | DTW-resampled axis-extension plots, including s-d plot |
| `E:\Matsuda_data\DTW_resample_output\projection_followup` | 9 | 0.61 MB | Projection follow-up on DTW-resampled data |
| `E:\Single_frame_smile\Analysis\Heatmap\output` | 591 | 57.40 MB | Heatmap output for selected representative sequences |

## 1. Big Picture

The `Analysis` folder contains a staged smile-dynamics research workflow.

The main chain is:

1. Extract VGG-Face features from face-frame sequences.
2. Convert each smile clip into a high-dimensional temporal trajectory.
3. Baseline-align each sequence using the first 5 frames.
4. Normalize each sequence to 20 time points.
5. Build representative/prototype trajectories for `polite`, `truesmile`, and `ambiguous`.
6. Compare these trajectories using direct distance, axis projection, off-axis deviation, DTW, synchronized minimum distance, and heatmap visualization.

The strongest repeated result across multiple analyses is:

- `ambiguous` is generally closer to `polite` than to `truesmile`.
- `truesmile` separates more clearly as the smile unfolds.
- Single straight-line axes are useful but too coarse to fully describe curved smile trajectories.

## 2. Data Foundation

### 2.1 Source Dataset

The standard source dataset used by the main pipeline is:

```text
E:\Matsuda_data\2-18meeting
```

Expected structure:

```text
E:\Matsuda_data\2-18meeting\
  polite\
  truesmile\
  ambiguous\
```

`video` and `videos` directories are ignored by the main sequence pipeline.

Frame sorting convention:

- Extract the last integer from the file stem.
- Sort numerically.
- Frame gaps are allowed.
- No interpolation or frame filling is done before feature extraction.

Frame rate convention:

- `fps = 30`

### 2.2 Feature Coverage

Verified locally:

| Class | Input sequences | `sequence_features.npy` | `sequence_features_rel.npy` | `normalized_sequence.npy` |
|---|---:|---:|---:|---:|
| `polite` | 41 | 41 | 41 | 41 |
| `truesmile` | 6 | 6 | 6 | 6 |
| `ambiguous` | 27 | 27 | 27 | 27 |

So for the main 74-sequence dataset, every sequence used in `analysis_sequence`, `analysis_projection`, `DTW`, and `analysis_minimum` has extracted feature vectors and normalized sequence data.

Important distinction:

- The main trajectory analyses use all 74 sequences.
- The heatmap analysis only uses three selected representative sequences, not all 74 sequences.

## 3. Shared Mathematical Conventions

Most analysis scripts use this notation:

```text
f(t)      raw feature vector
f0        baseline, usually mean of first 5 frames
f_rel(t)  f(t) - f0
d(t)      ||f_rel(t)||, magnitude curve
v(t)      ||f_rel(t) - f_rel(t-1)||, velocity curve
```

Main feature representation:

```text
VGG-Face fc7
shape per frame = 4096
sequence shape  = [T, 4096]
normalized shape = [20, 4096]
```

Prototype terminology:

- Method A: median trajectory, computed element-wise across normalized sequences.
- Method B: medoid trajectory, a real sequence chosen by minimum total Frobenius distance to other sequences in the same class.
- DTW representative: a separate real sequence chosen by minimum total intra-class DTW distance.

Do not mix these three meanings.

## 4. Feature Extraction

Folder:

```text
E:\Single_frame_smile\Analysis\feature_extractor
```

Important files:

- `feature_extractor_fc7.py`
- `feature_extractor_conv5_3.py`
- `face_comp_torch.py`
- `check_pt_order.py`
- `VGG-Face Feature Files Specification.md`
- `README.md`

What was done:

- Implemented VGG-Face feature extraction for face image sequences.
- `fc7` extraction returns `[T, 4096]` embeddings and saves `names` plus feature tensor.
- `conv5_3` extraction returns spatial maps `[T, 512, 7, 7]`.
- `gap512` is derived from `conv5_3` by global average pooling and returns `[T, 512]`.
- The code keeps frame order by sorting image filenames and running `DataLoader(shuffle=False)`.

Recommended use:

- Use `fc7` for main trajectory/distance/projection analysis.
- Use `conv5_3` or intermediate conv activations for spatial heatmap analysis.
- Use `gap512` when a lower-dimensional conv-derived trajectory is useful.

## 5. Background/RVM Analysis

Folder:

```text
E:\Single_frame_smile\Analysis\backgroundAnalysis
```

Important files:

- `README.MD`
- `compare_fc7_pt.py`
- `analyze_adjacent_diff.py`
- `plot_results.py`
- `compare_out_fc7\report.json`
- `compare_out_conv5_3\report.json`

Purpose:

- Compare features extracted with original background versus RVM/green-screen processed frames.
- Check whether background removal introduces a stable domain shift or changes neighbor structure.

Verified results from local reports:

- For `fc7`, cosine distance mean is about `0.1037`, which is large enough to treat background/RVM as a meaningful representation change.
- For conv GAP output, cosine distance mean is about `0.0498`, smaller than `fc7`.
- `knn_overlap_mean` remains around `0.84-0.87`, so neighbor structure is affected but not destroyed.
- Temporal smoothness is similar between compared variants.

Practical implication:

- Do not casually mix background and RVM-derived features in the same downstream comparison without documenting the domain source.

## 6. Early Keyframe Analysis: `analysis2-12`

Folder:

```text
E:\Single_frame_smile\Analysis\analysis2-12
```

Important files:

- `extract_window.py`
- `build_keyframe_pairs_csv.py`
- `extract_fc7_pair_diff.py`
- `analyze_fc7_pair_diff.py`
- `analyze_fc7_direction.py`
- `summary.md`
- `log.dat`

What was done:

1. Extracted frame windows from `.dat` ranges.
2. Built keyframe pair manifests from manually selected start/peak frames.
3. Extracted `vs` and `vp` fc7 features.
4. Computed `diff = vp - vs`.
5. Ran standard clustering on feature differences.
6. Ran direction clustering after filtering low-magnitude samples and unit-normalizing difference vectors.

Stage conclusion:

- Basic clustering mostly produced a large main cluster plus small outliers.
- Directional `kmeans_unit` gave a more usable split for manual review.
- The result is not strong enough to directly label clusters as true/non-true smile without human inspection.

Current role:

- Historical/diagnostic experiment.
- Useful as reference code for start-to-peak feature-difference extraction.
- Not the current main full-sequence pipeline.

## 7. Base-to-Peak Macro Feature Analysis

Folder:

```text
E:\Single_frame_smile\Analysis\vggface_feature_analysis
```

Important files:

- `extract_base_to_peak.py`
- `plot_base_to_peak_outputs.py`
- `plot_polyline_resample20.py`
- `compare_delta_over_percent.py`
- `analyze_temporal_macro_dynamics.py`

What was done:

- Extracted start-to-peak segments from existing feature `.pt` files.
- Computed baseline-relative `delta`, magnitude, velocity, progress along endpoint direction, deviation from endpoint direction, alignment, and curvature proxies.
- Supported resampling to a normalized time grid for group-level comparison.

Current role:

- Earlier macro-dynamics exploration.
- Many ideas here were later formalized in `analysis_sequence` and `analysis_projection`.

## 8. Main Sequence Pipeline: `analysis_sequence`

Folder:

```text
E:\Single_frame_smile\Analysis\analysis_sequence
```

Primary output root:

```text
E:\Matsuda_data\2-27meeting
```

Important files:

- `requirements.md`
- `coding_explanation.md`
- `coding_explanation_en.md`
- `run_pipeline.ps1`
- `common\base.py`
- `01_extract_features.py`
- `02_baseline_align.py`
- `03_compute_magnitude.py`
- `04_compute_velocity.py`
- `05_compute_duration_stats.py`
- `06_time_normalize.py`
- `07_build_prototypes.py`
- `08_class_difference_vectors.py`
- `09_segment_vectors.py`
- `10_projection_pca.py`
- `11_class_distance_curve.py`
- `12_projection_scores.py`
- `13_generate_visualizations.py`
- `14_generate_dataset_report.py`

Pipeline steps:

1. Extract VGG-Face `fc7` features for every frame of every sequence.
2. Save `sequence_features.npy` and `frame_names.json`.
3. Baseline-align with `f0 = mean(first 5 frames)`.
4. Save `sequence_features_rel.npy` and `baseline_f0.npy`.
5. Compute magnitude curve `||f_rel(t)||`.
6. Compute velocity curve `||f_rel(t)-f_rel(t-1)||`.
7. Compute duration statistics at 30 fps.
8. Linearly resample every sequence to `N=20`.
9. Copy sampled frames as `000.png` to `019.png`.
10. Build Method A median prototypes.
11. Build Method B medoid prototypes.
12. Compute class difference vectors and segment vectors.
13. Run PCA trajectory visualization.
14. Generate class distance curves, projection scores, summary plots, and dataset report.

Current verified dataset report count:

- `polite`: 41 sequences
- `truesmile`: 6 sequences
- `ambiguous`: 27 sequences

Current verified prototype metadata:

```text
polite:    num_sequences=41, shape=[20,4096], medoid_sequence_id=13
truesmile: num_sequences=6,  shape=[20,4096], medoid_sequence_id=2
ambiguous: num_sequences=27, shape=[20,4096], medoid_sequence_id=27
```

Important outputs:

```text
E:\Matsuda_data\2-27meeting\metrics\sequence_features\<class>\<seq>\sequence_features.npy
E:\Matsuda_data\2-27meeting\metrics\sequence_features_rel\<class>\<seq>\sequence_features_rel.npy
E:\Matsuda_data\2-27meeting\metrics\normalized\<class>\<seq>\normalized_sequence.npy
E:\Matsuda_data\2-27meeting\metrics\normalized_frames\<class>\<seq>\000.png ... 019.png
E:\Matsuda_data\2-27meeting\prototypes\prototype_<class>.npy
E:\Matsuda_data\2-27meeting\prototypes\prototype_<class>_medoid.npy
E:\Matsuda_data\2-27meeting\prototypes\prototype_meta.json
E:\Matsuda_data\2-27meeting\csv\dataset_report.csv
```

How to rerun:

```powershell
cd E:\Single_frame_smile\Analysis\analysis_sequence
powershell -ExecutionPolicy Bypass -File .\run_pipeline.ps1
```

Rerun caution:

- This can regenerate a large amount of data under `E:\Matsuda_data\2-27meeting`.
- Only rerun when the source dataset, feature extractor, or preprocessing definition changes.

## 9. Projection Analysis: `analysis_projection`

Folder:

```text
E:\Single_frame_smile\Analysis\analysis_projection
```

Primary output root:

```text
E:\Matsuda_data\3-10meeting
```

Verification snapshot inside repository:

```text
E:\Single_frame_smile\Analysis\analysis_projection\_verification_output\3-10meeting
```

Important files:

- `projection_analysis_cn.md`
- `projection_analysis_en.md`
- `work_summary_report.md`
- `run_projection_pipeline.ps1`
- `common.py`
- `01_build_projection_prototypes.py`
- `02_compute_direct_distance.py`
- `03_compute_projection_metrics.py`
- `04_compute_per_sequence_metrics.py`
- `05_compute_statistics.py`
- `06_generate_plots.py`
- `07_generate_report.py`
- `08_polite_axis_deviation_analysis.py`
- `09_shared_geometry_analysis.py`
- `10_list_all_pair_min_distances.py`
- `11_analyze_min_distance_positions.py`
- `12_excluded_prefix_min_distance_analysis.py`
- `13_last_ten_frames_min_distance_analysis.py`

Core idea:

- Reuse `analysis_sequence` outputs instead of re-extracting features.
- Build two parallel prototype methods:
  - Method A: median trajectory.
  - Method B: medoid/real-sequence trajectory.
- Use `truesmile` first-to-last prototype vector as the true-smile reference axis.
- Compute direct distance, along-axis progress, and off-axis deviation.
- Apply similar metrics to every normalized sequence for sample-level statistics.

Method A/B outputs:

```text
E:\Matsuda_data\3-10meeting\methodA\csv
E:\Matsuda_data\3-10meeting\methodA\plots
E:\Matsuda_data\3-10meeting\methodA\prototypes
E:\Matsuda_data\3-10meeting\methodA\report
E:\Matsuda_data\3-10meeting\methodB\csv
E:\Matsuda_data\3-10meeting\methodB\plots
E:\Matsuda_data\3-10meeting\methodB\prototypes
E:\Matsuda_data\3-10meeting\methodB\report
```

Key verified Method A result:

- `polite` vs `truesmile`: starts at `0.0939`, peaks at `0.4611` at `t=15`, ends at `0.4212`.
- `polite` vs `ambiguous`: starts at `0.0212`, peaks at `0.2033` at `t=11`, ends at `0.1675`.
- `truesmile` vs `ambiguous`: starts at `0.0956`, peaks at `0.4064` at `t=15`, ends at `0.3598`.
- Along true-smile axis at end:
  - `polite`: `0.2078`
  - `truesmile`: `1.0000`
  - `ambiguous`: `0.3727`

Key verified Method B result:

- `polite` vs `truesmile`: starts at `0.1527`, peaks at `0.6603` at `t=14`, ends at `0.6228`.
- `polite` vs `ambiguous`: starts at `0.0962`, peaks at `0.5107` at `t=17`, ends at `0.4949`.
- `truesmile` vs `ambiguous`: starts at `0.1439`, peaks at `0.7374` at `t=19`, ends at `0.7374`.
- Along true-smile axis at end:
  - `polite`: `0.1485`
  - `truesmile`: `1.0000`
  - `ambiguous`: `-0.0242`

Interpretation from projection reports:

- `polite` and `ambiguous` are closer to each other than either is to `truesmile`.
- Class distance is small near the beginning and grows as the smile unfolds.
- `polite` and `ambiguous` do not strongly advance along the true-smile axis.
- Method B can show stronger separation because it uses real representative sequences.

Polite-axis extension:

- Implemented in `08_polite_axis_deviation_analysis.py`.
- It uses the polite-smile prototype vector as the base axis.
- Verified Method A:
  - `truesmile` prototype off-axis ratio relative to polite axis ends around `2.0775`.
  - `ambiguous` ends around `0.8007`.
- Verified Method B:
  - `truesmile` ends around `1.6107`.
  - `ambiguous` ends around `0.8435`.

Interpretation:

- `polite` likely has its own stable dynamic direction.
- `truesmile` is not just "more intense polite"; it moves toward a different region/direction.

Shared geometry scripts:

- `09_shared_geometry_analysis.py`: original-sequence cross-time curve minimum distance and tangent-relative nearest-point decomposition.
- `10_list_all_pair_min_distances.py`: all-pair raw minimum-distance list.
- `11_analyze_min_distance_positions.py`: analyzes where minimum-distance positions occur.
- `12_excluded_prefix_min_distance_analysis.py`: repeats minimum-distance analysis after excluding early prefix frames.
- `13_last_ten_frames_min_distance_analysis.py`: focuses on last frames.

Caution:

- Some shared-geometry outputs are in `E:\Matsuda_data\3-10meeting\shared` when scripts are run.
- The core Method A/B report output is complete in `methodA` and `methodB`; shared geometry is more exploratory.

How to rerun:

```powershell
cd E:\Single_frame_smile\Analysis\analysis_projection
powershell -ExecutionPolicy Bypass -File .\run_projection_pipeline.ps1
```

## 10. Synchronized Minimum Distance: `analysis_minimum`

Folder:

```text
E:\Single_frame_smile\Analysis\analysis_minimum
```

Primary output root:

```text
E:\Matsuda_data\analysis_minimum_output
```

Important files:

- `minimum_requirements_cn.md`
- `minimum_common.py`
- `run_minimum_pipeline.py`
- `run_raw_minimum_debug.py`
- `result.md`

Purpose:

- Redefine minimum distance for time-aligned normalized curves.
- Use:

```text
d_sync_min(C1, C2) = min_t ||C1(t) - C2(t)||
```

This is different from cross-time minimum distance:

```text
min_{t1,t2} ||C1(t1) - C2(t2)||
```

Inputs:

- `E:\Matsuda_data\2-27meeting\metrics\normalized`
- `E:\Matsuda_data\2-27meeting\metrics\normalized_frames`
- `E:\Matsuda_data\2-27meeting\prototypes`

Verified result:

- Sequence pair count: `2701`
- Most common synchronized minimum time is usually `t=1`.
- `ambiguous_vs_polite` has lower mean/median synchronized minimum distance than either class versus `truesmile`.

Verified sequence-level statistics:

- `ambiguous_vs_polite`: mean `0.1496`, median `0.1332`, most_common_t `1`.
- `polite_vs_truesmile`: mean `0.2184`, median `0.2212`, most_common_t `1`.
- `ambiguous_vs_truesmile`: mean `0.2284`, median `0.2208`, most_common_t `1`.

Verified prototype-level statistics:

- Method A:
  - `polite` vs `truesmile`: min `0.0934` at `t=1`
  - `polite` vs `ambiguous`: min `0.0149` at `t=1`
  - `truesmile` vs `ambiguous`: min `0.0927` at `t=1`
- Method B:
  - `polite` vs `truesmile`: min `0.1527` at `t=0`
  - `polite` vs `ambiguous`: min `0.0429` at `t=1`
  - `truesmile` vs `ambiguous`: min `0.1439` at `t=0`

Interpretation:

- Early normalized stages are the most similar.
- `ambiguous` remains closer to `polite` than to `truesmile` under synchronized stage comparison.

## 11. DTW Analysis

Folder:

```text
E:\Single_frame_smile\Analysis\DTW
```

Primary output root:

```text
E:\Matsuda_data\DTW_analysis
```

Important files:

- `dtw_requirements_cn.md`
- `dtw_common.py`
- `run_dtw_pipeline.py`
- `plot_dtw_conclusion_summary.py`
- `result.md`

Purpose:

- Compare dynamic similarity while allowing temporal warping.
- Use original variable-length `sequence_features_rel.npy`, not the already-resampled 20-point sequences.

Branches computed:

- `magnitude_dtw`
- `magnitude_dtw_band`
- `velocity_dtw`
- `velocity_dtw_band`
- `pca10_dtw`
- `pca10_dtw_band`
- `pca20_dtw`
- `pca20_dtw_band`
- `pca30_dtw`
- `pca30_dtw_band`

Feature branches:

- Magnitude: `d(t)=||f_rel(t)||`
- Velocity: `v(t)=||f_rel(t)-f_rel(t-1)||`
- PCA branches: project original `f_rel(t)` to 10, 20, or 30 dimensions and run multivariate DTW.
- Band variants use Sakoe-Chiba constraint with radius `20%` of the longer sequence length.

Verified high-level result:

- In every branch, the closest inter-class pair by median DTW distance is `ambiguous_vs_polite`.
- Most compact intra-class pair is usually `polite_vs_polite`, with one magnitude-band case reporting `ambiguous_vs_ambiguous`.

Representative examples from the verified report:

- `magnitude_dtw`: closest inter-class median is `ambiguous_vs_polite = 0.7644`.
- `velocity_dtw`: closest inter-class median is `ambiguous_vs_polite = 0.3725`.
- `pca10_dtw`: closest inter-class median is `ambiguous_vs_polite = 172.0174`.
- `pca20_dtw`: closest inter-class median is `ambiguous_vs_polite = 231.7078`.
- `pca30_dtw`: closest inter-class median is `ambiguous_vs_polite = 273.4330`.

Interpretation:

- The `ambiguous` closer-to-`polite` pattern is stable across magnitude, velocity, and low-dimensional high-feature trajectories.
- DTW supports the same broad conclusion as projection and synchronized minimum distance.

## 12. DTW Resample

Folder:

```text
E:\Single_frame_smile\Analysis\DTW_resample
```

Primary output root:

```text
E:\Matsuda_data\DTW_resample_output
```

Important files:

- `todo.md`
- `dtw_resample_common.py`
- `run_dtw_resample_pipeline.py`
- `run_dtw_projection_followup.py`
- `run_dtw_axis_extension.py`
- `result.md`
- `projection_followup_result.md`

Purpose:

- Build a new 20-point representation that first respects intra-class temporal variation.
- For each class:
  1. Load original variable-length `sequence_features_rel.npy`.
  2. Compute intra-class DTW distance matrix.
  3. Choose the DTW representative sequence with minimum total intra-class DTW distance.
  4. Align all class sequences to that representative timeline.
  5. Resample aligned sequences to 20 points.

Verified DTW representative sequences:

- `polite`: representative sequence `13`, centrality `159.6338`
- `truesmile`: representative sequence `3`, centrality `35.1264`
- `ambiguous`: representative sequence `27`, centrality `111.1097`

Important outputs:

```text
E:\Matsuda_data\DTW_resample_output\csv\representative_sequences.csv
E:\Matsuda_data\DTW_resample_output\csv\all_sequences_to_representative.csv
E:\Matsuda_data\DTW_resample_output\csv\dtw_alignment_paths.csv
E:\Matsuda_data\DTW_resample_output\metrics\aligned_to_representative\<class>\<seq>\aligned_sequence.npy
E:\Matsuda_data\DTW_resample_output\metrics\resampled20_aligned\<class>\<seq>\aligned_resampled20.npy
E:\Matsuda_data\DTW_resample_output\metrics\resampled20_aligned\<class>\<seq>\alignment_mapping.json
```

Media outputs:

- Representative source videos and clips are copied/exported under:

```text
E:\Matsuda_data\DTW_resample_output\media\<class>\<representative_seq>
```

## 13. DTW-Resampled Projection Follow-Up

Folder:

```text
E:\Single_frame_smile\Analysis\DTW_resample
```

Script:

```text
run_dtw_projection_followup.py
```

Output root:

```text
E:\Matsuda_data\DTW_resample_output\projection_followup
```

Purpose:

- Treat each DTW representative sequence as the class prototype.
- Run true-smile-axis projection and off-axis analysis on the DTW-aligned-and-resampled 20-point data.

Verified representative sequences:

- `polite`: `13`
- `truesmile`: `3`
- `ambiguous`: `27`

Verified prototype metrics:

- `polite`: along_end `0.0575`, along_peak `0.0768`, off_end `0.4803`, off_peak `0.4874`
- `truesmile`: along_end `1.0000`, along_peak `1.0000`, off_end `0.0000`, off_peak `0.5993`
- `ambiguous`: along_end `0.1322`, along_peak `0.1665`, off_end `0.3860`, off_peak `0.5306`

Interpretation:

- After DTW alignment, `polite` and `ambiguous` still show limited true-smile-axis progress.
- `ambiguous` progresses somewhat more than `polite` on the true-smile axis but remains far below `truesmile`.

## 14. DTW-Resampled Axis Extension and s-d Plot

Folder:

```text
E:\Single_frame_smile\Analysis\DTW_resample
```

Script:

```text
run_dtw_axis_extension.py
```

Output root:

```text
E:\Matsuda_data\DTW_resample_output\axis_extension
```

Generated plot types:

- `t_s_axis_<axis_class>.png`: time versus projection length.
- `t_d_axis_<axis_class>.png`: time versus off-axis distance.
- `s_d_axis_<axis_class>.png`: projection length versus off-axis distance.

s-d plot code location:

```text
E:\Single_frame_smile\Analysis\DTW_resample\run_dtw_axis_extension.py
function: plot_s_d()
```

Current s-d outputs:

```text
E:\Matsuda_data\DTW_resample_output\axis_extension\plots\s_d_axis_truesmile.png
E:\Matsuda_data\DTW_resample_output\axis_extension\plots\s_d_axis_polite.png
E:\Matsuda_data\DTW_resample_output\axis_extension\plots\s_d_all_sequences_axis_truesmile.png
E:\Matsuda_data\DTW_resample_output\axis_extension\plots\s_d_all_sequences_axis_polite.png
E:\Matsuda_data\DTW_resample_output\axis_extension\plots\s_d_nearest6_axis_truesmile.png
E:\Matsuda_data\DTW_resample_output\axis_extension\plots\s_d_nearest6_axis_polite.png
```

Balanced nearest-representative selection:

- Script uses each class's intra-class DTW matrix under `E:\Matsuda_data\DTW_resample_output\csv\intra_class_dtw_matrix_<class>.csv`.
- For each class, it selects the 6 sequences nearest to that class's DTW representative sequence.
- This was added because `truesmile` only has 6 sequences, so plotting 6 per class gives a more balanced visual comparison.
- Selection list:

```text
E:\Matsuda_data\DTW_resample_output\axis_extension\csv\nearest6_to_representative_sequences.csv
```

Verified selected sequence IDs:

| Class | Representative | Nearest 6 sequence IDs |
|---|---|---|
| `polite` | `13` | `13`, `25`, `36`, `1`, `22`, `49` |
| `truesmile` | `3` | `3`, `5`, `0`, `2`, `4`, `1` |
| `ambiguous` | `27` | `27`, `23`, `22`, `4`, `17`, `26` |

Verified true-smile base axis summary:

- axis_norm `0.8119`
- `polite`: s_end `0.0466`, s_peak `0.0624`, d_end `0.3900`, d_peak `0.3958`
- `truesmile`: s_end `0.8119`, s_peak `0.8119`, d_end `0.0000`, d_peak `0.4866`
- `ambiguous`: s_end `0.1074`, s_peak `0.1352`, d_end `0.3134`, d_peak `0.4308`

Verified polite base axis summary:

- axis_norm `0.3928`
- `polite`: s_end `0.3928`, s_peak `0.3928`, d_end `0.0000`, d_peak `0.2269`
- `truesmile`: s_end `0.0964`, s_peak `0.1292`, d_end `0.8062`, d_peak `0.8197`
- `ambiguous`: s_end `0.0027`, s_peak `0.0638`, d_end `0.3313`, d_peak `0.4412`

Interpretation:

- With true-smile as axis, `truesmile` advances farthest along `s`.
- With polite as axis, `truesmile` has high off-axis distance `d`, supporting the idea that true smile and polite smile move in different directions.

Additional observation from the all-sequence s-d plots:

- After plotting all individual DTW-resampled sequences in the s-d plane, the class trajectories are heavily mixed.
- This means the all-sequence s-d plot does not provide strong evidence that the three smile classes have clearly different coarse spatial directions.
- The earlier prototype-only s-d plot can still describe representative trajectories, but it should not be used alone as proof that the full class distributions move in distinct directions.
- For claims about class-level direction differences, use more cautious wording: current evidence from all-sequence s-d visualization is inconclusive or weak.
- This weak result is important and should be preserved, because it prevents over-interpreting the prototype curves.

Additional observation from the nearest-6 balanced s-d plots:

- The nearest-6 plots are cleaner and easier to inspect than the all-sequence plots.
- They still should be treated as exploratory visualizations, because the selected sequences are intentionally close to each class's representative and therefore do not describe the full class distribution.
- Use them to inspect representative-neighborhood behavior, not to claim full class-level separation.

## 14b. Non-DTW Linear-Aligned s-d Plot

This section was added to compare the DTW-resampled s-d view against the earlier ordinary alignment method.

Code:

```text
E:\Single_frame_smile\Analysis\analysis_projection\run_linear_axis_extension.py
```

Input:

```text
E:\Matsuda_data\2-27meeting\metrics\normalized\<class>\<seq>\normalized_sequence.npy
```

Output root:

```text
E:\Matsuda_data\3-10meeting\linear_axis_extension
```

Method definitions:

- `methodA`: median prototype from the original linear-normalized projection pipeline.
- `methodB`: medoid prototype from the original linear-normalized projection pipeline; this keeps a real sequence as the prototype and is the closest comparison to DTW representative analysis.
- Nearest-6 selection is not DTW-based here. It selects the 6 sequences with smallest Frobenius/Euclidean distance to the class prototype in ordinary linearly normalized feature space.

Generated s-d plot outputs:

```text
E:\Matsuda_data\3-10meeting\linear_axis_extension\plots\s_d_axis_truesmile_methodA.png
E:\Matsuda_data\3-10meeting\linear_axis_extension\plots\s_d_axis_polite_methodA.png
E:\Matsuda_data\3-10meeting\linear_axis_extension\plots\s_d_all_sequences_axis_truesmile_methodA.png
E:\Matsuda_data\3-10meeting\linear_axis_extension\plots\s_d_all_sequences_axis_polite_methodA.png
E:\Matsuda_data\3-10meeting\linear_axis_extension\plots\s_d_nearest6_axis_truesmile_methodA.png
E:\Matsuda_data\3-10meeting\linear_axis_extension\plots\s_d_nearest6_axis_polite_methodA.png
E:\Matsuda_data\3-10meeting\linear_axis_extension\plots\s_d_axis_truesmile_methodB.png
E:\Matsuda_data\3-10meeting\linear_axis_extension\plots\s_d_axis_polite_methodB.png
E:\Matsuda_data\3-10meeting\linear_axis_extension\plots\s_d_all_sequences_axis_truesmile_methodB.png
E:\Matsuda_data\3-10meeting\linear_axis_extension\plots\s_d_all_sequences_axis_polite_methodB.png
E:\Matsuda_data\3-10meeting\linear_axis_extension\plots\s_d_nearest6_axis_truesmile_methodB.png
E:\Matsuda_data\3-10meeting\linear_axis_extension\plots\s_d_nearest6_axis_polite_methodB.png
```

Generated CSV/report outputs:

```text
E:\Matsuda_data\3-10meeting\linear_axis_extension\csv\nearest6_to_prototype_sequences_methodA.csv
E:\Matsuda_data\3-10meeting\linear_axis_extension\csv\nearest6_to_prototype_sequences_methodB.csv
E:\Matsuda_data\3-10meeting\linear_axis_extension\csv\prototype_metrics_axis_<axis>_<method>.csv
E:\Matsuda_data\3-10meeting\linear_axis_extension\csv\per_sequence_metrics_axis_<axis>_<method>.csv
E:\Matsuda_data\3-10meeting\linear_axis_extension\csv\statistics_axis_<axis>_<method>.csv
E:\Matsuda_data\3-10meeting\linear_axis_extension\report\linear_axis_extension_summary_methodA.md
E:\Matsuda_data\3-10meeting\linear_axis_extension\report\linear_axis_extension_summary_methodB.md
```

Verified methodB nearest-6 sequence IDs:

| Class | Prototype sequence | Nearest 6 sequence IDs |
|---|---|---|
| `polite` | `13` | `13`, `25`, `36`, `1`, `8`, `22` |
| `truesmile` | `2` | `2`, `1`, `3`, `5`, `4`, `0` |
| `ambiguous` | `27` | `27`, `4`, `22`, `23`, `17`, `8` |

Verified methodB true-smile base axis summary:

- axis_norm `0.6528`
- `polite`: s_end `0.0969`, s_peak `0.0990`, d_end `0.3806`, d_peak `0.3863`
- `truesmile`: s_end `0.6528`, s_peak `0.6528`, d_end `0.0000`, d_peak `0.4324`
- `ambiguous`: s_end `-0.0158`, s_peak `0.1083`, d_end `0.3309`, d_peak `0.4369`

Verified methodB polite base axis summary:

- axis_norm `0.3928`
- `polite`: s_end `0.3928`, s_peak `0.3928`, d_end `0.0000`, d_peak `0.2269`
- `truesmile`: s_end `0.1611`, s_peak `0.1677`, d_end `0.6326`, d_peak `0.6504`
- `ambiguous`: s_end `0.0027`, s_peak `0.0638`, d_end `0.3313`, d_peak `0.4412`

Interpretation:

- The ordinary linear-aligned methodB plot broadly preserves the representative-level pattern: true smile moves farthest along the true-smile axis, while polite and ambiguous remain much closer to the origin on that axis.
- The all-sequence ordinary-aligned s-d plots are still heavily mixed, so this view also does not provide strong evidence that the full class distributions have cleanly separated coarse directions.
- The nearest-6 ordinary-aligned plots are cleaner and useful for visual inspection, but remain representative-neighborhood plots rather than full-distribution evidence.

## 14c. Nearest-Baseline Curve Analysis

This section records the 2026-06-02 follow-up method requested after discussion of the s-d plot.

Purpose:

- Preserve the baseline curve as a full curve instead of reducing it only to a first-to-last straight axis.
- For each target stage on another curve, ask which stage of the baseline curve is geometrically closest.
- Use this to inspect whether a target smile transition follows the same stage progression as the baseline, remains closest to an early baseline stage, or has unstable temporal correspondence.

Code:

```text
E:\Single_frame_smile\Analysis\analysis_projection\run_nearest_baseline_curve_analysis.py
E:\Single_frame_smile\Analysis\analysis_projection\run_nearest_baseline_interactive_report.py
```

Explanation document for Pro-model review:

```text
E:\Single_frame_smile\Analysis\documents\nearest_baseline_curve_explanation_for_pro.md
```

Input data:

```text
E:\Matsuda_data\2-27meeting\metrics\normalized\<class>\<seq>\normalized_sequence.npy
E:\Matsuda_data\3-10meeting\methodA\prototypes\prototype_<class>_methodA.npy
E:\Matsuda_data\3-10meeting\methodB\prototypes\prototype_<class>_methodB.npy
E:\Matsuda_data\3-10meeting\linear_axis_extension\csv\nearest6_to_prototype_sequences_methodA.csv
E:\Matsuda_data\3-10meeting\linear_axis_extension\csv\nearest6_to_prototype_sequences_methodB.csv
```

Important scope:

- This first version uses the ordinary linear-normalized 20-point fc7 trajectories, not DTW-resampled trajectories.
- Baseline curves are `truesmile` and `polite` prototypes under methodA and methodB.
- Target curves are all class prototypes and the previous nearest-6 real sequences per class.
- Target stages are `5%, 10%, 15%, ..., 100%`.
- Baseline search uses a 1% interpolation grid from `0%` to `100%`.

Coordinate definition:

For target curve `C_2` and baseline curve `C_baseline`, find:

```text
tau*(t) = argmin_tau || C_2(t) - C_baseline(tau) ||_2
x_new(t) = tau*(t)
y_new(t) = || C_2(t) - C_baseline(tau*(t)) ||_2
```

Important correction:

- The new x-axis is nearest baseline progress.
- The new y-axis is the L2 length of the nearest vector linking `C_2(t)` to the nearest point on `C_baseline`.
- The new coordinate is not the nearest point's old fixed-axis `(s,d)` coordinate.

Static output root:

```text
E:\Matsuda_data\3-10meeting\nearest_baseline_curve
```

Static outputs verified:

- CSV count: 7
- Plot count: 25 PNG
- Report: `E:\Matsuda_data\3-10meeting\nearest_baseline_curve\report\nearest_baseline_curve_report.html`
- Prototype rows: 240
- Nearest-6 rows: 1440

Main CSV outputs:

```text
E:\Matsuda_data\3-10meeting\nearest_baseline_curve\csv\prototype_nearest_baseline_curve_all.csv
E:\Matsuda_data\3-10meeting\nearest_baseline_curve\csv\nearest6_nearest_baseline_curve_all.csv
E:\Matsuda_data\3-10meeting\nearest_baseline_curve\csv\endpoint_100_summary.csv
```

Endpoint summary at target stage `100%`:

| Method | Baseline | Target | Source | Nearest progress at 100% | Distance at 100% |
|---|---|---|---|---:|---:|
| methodA | truesmile | polite | prototype | 14.000 | 0.200 |
| methodA | truesmile | polite | nearest6 mean | 6.000 | 0.392 |
| methodA | truesmile | truesmile | prototype | 100.000 | 0.000 |
| methodA | truesmile | truesmile | nearest6 mean | 84.833 | 0.546 |
| methodA | truesmile | ambiguous | prototype | 18.000 | 0.207 |
| methodA | truesmile | ambiguous | nearest6 mean | 8.667 | 0.396 |
| methodA | polite | polite | prototype | 100.000 | 0.000 |
| methodA | polite | polite | nearest6 mean | 63.333 | 0.381 |
| methodA | polite | truesmile | prototype | 97.000 | 0.420 |
| methodA | polite | truesmile | nearest6 mean | 71.333 | 0.687 |
| methodA | polite | ambiguous | prototype | 98.000 | 0.166 |
| methodA | polite | ambiguous | nearest6 mean | 46.333 | 0.379 |
| methodB | truesmile | polite | prototype | 1.000 | 0.358 |
| methodB | truesmile | polite | nearest6 mean | 2.833 | 0.393 |
| methodB | truesmile | truesmile | prototype | 100.000 | 0.000 |
| methodB | truesmile | truesmile | nearest6 mean | 55.667 | 0.570 |
| methodB | truesmile | ambiguous | prototype | 4.000 | 0.322 |
| methodB | truesmile | ambiguous | nearest6 mean | 4.167 | 0.404 |
| methodB | polite | polite | prototype | 100.000 | 0.000 |
| methodB | polite | polite | nearest6 mean | 35.500 | 0.330 |
| methodB | polite | truesmile | prototype | 93.000 | 0.621 |
| methodB | polite | truesmile | nearest6 mean | 43.167 | 0.712 |
| methodB | polite | ambiguous | prototype | 17.000 | 0.321 |
| methodB | polite | ambiguous | nearest6 mean | 8.500 | 0.398 |

Current interpretation:

- With `truesmile` as baseline, the `polite` endpoint is closest to a very early true-smile baseline stage: methodA prototype `14%`, methodB prototype `1%`, nearest-6 means around `6%` and `2.833%`.
- This argues against saying polite smile is simply a weakened true smile that progresses along the full true-smile pathway.
- The safer wording is that polite-smile endpoints are geometrically closest to early true-smile baseline stages, but still have nonzero distance from that baseline.
- With `polite` as baseline, true-smile prototypes can map to late polite progress (`97%` or `93%`) but with relatively large distances (`0.420` or `0.621`), so high nearest progress alone is not enough to claim close similarity.
- `ambiguous` is method-sensitive in this view, especially when using polite as baseline. Treat ambiguous conclusions as provisional.

Interactive output root:

```text
E:\Matsuda_data\3-10meeting\nearest_baseline_curve_interactive
```

Interactive report:

```text
E:\Matsuda_data\3-10meeting\nearest_baseline_curve_interactive\interactive_nearest6_curves.html
```

Interactive report status:

- Standalone HTML; no external CDN or image dependency.
- Contains 12 interactive SVG charts.
- Contains 252 hoverable curves.
- Each curve has a transparent hit path for easier selection.
- Hovering a curve highlights it and shows `method / baseline / class / seq / rank`.
- Three top filter checkboxes control visible target classes: `polite`, `truesmile`, and `ambiguous`.

Portable Pro-review package:

```text
E:\Matsuda_data\3-10meeting\nearest_baseline_for_pro_20260602_111141.zip
```

Package contents:

- Explanation Markdown document.
- Static HTML report with adjusted local image paths.
- 25 PNG plots.
- 3 key CSV files.
- Nearest-baseline analysis code.

Cautions:

- This is still descriptive analysis, not a statistical test.
- Distance is unnormalized high-dimensional fc7 L2 distance.
- The nearest point is searched on a 1% interpolation grid, not solved as a continuous optimization problem.
- Connected new-curve lines are connected in target-stage order; they should not be read as actual movement along the baseline.
- If a class's nearest progress jumps backward and forward, that indicates unstable temporal correspondence and should be summarized quantitatively.

## 15. Heatmap Analysis

Folder:

```text
E:\Single_frame_smile\Analysis\Heatmap
```

Important files:

- `requirements.md`
- `source.dat`
- `发现md`
- `codes\heatmap_pipeline.py`
- `codes\heatmap_multilayer_pipeline.py`
- `codes\build_still_images.py`
- `codes\plot_heatmap_threshold_metrics.py`
- `codes\plot_heatmap_threshold_metrics_075.py`
- `codes\requirements2.md`

Current output root:

```text
E:\Single_frame_smile\Analysis\Heatmap\output
```

Verified current `summary.json`:

- model path: `E:\Single_frame_smile\data\models\vggface.pth`
- output dir: `E:\Single_frame_smile\Analysis\Heatmap\output`
- target layer: `conv5_3_by_project_convention_maxp_5_3`
- alpha: `0.4`
- interpolation: `bilinear`
- colormap: `turbo`
- device: `cuda:0`

Selected source sequences:

- `polite`: `E:\Matsuda_data\2-18meeting\polite\13`, 45 frames
- `truesmile`: `E:\Matsuda_data\2-18meeting\truesmile\3`, 60 frames
- `ambiguous`: `E:\Matsuda_data\2-18meeting\ambiguous\27`, 40 frames

Current output counts by top-level output directory:

- `polite`: 180 files, including 135 PNG and 45 NPY
- `truesmile`: 240 files, including 180 PNG and 60 NPY
- `ambiguous`: 160 files, including 120 PNG and 40 NPY
- `still_images`: 10 PNG summary images

What the first heatmap pipeline did:

- For every PNG in the three selected sequences:
  - copy/save original image
  - extract high-level conv response
  - aggregate spatial heatmap
  - save heatmap PNG
  - save overlay PNG
  - save heatmap NPY

Important interpretation:

- These are high-level convolution response visualizations, not eye-tracking maps and not causal classification explanations.
- The current output uses one selected layer by project convention, not the full multilayer/aggregation v2 output.

Multilayer heatmap code status:

- `codes\heatmap_multilayer_pipeline.py` exists.
- `codes\requirements2.md` defines the extended feature-difference heatmap plan:
  - layers: `maxp_5_3`, `relu_5_3`, `relu_4_3`
  - aggregations: A, B, C, D
  - raw heatmap time-series plots
- Verified output root does not contain `summary_v2.json`, so the multilayer v2 output is not verified as completed in the current output directory.

Notes from `发现md` and visual inspection:

- The heatmap often emphasizes face center, brow/nose/nasolabial regions, and broader facial response areas.
- It does not simply focus only on mouth corners.
- Coarse 7x7 spatial resolution can create blocky or cross-like visual artifacts.
- Heatmap area/intensity appears related to smile intensity and can grow then shrink over time.

## 16. Legacy Feature Extractor Experiments

Folder:

```text
E:\Single_frame_smile\Analysis\feature_extractor_old
```

Files:

- `VGG16_feature_extractor.py`
- `ResNet_18_feature_extractor.py`
- `ResNet_50_feature_extractor.py`
- `PCA.py`
- `t-SNE.py`

Current role:

- Legacy experiments for extracting generic visual features and plotting PCA/t-SNE.
- Not part of the current canonical smile trajectory pipeline.

## 17. Main Research Conclusions So Far

Verified/inferred from the current reports:

1. Full-sequence trajectory analysis is the current main direction, not single-frame or start/peak-only analysis.
2. `polite`, `truesmile`, and `ambiguous` occupy different dynamic paths in VGG-Face feature space.
3. Early/neutral phases are more similar; class differences become larger during smile unfolding.
4. `ambiguous` is repeatedly closer to `polite` than to `truesmile`.
5. `truesmile` advances most strongly along the true-smile axis.
6. `polite` and `ambiguous` have limited true-smile-axis progress and meaningful off-axis behavior.
7. Polite-axis analysis suggests polite smile has its own dynamic direction.
8. DTW confirms the `ambiguous` closer-to-`polite` result across magnitude, velocity, and PCA trajectory branches.
9. Synchronized minimum distance also supports early-stage similarity and `ambiguous`/`polite` closeness.
10. Straight-line first-to-last axes are useful summaries but too simple for fully curved trajectories.
11. All-sequence s-d plots show substantial overlap/mixing between classes, so they do not strongly support a class-level claim that the coarse spatial directions are distinct.

## 18. Known Limitations

1. Sample imbalance:
   - `truesmile` has only 6 sequences.
   - `polite` has 41 and `ambiguous` has 27.
   - This affects prototype robustness and class-level statistics.

2. Single-axis simplification:
   - True-smile and polite axes are first-to-last vectors.
   - Real smile trajectories can be curved, so along/off-axis metrics are incomplete summaries.
   - When all individual sequences are plotted in s-d space, trajectories overlap heavily, so the axis view is not sufficient evidence for separated class-level directions.

3. Feature scale and direction are mixed:
   - Distances and projections use raw or baseline-relative feature vectors.
   - They represent combined magnitude/direction differences, not pure angular similarity.

4. Method A and sample-level statistics are not directly equivalent:
   - Method A median prototype is built in high-dimensional feature space first.
   - Sample mean/median bands are computed after nonlinear projection/deviation metrics.
   - Therefore prototype curves can sit outside intuitive sample bands.

5. Heatmap interpretation:
   - Current heatmaps show convolutional response distribution.
   - They should not be described as proof of causal model decision regions.

6. Background/RVM domain shift:
   - Background preprocessing can alter fc7 representations.
   - Keep feature source consistent when comparing trajectories.

## 19. Recommended Next Steps

Most useful next analyses:

1. Compare linear-resampled 20-point results versus DTW-resampled 20-point results in one unified report.
2. Move beyond a single straight axis by using trajectory-to-trajectory or subspace distance.
3. Quantify whether polite and true smile form separate dynamic directions using angle/subspace analyses.
4. Add statistics that account for class imbalance, especially small `truesmile` sample size.
5. Complete or verify the multilayer feature-difference heatmap pipeline if spatial interpretation remains important.
6. Keep one canonical output root per analysis to avoid mixing verification output with current production output.
7. Treat the all-sequence s-d plot as a negative/weak-evidence check; if direction separation is still a research question, test it with explicit distributional statistics rather than visual prototype curves.
8. For nearest-baseline curves, add summary metrics: endpoint nearest progress, maximum nearest progress, monotonicity score, number of backward jumps, mean nearest distance, normalized distance by baseline endpoint norm, and normalized distance by baseline arc length.
9. Consider repeating the nearest-baseline analysis on DTW-resampled trajectories after the linear-aligned result is reviewed.

## 20. Safe Commands

Read-only inspection:

```powershell
rg --files E:\Single_frame_smile\Analysis
Get-ChildItem -Recurse E:\Single_frame_smile\Analysis
Get-Content -Raw -Encoding UTF8 E:\Single_frame_smile\Analysis\analysis_projection\work_summary_report.md
```

Main pipeline rerun:

```powershell
cd E:\Single_frame_smile\Analysis\analysis_sequence
powershell -ExecutionPolicy Bypass -File .\run_pipeline.ps1
```

Projection pipeline rerun:

```powershell
cd E:\Single_frame_smile\Analysis\analysis_projection
powershell -ExecutionPolicy Bypass -File .\run_projection_pipeline.ps1
```

Linear-aligned s-d rerun:

```powershell
cd E:\Single_frame_smile\Analysis\analysis_projection
python .\run_linear_axis_extension.py
python .\run_linear_axis_extension.py --method methodA
```

Nearest-baseline curve rerun:

```powershell
cd E:\Single_frame_smile\Analysis\analysis_projection
python .\run_nearest_baseline_curve_analysis.py
python .\run_nearest_baseline_interactive_report.py
```

DTW pipeline rerun:

```powershell
cd E:\Single_frame_smile\Analysis\DTW
python .\run_dtw_pipeline.py
```

DTW resample rerun:

```powershell
cd E:\Single_frame_smile\Analysis\DTW_resample
python .\run_dtw_resample_pipeline.py
python .\run_dtw_projection_followup.py
python .\run_dtw_axis_extension.py
```

Synchronized minimum rerun:

```powershell
cd E:\Single_frame_smile\Analysis\analysis_minimum
python .\run_minimum_pipeline.py
```

Do not rerun these blindly if the user only asks for explanation. They regenerate outputs under `E:\Matsuda_data`.

## 21. Files Most Likely Needed by the Next Agent

For understanding the whole project:

- `E:\Single_frame_smile\Analysis\analysis_sequence\requirements.md`
- `E:\Single_frame_smile\Analysis\analysis_sequence\coding_explanation_en.md`
- `E:\Single_frame_smile\Analysis\analysis_projection\projection_analysis_en.md`
- `E:\Single_frame_smile\Analysis\analysis_projection\work_summary_report.md`
- `E:\Single_frame_smile\Analysis\DTW\dtw_requirements_cn.md`
- `E:\Single_frame_smile\Analysis\DTW_resample\todo.md`
- `E:\Single_frame_smile\Analysis\analysis_minimum\minimum_requirements_cn.md`
- `E:\Single_frame_smile\Analysis\Heatmap\requirements.md`
- `E:\Single_frame_smile\Analysis\Heatmap\codes\requirements2.md`

For current results:

- `E:\Matsuda_data\2-27meeting\csv\dataset_report.csv`
- `E:\Matsuda_data\2-27meeting\prototypes\prototype_meta.json`
- `E:\Matsuda_data\3-10meeting\methodA\report\projection_summary_methodA.md`
- `E:\Matsuda_data\3-10meeting\methodB\report\projection_summary_methodB.md`
- `E:\Matsuda_data\3-10meeting\methodA\report\polite_axis_summary_methodA.md`
- `E:\Matsuda_data\3-10meeting\methodB\report\polite_axis_summary_methodB.md`
- `E:\Matsuda_data\3-10meeting\linear_axis_extension\report\linear_axis_extension_summary_methodA.md`
- `E:\Matsuda_data\3-10meeting\linear_axis_extension\report\linear_axis_extension_summary_methodB.md`
- `E:\Matsuda_data\DTW_analysis\report\dtw_result_report.md`
- `E:\Matsuda_data\DTW_resample_output\report\dtw_resample_summary.md`
- `E:\Matsuda_data\DTW_resample_output\projection_followup\report\dtw_projection_followup_summary.md`
- `E:\Matsuda_data\DTW_resample_output\axis_extension\report\axis_extension_summary.md`
- `E:\Matsuda_data\analysis_minimum_output\report\sync_min_distance_summary.md`
- `E:\Single_frame_smile\Analysis\Heatmap\output\summary.json`
- `E:\Single_frame_smile\Analysis\documents\nearest_baseline_curve_explanation_for_pro.md`
- `E:\Matsuda_data\3-10meeting\nearest_baseline_curve\report\nearest_baseline_curve_report.html`
- `E:\Matsuda_data\3-10meeting\nearest_baseline_curve\csv\endpoint_100_summary.csv`
- `E:\Matsuda_data\3-10meeting\nearest_baseline_curve_interactive\interactive_nearest6_curves.html`

For DTW-resampled s-d plot specifically:

- Code: `E:\Single_frame_smile\Analysis\DTW_resample\run_dtw_axis_extension.py`
- Function: `plot_s_d`
- All-sequence function: `plot_s_d_all_sequences`
- Balanced nearest-representative function: `plot_s_d_nearest_sequences`
- Outputs:
  - `E:\Matsuda_data\DTW_resample_output\axis_extension\plots\s_d_axis_truesmile.png`
  - `E:\Matsuda_data\DTW_resample_output\axis_extension\plots\s_d_axis_polite.png`
  - `E:\Matsuda_data\DTW_resample_output\axis_extension\plots\s_d_all_sequences_axis_truesmile.png`
  - `E:\Matsuda_data\DTW_resample_output\axis_extension\plots\s_d_all_sequences_axis_polite.png`
  - `E:\Matsuda_data\DTW_resample_output\axis_extension\plots\s_d_nearest6_axis_truesmile.png`
  - `E:\Matsuda_data\DTW_resample_output\axis_extension\plots\s_d_nearest6_axis_polite.png`
  - `E:\Matsuda_data\DTW_resample_output\axis_extension\csv\nearest6_to_representative_sequences.csv`

For non-DTW linear-aligned s-d plot specifically:

- Code: `E:\Single_frame_smile\Analysis\analysis_projection\run_linear_axis_extension.py`
- Main outputs:
  - `E:\Matsuda_data\3-10meeting\linear_axis_extension\plots\s_d_nearest6_axis_truesmile_methodB.png`
  - `E:\Matsuda_data\3-10meeting\linear_axis_extension\plots\s_d_nearest6_axis_polite_methodB.png`
  - `E:\Matsuda_data\3-10meeting\linear_axis_extension\plots\s_d_all_sequences_axis_truesmile_methodB.png`
  - `E:\Matsuda_data\3-10meeting\linear_axis_extension\plots\s_d_all_sequences_axis_polite_methodB.png`
  - `E:\Matsuda_data\3-10meeting\linear_axis_extension\csv\nearest6_to_prototype_sequences_methodB.csv`

For nearest-baseline curve analysis specifically:

- Static analysis code: `E:\Single_frame_smile\Analysis\analysis_projection\run_nearest_baseline_curve_analysis.py`
- Interactive report code: `E:\Single_frame_smile\Analysis\analysis_projection\run_nearest_baseline_interactive_report.py`
- Explanation for Pro review: `E:\Single_frame_smile\Analysis\documents\nearest_baseline_curve_explanation_for_pro.md`
- Static output root: `E:\Matsuda_data\3-10meeting\nearest_baseline_curve`
- Interactive HTML: `E:\Matsuda_data\3-10meeting\nearest_baseline_curve_interactive\interactive_nearest6_curves.html`
- Portable package: `E:\Matsuda_data\3-10meeting\nearest_baseline_for_pro_20260602_111141.zip`

## 22. Verification Performed for This Handoff

Verified:

- `Analysis/documents` was empty before adding this file.
- Existing top-level `Analysis` folders were listed.
- `git status --short` was checked; unrelated dirty/untracked files existed before this document edit.
- Main feature coverage counts matched across input, extracted, baseline-aligned, and normalized sequence outputs.
- `prototype_meta.json` was read and medoid IDs were recorded.
- Current output root file counts and sizes were checked.
- Projection, polite-axis, DTW, DTW-resample, axis-extension, and synchronized-minimum summary reports were read.
- Heatmap `summary.json` was read; `summary_v2.json` was not present.
- `run_linear_axis_extension.py` was run for `methodA` and `methodB`.
- Several generated linear-aligned s-d plots were opened for visual sanity checks.
- On 2026-06-02, `run_nearest_baseline_curve_analysis.py` passed `python -m py_compile`.
- On 2026-06-02, `run_nearest_baseline_curve_analysis.py` was run and produced 240 prototype rows, 1440 nearest-6 rows, 25 plots, and the static HTML report.
- On 2026-06-02, static nearest-baseline HTML image references were checked: 25 image references, 0 missing.
- On 2026-06-02, representative nearest-baseline plots were visually opened for sanity checks.
- On 2026-06-02, `nearest_baseline_curve_explanation_for_pro.md` was created and read back.
- On 2026-06-02, `nearest_baseline_for_pro_20260602_111141.zip` was created; it contains the explanation document, static HTML, 25 plots, 3 CSV files, and the code file.
- On 2026-06-02, `run_nearest_baseline_interactive_report.py` passed `python -m py_compile`.
- On 2026-06-02, `run_nearest_baseline_interactive_report.py` was run and produced `interactive_nearest6_curves.html`.
- On 2026-06-02, the interactive HTML was statically checked: 12 chart cards, 252 curve groups, 252 `data-target-class` attributes, 3 default-checked class filters, and hover handlers present.
- On 2026-06-02, automatic in-app browser verification for the local `file://` interactive HTML was blocked by browser URL policy; static HTML/JS structure checks were used instead.

Not verified:

- No full raw-image-to-feature pipeline was rerun.
- No visual quality review was done for every generated plot.
- No code tests were run.
- Nearest-baseline analysis has not yet been rerun on DTW-resampled trajectories.
- No statistical hypothesis test has been added for nearest-baseline progress or distance metrics.
