# Projection Analysis Requirements (English)

## 1. Goal

Based on the existing outputs from `analysis_sequence`, add a new projection analysis centered on a "true-smile axis" to answer two questions:

1. How far does each smile category move along the true-smile axis over time?
2. How far does each smile category deviate away from the true-smile axis over time?

This analysis must support both prototype definitions:

- Method A: median prototype
- Method B: medoid prototype

Both methods must be treated as parallel primary analyses. For Method B, because the prototype corresponds to a real sequence, the plots should preserve the real `sequence_id` and support visual linkage to the actual frames whenever possible.

---

## 2. What Already Exists

The directory `E:\Single_frame_smile\Analysis\analysis_sequence` has already defined or produced the following components:

1. Input data structure  
   The source dataset is organized as:

   ```text
   E:\Matsuda_data\2-18meeting\
   ├── polite\
   ├── truesmile\
   └── ambiguous\
   ```

   Each sequence folder contains frame-wise images for one smile clip.

2. Feature extraction  
   For each frame, VGG-Face `fc7` features are extracted:

   ```text
   f(t) ∈ R^D, D = 4096
   ```

3. Baseline alignment  
   The baseline is defined as the mean of the first five frames:

   ```text
   f0 = mean(f(0), f(1), f(2), f(3), f(4))
   f_rel(t) = f(t) - f0
   ```

4. Time normalization  
   Each `f_rel` sequence is linearly resampled to 20 time points:

   ```text
   f_norm(t), t = 0, 1, ..., 19
   ```

5. Prototype construction  
   Two prototype definitions already exist:

   - Method A: median trajectory
   - Method B: medoid trajectory

6. Existing outputs  
   The current pipeline has already defined or produced:

   - `metrics/sequence_features/...`
   - `metrics/sequence_features_rel/...`
   - `metrics/normalized/...`
   - `metrics/normalized_frames/...`
   - `prototypes/prototype_<class>.npy`
   - `prototypes/prototype_<class>_medoid.npy`
   - `plots/...`
   - `csv/...`

---

## 3. Core Idea of the New Analysis

### 3.1 True-Smile Axis

Under one prototype definition, let the prototype of the true-smile class be:

```text
p_true(t), t = 0, 1, ..., 19
```

Define the true-smile axis as the line from the first to the last prototype point:

```text
g = p_true(19) - p_true(0)
```

Define the unit direction vector as:

```text
u = g / ||g||
```

Notes:

- `g` is not the full true-smile trajectory. It is a first-order semantic axis approximation.
- This analysis explicitly accepts this first-to-last-line approximation.

### 3.2 Initial Bias

Because each class uses its own `f0` for baseline alignment, the start points in `f_rel` space must not be interpreted as sharing one common origin.

Therefore, the initial bias must be defined separately as the difference between raw baseline vectors, instead of being mixed into the dynamic projection analysis.

For class `c`, define its baseline prototype as the representative baseline vector of that class:

- Method A: baseline median
- Method B: the baseline of the medoid sequence

Denote it as:

```text
b_c ∈ R^D
```

The initial bias relative to true smile is then:

```text
offset_c = || b_c - b_true ||_2
```

This quantity describes the static difference before or at the start of the smile and should not be mixed into Question A / Question B.

---

## 4. Dynamic Projection Analysis Definitions

For the prototype of any class `c`:

```text
p_c(t), t = 0, 1, ..., 19
```

Define the dynamic vector relative to its own starting point:

```text
d_c(t) = p_c(t) - p_c(0)
```

Therefore:

```text
d_c(0) = 0
```

This is intentional. The goal is to analyze only the dynamics after the class-specific starting point.

### 4.1 Question A: How far does it move along the true-smile axis?

Define the projection length of class `c` at time `t` onto the true-smile axis as:

```text
a_c(t) = < d_c(t), u >
```

where `<·,·>` denotes the inner product.

To obtain a normalized ratio, define:

```text
ratio_along_c(t) = a_c(t) / ||g||
```

Interpretation:

- `ratio_along_c(t) = 0`: still at its own starting point
- `ratio_along_c(t) = 1`: has advanced along the axis by the same amount as the full true-smile start-to-end length
- `ratio_along_c(t) > 1`: exceeds the true-smile end position along that axis
- `ratio_along_c(t) < 0`: moves in the opposite direction

### 4.2 Question B: How far does it deviate from the true-smile axis?

First define the projection vector of `d_c(t)` onto the true-smile axis:

```text
proj_c(t) = proj_g(d_c(t))
```

Then define the residual deviation vector:

```text
r_c(t) = d_c(t) - proj_c(t)
```

The off-axis distance is:

```text
dist_off_c(t) = || r_c(t) ||_2
```

The normalized off-axis ratio is:

```text
ratio_off_c(t) = dist_off_c(t) / ||g||
```

Interpretation:

- Smaller `ratio_off_c(t)` means the class is closer to the true-smile axis at that time point.
- Larger `ratio_off_c(t)` means the class may still be moving, but its path deviates more strongly from the true-smile axis.

---

## 5. Why Initial Bias and Dynamic Deviation Must Be Separate

This analysis explicitly splits the interpretation into two layers:

1. Initial bias  
   Compare each class baseline `f0` against the true-smile baseline.

2. Dynamic projection  
   Compare how each class, starting from its own origin, advances along and deviates from the true-smile axis.

This split is required because:

- Directly comparing absolute points across classes in `f_rel` space would mix incompatible baseline reference frames.
- With `d_c(t) = p_c(t) - p_c(0)`, the value at `t=0` is always zero, which is desirable for dynamic-shape analysis.
- Initial differences are meaningful, but they should be reported separately rather than embedded into one dynamic metric.

---

## 6. Parallel Analysis Requirements for Method A and Method B

### 6.1 Method A

Use the median prototype:

```text
p_c^A(t)
```

Then compute:

- `g^A`
- `u^A`
- `offset_c^A`
- `ratio_along_c^A(t)`
- `ratio_off_c^A(t)`

### 6.2 Method B

Use the medoid prototype:

```text
p_c^B(t)
```

Then compute:

- `g^B`
- `u^B`
- `offset_c^B`
- `ratio_along_c^B(t)`
- `ratio_off_c^B(t)`

Additional Method B requirements:

- Save the real `sequence_id` of the prototype
- Save the corresponding `normalized_frames`
- Allow the plots to reference or highlight the real prototype frames

---

## 7. Analysis Steps

### Step 1. Read Existing Outputs

Purpose:

- Reuse outputs from `analysis_sequence` and avoid recomputing features.

Inputs:

- `sequence_features.npy`
- `sequence_features_rel.npy`
- `normalized_sequence.npy`
- `sampled_frames.json`
- `prototype_*.npy`
- `prototype_*_medoid.npy`

### Step 2. Build Baseline Prototypes

Purpose:

- Build a representative baseline vector for each class to analyze initial bias.

Definitions:

- Method A: element-wise median across all sample baselines `f0`
- Method B: use the `f0` from the medoid sequence

Outputs:

- Baseline prototype per class
- Baseline offset relative to true smile

### Step 3. Build the True-Smile Axis

Purpose:

- Define the reference direction used for all projection analyses.

Definitions:

```text
g = p_true(19) - p_true(0)
u = g / ||g||
```

Notes:

- Method A and Method B each build their own `g`
- If `||g||` is extremely small, the analysis must fail clearly or mark the result as invalid

### Step 4. Compute Along-Axis Progress for Each Class

Purpose:

- Measure how far each class advances along the true-smile direction over time.

Definitions:

```text
d_c(t) = p_c(t) - p_c(0)
a_c(t) = < d_c(t), u >
ratio_along_c(t) = a_c(t) / ||g||
```

Outputs:

- Projection length at all 20 time points
- Normalized progress ratio at all 20 time points

### Step 5. Compute Off-Axis Deviation for Each Class

Purpose:

- Measure how far each class deviates away from the true-smile axis over time.

Definitions:

```text
proj_c(t) = proj_g(d_c(t))
r_c(t) = d_c(t) - proj_c(t)
dist_off_c(t) = ||r_c(t)||_2
ratio_off_c(t) = dist_off_c(t) / ||g||
```

Outputs:

- Absolute off-axis distance at all 20 time points
- Normalized off-axis ratio at all 20 time points

### Step 6. Optional Per-Sequence Supplementary Analysis

Purpose:

- Compare not only class prototypes, but also within-class distributions.

Method:

- Apply the same formulas to every normalized sample sequence `f_norm_i(t)`:
  - `ratio_along_i(t)`
  - `ratio_off_i(t)`

Uses:

- Compute class mean and standard deviation
- Inspect within-class dispersion
- Check whether the prototype reflects the overall class trend

### Step 7. Generate Plots and Summary Tables

Purpose:

- Produce outputs suitable for interpretation and reporting.

---

## 8. Required Outputs

Recommended output root:

```text
E:\Matsuda_data\projection_analysis\
```

Recommended subfolders:

```text
prototypes\
csv\
plots\
report\
```

### 8.1 Prototype Metadata

Recommended files:

- `prototypes\projection_meta_methodA.json`
- `prototypes\projection_meta_methodB.json`

Suggested content:

- prototype method
- class list
- time length `T=20`
- feature dimension `D=4096`
- true-smile axis length `||g||`
- real `sequence_id` for Method B

### 8.2 Initial Bias Results

Recommended files:

- `csv\baseline_offsets_methodA.csv`
- `csv\baseline_offsets_methodB.csv`

Suggested columns:

```text
method
class
offset_to_truesmile
baseline_norm
sequence_id   # used by Method B, optional for Method A
```

### 8.3 Dynamic Projection Results

Recommended files:

- `csv\projection_along_methodA.csv`
- `csv\projection_along_methodB.csv`
- `csv\projection_off_methodA.csv`
- `csv\projection_off_methodB.csv`

Suggested columns:

```text
method
class
time_index
projection_length
projection_ratio
off_axis_distance
off_axis_ratio
```

### 8.4 Per-Sequence Supplementary Results

Recommended files:

- `csv\projection_per_sequence_methodA.csv`
- `csv\projection_per_sequence_methodB.csv`

Suggested columns:

```text
method
class
sequence_id
time_index
projection_ratio
off_axis_ratio
```

### 8.5 Summary Reports

Recommended files:

- `report\projection_summary_methodA.md`
- `report\projection_summary_methodB.md`

Suggested content:

- ranking of initial biases
- overall trend of progress along the true-smile axis
- overall trend of deviation away from the true-smile axis
- similarities and differences between Method A and Method B

---

## 9. Plot Requirements

### Plot 1. Baseline Initial-Bias Bar Chart

Suggested files:

- `plots\baseline_offsets_methodA.png`
- `plots\baseline_offsets_methodB.png`

Content:

- x-axis: class
- y-axis: `offset_c`

Purpose:

- Report baseline differences separately from dynamic analysis

### Plot 2. Prototype Along-Axis Progress Curves

Suggested files:

- `plots\projection_along_methodA.png`
- `plots\projection_along_methodB.png`

Content:

- x-axis: `time_index = 0..19`
- y-axis: `ratio_along_c(t)`
- all three class curves on the same plot

Purpose:

- Compare how fast and how far each class advances along the true-smile axis

### Plot 3. Prototype Off-Axis Deviation Curves

Suggested files:

- `plots\projection_off_methodA.png`
- `plots\projection_off_methodB.png`

Content:

- x-axis: `time_index = 0..19`
- y-axis: `ratio_off_c(t)`
- all three class curves on the same plot

Purpose:

- Show when each class starts to deviate clearly from the true-smile axis

### Plot 4. Along-vs-Off Phase Plot

Suggested files:

- `plots\projection_phase_methodA.png`
- `plots\projection_phase_methodB.png`

Content:

- x-axis: `ratio_along_c(t)`
- y-axis: `ratio_off_c(t)`
- each class forms a time trajectory in this 2D plane

Purpose:

- Visualize progress and deviation jointly

### Plot 5. Per-Sequence Confidence-Band Plots (Recommended)

Suggested files:

- `plots\projection_along_band_methodA.png`
- `plots\projection_off_band_methodA.png`
- `plots\projection_along_band_methodB.png`
- `plots\projection_off_band_methodB.png`

Content:

- prototype curve
- sample mean curve
- sample standard deviation or quantile band

Purpose:

- Compare the prototype against the class distribution

### Plot 6. Method B Prototype Frame Montage (Important)

Suggested files:

- `plots\prototype_frames_methodB_<class>.png`

Content:

- show the 20 normalized frames of the real sequence selected by Method B
- the title should include `class` and `sequence_id`

Purpose:

- strengthen the interpretability of Method B as a real exemplar

---

## 10. Interpretation Principles

1. A large `ratio_along` does not automatically mean "more true-smile-like"  
   It only means more progress along the true-smile axis.

2. A small `ratio_off` means closer adherence to the true-smile axis  
   It must still be interpreted together with `ratio_along`.

3. A typical "closer to true smile" pattern is:

- larger `ratio_along`
- smaller `ratio_off`

4. Initial bias and dynamic deviation are different layers:

- initial bias answers "how similar are the starting states?"
- dynamic deviation answers "how similar is the trajectory shape?"

5. The true-smile axis is a first-to-last-point approximation, not a guarantee that the full true-smile trajectory is linear.

---

## 11. Final Requirement Summary

This analysis must:

1. Use outputs from `analysis_sequence` as its input foundation.
2. Explicitly separate baseline initial bias from dynamic projection deviation.
3. Use the mean of the first five frames to define `f0`.
4. Use 20 time points with indices `0..19`.
5. Define the axis from the first and last point of the true-smile prototype.
6. Use the following normalized metrics:

   ```text
   ratio_along_c(t) = projection_length / ||g||
   ratio_off_c(t)   = off_axis_distance / ||g||
   ```

7. Treat Method A and Method B as two parallel primary analyses.
8. Preserve the real `sequence_id` for Method B and support linkage to real frames.
