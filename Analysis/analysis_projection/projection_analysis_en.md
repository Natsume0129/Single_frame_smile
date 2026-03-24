# Projection Analysis Requirements (English)

## 1. Goal

Based on the existing outputs from `analysis_sequence`, add a new true-smile-referenced projection analysis and difference analysis.

All outputs must be separated by prototype trajectory type, at least into:

- Method A outputs
- Method B outputs

The analysis should answer five main lines:

1. Direct difference line  
   How large is the difference between a reference class prototype and the other class prototypes at each time point, and how does that difference evolve over time?

2. Along-axis progress line  
   How far does each class move along the true-smile axis over time?

3. Off-axis deviation line  
   How far does each class deviate away from the true-smile axis over time?

4. Curve-to-curve minimum point distance line  
   Given two trajectories, what is the minimum Euclidean distance between any sampled point on the first curve and any sampled point on the second curve?

5. Tangent-relative nearest-point decomposition line  
   For each point on one curve, when the nearest point on another curve is found, how much of that displacement lies along the local tangent direction, and how much lies in the normal residual direction?

This analysis must support both prototype definitions:

- Method A: median trajectory
- Method B: medoid trajectory

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

## 3. Formal Definition of the Prototypes

### 3.1 Basic Notation

For the `i`-th normalized sequence in class `c`, denote:

```text
f_i^c(t), t = 0, 1, ..., 19
```

Each sequence has shape:

```text
[20, D]
```

### 3.2 Method A: Median Trajectory

Method A does not choose one real sequence. Instead, it constructs a new representative trajectory by taking the median at every time point and every feature dimension across all normalized sequences in that class.

Definition:

```text
p_c^A(t)_d = median_i(f_i^c(t)_d)
```

Equivalently:

```text
p_c^A(t) = median_i(f_i^c(t))
```

where the median is computed element-wise.

Notes:

- Method A uses all time points `t=0..19`.
- Method A outputs a new prototype trajectory.
- This trajectory usually does not correspond to a real file.

### 3.3 Method B: Medoid Trajectory

Method B selects one real sequence inside the class that is globally the most representative.

Let each normalized sequence matrix be:

```text
F_i^c ∈ R^{20×D}
```

Define the sequence-level distance between any two sequences as:

```text
d(F_i^c, F_j^c) = ||F_i^c - F_j^c||_F
```

where `||·||_F` is the Frobenius norm.

Then the medoid index is:

```text
i_c^* = argmin_i Σ_j d(F_i^c, F_j^c)
```

The prototype is:

```text
p_c^B(t) = f_{i_c^*}^c(t)
```

Notes:

- Method B uses the full sequence-level cost across time.
- Method B outputs a real sequence.
- Method B must preserve the corresponding `sequence_id` and frame mapping.

---

## 4. Three Main Analysis Lines

### 4.1 Main Line 1: Direct Difference Analysis

This line directly compares two prototype vectors at the same time point.

For any two classes `a` and `b`, define the difference at time `t` as:

```text
diff_{a,b}(t) = || p_a(t) - p_b(t) ||_2
```

where:

- `p_a(t) - p_b(t)` is the difference vector at the same time point
- `||·||_2` is the Euclidean norm

Interpretation:

- Smaller `diff_{a,b}(t)` means the two classes are closer at that time point.
- Larger `diff_{a,b}(t)` means the two classes are more different at that time point.
- The value at `t=0` can be directly interpreted as the starting-state difference.
- This curve can answer whether the difference already exists at the beginning, whether it grows later, when it reaches a maximum, and whether it shrinks again.

### 4.2 Main Line 2: How Far Does It Move Along the True-Smile Axis?

Under one prototype setting, let the true-smile prototype be:

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
- This analysis explicitly accepts the first-to-last-line approximation.

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

Define the projection length onto the true-smile axis as:

```text
a_c(t) = < d_c(t), u >
```

The normalized progress ratio is:

```text
ratio_along_c(t) = a_c(t) / ||g||
```

Interpretation:

- `ratio_along_c(t) = 0`: still at its own start point
- `ratio_along_c(t) = 1`: has progressed along the axis by the same amount as the full true-smile start-to-end length
- `ratio_along_c(t) > 1`: exceeds the true-smile end position along that axis
- `ratio_along_c(t) < 0`: moves in the opposite direction

### 4.3 Main Line 3: How Far Does It Deviate from the True-Smile Axis?

First define the projection vector of `d_c(t)` onto the true-smile axis:

```text
proj_c(t) = proj_g(d_c(t))
```

Then define the deviation vector:

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

- Smaller `ratio_off_c(t)` means the class stays closer to the true-smile axis.
- Larger `ratio_off_c(t)` means the class may still be moving, but its path deviates more strongly from the true-smile axis.

### 4.4 Main Line 4: Curve-to-Curve Minimum Point Distance

This line compares two trajectories using their original, pre-resampling sampled points.

Important requirement:

- This analysis should use the sequence data **before time normalization / resampling**
- The reason is that we want the true minimum among real sampled points
- This also allows us to recover the corresponding original frames directly

For two curves:

```text
C1(t1), t1 = 0, 1, ..., T1-1
C2(t2), t2 = 0, 1, ..., T2-1
```

define:

```text
d_min(C1, C2) = min_{t1, t2} || C1(t1) - C2(t2) ||_2
```

and the corresponding minimum point pair:

```text
(t1*, t2*) = argmin_{t1, t2} || C1(t1) - C2(t2) ||_2
```

Interpretation:

- This tells us the closest approach between two trajectories
- It also tells us at which two real frames this closest approach happens

### 4.5 Main Line 5: Tangent-Relative Nearest-Point Decomposition

This line should also use the original, pre-resampling sampled points.

Important requirement:

- This analysis should use the sequence data **before time normalization / resampling**
- The reason is that resampling makes the curve more discrete and may distort the local tangent estimate

For a point `C1(t)` on curve `C1`, define the local tangent vector using neighboring original sampled points.

Recommended discrete definition:

```text
tau(t) = C1(t+1) - C1(t-1)      for interior points
tau(0) = C1(1) - C1(0)
tau(T1-1) = C1(T1-1) - C1(T1-2)
```

and the unit tangent direction:

```text
u_tan(t) = tau(t) / ||tau(t)||
```

Then, for each `C1(t)`, first find the nearest point on `C2`:

```text
t2*(t) = argmin_{t2} || C2(t2) - C1(t) ||_2
```

Define the displacement vector:

```text
d(t) = C2(t2*(t)) - C1(t)
```

Then decompose this displacement into:

- tangent-aligned component
- normal residual component

Definitions:

```text
d_parallel(t) = <d(t), u_tan(t)> u_tan(t)
d_normal(t) = d(t) - d_parallel(t)
```

and the corresponding scalar magnitudes:

```text
dist_parallel(t) = ||d_parallel(t)||_2
dist_normal(t) = ||d_normal(t)||_2
dist_total(t) = ||d(t)||_2
```

Interpretation:

- `dist_total(t)` tells us how close `C2` gets to `C1(t)`
- `dist_parallel(t)` tells us how much of that nearest-point displacement is along the local direction of `C1`
- `dist_normal(t)` tells us how much of that nearest-point displacement is away from the local direction of `C1`

This avoids the high-dimensional problem of defining one unique normal line. Instead, it uses the tangent direction and the residual decomposition, which remains well-defined in high-dimensional space.

---

## 5. Role of Baseline and Initial Bias

This requirement no longer treats `initial bias` as a primary analysis step. Instead, it is a supplementary interpretation attached to Main Line 1.

Reasons:

1. The class-specific `f0` vectors are naturally different  
   because `f0` is the mean of the first five frames of each sequence, not the same image.

2. In the dynamic projection analysis, we use:

   ```text
   d_c(t) = p_c(t) - p_c(0)
   ```

   This is intentional, because the goal is to analyze the dynamic path after each class-specific starting point. Therefore `d_c(0)=0` is by design.

3. If the research focus is whether the difference already exists at the beginning and how it evolves over time, then the more natural quantity is simply:

   ```text
   diff_{a,b}(0)
   ```

That means:

- The direct difference at `t=0` is the most direct expression of the starting-state difference.
- If needed, baseline `f0` differences can still be reported as supplementary information.
- But `initial bias` is no longer a primary Method A / Method B step.

---

## 6. Parallel Analysis Requirements for Method A and Method B

### 6.1 Method A

Use the median prototype:

```text
p_c^A(t)
```

Then compute:

- `diff_{a,b}^A(t)`
- `g^A`
- `u^A`
- `ratio_along_c^A(t)`
- `ratio_off_c^A(t)`

### 6.2 Method B

Use the medoid prototype:

```text
p_c^B(t)
```

Then compute:

- `diff_{a,b}^B(t)`
- `g^B`
- `u^B`
- `ratio_along_c^B(t)`
- `ratio_off_c^B(t)`

Additional Method B requirements:

- Save the real `sequence_id` of the prototype
- Save the corresponding `normalized_frames`
- Allow the plots to reference or highlight the real prototype frames

### 6.3 Shared Geometry Analyses

Main Line 4 and Main Line 5 are different from Method A / Method B prototype analyses.

They should be treated as **shared geometry analyses** because:

- they are based on original pre-resampling sampled points
- they are intended to preserve real-frame correspondence
- they do not depend on one prototype construction method

So these outputs do not need to be duplicated under both `methodA` and `methodB`.

Recommended organization:

- `shared/csv/...`
- `shared/plots/...`
- `shared/report/...`

### 6.4 Output Organization Requirement

All outputs must be separated by prototype trajectory type. Method A and Method B results must not be mixed into the same output file.

Recommended organization:

- filename suffixes:
  - `..._methodA.*`
  - `..._methodB.*`
- or separate subdirectories:
  - `methodA/...`
  - `methodB/...`

If both are used together, that is also acceptable, but it must always be obvious:

- whether a result belongs to Method A or Method B
- and which prototype trajectory definition it corresponds to

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

### Step 2. Build Trajectory Prototypes

Purpose:

- Derive one representative prototype trajectory from multiple normalized sequences in each class.

Method A:

- Take the median across all samples at every time point and every dimension
- Produce a new `[20, D]` prototype

Method B:

- Use full-sequence distance to compute the medoid
- Select one real sample as the `[20, D]` prototype

Outputs:

- `p_c^A(t)`
- `p_c^B(t)`
- Method B `sequence_id`

### Step 3. Compute Direct Difference Curves

Purpose:

- Directly compare class differences at each time point and track their temporal evolution.

Definition:

```text
diff_{a,b}(t) = || p_a(t) - p_b(t) ||_2
```

Outputs:

- `polite vs truesmile`
- `ambiguous vs truesmile`
- `polite vs ambiguous`

for all `t=0..19`.

### Step 4. Build the True-Smile Axis

Purpose:

- Define the reference direction for all projection analyses.

Definitions:

```text
g = p_true(19) - p_true(0)
u = g / ||g||
```

Notes:

- Method A and Method B each build their own `g`
- If `||g||` is extremely small, the analysis must fail clearly or mark the result as invalid

### Step 5. Compute Along-Axis Progress

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

### Step 6. Compute Off-Axis Deviation

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

### Step 7. Per-Sequence Supplementary Analysis (Required)

Purpose:

- Apply the same analysis not only to prototype trajectories, but also to every normalized sequence.

Method:

- Apply the same formulas to every normalized sample sequence `f_norm_i(t)`:
  - time-wise direct difference
  - `ratio_along_i(t)`
  - `ratio_off_i(t)`

Uses:

- Compute class mean and standard deviation
- Inspect within-class dispersion
- Check whether the prototype reflects the overall class trend
- Perform within-class statistics and between-class statistical comparisons

### Step 8. Compute Curve-to-Curve Minimum Point Distance

Purpose:

- Measure the closest point-pair distance between two original trajectories.
- Preserve the real frame correspondence.

Inputs:

- original sequence features before resampling
- frame-name mapping or frame index mapping

Outputs:

- minimum distance value
- corresponding `(t1*, t2*)`
- corresponding sequence IDs and frame IDs
- corresponding image pair

### Step 9. Compute Tangent-Relative Nearest-Point Decomposition

Purpose:

- For each point on one original curve, measure whether the nearest approach from another curve is mainly along the tangent direction or mainly off that direction.

Inputs:

- original sequence features before resampling
- frame-name mapping or frame index mapping

Outputs:

- `dist_total(t)`
- `dist_parallel(t)`
- `dist_normal(t)`
- nearest point index on the second curve
- corresponding frame references

### Step 10. Generate Plots and Summary Tables

Purpose:

- Produce outputs suitable for interpretation and reporting.

---

## 8. Required Outputs

Recommended output root:

```text
E:\Matsuda_data\projection_analysis\
```

Preferred organization:

```text
E:\Matsuda_data\projection_analysis\
├── methodA\
│   ├── csv\
│   ├── plots\
│   ├── prototypes\
│   └── report\
└── methodB\
    ├── csv\
    ├── plots\
    ├── prototypes\
    └── report\
```

This organization is preferred over mixing all results in one directory.

### 8.1 Prototype Metadata

Recommended files:

- `methodA\prototypes\projection_meta_methodA.json`
- `methodB\prototypes\projection_meta_methodB.json`

Suggested content:

- prototype method
- class list
- time length `T=20`
- feature dimension `D=4096`
- true-smile axis length `||g||`
- real `sequence_id` for Method B

### 8.2 Direct Difference Results

Recommended files:

- `methodA\csv\direct_distance_methodA.csv`
- `methodB\csv\direct_distance_methodB.csv`

Suggested columns:

```text
method
anchor_class
target_class
time_index
difference_norm
```

Notes:

- `anchor_class` is the reference class
- `target_class` is the other class being compared against it

### 8.3 Dynamic Projection Results

Recommended files:

- `methodA\csv\projection_along_methodA.csv`
- `methodB\csv\projection_along_methodB.csv`
- `methodA\csv\projection_off_methodA.csv`
- `methodB\csv\projection_off_methodB.csv`

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

### 8.4 Baseline Supplementary Results (Optional)

Recommended files:

- `methodA\csv\baseline_offset_supplement_methodA.csv`
- `methodB\csv\baseline_offset_supplement_methodB.csv`

Suggested columns:

```text
method
class
baseline_offset_to_truesmile
sequence_id   # used by Method B, optional for Method A
```

Notes:

- This part is supplementary only, not a primary output.

### 8.5 Per-Sequence Results

Recommended files:

- `methodA\csv\projection_per_sequence_methodA.csv`
- `methodB\csv\projection_per_sequence_methodB.csv`
- `methodA\csv\per_sequence_direct_distance_methodA.csv`
- `methodB\csv\per_sequence_direct_distance_methodB.csv`

Suggested columns set 1:

```text
method
class
sequence_id
time_index
projection_ratio
off_axis_ratio
```

Suggested columns set 2:

```text
method
anchor_class
target_class
sequence_id
time_index
difference_norm
```

Notes:

- `anchor_class` is the reference prototype class
- `target_class` is the class of the sequence

### 8.6 Per-Sequence Statistical Summaries

Recommended files:

- `methodA\csv\projection_statistics_methodA.csv`
- `methodB\csv\projection_statistics_methodB.csv`
- `methodA\csv\direct_distance_statistics_methodA.csv`
- `methodB\csv\direct_distance_statistics_methodB.csv`

Suggested columns:

```text
method
metric_type
class
anchor_class
time_index
mean
std
median
q1
q3
```

### 8.7 Shared Geometry Outputs

Recommended files:

- `shared\csv\curve_min_distance.csv`
- `shared\csv\tangent_relative_distance.csv`
- `shared\plots\curve_min_distance_examples.png`
- `shared\plots\tangent_relative_total_distance.png`
- `shared\plots\tangent_relative_parallel_distance.png`
- `shared\plots\tangent_relative_normal_distance.png`
- `shared\report\curve_geometry_summary.md`

Suggested columns for `curve_min_distance.csv`:

```text
curve1_class
curve1_sequence_id
curve1_time_index
curve1_frame_name
curve2_class
curve2_sequence_id
curve2_time_index
curve2_frame_name
min_distance
```

Suggested columns for `tangent_relative_distance.csv`:

```text
curve1_class
curve1_sequence_id
curve1_time_index
curve1_frame_name
curve2_class
curve2_sequence_id
nearest_time_index_on_curve2
nearest_frame_name_on_curve2
dist_total
dist_parallel
dist_normal
```

Notes:

- These outputs should use original pre-resampling sequence points
- These outputs should preserve real frame correspondence

### 8.8 Summary Reports

Recommended files:

- `methodA\report\projection_summary_methodA.md`
- `methodB\report\projection_summary_methodB.md`

Suggested content:

- key findings from the direct-difference curves
- overall trend of progress along the true-smile axis
- overall trend of deviation away from the true-smile axis
- key findings from within-class and between-class statistics
- similarities and differences between Method A and Method B

---

## 9. Plot Requirements

### Plot 1. Anchor-Based Direct-Difference Curves (Required)

Suggested files:

- `methodA\plots\direct_distance_anchor_polite_methodA.png`
- `methodA\plots\direct_distance_anchor_truesmile_methodA.png`
- `methodA\plots\direct_distance_anchor_ambiguous_methodA.png`
- `methodB\plots\direct_distance_anchor_polite_methodB.png`
- `methodB\plots\direct_distance_anchor_truesmile_methodB.png`
- `methodB\plots\direct_distance_anchor_ambiguous_methodB.png`

Content:

- x-axis: `time_index = 0..19`
- y-axis: `difference_norm`
- each plot fixes one anchor class as the reference
- each plot contains two curves, corresponding to the other two classes relative to that anchor class

Purpose:

- observe whether class differences already exist at the beginning or open up later
- identify when the difference is maximal and whether it shrinks afterward

Notes:

- there should be 3 plots for the 3 anchor classes
- Method A and Method B each require a full set

### Plot 2. Projection Along True-Smile Axis Curves (Required)

Suggested files:

- `methodA\plots\projection_along_methodA.png`
- `methodB\plots\projection_along_methodB.png`

Content:

- x-axis: `time_index = 0..19`
- y-axis: percentage or ratio
- three curves corresponding to:
  - polite prototype trajectory
  - truesmile prototype trajectory
  - ambiguous prototype trajectory

Purpose:

- compare how fast and how far each class advances along the true-smile axis

Additional requirement:

- the recommended display range is around `0` to `1`
- if values exceed `1`, they must still be shown without clipping

### Plot 3. Deviation from True-Smile Axis Curves (Required)

Suggested files:

- `methodA\plots\projection_off_methodA.png`
- `methodB\plots\projection_off_methodB.png`

Content:

- x-axis: `time_index = 0..19`
- y-axis: percentage or ratio
- three curves corresponding to:
  - polite prototype trajectory
  - truesmile prototype trajectory
  - ambiguous prototype trajectory

Purpose:

- show when each class starts to deviate clearly from the true-smile axis

Additional requirement:

- the y-axis must remain consistent with `ratio_off_c(t)`

### Plot 4. Along-vs-Off Phase Plot (Recommended)

Suggested files:

- `methodA\plots\projection_phase_methodA.png`
- `methodB\plots\projection_phase_methodB.png`

Content:

- x-axis: `ratio_along_c(t)`
- y-axis: `ratio_off_c(t)`
- each class forms a time trajectory in this 2D plane

Purpose:

- visualize progress and deviation jointly

### Plot 5. Per-Sequence Statistical-Band Plots (Strongly Recommended)

Suggested files:

- `methodA\plots\projection_along_band_methodA.png`
- `methodA\plots\projection_off_band_methodA.png`
- `methodB\plots\projection_along_band_methodB.png`
- `methodB\plots\projection_off_band_methodB.png`

Content:

- prototype curve
- sample mean curve
- sample standard deviation or quantile band

Purpose:

- compare the prototype against the class distribution
- observe whether within-class variance changes over time

### Plot 6. Per-Sequence Direct-Difference Distribution Plots (Recommended)

Suggested files:

- `methodA\plots\direct_distance_band_anchor_polite_methodA.png`
- `methodA\plots\direct_distance_band_anchor_truesmile_methodA.png`
- `methodA\plots\direct_distance_band_anchor_ambiguous_methodA.png`
- `methodB\plots\direct_distance_band_anchor_polite_methodB.png`
- `methodB\plots\direct_distance_band_anchor_truesmile_methodB.png`
- `methodB\plots\direct_distance_band_anchor_ambiguous_methodB.png`

Content:

- x-axis: `time_index = 0..19`
- y-axis: sample-level `difference_norm`
- for one anchor class, show the mean curve and variance band of the other two classes relative to the anchor prototype

Purpose:

- examine not only prototype-level differences, but also distribution-level class separation
- assess whether the class difference is stable or driven by a few samples

### Plot 7. Method B Prototype Frame Montage (Important)

Suggested files:

- `methodB\plots\prototype_frames_methodB_<class>.png`

Content:

- show the 20 normalized frames of the real sequence selected by Method B
- the title should include `class` and `sequence_id`

Purpose:

- strengthen the interpretability of Method B as a real exemplar

### Plot 8. Baseline Supplementary Plot (Optional)

Suggested files:

- `methodA\plots\baseline_offset_supplement_methodA.png`
- `methodB\plots\baseline_offset_supplement_methodB.png`

Purpose:

- only as a supplementary description of starting-state difference, not a primary plot

### Plot 9. Curve-to-Curve Minimum Distance Example Plot (Recommended)

Suggested files:

- `shared\plots\curve_min_distance_examples.png`

Content:

- selected example pairs with the globally nearest two real sampled frames highlighted
- include sequence IDs and frame names

Purpose:

- show the actual visual meaning of the minimum point-to-point distance

### Plot 10. Tangent-Relative Distance Curves (Recommended)

Suggested files:

- `shared\plots\tangent_relative_total_distance.png`
- `shared\plots\tangent_relative_parallel_distance.png`
- `shared\plots\tangent_relative_normal_distance.png`

Content:

- x-axis: original time index of `C1`
- y-axis: distance magnitude
- can be shown for selected class pairs or selected sequences

Purpose:

- distinguish whether nearest approach happens mainly along the local motion direction or mainly away from it

---

## 10. Per-Sequence Statistical Analysis Requirements

In addition to the prototype trajectories, the same analysis must be applied to every normalized sequence.

For each sample-level sequence, the analysis must at least compute:

1. time-wise direct difference relative to an anchor prototype
2. `ratio_along` relative to the true-smile axis
3. `ratio_off` relative to the true-smile axis

Based on sample-level results, the analysis must at least provide:

1. per-class, per-time-point mean
2. per-class, per-time-point standard deviation
3. per-class, per-time-point median
4. per-class, per-time-point interquartile range or confidence band

If feasible, it is recommended to further include:

1. peak-time comparison
2. area-under-curve (AUC) comparison
3. final-time-point comparison
4. between-class statistical tests

The goals of the sample-level statistical analysis are:

- to check whether the prototype conclusions reflect the class-level trend
- to measure within-class dispersion
- to assess whether the between-class differences are statistically stable

The new shared geometry analyses should also support:

- summary statistics over repeated sequence pairs
- selection of representative example pairs for visualization

---

## 11. Interpretation Principles

1. `diff_{a,b}(t)` answers: "how different are the two classes at this time point?"

2. `ratio_along` answers: "does this class move along the true-smile direction, and by how much?"

3. `ratio_off` answers: "does this class deviate away from the true-smile axis, and by how much?"

4. A large `ratio_along` does not automatically mean "more true-smile-like"  
   It only means more progress along the true-smile axis.

5. A small `ratio_off` means closer adherence to the true-smile axis  
   It must still be interpreted together with `ratio_along`.

6. A typical "closer to true smile" pattern is:

- larger `ratio_along`
- smaller `ratio_off`

7. The direct difference at `t=0` is the most direct expression of the starting-state difference.

8. The true-smile axis is a first-to-last-point approximation, not a guarantee that the full true-smile trajectory is linear.

---

## 12. Final Requirement Summary

This analysis must:

1. Use outputs from `analysis_sequence` as its input foundation.
2. Explicitly define Method A as the median-trajectory method and Method B as the minimum-cost medoid method.
3. Use the mean of the first five frames to define `f0`.
4. Use 20 time points with indices `0..19`.
5. Define the axis from the first and last point of the true-smile prototype.
6. Treat the following five lines as formal outputs:

- time-wise direct difference
- projection progress along the true-smile axis
- deviation distance away from the true-smile axis
- curve-to-curve minimum point distance using original pre-resampling sequence points
- tangent-relative nearest-point decomposition using original pre-resampling sequence points

7. Use the following normalized metrics:

   ```text
   ratio_along_c(t) = projection_length / ||g||
   ratio_off_c(t)   = off_axis_distance / ||g||
   ```

8. Treat Method A and Method B as two parallel primary analyses.
9. The minimum required core plots are:

- 3 anchor-based direct-difference plots per method
- 1 projection-along plot per method
- 1 off-axis-deviation plot per method

10. The new shared geometry analyses should preserve real frame correspondence and use original pre-resampling sequence points.
11. Preserve the real `sequence_id` for Method B and support linkage to real frames.
12. In addition to the prototype trajectories, the same type of analysis must be applied to every normalized sequence, followed by sample-level statistical summaries.
