# Context Document for Further Deep Analysis

## 1. What this project is about

We are studying **dynamic smile trajectories** in a high-dimensional facial feature space.

The main goal is **not** to classify a single frame.  
Instead, we want to compare how different kinds of smiles evolve over time.

The three smile classes are:

- `polite`
- `truesmile`
- `ambiguous`

Our working idea is:

- each smile sequence can be represented as a trajectory in feature space
- different smile types may follow different temporal paths
- the differences may become clearer as the smile unfolds

---

## 2. What data representation we are using

For each frame in a smile sequence:

- we extract a VGG-Face `fc7` feature vector
- feature dimension is `4096`

So one smile sequence is a time series of 4096-dimensional vectors.

To make different sequences comparable, we use the following preprocessing:

1. **Baseline alignment**

We define:

```text
f0 = mean of the first 5 frames
f_rel(t) = f(t) - f0
```

The purpose is to reduce static factors and focus more on dynamic change.

2. **Time normalization**

Each sequence is resampled to `20` time points:

```text
f_norm(t), t = 0, 1, ..., 19
```

So every sequence finally has shape:

```text
[20, 4096]
```

---

## 3. What we already have

### 3.1 Existing code base

There are two related code folders:

- [analysis_sequence](/e:/Single_frame_smile/Analysis/analysis_sequence)
- [analysis_projection](/e:/Single_frame_smile/Analysis/analysis_projection)

`analysis_sequence` contains the earlier pipeline for:

- feature extraction
- baseline alignment
- time normalization
- prototype construction
- general plots and reports

`analysis_projection` is the newer module we built for projection-based and distance-based analysis.

### 3.2 Main documents

Important documents in [analysis_projection](/e:/Single_frame_smile/Analysis/analysis_projection):

- [projection_analysis_cn.md](/e:/Single_frame_smile/Analysis/analysis_projection/projection_analysis_cn.md)
- [projection_analysis_en.md](/e:/Single_frame_smile/Analysis/analysis_projection/projection_analysis_en.md)
- [work_summary_report.md](/e:/Single_frame_smile/Analysis/analysis_projection/work_summary_report.md)
- [slide.md](/e:/Single_frame_smile/Analysis/analysis_projection/slide.md)
- [slide_en.md](/e:/Single_frame_smile/Analysis/analysis_projection/slide_en.md)

### 3.3 Main analysis scripts

Important scripts in [analysis_projection](/e:/Single_frame_smile/Analysis/analysis_projection):

- [01_build_projection_prototypes.py](/e:/Single_frame_smile/Analysis/analysis_projection/01_build_projection_prototypes.py)
- [02_compute_direct_distance.py](/e:/Single_frame_smile/Analysis/analysis_projection/02_compute_direct_distance.py)
- [03_compute_projection_metrics.py](/e:/Single_frame_smile/Analysis/analysis_projection/03_compute_projection_metrics.py)
- [04_compute_per_sequence_metrics.py](/e:/Single_frame_smile/Analysis/analysis_projection/04_compute_per_sequence_metrics.py)
- [05_compute_statistics.py](/e:/Single_frame_smile/Analysis/analysis_projection/05_compute_statistics.py)
- [06_generate_plots.py](/e:/Single_frame_smile/Analysis/analysis_projection/06_generate_plots.py)
- [07_generate_report.py](/e:/Single_frame_smile/Analysis/analysis_projection/07_generate_report.py)
- [08_polite_axis_deviation_analysis.py](/e:/Single_frame_smile/Analysis/analysis_projection/08_polite_axis_deviation_analysis.py)

### 3.4 Existing outputs

The current main output directory is:

- [3-10meeting](/e:/Matsuda_data/3-10meeting)

It contains:

- `methodA/`
- `methodB/`

Each method contains:

- `csv/`
- `plots/`
- `prototypes/`
- `report/`

---

## 4. How prototypes are defined

We use two prototype definitions.

### Method A: median trajectory

For each class:

- at each time point
- at each feature dimension

we take the median across all sequences.

This gives a new synthetic trajectory:

```text
p_c^A(t)
```

It is a statistical center trajectory, but it usually does **not** correspond to a real file.

### Method B: medoid trajectory

For each class:

- compute pairwise full-sequence distance using Frobenius norm
- choose the real sequence with minimum total distance to all others

This gives:

```text
p_c^B(t)
```

It is a real sequence and keeps a real `sequence_id`.

---

## 5. What analyses we already performed

We currently have three main analysis lines.

### 5.1 Direct distance between class prototypes

For the same time point:

```text
diff_{a,b}(t) = ||p_a(t) - p_b(t)||_2
```

This is used to study:

- whether classes are already close or different at the beginning
- whether they separate more as time goes on
- which classes are closer to each other

### 5.2 Projection along the true-smile axis

We define the true-smile axis as:

```text
g = p_true(19) - p_true(0)
u = g / ||g||
```

For each class trajectory:

```text
d_c(t) = p_c(t) - p_c(0)
```

Then we project `d_c(t)` onto `u`.

This gives a ratio that means:

- how much the trajectory moves along the true-smile main direction

### 5.3 Deviation from the true-smile axis

For the same `d_c(t)`:

- compute projection onto the axis
- compute residual vector
- use residual norm as off-axis deviation

This gives a ratio that means:

- how far the trajectory is away from the true-smile axis

### 5.4 Additional polite-axis analysis

Later, based on feedback, we also repeated the off-axis analysis using:

- the `polite smile` axis as the base axis

This was done in:

- [08_polite_axis_deviation_analysis.py](/e:/Single_frame_smile/Analysis/analysis_projection/08_polite_axis_deviation_analysis.py)

---

## 6. What results we have so far

### 6.1 Stable observations

From the current results, these patterns look relatively stable:

1. Distances at the beginning are usually smaller.
2. As time goes on, distances between smile categories generally become larger.
3. `polite` and `ambiguous` are usually closer to each other than either is to `truesmile`.
4. `truesmile` clearly differs from the other two categories.
5. `polite` and `ambiguous` do **not** strongly progress along the true-smile axis.
6. They seem to move toward other regions of feature space.

### 6.2 Important interpretation

We do **not** interpret the small initial differences mainly as smile differences.

A better interpretation is:

- the beginning is closer to a neutral state
- the main smile-related differences become clearer after the smile develops

### 6.3 Important additional finding

Even `truesmile` itself shows considerable deviation from the true-smile axis during middle stages.

This suggests:

- the true-smile trajectory itself is **not** a straight line
- the line from start to end is only a rough global reference

This is a very important limitation of the current axis-based method.

### 6.4 Polite-axis result

When using the `polite smile` axis instead:

- polite smile shows smaller deviation from its own axis
- true smile shows much larger deviation from the polite axis

This suggests:

- polite smile may also have its own dynamic direction
- polite smile should not be understood only as “not true smile”

---

## 7. What problems or limitations we already noticed

These are important.

### 7.1 One single axis is probably too simple

The current axis is only:

- one line connecting the start and end of a prototype

But real trajectories may be:

- curved
- stage-dependent
- non-linear

So this axis is useful, but incomplete.

### 7.2 Prototype curves and sample-derived averages are not directly comparable

In some plots:

- the prototype curve is often much lower than the sample mean and sample band

This is because:

- prototype is built first in feature space
- then the metric is computed on that prototype

while the sample mean is:

- compute the metric for each sample first
- then average in metric space

Because the metric is non-linear, these are not equivalent.

### 7.3 Current distances mix direction and magnitude

The current vectors are **not** L2-normalized before distance/projection/deviation computation.

So the current metrics reflect both:

- direction difference
- vector magnitude difference

This means the current results should be interpreted as:

- geometric differences in feature space

not as pure angle similarity.

### 7.4 High-dimensional space issue

We are working in a 4096-dimensional space.

So using a single axis to judge whether another smile “moves like true smile” may be too strict.

It is possible that:

- other smiles do change dynamically
- but mainly in directions different from the chosen true-smile axis

---

## 8. Current working conclusions

At the current stage, our conclusions are:

1. Different smile categories do show different temporal trajectories in feature space.
2. Expressions are more similar in the neutral stage.
3. As the smile becomes stronger, category differences become larger.
4. `polite` and `ambiguous` are closer to each other.
5. `polite` and `ambiguous` do not strongly follow the true-smile global axis.
6. `truesmile` itself is also not fully linear.
7. Therefore, the current one-axis model is a useful first step, but not enough to fully describe smile dynamics.

---

## 9. What kind of help we want from a deep-thinking model

We want help exploring **better analysis methods**.

The key question is:

Given the current results and limitations, what more meaningful or more robust analysis framework should be used next?

In particular, we want ideas about:

1. Better ways to model smile dynamics than one single start-to-end axis
2. Whether we should use:
- subspace methods
- trajectory alignment methods
- manifold or nonlinear methods
- local tangent directions instead of one global axis
3. How to compare smile trajectories in a way that is:
- interpretable
- robust
- meaningful in high-dimensional space
4. How to distinguish:
- smile intensity
- smile direction
- smile style
- dynamic smoothness / rigidity
5. How to better analyze the intentional polite-smile case of Matsuda-kun
6. Whether there are more suitable metrics than the current:
- Euclidean distance
- Frobenius distance
- projection ratio
- off-axis ratio

---

## 10. Suggested questions for further analysis

If another model is asked to think deeply about this project, these are the most useful questions:

1. Is the current preprocessing pipeline reasonable for dynamic smile analysis?
2. What are the strengths and weaknesses of:
- baseline alignment by first-5-frame mean
- time resampling to 20 points
- median / medoid prototypes
3. What better mathematical model can replace the current single-axis analysis?
4. How should we model a smile trajectory if the true-smile trajectory itself is curved?
5. Is there a better way to compare classes than direct pointwise Euclidean distance?
6. How should we analyze whether polite smile has its own stable dynamic direction?
7. How should we analyze whether ambiguous smile is truly intermediate, or just heterogeneous?
8. How can we connect these feature-space results back to interpretable facial behavior?

---

## 11. Important file references

### Core method documents

- [projection_analysis_cn.md](/e:/Single_frame_smile/Analysis/analysis_projection/projection_analysis_cn.md)
- [projection_analysis_en.md](/e:/Single_frame_smile/Analysis/analysis_projection/projection_analysis_en.md)
- [work_summary_report.md](/e:/Single_frame_smile/Analysis/analysis_projection/work_summary_report.md)

### Main current reports

- [projection_summary_methodA.md](/e:/Matsuda_data/3-10meeting/methodA/report/projection_summary_methodA.md)
- [projection_summary_methodB.md](/e:/Matsuda_data/3-10meeting/methodB/report/projection_summary_methodB.md)
- [polite_axis_summary_methodA.md](/e:/Matsuda_data/3-10meeting/methodA/report/polite_axis_summary_methodA.md)
- [polite_axis_summary_methodB.md](/e:/Matsuda_data/3-10meeting/methodB/report/polite_axis_summary_methodB.md)

### Main scripts

- [03_compute_projection_metrics.py](/e:/Single_frame_smile/Analysis/analysis_projection/03_compute_projection_metrics.py)
- [04_compute_per_sequence_metrics.py](/e:/Single_frame_smile/Analysis/analysis_projection/04_compute_per_sequence_metrics.py)
- [06_generate_plots.py](/e:/Single_frame_smile/Analysis/analysis_projection/06_generate_plots.py)
- [08_polite_axis_deviation_analysis.py](/e:/Single_frame_smile/Analysis/analysis_projection/08_polite_axis_deviation_analysis.py)

---

## 12. Final note

This document is meant for a model or researcher who has **no previous context**.

The most important thing to understand is:

- we already have a working dynamic smile analysis pipeline
- we already obtained meaningful differences between smile categories
- but the current one-axis model is too simple
- the next step is to think of better and more principled ways to analyze these trajectories

