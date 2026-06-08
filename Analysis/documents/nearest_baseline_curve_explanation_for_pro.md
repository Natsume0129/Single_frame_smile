# Nearest-Baseline Curve Analysis: Explanation Document for Pro Model Review

This document explains the current nearest-baseline curve analysis from zero. It is written so that a stronger reasoning model can review the method, interpret the current results, and suggest whether the analysis supports a meaningful claim about smile-transition trajectories.

## 1. Research Context

We analyze facial-expression transition trajectories, especially smile-related transitions. Each expression sequence is represented as a time-normalized trajectory in a high-dimensional feature space. In the current implementation, the feature is `fc7`, and each sequence is linearly normalized to 20 time points.

The three expression classes are:

- `truesmile`
- `polite`
- `ambiguous`

The key scientific question is whether different smile types follow the same underlying transition path, follow a weakened version of another path, or follow a different path.

## 2. Data Used in This Analysis

This analysis uses the existing linear-normalized data, not DTW-resampled data.

Input sequence data:

```text
E:\Matsuda_data\2-27meeting\metrics\normalized\<class>\<sequence_id>\normalized_sequence.npy
```

Prototype data:

```text
E:\Matsuda_data\3-10meeting\methodA\prototypes\prototype_<class>_methodA.npy
E:\Matsuda_data\3-10meeting\methodB\prototypes\prototype_<class>_methodB.npy
```

Nearest-6 sample lists from the previous analysis:

```text
E:\Matsuda_data\3-10meeting\linear_axis_extension\csv\nearest6_to_prototype_sequences_methodA.csv
E:\Matsuda_data\3-10meeting\linear_axis_extension\csv\nearest6_to_prototype_sequences_methodB.csv
```

`methodA` and `methodB` are two existing prototype-construction settings. This document does not assume that one is correct. Instead, it treats agreement or disagreement between methodA and methodB as part of the result.

## 3. Original Fixed-Axis s-d Plot

Before the new analysis, we used a fixed-axis s-d plot.

For a baseline curve \(C_{\mathrm{baseline}}\), define the main transition axis:

$$
\mathbf{a} = C_{\mathrm{baseline}}(100\%) - C_{\mathrm{baseline}}(0\%)
$$

For a point \(C_2(t)\) on a target curve, define:

$$
\Delta(t) = C_2(t) - C_2(0\%)
$$

The old fixed-axis coordinate was approximately:

$$
s(t) = \frac{\Delta(t) \cdot \mathbf{a}}{\|\mathbf{a}\|}
$$

$$
d(t) = \left\|\Delta(t) - s(t)\frac{\mathbf{a}}{\|\mathbf{a}\|}\right\|
$$

The old plot answers:

> How far does the target transition move along the selected baseline direction, and how far does it deviate from that direction?

This is useful, but it compresses the baseline curve into one straight axis. It does not directly ask which stage of the baseline curve is closest to each stage of the target curve.

## 4. New Nearest-Baseline Coordinate Definition

The new method keeps the full baseline curve \(C_{\mathrm{baseline}}\). It does not replace the baseline curve with only a start-to-end axis.

For each target stage \(t\) on another curve \(C_2\), for example:

$$
t \in \{5\%, 10\%, 15\%, \ldots, 100\%\}
$$

we search over the baseline curve and find the closest point:

$$
\tau^*(t) =
\arg\min_{\tau \in [0\%,100\%]}
\left\| C_2(t) - C_{\mathrm{baseline}}(\tau) \right\|_2
$$

The nearest vector is:

$$
\mathbf{v}_{\mathrm{nearest}}(t)
= C_2(t) - C_{\mathrm{baseline}}(\tau^*(t))
$$

The new coordinate is:

$$
x_{\mathrm{new}}(t) = \tau^*(t)
$$

$$
y_{\mathrm{new}}(t) =
\left\| \mathbf{v}_{\mathrm{nearest}}(t) \right\|_2
=
\left\| C_2(t) - C_{\mathrm{baseline}}(\tau^*(t)) \right\|_2
$$

So the new plot is not the old fixed-axis s-d plot.

The new x-axis means:

> Which progress stage of the baseline curve is most similar to the current target stage?

The new y-axis means:

> How far is the current target point from that nearest baseline stage?

## 5. Implementation Details

Current script:

```text
E:\Single_frame_smile\Analysis\analysis_projection\run_nearest_baseline_curve_analysis.py
```

Current output root:

```text
E:\Matsuda_data\3-10meeting\nearest_baseline_curve
```

The current implementation uses:

- target stages: 5%, 10%, 15%, ..., 100%
- baseline search grid: 0%, 1%, 2%, ..., 100%
- distance metric: high-dimensional L2 distance in fc7 trajectory space
- baseline classes: `truesmile` and `polite`
- target curves: all three class prototypes and nearest-6 real sequences for each class

The target point and baseline search points are linearly interpolated from the 20 normalized points.

## 6. What Each Figure Type Means

The HTML report is here:

```text
E:\Matsuda_data\3-10meeting\nearest_baseline_curve\report\nearest_baseline_curve_report.html
```

The generated figures are here:

```text
E:\Matsuda_data\3-10meeting\nearest_baseline_curve\plots
```

### 6.1 `demo_nearest_baseline_definition.png`

This is a simple 2D conceptual diagram. It shows a baseline curve and a target curve. For several target stages, a dashed line links the target point to its nearest point on the baseline curve.

This figure explains the geometry, not the real data.

### 6.2 `prototype_new_curve_baseline_<baseline>_<method>.png`

This is the main prototype-level new curve.

Each point is:

$$
\left(x_{\mathrm{new}}(t), y_{\mathrm{new}}(t)\right)
$$

The points are connected in the order of the target curve stage \(t\). Therefore, if the curve moves from left to right, the target transition is matching later and later stages of the baseline. If it stays near the left side, the target transition remains closest to an early stage of the baseline. If it jumps back and forth, the temporal correspondence is unstable.

### 6.3 `prototype_nearest_progress_baseline_<baseline>_<method>.png`

This figure plots:

- x-axis: target stage on \(C_2\)
- y-axis: nearest baseline progress \(\tau^*(t)\)

The diagonal line means:

$$
\tau^*(t) = t
$$

If a target curve follows the diagonal, then its stages correspond well to the same stages on the baseline curve. If the curve is much lower than the diagonal, then the target curve is staying close to earlier baseline stages.

### 6.4 `prototype_nearest_distance_baseline_<baseline>_<method>.png`

This figure plots:

- x-axis: target stage on \(C_2\)
- y-axis: nearest-vector length

This shows how far the target curve is from the closest baseline point at each stage. Low distance means the target point is geometrically close to the baseline curve. High distance means the target point is not well explained by the baseline curve.

### 6.5 `nearest6_new_curve_baseline_<baseline>_<method>.png`

This repeats the new curve analysis for the six real sequences that were previously selected as closest to each class prototype.

Thin lines are individual nearest-6 sequences. Dashed lines are prototype curves. This figure checks whether the prototype-level behavior is also visible in real sample trajectories.

### 6.6 `nearest6_nearest_progress_band_baseline_<baseline>_<method>.png`

This figure summarizes the nearest-6 real sequences.

- solid line: mean nearest baseline progress
- shaded band: interquartile range across the nearest-6 sequences
- dashed line: prototype result

This figure answers:

> At each target stage, which stage of the baseline are the real sequences closest to on average?

### 6.7 `nearest6_nearest_distance_band_baseline_<baseline>_<method>.png`

This figure summarizes the nearest-6 distance values.

- solid line: mean nearest-vector length
- shaded band: interquartile range across the nearest-6 sequences
- dashed line: prototype result

This figure answers:

> At each target stage, how far are the real sequences from their nearest baseline stage?

## 7. Current Endpoint Summary

The endpoint summary at target stage 100% is:

| method | baseline | target | source | nearest progress at 100% | distance at 100% |
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

## 8. Preliminary Interpretation of the Current Results

When `truesmile` is used as the baseline, the `polite` prototype at 100% is closest to a very early point on the true-smile baseline:

- methodA: 14%
- methodB: 1%

The nearest-6 polite real sequences show a similar pattern:

- methodA mean: 6%
- methodB mean: 2.833%

This suggests that, relative to the true-smile baseline, polite-smile trajectories do not simply progress to the late true-smile stage. Even at their own endpoint, they remain closest to the early part of the true-smile curve.

This is important because it argues against a simple interpretation that polite smile is only a weaker version of true smile. If polite smile were just a weaker true smile moving along the same path, we might expect the nearest baseline progress to increase steadily toward a later true-smile stage, even if the distance or intensity were smaller.

However, the distance values are not zero. The result is therefore not:

> Polite smile equals early true smile.

Rather, the safer interpretation is:

> Polite smile endpoints are geometrically closest to early true-smile baseline stages, but still have a measurable distance from that baseline.

When `polite` is used as the baseline, the `truesmile` prototype endpoint is close to late polite-baseline progress:

- methodA: 97%
- methodB: 93%

But the distance is relatively large:

- methodA: 0.420
- methodB: 0.621

This means true smile may reach a region that is late along the polite baseline, but it is not necessarily close in absolute geometric distance.

The `ambiguous` class is less stable across methodA and methodB. For example, with `polite` baseline:

- methodA ambiguous prototype endpoint is near 98%
- methodB ambiguous prototype endpoint is near 17%

This method disagreement suggests that ambiguous expressions are sensitive to prototype construction or may not have a single stable relationship to the baseline curves.

## 9. Key Questions for Pro Model Analysis

Please analyze the report and figures with these questions in mind:

1. Does the nearest-baseline progress curve provide evidence that polite smile is not merely a weakened true smile?
2. When `truesmile` is the baseline, why do polite and ambiguous endpoints remain closest to early true-smile progress?
3. Does the progress curve increase monotonically, stay flat, or jump? What does that imply about temporal correspondence?
4. How should we interpret cases where nearest progress is low but nearest distance is also low?
5. How should we interpret cases where nearest progress is high but nearest distance is large?
6. Are methodA and methodB consistent enough to support a stable conclusion?
7. Does the nearest-6 real-sequence result support or weaken the prototype-level interpretation?
8. Should distance be normalized by baseline length or baseline arc length before comparing across baselines and methods?
9. Are there statistical tests or summary metrics that should be added before making a strong claim?

## 10. Important Caveats

The nearest-baseline analysis is descriptive at this stage. It has not yet been converted into a statistical hypothesis test.

Important limitations:

- The distance is high-dimensional L2 distance in fc7 space.
- The distance is currently not normalized by baseline endpoint length or baseline arc length.
- The nearest point is searched on a 1% interpolation grid along the baseline curve.
- The nearest point can move non-monotonically along the baseline. This is informative, but it also means the connected curve should be interpreted as target-stage order, not as continuous movement along the baseline.
- Prototype results can be cleaner than real-sequence results. The nearest-6 plots are needed to check whether the prototype behavior is representative.
- methodA and methodB may produce different prototype geometries, especially for ambiguous expressions.

## 11. Suggested Next Analysis

Useful next quantitative summaries would be:

- endpoint nearest progress at 100%
- mean nearest progress across all stages
- maximum nearest progress reached by each target curve
- monotonicity score of \(\tau^*(t)\)
- number of backward jumps in \(\tau^*(t)\)
- mean and maximum nearest distance
- normalized distance using baseline endpoint norm
- normalized distance using baseline arc length
- bootstrap confidence intervals across real sequences
- comparison with DTW-resampled trajectories

These summaries would make it easier to write a results paragraph and defend the interpretation.

## 12. Suggested Prompt for Pro Model

Please review the nearest-baseline curve analysis described above. The central question is whether polite smile and ambiguous smile follow the same transition pathway as true smile, a weakened version of true smile, or a distinct pathway.

Focus on the generated HTML report and figures:

```text
E:\Matsuda_data\3-10meeting\nearest_baseline_curve\report\nearest_baseline_curve_report.html
```

Please provide:

1. A plain-language interpretation of each figure type.
2. A scientific interpretation of the current endpoint summary.
3. Whether the results support or weaken the hypothesis that polite smile is only a weakened true smile.
4. Whether methodA and methodB are consistent enough.
5. What additional quantitative metrics or statistical tests should be added.
6. A concise results paragraph suitable for discussion with a supervisor.

