# 3-31 DTW_resample 汇报PPT初稿

## Slide 1. Title

**This week: DTW-based resampling and follow-up projection analysis**

- DTW medoid for each smile class
- DTW alignment within each class
- resample aligned sequences to 20 points
- rerun projection analysis 4.2 and 4.3 on the new data

---

## Slide 2. Motivation

**Why do we need DTW-based resampling?**

- In the previous analysis, all sequences were linearly resampled to 20 points.
- This is simple, but it assumes the same temporal pace across sequences.
- In reality, smile timing can vary a lot.
- So we want a class-wise alignment method that respects temporal variation.

**Main idea**

- find a representative real sequence in each class
- align all class sequences to this representative with DTW
- then resample the aligned result to 20 points

---

## Slide 3. Input Data

**Data source**

- use `sequence_features_rel.npy`
- this means:
  - baseline aligned
  - original variable-length sequence
  - not yet resampled to 20 points

**Why use `f_rel`?**

- we still focus on dynamic change relative to the starting state
- we keep consistency with the previous projection analysis

---

## Slide 4. DTW Representative Sequence

**Step 1: build intra-class DTW distance matrix**

For one class:

- sequence 1
- sequence 2
- sequence 3
- ...

We compute pairwise DTW distance:

`D(i, j) = DTW(S_i, S_j)`

Then for each sequence:

`cost_i = Σ_j D(i, j)`

The sequence with the smallest total cost is selected as the representative sequence.

This is the **DTW medoid**.

---

## Slide 5. DTW Alignment and Resampling

**Step 2: align all sequences to the representative sequence**

- for each class
- fix the representative sequence as reference
- align every other sequence to it with DTW
- use Sakoe-Chiba band = 20%

**Step 3: resample to 20 points**

- after DTW alignment
- resample the aligned sequence to 20 points

So now we have:

- one representative sequence for each class
- one aligned-and-resampled 20-point sequence for every sample

---

## Slide 6. Additional Output

For each representative sequence, we also output:

- the original source video
- a clip for that representative sequence
- its own 20-point resampled version

This makes the representative sequence easier to inspect visually.

Representative sequences currently selected:

- polite: sequence 13
- truesmile: sequence 3
- ambiguous: sequence 27

---

## Slide 7. Follow-up Analysis

After getting DTW-resampled sequences, we reran the equivalent of:

- 4.2 projection along the true-smile axis
- 4.3 deviation from the true-smile axis

But now:

- prototype trajectory = DTW representative sequence
- participant sequences = DTW-aligned and resampled 20-point sequences

So the analysis pipeline is the same in meaning,
but the time normalization is now DTW-guided instead of only linear.

---

## Slide 8. Projection Along True-Smile Axis

**Main result**

- truesmile still shows the strongest progression along the true-smile axis
- polite and ambiguous still progress much less along this axis

Current prototype-level result:

- polite: `along_end ≈ 0.0575`
- truesmile: `along_end = 1.0000`
- ambiguous: `along_end ≈ 0.1322`

**Interpretation**

- Even after DTW-based alignment, polite and ambiguous do not strongly move along the true-smile main direction.

---

## Slide 9. Deviation from True-Smile Axis

**Main result**

- polite and ambiguous still show considerable off-axis deviation
- truesmile ends at zero by definition, but still has non-zero deviation in the middle stage

Current prototype-level result:

- polite: `off_end ≈ 0.4803`
- truesmile: `off_end = 0.0000`
- ambiguous: `off_end ≈ 0.3860`

**Interpretation**

- The true-smile trajectory is still not a straight line.
- Polite and ambiguous are still mainly changing in other directions.

---

## Slide 10. Comparison with Previous Analysis

**What changed?**

- We replaced simple linear time normalization with DTW-based within-class alignment.

**What did not change?**

- truesmile remains the most consistent class along the true-smile direction
- polite and ambiguous are still not strongly aligned with the true-smile axis

**Meaning**

- The previous conclusion does not seem to be only an artifact of simple linear resampling.
- Even after DTW-based resampling, the overall geometric relation is similar.

---

## Slide 11. Current Conclusion

1. DTW can be used to define a class-wise representative real sequence.
2. DTW-based alignment gives a more natural resampling scheme than simple linear normalization.
3. However, even after this improved resampling:
   - polite and ambiguous still do not strongly follow the true-smile axis
   - they still appear to move toward other regions in feature space
4. This makes the current conclusion more robust.

---

## Slide 12. Next Step

Possible next steps:

1. compare original linear resampling and DTW-based resampling more systematically
2. test whether polite-axis results also remain stable after DTW-resampling
3. analyze Matsuda-kun’s intentional polite smile in this new aligned space
4. decide whether DTW-resampled sequences should become the main input for later analyses

---

## Notes for Presentation

- Keep the focus on the logic:
  - DTW medoid
  - DTW alignment
  - resample to 20 points
  - rerun 4.2 and 4.3
- Do not spend too much time on implementation details.
- The strongest message is:
  - even after DTW-based alignment, polite and ambiguous are still not following the true-smile axis.
