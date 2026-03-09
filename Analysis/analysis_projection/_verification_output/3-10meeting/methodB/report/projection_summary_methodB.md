# Projection Summary (methodB)

## Direct Distance
### Anchor: polite
- truesmile: start=0.1527, peak=0.6603 at t=14, end=0.6228
- ambiguous: start=0.0962, peak=0.5107 at t=17, end=0.4949
### Anchor: truesmile
- polite: start=0.1527, peak=0.6603 at t=14, end=0.6228
- ambiguous: start=0.1439, peak=0.7374 at t=19, end=0.7374
### Anchor: ambiguous
- polite: start=0.0962, peak=0.5107 at t=17, end=0.4949
- truesmile: start=0.1439, peak=0.7374 at t=19, end=0.7374

## Axis Metrics
- polite: along_end=0.1485, along_peak=0.1516, off_end=0.5830, off_peak=0.5917
- truesmile: along_end=1.0000, along_peak=1.0000, off_end=0.0000, off_peak=0.6623
- ambiguous: along_end=-0.0242, along_peak=0.1659, off_end=0.5069, off_peak=0.6693

## Per-Sequence Statistics
### projection_ratio
- polite: mean_end=0.1361, mean_peak=0.1615 at t=16, iqr_end=(0.0256, 0.2050)
- truesmile: mean_end=0.4482, mean_peak=0.4482 at t=19, iqr_end=(0.2826, 0.4758)
- ambiguous: mean_end=0.1932, mean_peak=0.2133 at t=15, iqr_end=(0.1210, 0.2969)
### off_axis_ratio
- polite: mean_end=0.8024, mean_peak=0.8158 at t=16, iqr_end=(0.6673, 0.9843)
- truesmile: mean_end=1.0077, mean_peak=1.1697 at t=17, iqr_end=(1.1267, 1.2220)
- ambiguous: mean_end=0.9089, mean_peak=0.9248 at t=17, iqr_end=(0.7690, 1.0903)

## Notes
- Method A uses median prototype trajectories.
- Method B uses medoid prototype trajectories and preserves real sequence IDs.
