$ErrorActionPreference = "Stop"

$scripts = @(
  "01_extract_features.py",
  "02_baseline_align.py",
  "03_compute_magnitude.py",
  "04_compute_velocity.py",
  "05_compute_duration_stats.py",
  "06_time_normalize.py",
  "07_build_prototypes.py",
  "08_class_difference_vectors.py",
  "09_segment_vectors.py",
  "10_projection_pca.py",
  "11_class_distance_curve.py",
  "12_projection_scores.py",
  "13_generate_visualizations.py",
  "14_generate_dataset_report.py"
)

foreach ($s in $scripts) {
  Write-Host "Running $s ..."
  python $s
}

Write-Host "Pipeline done."

