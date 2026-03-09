$ErrorActionPreference = "Stop"

$analysisInputRoot = "E:\Matsuda_data\2-27meeting"
$outputRoot = "E:\Matsuda_data\3-10meeting"

$scripts = @(
  "01_build_projection_prototypes.py",
  "02_compute_direct_distance.py",
  "03_compute_projection_metrics.py",
  "04_compute_per_sequence_metrics.py",
  "05_compute_statistics.py",
  "06_generate_plots.py",
  "07_generate_report.py"
)

foreach ($script in $scripts) {
  Write-Host "Running $script ..."
  python $script --analysis_input_root $analysisInputRoot --output_root $outputRoot
  if ($LASTEXITCODE -ne 0) {
    throw "Script failed: $script"
  }
}

Write-Host "Projection pipeline done."
