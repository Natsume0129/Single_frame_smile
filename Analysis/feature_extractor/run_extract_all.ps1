# run_extract_all.ps1
# =========================
# Config
# =========================
$PY = "python"

$fc7Script   = "feature_extractor_fc7.py"
$convScript  = "feature_extractor_conv5_3.py"

$fc7Weights  = "E:\Single_frame_smile\data\models\vggface.pth"
$convWeights = "E:\Single_frame_smile\data\models\vggface_conv.pth"

$device = "cuda:0"

$rootIn  = "E:\Matsuda_data\vgg-face_analysis\classic_segments\processedbyrvm"
$rootOut = "E:\Matsuda_data\vgg-face_analysis\classic_segments\extracted_features_rvm"

# =========================
# Helpers
# =========================
function Get-RelativePath([string]$base, [string]$path) {
  $baseFull = (Resolve-Path $base).Path.TrimEnd('\')
  $pathFull = (Resolve-Path $path).Path
  if ($pathFull.StartsWith($baseFull, [System.StringComparison]::OrdinalIgnoreCase)) {
    return $pathFull.Substring($baseFull.Length).TrimStart('\')
  }
  return $pathFull
}

# 判断是否“叶子文件夹”：没有子目录；且包含至少一张图片
$imgExt = @(".jpg", ".jpeg", ".png", ".bmp", ".webp")

# =========================
# Collect target folders
# =========================
$allDirs = Get-ChildItem -Path $rootIn -Directory -Recurse

$leafDirs = $allDirs | Where-Object {
  $hasSubDir = @(Get-ChildItem -Path $_.FullName -Directory -Force -ErrorAction SilentlyContinue).Count -gt 0
  if ($hasSubDir) { return $false }

  $hasImg = @(Get-ChildItem -Path $_.FullName -File -Force -ErrorAction SilentlyContinue |
              Where-Object { $imgExt -contains $_.Extension.ToLower() }).Count -gt 0
  return $hasImg
}

Write-Host "Found leaf image folders:" $leafDirs.Count

# =========================
# Run
# =========================
foreach ($d in $leafDirs) {
  $imgDir = $d.FullName
  $rel    = Get-RelativePath $rootIn $imgDir

  # 输出目录：保留相对路径，避免同名覆盖
  $outDir = Join-Path $rootOut $rel
  New-Item -ItemType Directory -Force -Path $outDir | Out-Null

  # 1) fc7 输出：一个 .pt 文件
  $fc7Out = Join-Path $outDir "vggface_fc7_withbg.pt"
  Write-Host "== FC7 ==" -ForegroundColor Cyan
  Write-Host "img_dir: $imgDir"
  Write-Host "save   : $fc7Out"

  & $PY $fc7Script `
    --weights $fc7Weights `
    --img_dir  $imgDir `
    --save     $fc7Out `
    --mode     fc7 `
    --device   $device

  if ($LASTEXITCODE -ne 0) {
    Write-Warning "FC7 failed: $imgDir (exit=$LASTEXITCODE). Continue..."
    continue
  }

  # 2) conv5_3 输出：你的脚本是 --save 给一个前缀目录/前缀名（示例里没加 .pt）
  # 这里用 outDir 下的 "vggface_conv5_3_rvm" 作为前缀，最终产物由脚本决定（比如 *_conv5_3.pt / *_gap.pt 等）
  $convPrefix = Join-Path $outDir "vggface_conv5_3_rvm"
  Write-Host "== Conv5_3 ==" -ForegroundColor Green
  Write-Host "img_dir: $imgDir"
  Write-Host "save   : $convPrefix"

  & $PY $convScript `
    --weights $convWeights `
    --img_dir  $imgDir `
    --save     $convPrefix `
    --device   $device

  if ($LASTEXITCODE -ne 0) {
    Write-Warning "Conv5_3 failed: $imgDir (exit=$LASTEXITCODE). Continue..."
    continue
  }
}

Write-Host "Done."
