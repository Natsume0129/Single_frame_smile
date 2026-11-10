param(
    [ValidateSet("check", "prepare", "run")][string]$Stage = "check",
    [string]$Config = "$PSScriptRoot\config.example.json",
    [string]$Person = ""
)
$ErrorActionPreference = "Stop"
$repoRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$pythonPath = Join-Path $repoRoot ".venv\Scripts\python.exe"
$pipelineArgs = @("$PSScriptRoot\pipeline.py", $Stage, "--config", $Config)
if ($Person) { $pipelineArgs += @("--person", $Person) }
& $pythonPath @pipelineArgs
if ($LASTEXITCODE -ne 0) { throw "Experiment failed. See the reported path and logs." }
