# Unattended overnight run. Logs everything, survives individual failures,
# and writes a summary you can read in the morning.
#
#   .\scripts\run_overnight.ps1
#
# Order is deliberate: cheapest and most informative first, so if you only get
# three hours of GPU time you still have the week-2 deliverable.

param(
    [string]$Model = "Qwen/Qwen3-1.7B",
    [int]$Generations = 5
)

$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot\..
$env:PYTHONPATH = "src"
$py = ".\.venv\Scripts\python.exe"

$stamp = Get-Date -Format "yyyyMMdd_HHmm"
$log = "logs\overnight_$stamp.log"
New-Item -ItemType Directory -Force -Path logs | Out-Null

function Step($name, $argList) {
    Write-Host "`n>>> $name" -ForegroundColor Yellow
    Add-Content $log "`n===== $name  ($(Get-Date -Format 'HH:mm:ss')) ====="
    $t0 = Get-Date
    & $py @argList | Tee-Object -FilePath $log -Append
    $mins = ((Get-Date) - $t0).TotalMinutes
    if ($LASTEXITCODE -eq 0) {
        Write-Host ("    ok ({0:n1} min)" -f $mins) -ForegroundColor Green
        Add-Content $log "----- OK ($([math]::Round($mins,1)) min)"
    } else {
        Write-Host ("    FAILED exit $LASTEXITCODE ({0:n1} min)" -f $mins) -ForegroundColor Red
        Add-Content $log "----- FAILED exit $LASTEXITCODE"
    }
}

Write-Host "logging to $log" -ForegroundColor Cyan

# 1. Sanity: environment + metric unit tests. Fast, and catches a broken env
#    before you burn six hours discovering it.
Step "unit tests"  @("tests\test_units.py")
Step "check-env"   @("-m", "collapse.cli", "check-env")

# 2. English generation-0 baseline only. This is the week-2 deliverable and the
#    insurance policy: it stands alone as the static cross-lingual paper.
Step "en baseline g0" @("-m", "collapse.cli", "run", "--config", "configs/en_g5.yaml",
                        "--run-name", "en_g0_baseline",
                        "--set", "loop.generations=1", "model.name=$Model")

# 3. Bangla generation-0 baseline.
Step "bn baseline g0" @("-m", "collapse.cli", "run", "--config", "configs/bn_g5.yaml",
                        "--run-name", "bn_g0_baseline",
                        "--set", "loop.generations=1", "model.name=$Model")

# 4. The smoke-test arm from plan week 3: 100% synthetic, one language, full
#    depth. If nothing moves here, nothing will move anywhere.
Step "en r100 full loop" @("-m", "collapse.cli", "run", "--config", "configs/en_g5.yaml",
                           "--run-name", "en_qwen3_r100",
                           "--set", "loop.ratio=1.0", "model.name=$Model",
                           "loop.generations=$Generations")
Step "analyze en r100" @("-m", "collapse.cli", "analyze", "--run", "runs/en_qwen3_r100")

# 5. Same in Bangla -- the H2 comparison.
Step "bn r100 full loop" @("-m", "collapse.cli", "run", "--config", "configs/bn_g5.yaml",
                           "--run-name", "bn_qwen3_r100",
                           "--set", "loop.ratio=1.0", "model.name=$Model",
                           "loop.generations=$Generations")
Step "analyze bn r100" @("-m", "collapse.cli", "analyze", "--run", "runs/bn_qwen3_r100")

Write-Host "`n=== overnight complete. log: $log ===" -ForegroundColor Cyan
Write-Host "Read first:  runs\en_qwen3_r100\analysis.json  (the H1 verdict)"
Get-ChildItem runs -Directory | ForEach-Object {
    $s = Join-Path $_.FullName "summary.json"
    if (Test-Path $s) { Write-Host ("  {0,-24} {1}" -f $_.Name, "summary.json OK") }
    else              { Write-Host ("  {0,-24} {1}" -f $_.Name, "INCOMPLETE") -ForegroundColor DarkYellow }
}
