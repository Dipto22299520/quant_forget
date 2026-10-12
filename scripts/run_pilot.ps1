# PILOT: full pipeline, small data, multiple seeds -> GO/NO-GO decision.
#
#   .\scripts\run_pilot.ps1                      # en, 3 seeds  (~3 h)
#   .\scripts\run_pilot.ps1 -Langs en,bn         # both         (~6 h)
#   .\scripts\run_pilot.ps1 -Seeds 1,2,3,4,5     # more seeds
#   .\scripts\run_pilot.ps1 -DryRun
#
# Resumable: finished generations are skipped, so a crash costs one generation.
# Each seed is a SEPARATE run directory so the verdict can measure seed noise.

param(
    [string[]]$Langs = @("en"),
    [string]  $ConfigPrefix = "pilot",   # "pilot" = open QA, "pilot_mmlu" = MCQ
    [string]  $NamePrefix   = "pilot",
    [int[]]   $Seeds = @(1, 2, 3),
    [string]  $Model = "Qwen/Qwen3-1.7B",
    [int]     $Generations = 6,
    [switch]  $DryRun
)

$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot\..
$env:PYTHONPATH = "src"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$py = ".\.venv\Scripts\python.exe"

$stamp = Get-Date -Format "yyyyMMdd_HHmm"
New-Item -ItemType Directory -Force -Path logs | Out-Null
$log = "logs\pilot_$stamp.log"

$total = $Langs.Count * $Seeds.Count
$i = 0
$started = Get-Date
Write-Host "=== PILOT: $total runs ($($Langs.Count) lang x $($Seeds.Count) seeds), $Generations generations ===" -ForegroundColor Cyan
Write-Host "log: $log`n"

foreach ($lang in $Langs) {
    foreach ($seed in $Seeds) {
        $i++
        $name = "${NamePrefix}_${lang}_s${seed}"
        Write-Host "[$i/$total] $name" -ForegroundColor Yellow

        # One seed value drives data sampling, generation sampling AND training,
        # so seeds are genuinely independent replicates rather than sharing a
        # fixed data order.
        $cmd = @(
            "-m", "collapse.cli", "run",
            "--config", "configs/${ConfigPrefix}_$lang.yaml",
            "--run-name", $name,
            "--set", "seed=$seed", "gen.seed=$seed", "train.seed=$seed",
            "model.name=$Model", "loop.generations=$Generations"
        )
        if ($DryRun) { Write-Host "   $py $($cmd -join ' ')"; continue }

        $t0 = Get-Date
        & $py @cmd | Tee-Object -FilePath $log -Append | Select-String -Pattern "gen \d+ \|"
        if ($LASTEXITCODE -ne 0) {
            Write-Host "   FAILED (exit $LASTEXITCODE) -- continuing" -ForegroundColor Red
            continue
        }
        Write-Host ("   done in {0:n1} min" -f ((Get-Date) - $t0).TotalMinutes) -ForegroundColor Green
    }
}

if ($DryRun) { exit 0 }

Write-Host ("`n=== pilot finished in {0:n1} h ===" -f ((Get-Date) - $started).TotalHours) -ForegroundColor Cyan

foreach ($lang in $Langs) {
    Write-Host "`n########## VERDICT: $lang ##########" -ForegroundColor Cyan
    & $py -m collapse.cli pilot --prefix "${NamePrefix}_${lang}_s" | Tee-Object -FilePath $log -Append
}

Write-Host "`nfull log: $log"
Write-Host "verdict json: runs\pilot_<lang>_sverdict.json"
