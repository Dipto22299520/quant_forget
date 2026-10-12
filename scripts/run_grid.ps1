# Core grid runner. Sequential by design -- one 16GB card, one run at a time.
#
# Every run is RESUMABLE: finished generations are skipped on restart, so if this
# dies at 4am you just run it again and it picks up where it stopped.
#
#   .\scripts\run_grid.ps1                    # full core grid
#   .\scripts\run_grid.ps1 -Ratios 1.0        # just the 100% synthetic arm
#   .\scripts\run_grid.ps1 -Langs en          # one language
#   .\scripts\run_grid.ps1 -DryRun            # print commands, run nothing

param(
    [string[]]$Langs      = @("en", "bn"),
    [double[]]$Ratios     = @(1.0, 0.75, 0.5),
    [string] $Model       = "Qwen/Qwen3-1.7B",
    [string] $Tag         = "qwen3",
    [int]    $Generations = 5,
    [switch] $DryRun
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..
$env:PYTHONPATH = "src"
$py = ".\.venv\Scripts\python.exe"

$configs = @{ "en" = "configs/en_g5.yaml"; "bn" = "configs/bn_g5.yaml" }
$total = $Langs.Count * $Ratios.Count
$i = 0
$started = Get-Date

Write-Host "=== core grid: $total runs, model=$Model, gens=$Generations ===" -ForegroundColor Cyan

foreach ($lang in $Langs) {
    foreach ($ratio in $Ratios) {
        $i++
        $pct = [int]($ratio * 100)
        $name = "${lang}_${Tag}_r${pct}"
        $cfg = $configs[$lang]

        Write-Host "`n[$i/$total] $name" -ForegroundColor Yellow
        $cmd = @(
            "-m", "collapse.cli", "run",
            "--config", $cfg,
            "--run-name", $name,
            "--set", "loop.ratio=$ratio", "model.name=$Model",
            "loop.generations=$Generations"
        )

        if ($DryRun) { Write-Host "  $py $($cmd -join ' ')"; continue }

        $t0 = Get-Date
        & $py @cmd
        if ($LASTEXITCODE -ne 0) {
            # Keep going: one bad arm should not cost you the whole night.
            Write-Host "  FAILED (exit $LASTEXITCODE) -- continuing to next run" -ForegroundColor Red
            continue
        }
        Write-Host ("  done in {0:n1} min" -f ((Get-Date) - $t0).TotalMinutes) -ForegroundColor Green

        & $py -m collapse.cli analyze --run "runs/$name"
    }
}

if (-not $DryRun) {
    Write-Host ("`n=== grid finished in {0:n1} h ===" -f ((Get-Date) - $started).TotalHours) -ForegroundColor Cyan
    Write-Host "Results: runs\<name>\summary.json, analysis.json, figures\"
}
