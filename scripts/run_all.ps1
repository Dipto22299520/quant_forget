# MASTER RUNNER -- the full corrected study, in priority order.
#
#   .\scripts\run_all.ps1                 # everything, in order
#   .\scripts\run_all.ps1 -Stages A,B     # only some stages
#   .\scripts\run_all.ps1 -DryRun         # print the plan, run nothing
#
# Ordered so that if you only get one night, you still have the statistical core.
# Every run is resumable and each is a separate directory, so stopping and
# restarting costs at most one generation.
#
#   A  primary      2 langs x ratio 1.0 x 3 seeds   -- the H1/H2 core        (~10 h)
#   B  ratio sweep  2 langs x {0.75, 0.5, 0.0} x 1 seed -- dose-response + SFT ctrl
#   C  controls     full fine-tune + 2nd model      -- rules out artefacts    (~5 h)
#   D  H4 filters   2 langs x {selfverify, cal}     -- mitigation failure     (~7 h)
#   F  replication  r50 x 2 seeds x 2 langs, bn fullft x 2 seeds            (~17 h)
#   M  method screen  English, 6 mitigations + 2 CAL seeds                  (~5 h)
#   K  samples per prompt  English, K in {2,4}, with predictions        (~2 h)
#   E  analysis     no GPU
#
# Progress bars stream to the console; log lines are teed to logs\.

param(
    [string[]]$Stages = @("A", "B", "C", "D", "F", "E"),
    [int[]]   $Seeds  = @(1, 2, 3),
    # Seeds for the Stage M method arms: screen at 1, then promote with 2,3.
    [int[]]   $MethodSeeds = @(1),
    [string[]]$Langs  = @("en", "bn"),
    [string]  $Model  = "Qwen/Qwen3-1.7B",
    # Run-name prefix. Empty for the main study; set it when running the same
    # arms with another model so the names do not collide with (and get skipped
    # as) the finished 1.7B arms, e.g.  -Model Qwen/Qwen3-4B -Tag q4b_ -Langs en
    [string]  $Tag    = "",
    [string]  $Model2 = "Qwen/Qwen2.5-1.5B",
    [int]     $Generations = 6,
    # Micro-batch for the LoRA arms. effective batch = PerDeviceBs x GradAccum,
    # held at 16 so the optimisation trajectory matches Stage A exactly; only GPU
    # utilisation changes. bs=8 peaks near 11GB of 17.1GB; bs=16 OOMs because
    # weighted_lm_loss materialises fp32 logits (bs x seq x 152k vocab).
    [int]     $PerDeviceBs = 8,
    [int]     $GradAccum   = 2,
    # Bangla sequences are 2x longer (1536-token window, see configs/main_bn.yaml)
    # so the micro-batch is halved; effective batch stays 16.
    [int]     $BnPerDeviceBs = 4,
    [int]     $BnGradAccum   = 4,
    [switch]  $DryRun
)

function Bs($lang)    { if ($lang -eq "bn") { $BnPerDeviceBs } else { $PerDeviceBs } }
function Accum($lang) { if ($lang -eq "bn") { $BnGradAccum }   else { $GradAccum } }

$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot\..
$env:PYTHONPATH = "src"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$py = ".\.venv\Scripts\python.exe"

$stamp = Get-Date -Format "yyyyMMdd_HHmm"
New-Item -ItemType Directory -Force -Path logs | Out-Null
$log = "logs\runall_$stamp.log"
$started = Get-Date
$done = 0; $failed = @()

function Run-Arm($name, $config, $overrides) {
    $script:done++
    # An arm with summary.json is FINISHED. Skip it before check_resume runs:
    # otherwise a later code change makes the manifest reject completed work that
    # never needed re-running, and it gets reported as a failure.
    if (Test-Path "runs\$name\summary.json") {
        # A finished Bangla arm from the invalid 768-token campaign (README
        # finding #5) must not be mistaken for a result. Refuse loudly rather
        # than skip silently: the user has to move it aside on purpose.
        if ($name -match "_bn_") {
            $fp = (Get-Content "runs\$name\summary.json" -Raw | ConvertFrom-Json).metrics[0].code_fingerprint
            if ($fp -in @("a96c1f931adc", "10c791652a94")) {
                Write-Host "`n[$script:done] $name  -- INVALID Bangla run (fingerprint $fp, pre-fix truncation bug)." -ForegroundColor Red
                Write-Host "    Move it aside, then re-run:" -ForegroundColor Red
                Write-Host "      New-Item -ItemType Directory -Force runs_invalid_bn768 | Out-Null" -ForegroundColor Yellow
                Write-Host "      Move-Item runs\$name runs_invalid_bn768\" -ForegroundColor Yellow
                exit 2
            }
        }
        Write-Host "`n[$script:done] $name  -- already complete, skipping" -ForegroundColor DarkGray
        return
    }
    Write-Host "`n[$script:done] $name" -ForegroundColor Yellow
    $cmd = @("-m", "collapse.cli", "run", "--config", $config, "--run-name", $name,
             "--set") + $overrides
    if ($DryRun) { Write-Host "    $py $($cmd -join ' ')" -ForegroundColor DarkGray; return }

    Add-Content $log "`n===== $name  ($(Get-Date -Format 'HH:mm:ss')) ====="
    $t0 = Get-Date
    # stdout -> log, stderr (tqdm bars) -> console
    & $py @cmd | Tee-Object -FilePath $log -Append | Select-String -Pattern "gen \d+ \||sft mix"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "    FAILED (exit $LASTEXITCODE) -- continuing" -ForegroundColor Red
        $script:failed += $name
    } else {
        Write-Host ("    done in {0:n1} min" -f ((Get-Date) - $t0).TotalMinutes) -ForegroundColor Green
    }
}

Write-Host "=== FULL STUDY: stages $($Stages -join ',') ===" -ForegroundColor Cyan
Write-Host "log: $log`n"

# ---------------------------------------------------------------- STAGE A
# The statistical core. Three seeds per language at 100% synthetic: this is what
# the pre-declared H1 test consumes, and the only stage with seed replication.
if ($Stages -contains "A") {
    Write-Host "`n########## STAGE A: primary arms ##########" -ForegroundColor Cyan
    foreach ($lang in $Langs) {
        foreach ($s in $Seeds) {
            Run-Arm "${Tag}main_${lang}_r100_s${s}" "configs/main_$lang.yaml" `
                @("seed=$s", "gen.seed=$s", "train.seed=$s",
                  "model.name=$Model", "loop.generations=$Generations", "loop.ratio=1.0",
                  "train.per_device_bs=$(Bs $lang)", "train.grad_accum=$(Accum $lang)")
        }
    }
}

# ---------------------------------------------------------------- STAGE B
# Dose-response. One seed each: if collapse is real it should scale with the
# synthetic fraction, and a monotone trend across ratios is worth more than a
# third seed at a single ratio. ratio 0.0 is the continued-SFT control: the same
# number of recursive fine-tuning steps on REAL data only. If calibration drifts
# there too, the drift is an SFT effect, not a synthetic-data effect.
if ($Stages -contains "B") {
    Write-Host "`n########## STAGE B: ratio sweep ##########" -ForegroundColor Cyan
    foreach ($lang in $Langs) {
        foreach ($r in @(0.75, 0.5, 0.0)) {
            $pct = [int]($r * 100)
            Run-Arm "${Tag}main_${lang}_r${pct}_s1" "configs/main_$lang.yaml" `
                @("seed=1", "gen.seed=1", "train.seed=1",
                  "model.name=$Model", "loop.generations=$Generations", "loop.ratio=$r",
                  "train.per_device_bs=$(Bs $lang)", "train.grad_accum=$(Accum $lang)")
        }
    }
}

# ---------------------------------------------------------------- STAGE C
# Controls. Full fine-tuning answers "is this a LoRA artefact?"; a second model
# family answers "is this a Qwen3 artefact?". Both are questions a reviewer asks.
if ($Stages -contains "C") {
    Write-Host "`n########## STAGE C: controls ##########" -ForegroundColor Cyan
    foreach ($lang in $Langs) {
        Run-Arm "${Tag}ctrl_${lang}_fullft_s1" "configs/main_$lang.yaml" `
            @("seed=1", "gen.seed=1", "train.seed=1", "model.name=$Model",
              "loop.generations=$Generations", "train.mode=full", "train.lr=2.0e-5",
              "train.per_device_bs=1", "train.grad_accum=16")
        Run-Arm "${Tag}ctrl_${lang}_model2_s1" "configs/main_$lang.yaml" `
            @("seed=1", "gen.seed=1", "train.seed=1", "model.name=$Model2",
              "loop.generations=$Generations",
              "train.per_device_bs=$(Bs $lang)", "train.grad_accum=$(Accum $lang)")
    }
}

# ---------------------------------------------------------------- STAGE D
# H4: does confidence-based mitigation degrade exactly where it is needed?
# selfverify filters the DATA on internal confidence; cal changes the LOSS.
if ($Stages -contains "D") {
    Write-Host "`n########## STAGE D: H4 mitigation arms ##########" -ForegroundColor Cyan
    foreach ($lang in $Langs) {
        Run-Arm "${Tag}h4_${lang}_selfverify_s1" "configs/main_$lang.yaml" `
            @("seed=1", "gen.seed=1", "train.seed=1", "model.name=$Model",
              "loop.generations=$Generations", "loop.filter=selfverify",
              "loop.filter_threshold=0.5",
              "train.per_device_bs=$(Bs $lang)", "train.grad_accum=$(Accum $lang)",
              # Filtering removes candidates, so a strict ratio would refuse to
              # build the mix. Shrink the target instead of topping up with real
              # data, which would change the treatment.
              "loop.sft_size=1500")
        # The fixed 0.5 threshold above kept 99.7% of candidates (median
        # confidence rose 0.81 -> 0.97 across the loop), so that arm is a
        # halved-budget control. This one keeps the top half by confidence,
        # re-ranked every generation, so the filter actually selects.
        Run-Arm "${Tag}h4_${lang}_selfverify_top_s1" "configs/main_$lang.yaml" `
            @("seed=1", "gen.seed=1", "train.seed=1", "model.name=$Model",
              "loop.generations=$Generations", "loop.filter=selfverify_top",
              "loop.filter_threshold=0.5",
              "train.per_device_bs=$(Bs $lang)", "train.grad_accum=$(Accum $lang)",
              "loop.sft_size=1500")
        Run-Arm "${Tag}h4_${lang}_cal_s1" "configs/main_$lang.yaml" `
            @("seed=1", "gen.seed=1", "train.seed=1", "model.name=$Model",
              "loop.generations=$Generations", "train.loss=cal", "train.cal_gamma=1.0",
              "train.per_device_bs=$(Bs $lang)", "train.grad_accum=$(Accum $lang)")
    }
}

# ---------------------------------------------------------------- STAGE F
# Replication seeds for the two results the paper now leans on. Ordered
# cheapest-and-most-load-bearing first, so a partial night still helps:
#   en r50     secures the English dose-response headline     (~35 min each)
#   bn r50     secures the Bangla dose-response headline       (~3.5 h each)
#   bn fullft  the one control that contradicted H2 at n=1    (~4.6 h each)
if ($Stages -contains "F") {
    Write-Host "`n########## STAGE F: replication seeds ##########" -ForegroundColor Cyan
    foreach ($lang in $Langs) {
        foreach ($s in @(2, 3)) {
            Run-Arm "${Tag}main_${lang}_r50_s${s}" "configs/main_$lang.yaml" `
                @("seed=$s", "gen.seed=$s", "train.seed=$s",
                  "model.name=$Model", "loop.generations=$Generations", "loop.ratio=0.5",
                  "train.per_device_bs=$(Bs $lang)", "train.grad_accum=$(Accum $lang)")
        }
    }
    foreach ($s in @(2, 3)) {
        Run-Arm "${Tag}ctrl_bn_fullft_s${s}" "configs/main_bn.yaml" `
            @("seed=$s", "gen.seed=$s", "train.seed=$s", "model.name=$Model",
              "loop.generations=$Generations", "train.mode=full", "train.lr=2.0e-5",
              "train.per_device_bs=1", "train.grad_accum=16")
    }
}

# ---------------------------------------------------------------- STAGE M
# Method screen (English, Qwen3-1.7B, 100% synthetic unless stated). Primary
# endpoint, declared before any of these ran: AUROC at g5 -- temperature drift is
# removable post hoc without labels (scripts/label_free.py), so a training-time
# method has to earn its place on discrimination, not ECE. Promote a method if
# g5 AUROC beats the CE-baseline mean by >= 0.03 with accuracy >= CE mean - 0.01.
# AUROC is ranked on EXACT log-space option logits: the stored float32
# probabilities saturate (283/600 items at exactly 1.0), and a method that merely
# softens the model would otherwise "gain" AUROC by making ties rankable.
# Scored by scripts/screen_report.py.
if ($Stages -contains "M") {
    Write-Host "`n########## STAGE M: method screen ##########" -ForegroundColor Cyan
    $base = @("model.name=$Model", "loop.generations=$Generations",
              "train.per_device_bs=$PerDeviceBs", "train.grad_accum=$GradAccum")
    # PROMOTED (28 Sept, by the declared AUROC rule): one epoch per round, seed 1
    # gave g5 AUROC 0.735 >= 0.734 with accuracy 0.570. Seeds 2-3 replicate it.
    # The combination arm tests the mechanism's prediction that removing BOTH
    # drivers (repetition and top-p truncation) lands nearest the fixed point.
    foreach ($s in @(2, 3)) {
        Run-Arm "${Tag}m_en_epochs1_s$s" "configs/main_en.yaml" (@("seed=$s", "gen.seed=$s", "train.seed=$s") + $base + @("train.epochs=1"))
    }
    Run-Arm "${Tag}m_en_e1anc_s1" "configs/main_en.yaml" (@("seed=1", "gen.seed=1", "train.seed=1") + $base + @("train.epochs=1", "gen.sampler=ancestral"))
    foreach ($s in $MethodSeeds) {
        $sd = @("seed=$s", "gen.seed=$s", "train.seed=$s")
        # MECHANISM ARMS (declared rule, before any ran): the entropy-neutral
        # sampler "works" if self-target entropy at g4 stays >= 0.9 x its g0 value
        # (CE: 0.39x), raw ECE drift g0->g5 <= +0.005 (CE mean +0.015), and
        # accuracy >= CE mean - 0.01. The two diagnostics say WHICH part of the
        # loop sharpens: the sampler (ancestral) or repeated fitting of one sample
        # per prompt (epochs=1, same optimiser steps as the half-budget control).
        Run-Arm "${Tag}m_en_ensampler_s$s" "configs/main_en.yaml" ($sd + $base + @("gen.sampler=entropy_neutral"))
        Run-Arm "${Tag}m_en_ancestral_s$s" "configs/main_en.yaml" ($sd + $base + @("gen.sampler=ancestral"))
        Run-Arm "${Tag}m_en_epochs1_s$s"   "configs/main_en.yaml" ($sd + $base + @("train.epochs=1"))
        # prior work: truncated CE (Shabgahi et al. 2025), stronger ForTIFAI CAL
        Run-Arm "${Tag}m_en_tce_s$s"    "configs/main_en.yaml" ($sd + $base + @("train.loss=tce", "train.tce_threshold=0.95"))
        Run-Arm "${Tag}m_en_calg3_s$s"  "configs/main_en.yaml" ($sd + $base + @("train.loss=cal", "train.cal_gamma=3.0"))
        # ours: CAL + a 5% reference anchor that is never reused across rounds
        Run-Arm "${Tag}m_en_calanchor_s$s" "configs/main_en.yaml" ($sd + $base + @("train.loss=cal", "train.cal_gamma=1.0", "loop.ratio=0.95", "loop.real_fresh=true"))
        # confidence filter that actually selects; the inert p>0.5 arm is its matched-budget control
        Run-Arm "${Tag}m_en_svtop_s$s"  "configs/main_en.yaml" ($sd + $base + @("loop.filter=selfverify_top", "loop.filter_threshold=0.5", "loop.sft_size=1500"))
        # standard regularisers
        Run-Arm "${Tag}m_en_ls_s$s"     "configs/main_en.yaml" ($sd + $base + @("train.loss=ls", "train.ls_eps=0.1"))
        Run-Arm "${Tag}m_en_ent_s$s"    "configs/main_en.yaml" ($sd + $base + @("train.loss=ent", "train.ent_beta=0.1"))
    }
    # plain CAL needs three seeds to be a baseline (s1 already exists)
    foreach ($s in @(2, 3)) {
        Run-Arm "${Tag}h4_en_cal_s$s" "configs/main_en.yaml" (@("seed=$s", "gen.seed=$s", "train.seed=$s") + $base + @("train.loss=cal", "train.cal_gamma=1.0"))
    }
}

# ---------------------------------------------------------------- STAGE K
# Samples per prompt at a fixed 3,000-pair budget. PREDICTIONS, written before
# any K arm ran, from the measured per-round entropy retention (model entropy
# on its own training targets, round to round, English 1.7B seed 1):
#   standard top-p, K=1 (CE)   0.79   measured
#   ancestral,      K=1        0.85   measured  -> fitting costs ~15%/round,
#                                                  the top-p sampler ~7%/round
# If one-sample fitting is plug-in estimation bias, its cost shrinks ~1/K:
#   ancestral, K=2 -> ~0.92    ancestral, K=4 -> ~0.96    top-p, K=4 -> ~0.89
if ($Stages -contains "K") {
    Write-Host "`n########## STAGE K: samples per prompt ##########" -ForegroundColor Cyan
    $base = @("model.name=$Model", "loop.generations=$Generations",
              "train.per_device_bs=$PerDeviceBs", "train.grad_accum=$GradAccum")
    foreach ($s in $MethodSeeds) {
        $sd = @("seed=$s", "gen.seed=$s", "train.seed=$s")
        Run-Arm "${Tag}m_en_k4anc_s$s" "configs/main_en.yaml" ($sd + $base + @("gen.sampler=ancestral", "loop.samples_per_prompt=4"))
        Run-Arm "${Tag}m_en_k2anc_s$s" "configs/main_en.yaml" ($sd + $base + @("gen.sampler=ancestral", "loop.samples_per_prompt=2"))
        Run-Arm "${Tag}m_en_k4_s$s"    "configs/main_en.yaml" ($sd + $base + @("loop.samples_per_prompt=4"))
    }
}

# ---------------------------------------------------------------- STAGE R
# Replication seeds (28 Sept), in priority order. The half-data control decides
# the CAL claim: at seed 1 it reached AUROC 0.739 against CAL's three-seed 0.742,
# so if it replicates, CAL's discrimination gain may be "less effective training".
# Identical settings to h4_en_selfverify_s1 (a p>0.5 filter that keeps ~all, then
# a random 1,500 of 3,000 pairs). Then the two arms that held calibration at the
# fixed point at one seed (TCE, entropy bonus), then the best single AUROC (K=2).
if ($Stages -contains "R") {
    Write-Host "`n########## STAGE R: replication seeds ##########" -ForegroundColor Cyan
    $base = @("model.name=$Model", "loop.generations=$Generations",
              "train.per_device_bs=$PerDeviceBs", "train.grad_accum=$GradAccum")
    $specs = @(
        @("h4_en_selfverify", @("loop.filter=selfverify", "loop.filter_threshold=0.5", "loop.sft_size=1500")),
        # ForTIFAI's primary setting is TCE with threshold 0.9 (arXiv 2509.08972);
        # the 0.95 arm stays as a one-seed threshold-sensitivity point.
        @("m_en_tce90", @("train.loss=tce", "train.tce_threshold=0.9")),
        @("m_en_ent",   @("train.loss=ent", "train.ent_beta=0.1")),
        @("m_en_k2anc", @("gen.sampler=ancestral", "loop.samples_per_prompt=2"))
    )
    foreach ($spec in $specs) {
        $seeds = if ($spec[0] -eq "m_en_tce90") { @(1, 2, 3) } else { @(2, 3) }
        foreach ($s in $seeds) {
            Run-Arm "${Tag}$($spec[0])_s$s" "configs/main_en.yaml" (@("seed=$s", "gen.seed=$s", "train.seed=$s") + $base + $spec[1])
        }
    }
}

# ---------------------------------------------------------------- STAGE E
if ($Stages -contains "E") {
    Write-Host "`n########## STAGE E: analysis (no GPU) ##########" -ForegroundColor Cyan
    if (-not $DryRun) {
        foreach ($lang in $Langs) {
            & $py -m collapse.cli h1 --prefix "main_${lang}_r100_s" | Tee-Object -FilePath $log -Append
        }
        Get-ChildItem runs -Directory -Filter "main_*" | ForEach-Object {
            if (Test-Path (Join-Path $_.FullName "summary.json")) {
                & $py -m collapse.cli analyze --run "runs/$($_.Name)" | Out-Null
            }
        }
        Write-Host "figures written to runs\<arm>\figures\" -ForegroundColor Green
    }
}

if (-not $DryRun) {
    Write-Host ("`n=== finished in {0:n1} h ===" -f ((Get-Date) - $started).TotalHours) -ForegroundColor Cyan
    if ($failed.Count) {
        Write-Host "FAILED arms: $($failed -join ', ')" -ForegroundColor Red
    }
    Write-Host "log: $log"
    Write-Host "`nRead first:  .\.venv\Scripts\python.exe -m collapse.cli h1 --prefix main_en_r100_s"
}
