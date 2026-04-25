# Week 7 Phase 1 — Bring bounded-mask 4k-budget headline cells to n=5 seeds.
#
# Adds train_seeds 44 and 55 to the bounded mask runs that currently sit at
# n=3, plus the missing m02_md2 4k cell.

$python = "D:\Anaconda\envs\chemgym\python.exe"

function Invoke-MaskedRun([string]$saveRoot, [string]$muCo, [int]$trainSteps, [int]$evalSteps, [int]$standardEvalSteps, [string]$seeds, [int]$maxDev) {
    New-Item -ItemType Directory -Force -Path $saveRoot | Out-Null
    $log = Join-Path $saveRoot "run.log"
    $err = Join-Path $saveRoot "run.err.log"
    if (Test-Path $log) { Remove-Item $log -Force }
    if (Test-Path $err) { Remove-Item $err -Force }
    & $python "ProjectMain\week4_action_reward_ablation.py" `
        --profiles "mutation_delta_strict_stop_masked" `
        --seeds $seeds `
        --mu-co $muCo `
        --train-steps $trainSteps `
        --eval-steps $evalSteps `
        --standard-eval-seeds $seeds `
        --standard-eval-steps $standardEvalSteps `
        --device "cuda" `
        --oracle-mode "hybrid" `
        --max-deviation-override $maxDev `
        --write-summary-md `
        --save-root $saveRoot *>> $log 2>> $err
}

# m02 md2 4k — currently missing entirely; build n=5 (11,22,33,44,55)
Invoke-MaskedRun "ProjectMain\checkpoints\week7_extra_seeds_m02_md2_4k_n5"      "-0.2" 4096 160 100 "11,22,33,44,55" 2

# m02 md6 4k — extend 11,22,33 to n=5
Invoke-MaskedRun "ProjectMain\checkpoints\week7_extra_seeds_m02_md6_4k_extra"   "-0.2" 4096 160 100 "44,55" 6

# m06 md2 4k — extend 11,22,33 to n=5
Invoke-MaskedRun "ProjectMain\checkpoints\week7_extra_seeds_m06_md2_4k_extra"   "-0.6" 4096 160 100 "44,55" 2

# m06 md6 4k — extend 11,22,33 to n=5
Invoke-MaskedRun "ProjectMain\checkpoints\week7_extra_seeds_m06_md6_4k_extra"   "-0.6" 4096 160 100 "44,55" 6

# Refresh the headline envelope report so the new seeds show up.
& $python "ProjectMain\week6_envelope_report.py"
