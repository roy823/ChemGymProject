# Week 7 Phase 3 — mu_CO phase-diagram queue.
#
# Bounded-mask md=4 at 4k budget across mu_CO in {-0.4, -0.8, -1.0}, seeds 11,22,33.
# The two endpoints -0.2 and -0.6 are already in week 6.

$python = "D:\Anaconda\envs\chemgym\python.exe"

function Invoke-PhaseRun([string]$saveRoot, [string]$muCo) {
    New-Item -ItemType Directory -Force -Path $saveRoot | Out-Null
    $log = Join-Path $saveRoot "run.log"
    $err = Join-Path $saveRoot "run.err.log"
    if (Test-Path $log) { Remove-Item $log -Force }
    if (Test-Path $err) { Remove-Item $err -Force }
    & $python "ProjectMain\week4_action_reward_ablation.py" `
        --profiles "mutation_delta_strict_stop_masked" `
        --seeds "11,22,33" `
        --mu-co $muCo `
        --train-steps 4096 `
        --eval-steps 160 `
        --standard-eval-seeds "11,22,33" `
        --standard-eval-steps 100 `
        --device "cuda" `
        --oracle-mode "hybrid" `
        --max-deviation-override 4 `
        --write-summary-md `
        --save-root $saveRoot *>> $log 2>> $err
}

Invoke-PhaseRun "ProjectMain\checkpoints\week7_phase_md4_m04_4k_s3" "-0.4"
Invoke-PhaseRun "ProjectMain\checkpoints\week7_phase_md4_m08_4k_s3" "-0.8"
Invoke-PhaseRun "ProjectMain\checkpoints\week7_phase_md4_m10_4k_s3" "-1.0"

# Refresh the envelope report so the new mu rows are available.
& $python "ProjectMain\week6_envelope_report.py"
