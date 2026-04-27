# Week 8 — overnight pilot queue (v1.2 §8 step 2 baselines).
#
# Three phases, ~10 GPU-hours total on RTX-4060 hybrid oracle (UMA + OC25
# ensemble). Run from repo root:
#     pwsh -File ProjectMain\week8\week8_overnight_pilot_queue.ps1
#
# Phase 1 (gating, ~2h): RE-MC default-ladder calibration pilots
#   per v1.2 §1 D12. 1 seed × 4096 pooled budget at each mu_CO. Confirms
#   inter-replica swap acceptance lies in [10%, 60%] for all adjacent
#   pairs of the default ladder T = {0.05, 0.10, 0.20, 0.35} eV.
#
# Phase 2 (~4h): canonical_MC main-table headline cell, mu_CO = -0.6 eV.
#   5 seeds × 4096 distinct-oracle-call budget under the static protocol
#   (max 32 steps per episode, T = 0.10 eV, swap action mode). Provides
#   the locked Exp-1 canonical_MC reference for the harder mu cell.
#
# Phase 3 (~4h): random_swap baseline, mu_CO = -0.6 eV.
#   5 seeds × 4096 distinct-oracle-call budget. Lower-bound calibration
#   per v1.2 §4.4. Same protocol as Phase 2 so cells are directly
#   comparable.
#
# Outputs land under ProjectMain\checkpoints\week8_*\:
#   - meta.json            run config snapshot
#   - static_budget_summary.csv   one row per seed (or per RE-MC seed/pool)
#   - seeds\<seed>\running_omega_trace.csv   per-step trace
#   - run.log / run.err.log
#
# Re-launch any failed cell by re-running just its Invoke-W8MC line; the
# summary CSV is wiped and rewritten per launch, the trace files per
# seed.

$python = "D:\Anaconda\envs\chemgym\python.exe"

function Invoke-W8MC {
    param(
        [string]$saveRoot,
        [string]$method,
        [string]$muCo,
        [int]$budget,
        [string]$seeds,
        [string]$reTemps = "0.05,0.10,0.20,0.35",
        [int]$reRound = 64,
        [double]$T = 0.10
    )
    New-Item -ItemType Directory -Force -Path $saveRoot | Out-Null
    $log = Join-Path $saveRoot "run.log"
    $err = Join-Path $saveRoot "run.err.log"
    if (Test-Path $log) { Remove-Item $log -Force }
    if (Test-Path $err) { Remove-Item $err -Force }
    Write-Host "[Week8-Queue] Launching $method mu=$muCo budget=$budget seeds=$seeds -> $saveRoot"
    & $python "ProjectMain\week8\week8_baselines_mc.py" `
        --method $method `
        --protocol static `
        --mu-co $muCo `
        --T $T `
        --total-oracle-budget $budget `
        --max-steps-per-episode 32 `
        --seeds $seeds `
        --re-temperatures $reTemps `
        --re-proposals-per-replica-per-round $reRound `
        --bulk-pd-fraction 0.08 `
        --n-active-layers 4 `
        --save-root $saveRoot `
        --oracle-mode "hybrid" `
        --record-trace *>> $log 2>> $err
    Write-Host "[Week8-Queue] Done   $method mu=$muCo. Tail of run.log:"
    Get-Content $log -Tail 6 | Write-Host
}

# Phase 1: RE-MC ladder calibration pilots (gating, D12)
Invoke-W8MC -saveRoot "ProjectMain\checkpoints\week8_remc_pilot_m06_4k_s1" `
            -method "replica_exchange_mc" -muCo "-0.6" -budget 4096 -seeds "11"
Invoke-W8MC -saveRoot "ProjectMain\checkpoints\week8_remc_pilot_m02_4k_s1" `
            -method "replica_exchange_mc" -muCo "-0.2" -budget 4096 -seeds "11"

# Phase 2: canonical_MC main-table cell (mu = -0.6, the harder mu)
Invoke-W8MC -saveRoot "ProjectMain\checkpoints\week8_canonicalmc_m06_4k_s5" `
            -method "canonical_mc" -muCo "-0.6" -budget 4096 -seeds "11,22,33,44,55"

# Phase 3: random_swap baseline (mu = -0.6)
Invoke-W8MC -saveRoot "ProjectMain\checkpoints\week8_randomswap_m06_4k_s5" `
            -method "random_swap" -muCo "-0.6" -budget 4096 -seeds "11,22,33,44,55"

Write-Host ""
Write-Host "[Week8-Queue] All phases complete. Summary CSVs:"
foreach ($d in @(
    "ProjectMain\checkpoints\week8_remc_pilot_m06_4k_s1\static_budget_summary.csv",
    "ProjectMain\checkpoints\week8_remc_pilot_m02_4k_s1\static_budget_summary.csv",
    "ProjectMain\checkpoints\week8_canonicalmc_m06_4k_s5\static_budget_summary.csv",
    "ProjectMain\checkpoints\week8_randomswap_m06_4k_s5\static_budget_summary.csv"
)) {
    if (Test-Path $d) {
        Write-Host "  $d"
    } else {
        Write-Host "  $d  (MISSING — check run.err.log)"
    }
}
Write-Host ""
Write-Host "[Week8-Queue] Inspect with:"
Write-Host "  type ProjectMain\checkpoints\week8_remc_pilot_m06_4k_s1\static_budget_summary.csv"
Write-Host "Look for exchange_acceptance in [0.10, 0.60] for D12 ladder healthy."
