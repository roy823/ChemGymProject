# Week 8 D12 ladder-adjustment queue.
#
# Purpose:
#   The default RE-MC ladder T={0.05,0.10,0.20,0.35} produced aggregate
#   exchange acceptance 1.000 at both mu_CO pilot cells, above the locked
#   D12 upper bound of 0.60. D12 allows one adjustment, changing only the
#   highest temperature within {0.30,0.35,0.40,0.50}. This queue uses the
#   largest allowed top temperature, 0.50 eV, and writes pair-level exchange
#   diagnostics via the patched week8_baselines_mc.py summary columns.
#
# Run from repo root:
#   powershell -NoProfile -ExecutionPolicy Bypass -File ProjectMain\week8\week8_d12_ladder_adjust_queue.ps1
#
# Decision rule:
#   If every adjacent pair in exchange_acceptance_by_pair is in [0.10,0.60]
#   for both mu_CO cells, lock this adjusted ladder for Exp-1 RE-MC.
#   Otherwise activate v1.2 risk R9 and drop RE-MC from the main table.

$python = "D:\Anaconda\envs\chemgym\python.exe"
$reTemps = "0.05,0.10,0.20,0.50"

function Invoke-W8REMCAdjusted {
    param(
        [string]$saveRoot,
        [string]$muCo
    )

    New-Item -ItemType Directory -Force -Path $saveRoot | Out-Null
    $log = Join-Path $saveRoot "run.log"
    $err = Join-Path $saveRoot "run.err.log"
    if (Test-Path $log) { Remove-Item $log -Force }
    if (Test-Path $err) { Remove-Item $err -Force }

    Write-Host "[Week8-D12] Launching adjusted RE-MC mu=$muCo ladder=$reTemps -> $saveRoot"
    & $python "ProjectMain\week8\week8_baselines_mc.py" `
        --method "replica_exchange_mc" `
        --protocol static `
        --mu-co $muCo `
        --total-oracle-budget 4096 `
        --max-steps-per-episode 32 `
        --seeds "11" `
        --re-temperatures $reTemps `
        --re-proposals-per-replica-per-round 64 `
        --bulk-pd-fraction 0.08 `
        --n-active-layers 4 `
        --save-root $saveRoot `
        --oracle-mode "hybrid" *>> $log 2>> $err

    Write-Host "[Week8-D12] Done mu=$muCo. Summary:"
    Import-Csv (Join-Path $saveRoot "static_budget_summary.csv") |
        Select-Object method,mu_co,T,seed,best_omega,n_exchange_attempts,n_exchange_accepts,exchange_acceptance,exchange_acceptance_by_pair |
        Format-List
}

Invoke-W8REMCAdjusted -saveRoot "ProjectMain\checkpoints\week8_remc_pilot_m06_4k_s1_ttop050" -muCo "-0.6"
Invoke-W8REMCAdjusted -saveRoot "ProjectMain\checkpoints\week8_remc_pilot_m02_4k_s1_ttop050" -muCo "-0.2"

Write-Host "[Week8-D12] Complete. Inspect:"
Write-Host "  type ProjectMain\checkpoints\week8_remc_pilot_m06_4k_s1_ttop050\static_budget_summary.csv"
Write-Host "  type ProjectMain\checkpoints\week8_remc_pilot_m02_4k_s1_ttop050\static_budget_summary.csv"
