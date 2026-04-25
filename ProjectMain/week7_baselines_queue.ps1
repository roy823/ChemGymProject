# Week 7 Phase 2 — feasibility-aware random / SA baselines under matched budget.
#
# Each call writes standard_eval_per_seed.csv + standard_eval_by_train_seed.csv
# under its own checkpoints subdirectory, schema-compatible with
# week4_action_reward_ablation.standard_eval_*.

$python = "D:\Anaconda\envs\chemgym\python.exe"

function Invoke-Baseline([string]$method, [string]$muCo, [int]$totalSteps, [string]$saveRoot, [int]$maxDev) {
    New-Item -ItemType Directory -Force -Path $saveRoot | Out-Null
    $log = Join-Path $saveRoot "run.log"
    $err = Join-Path $saveRoot "run.err.log"
    if (Test-Path $log) { Remove-Item $log -Force }
    if (Test-Path $err) { Remove-Item $err -Force }
    & $python "ProjectMain\week7_baselines_feasible.py" `
        --method $method `
        --mu-co $muCo `
        --total-steps $totalSteps `
        --seeds "11,22,33,44,55" `
        --max-deviation $maxDev `
        --save-root $saveRoot `
        --oracle-mode "hybrid" *>> $log 2>> $err
}

# 2k budget at both mu_CO
Invoke-Baseline "random_mutation" "-0.2" 2048 "ProjectMain\checkpoints\week7_baselines_random_m02_2k_md4" 4
Invoke-Baseline "sa_mutation"     "-0.2" 2048 "ProjectMain\checkpoints\week7_baselines_sa_m02_2k_md4"     4
Invoke-Baseline "random_mutation" "-0.6" 2048 "ProjectMain\checkpoints\week7_baselines_random_m06_2k_md4" 4
Invoke-Baseline "sa_mutation"     "-0.6" 2048 "ProjectMain\checkpoints\week7_baselines_sa_m06_2k_md4"     4

# 4k budget at both mu_CO
Invoke-Baseline "random_mutation" "-0.2" 4096 "ProjectMain\checkpoints\week7_baselines_random_m02_4k_md4" 4
Invoke-Baseline "sa_mutation"     "-0.2" 4096 "ProjectMain\checkpoints\week7_baselines_sa_m02_4k_md4"     4
Invoke-Baseline "random_mutation" "-0.6" 4096 "ProjectMain\checkpoints\week7_baselines_random_m06_4k_md4" 4
Invoke-Baseline "sa_mutation"     "-0.6" 4096 "ProjectMain\checkpoints\week7_baselines_sa_m06_4k_md4"     4

& $python "ProjectMain\week7_pareto_report.py"
