# Week 7 Phase 4 — best-structure audit driver.
# Pulls every relevant week 5/6/7 run, extracts the best feasible structure,
# computes Warren-Cowley + per-layer Pd fraction, writes a single REPORT.

$python = "D:\Anaconda\envs\chemgym\python.exe"

$saveRoot = "ProjectMain\checkpoints\week7_structure_audit"
New-Item -ItemType Directory -Force -Path $saveRoot | Out-Null
$log = Join-Path $saveRoot "run.log"
$err = Join-Path $saveRoot "run.err.log"
if (Test-Path $log) { Remove-Item $log -Force }
if (Test-Path $err) { Remove-Item $err -Force }

# Bounded-mask runs at both mu_CO and both budgets.
$inputRuns = @(
    "ProjectMain\checkpoints\week6_masksweep_m02_md2_2k_s3",
    "ProjectMain\checkpoints\week6_masksweep_m02_md6_2k_s3",
    "ProjectMain\checkpoints\week6_masksweep_m02_md6_4k_s3",
    "ProjectMain\checkpoints\week6_boundedopen_m02_md4_4k_s3",
    "ProjectMain\checkpoints\week6_masksweep_m06_md2_2k_s3",
    "ProjectMain\checkpoints\week6_masksweep_m06_md2_4k_s3",
    "ProjectMain\checkpoints\week6_masksweep_m06_md6_2k_s3",
    "ProjectMain\checkpoints\week6_masksweep_m06_md6_4k_s3",
    "ProjectMain\checkpoints\week6_boundedopen_m06_md4_4k_s3",
    "ProjectMain\checkpoints\week5_maskedopen_m02_strictstop_2k_s3",
    "ProjectMain\checkpoints\week5_maskedopen_m06_strictstop_2k_s3",
    "ProjectMain\checkpoints\week6_open_m02_strictstop_4k_s3",
    "ProjectMain\checkpoints\week6_open_m06_strictstop_4k_s3",
    "ProjectMain\checkpoints\week5_swapctrl_m02_strictstop_2k_s3",
    "ProjectMain\checkpoints\week5_swapctrl_m06_strictstop_2k_s3"
)

# Phase 3 outputs (added in priority order if present).
$phaseDirs = @(
    "ProjectMain\checkpoints\week7_phase_md4_m04_4k_s3",
    "ProjectMain\checkpoints\week7_phase_md4_m08_4k_s3",
    "ProjectMain\checkpoints\week7_phase_md4_m10_4k_s3"
)
foreach ($d in $phaseDirs) {
    if (Test-Path $d) { $inputRuns += $d }
}

& $python "ProjectMain\week7_structure_audit.py" `
    --input-runs $inputRuns `
    --save-root $saveRoot `
    --eval-steps 120 `
    --oracle-mode "hybrid" *>> $log 2>> $err

& $python "ProjectMain\week7_pareto_report.py"
