$python = "D:\Anaconda\envs\chemgym\python.exe"

$saveRoot = "ProjectMain\checkpoints\week6_masksweep_m06_md6_4k_s3"
New-Item -ItemType Directory -Force -Path $saveRoot | Out-Null
$log = Join-Path $saveRoot "run.log"
$err = Join-Path $saveRoot "run.err.log"
if (Test-Path $log) { Remove-Item $log -Force }
if (Test-Path $err) { Remove-Item $err -Force }
& $python "ProjectMain\week4_action_reward_ablation.py" --profiles "mutation_delta_strict_stop_masked" --seeds "11,22,33" --mu-co "-0.6" --train-steps "4096" --eval-steps "160" --standard-eval-seeds "11,22,33" --standard-eval-steps "100" --device "cuda" --oracle-mode "hybrid" --max-deviation-override "6" --write-summary-md --save-root $saveRoot *>> $log 2>> $err

$saveRoot = "ProjectMain\checkpoints\week6_masksweep_m02_md6_2k_s3"
New-Item -ItemType Directory -Force -Path $saveRoot | Out-Null
$log = Join-Path $saveRoot "run.log"
$err = Join-Path $saveRoot "run.err.log"
if (Test-Path $log) { Remove-Item $log -Force }
if (Test-Path $err) { Remove-Item $err -Force }
& $python "ProjectMain\week4_action_reward_ablation.py" --profiles "mutation_delta_strict_stop_masked" --seeds "11,22,33" --mu-co "-0.2" --train-steps "2048" --eval-steps "120" --standard-eval-seeds "11,22,33" --standard-eval-steps "80" --device "cuda" --oracle-mode "hybrid" --max-deviation-override "6" --write-summary-md --save-root $saveRoot *>> $log 2>> $err

$saveRoot = "ProjectMain\checkpoints\week6_masksweep_m02_md6_4k_s3"
New-Item -ItemType Directory -Force -Path $saveRoot | Out-Null
$log = Join-Path $saveRoot "run.log"
$err = Join-Path $saveRoot "run.err.log"
if (Test-Path $log) { Remove-Item $log -Force }
if (Test-Path $err) { Remove-Item $err -Force }
& $python "ProjectMain\week4_action_reward_ablation.py" --profiles "mutation_delta_strict_stop_masked" --seeds "11,22,33" --mu-co "-0.2" --train-steps "4096" --eval-steps "160" --standard-eval-seeds "11,22,33" --standard-eval-steps "100" --device "cuda" --oracle-mode "hybrid" --max-deviation-override "6" --write-summary-md --save-root $saveRoot *>> $log 2>> $err

& $python "ProjectMain\week6_mask_sweep_report.py"
& $python "ProjectMain\week6_envelope_report.py"
