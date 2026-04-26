$python = "D:\Anaconda\envs\chemgym\python.exe"

$saveRoot = "ProjectMain\checkpoints\week6_open_m06_strictstop_4k_s3"
New-Item -ItemType Directory -Force -Path $saveRoot | Out-Null
$log = Join-Path $saveRoot "run.log"
$err = Join-Path $saveRoot "run.err.log"
if (Test-Path $log) { Remove-Item $log -Force }
if (Test-Path $err) { Remove-Item $err -Force }
& $python "ProjectMain\week4\week4_action_reward_ablation.py" --profiles "mutation_delta_strict_stop" --seeds "11,22,33" --mu-co "-0.6" --train-steps "4096" --eval-steps "160" --standard-eval-seeds "11,22,33" --standard-eval-steps "100" --device "cuda" --oracle-mode "hybrid" --write-summary-md --save-root $saveRoot *>> $log 2>> $err

$saveRoot = "ProjectMain\checkpoints\week6_open_m02_strictstop_4k_s3"
New-Item -ItemType Directory -Force -Path $saveRoot | Out-Null
$log = Join-Path $saveRoot "run.log"
$err = Join-Path $saveRoot "run.err.log"
if (Test-Path $log) { Remove-Item $log -Force }
if (Test-Path $err) { Remove-Item $err -Force }
& $python "ProjectMain\week4\week4_action_reward_ablation.py" --profiles "mutation_delta_strict_stop" --seeds "11,22,33" --mu-co "-0.2" --train-steps "4096" --eval-steps "160" --standard-eval-seeds "11,22,33" --standard-eval-steps "100" --device "cuda" --oracle-mode "hybrid" --write-summary-md --save-root $saveRoot *>> $log 2>> $err

& $python "ProjectMain\week6\week6_envelope_report.py"
