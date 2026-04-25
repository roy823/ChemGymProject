# Week 7 — closed-loop refresh.
# After any new run lands under ProjectMain/checkpoints/, this script
# regenerates the Pareto report and the significance table, in that
# order so the significance markdown appends cleanly to a fresh REPORT.md.

$python = "D:\Anaconda\envs\chemgym\python.exe"
$saveRoot = "ProjectMain\checkpoints\week7_pareto_report"
$baselinesRoot = "ProjectMain\checkpoints"

& $python "ProjectMain\week7_pareto_report.py" --save-root $saveRoot --baselines-glob-root $baselinesRoot
& $python "ProjectMain\week7_significance_analysis.py" --save-root $saveRoot --baselines-glob-root $baselinesRoot

Write-Host "[Week7-CloseLoop] Refresh complete. Open $saveRoot\REPORT.md."
Write-Host "[Week7-CloseLoop] Manual interpretation lives in ProjectMain\week7_key_findings_2026_04_25.md (or whichever date stamp applies)."
