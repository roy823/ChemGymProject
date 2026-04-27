# Week 8 Overnight Pilot Result - 2026-04-27

Source outputs are under `ProjectMain/checkpoints/week8_*` and are ignored by
Git. This note records the checkpoint-derived results needed to decide the next
registered run.

## Smoke checks

- `week8_protocol.py`: passed under EMT fallback.
- `week8_baselines_mc.py`: passed under EMT fallback.
- Overnight queue stderr: empty at queue level; per-run stderr contained only
  Torch/MLIP warnings, no traceback.

## Completed hybrid-oracle runs

| Cell | Seed(s) | Budget | Key result | Wall time |
| --- | ---: | ---: | ---: | ---: |
| RE-MC default ladder, mu_CO = -0.6 | 11 | 4096 pooled | best_omega = -299.319 eV; exchange = 18/18 = 1.000 | 0.73 h |
| RE-MC default ladder, mu_CO = -0.2 | 11 | 4096 pooled | best_omega = -328.938 eV; exchange = 17/17 = 1.000 | 0.70 h |
| canonical_MC, mu_CO = -0.6 | 11,22,33,44,55 | 4096 each | mean best_omega = -304.508 eV; sd = 7.237 eV; mean accept = 0.388 | 3.53 h |
| random_swap, mu_CO = -0.6 | 11,22,33,44,55 | 4096 each | mean best_omega = -298.266 eV; sd = 0.668 eV | 3.17 h |

Per-seed static mu_CO = -0.6 comparison:

| Seed | canonical_MC best_omega | random_swap best_omega | canonical - random |
| ---: | ---: | ---: | ---: |
| 11 | -312.460 | -297.301 | -15.159 |
| 22 | -299.248 | -298.206 | -1.042 |
| 33 | -299.255 | -299.166 | -0.089 |
| 44 | -312.410 | -298.202 | -14.207 |
| 55 | -299.166 | -298.453 | -0.713 |

Mean paired delta `canonical_MC - random_swap` = -6.242 eV (lower omega is
better), sd = 7.721 eV.

## Decision

The canonical_MC and random_swap cells are usable for the Week-8 Exp-1
mu_CO = -0.6 static table.

The default RE-MC ladder does not pass D12. The observed aggregate exchange
acceptance is 1.000 at both mu_CO values, above the registered upper bound
0.60. The first pilot also exposed an implementation gap: the code recorded
only aggregate exchange acceptance, while D12 requires adjacent-pair acceptance.
The next run must therefore use the patched pair-level exchange columns before
locking or dropping RE-MC.

## Next experiment

Run the single allowed D12 adjustment with only the highest temperature changed:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File ProjectMain\week8\week8_d12_ladder_adjust_queue.ps1
```

The adjusted ladder is `T = {0.05, 0.10, 0.20, 0.50}`. Decision rule:

- If every adjacent pair in `exchange_acceptance_by_pair` is in `[0.10, 0.60]`
  for both mu_CO pilot cells, lock this adjusted ladder for the remaining
  Exp-1 RE-MC runs.
- If any adjacent pair remains outside `[0.10, 0.60]`, activate risk-register
  entry R9: drop RE-MC from the main table and keep canonical_MC as the main MC
  baseline, with RE-MC reported as a failed-thermalization SI sensitivity.
