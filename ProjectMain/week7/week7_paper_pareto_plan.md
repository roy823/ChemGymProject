# Week 7 Paper-Grade Evidence Plan

Date: 2026-04-25
Target venues: Nat Commun, npj Comput Mater, JACS Au, Angewandte (RL for materials track).

## 1. Why this week

The week-5 gap analysis already named the missing pieces. Week 6 closed the
`max_deviation` Pareto axis at two chemical potentials and delivered the Bounded
vs Open vs Fixed-Swap headline. The current weakness is no longer the algorithm
or the reward — it is the *evidence package* a top-journal reviewer would
demand:

- `n=3` on most headline cells, which is below the `n>=5` bar that recent
  RL-for-materials papers use for averaged learning curves and final tables.
- No budget-matched random / simulated-annealing baseline reported under the
  same feasibility metrics. AIMATDESIGN (`npj Comput Mater 2026`,
  https://www.nature.com/articles/s41524-025-01894-x) shows shared-budget
  prediction count `SR_legal / SR_cls / SR80 / SR_done` as the headline panel.
- No `mu_CO` phase diagram. The operando claim in the manuscript needs at
  least 4-5 chemical-potential points to support a phase boundary, not 2.
- No physical validation of the discovered surface. `sro_analysis.py` ships
  Warren-Cowley parameters but the best `feasible` structures from week 5 / 6
  have never been audited. *Nat Commun 2021* on adsorbate-driven Cu-Pd
  restructuring (https://www.nature.com/articles/s41467-021-21555-z) is the
  closest physical reference and should be the comparator.
- No oracle-call / wall-clock accounting. HDRL-FP (`Nat Commun 2024`,
  https://www.nature.com/articles/s41467-024-50531-6) reports both, and the
  reviewers will ask.

These are not new ideas. Each is a finishing touch on the Pareto story we
already have.

## 2. Phases and acceptance criteria

The phases are independent and can be queued in parallel (Phase 1 and 3 share
GPU, Phase 2 is CPU-friendly, Phase 4 is offline).

### Phase 1: Statistical power (raise headline cells to n=5)

Add training seeds `44, 55` to the bounded-mask 4k-budget runs that are
currently at `n=3`:

- `mu_CO = -0.2`, `max_deviation = 2` (does not yet exist at 4k -> seeds 11,22,33,44,55)
- `mu_CO = -0.2`, `max_deviation = 6`, 4k -> seeds 44, 55
- `mu_CO = -0.6`, `max_deviation = 2`, 4k -> seeds 44, 55
- `mu_CO = -0.6`, `max_deviation = 6`, 4k -> seeds 44, 55

Acceptance: each headline cell has `n>=5` and `ci95_feasible_best_omega < 5 eV`.
Driver: `week7_extra_seeds_queue.ps1`.

### Phase 2: Budget-matched random / SA / fixed-swap baselines under feasibility metrics

Run random search and simulated annealing against the *same* env config as
`mutation_delta_strict_stop_masked` (with the chosen `max_deviation`), at
matched oracle-call budgets `2048` and `4096`, recording the same
`feasible_best_omega`, `mean_constraint_d_frac`, etc. Output schema must match
the `standard_eval_summary.csv` produced by `week4_action_reward_ablation.py`
so the existing `week6_envelope_report.py` can ingest it without changes.

Driver: `week7_baselines_feasible.py` -> writes per-seed and per-train-seed CSV
under `checkpoints/week7_baselines_<method>_<mu>_<budget>/`.

Acceptance: at each `(mu_CO, budget)`, MaskablePPO+mask leads random and SA by
at least 5 eV in `mean_feasible_best_omega` with non-overlapping CI95.

### Phase 3: mu_CO phase scan

`mu_CO in {-0.2, -0.4, -0.6, -0.8, -1.0}` x `max_deviation = 4` x seeds
`{11, 22, 33}` at 4k budget. The two endpoints `-0.2, -0.6` already exist and
will be reused, so this phase only needs `-0.4, -0.8, -1.0`.

Driver: `week7_mu_phase_diagram_queue.ps1`.

Acceptance: monotone trend of `n_CO`, `theta_Pd_surface`, and best feasible
Omega as a function of `mu_CO`, with phase boundary visible (kink) where the
CO surface coverage saturates.

### Phase 4: Best-structure physical audit (Warren-Cowley + layer-resolved Pd fraction)

Reload `latest_model.zip` from each completed run, replay one greedy episode,
extract the metal slab at the step with minimum *feasible* Omega, and:

- Save the structure as `.cif` and `.xyz` next to the run directory.
- Compute Warren-Cowley `alpha(Cu-Cu)`, `alpha(Cu-Pd)`, `alpha(Pd-Pd)` for
  the metal subsystem using the existing `chem_gym.analysis.sro_analysis.calculate_wcp`.
- Compute the per-layer Pd fraction (top, sub, sub2, sub3) and surface-vs-bulk
  ratio.
- Aggregate to a CSV indexed by `(mu_CO, max_deviation, train_seed)` and a
  Markdown report.

Driver: `week7_structure_audit.py`.

Acceptance: at strong CO chemical potential (`mu_CO = -1.0`), the audit shows
`Pd_top / Pd_bulk >= 1.5` and a non-zero `alpha(Pd-Pd)` clustering signal. At
weak CO chemical potential (`mu_CO = -0.2`), Pd is more dispersed
(`alpha(Pd-Pd) ~ 0`). This recovers the literature-known adsorbate-driven
segregation trend and is the physical-validation paragraph in the manuscript.

### Phase 5: Aggregate paper-grade report

Single Markdown that pulls together the four upstream CSV products into the
five tables a reviewer expects:

1. Headline: `MaskablePPO+mask` vs `Random` vs `SA` vs `Fixed-Swap` at matched
   budgets.
2. Pareto: feasible best Omega vs `max_deviation` at both `mu_CO` and both
   budgets.
3. Phase: feasible best Omega, theta_Pd_surface, n_CO across `mu_CO`.
4. Structure: Warren-Cowley parameters and per-layer Pd fraction across
   `mu_CO` and `max_deviation`.
5. Compute: oracle calls per method, wall-clock seconds per method, throughput
   ratio of MaskablePPO vs Random vs SA.

Driver: `week7_pareto_report.py` -> `checkpoints/week7_pareto_report/REPORT.md`
plus the four input CSVs aggregated as a single `summary.csv` for the paper
SI.

Acceptance: a single Markdown table set ready to drop into the manuscript LaTeX
without manual editing.

## 3. File map

```
ProjectMain/
  week7_paper_pareto_plan.md             <-- this file
  week7_baselines_feasible.py            <-- random + SA driver, feasibility-aware
  week7_extra_seeds_queue.ps1            <-- Phase 1 launcher
  week7_mu_phase_diagram_queue.ps1       <-- Phase 3 launcher
  week7_structure_audit.py               <-- Phase 4 driver
  week7_pareto_report.py                 <-- Phase 5 aggregator
  checkpoints/
    week7_baselines_random_m02_2k/       <-- Phase 2 outputs
    week7_baselines_random_m02_4k/
    week7_baselines_sa_m02_2k/
    week7_baselines_sa_m02_4k/
    week7_baselines_random_m06_2k/
    week7_baselines_random_m06_4k/
    week7_baselines_sa_m06_2k/
    week7_baselines_sa_m06_4k/
    week7_phase_md4_m04_4k_s3/           <-- Phase 3 outputs
    week7_phase_md4_m08_4k_s3/
    week7_phase_md4_m10_4k_s3/
    week7_extra_seeds_<...>/             <-- Phase 1 outputs
    week7_structure_audit/               <-- Phase 4 outputs
    week7_pareto_report/                 <-- Phase 5 outputs
```

## 4. Compute budget estimate

| Phase | Runs | Steps per run | Wall-clock per run (RTX-class GPU) | Total |
| --- | ---: | ---: | ---: | ---: |
| 1 | 12 | 4096 | ~1.5 h | ~18 h |
| 2 | 16 | 2048 / 4096 | ~10 min CPU each (no model fwd) | ~3 h |
| 3 | 9 | 4096 | ~1.5 h | ~14 h |
| 4 | ~30 (replays) | <100 ds steps | ~1 min each | ~30 min |
| 5 | 1 | offline | minutes | minutes |

Total: roughly two GPU-days plus a few CPU-hours, which is realistic to push
through over a single weekend or as overnight queues.

## 5. Order of operations

1. Land this plan + the four scripts in a single commit so the code review
   surface is small.
2. Smoke-test each script with `--help` and (for `week7_baselines_feasible`)
   a 32-step EMT-fallback sanity run (no oracle, no GPU). This catches
   import/argparse breakage without spending oracle calls.
3. Push to `origin/main`.
4. Launch Phase 2 (CPU-only) immediately; queue Phase 1 + 3 on GPU; run
   Phase 4 once any model checkpoint is available; assemble Phase 5 last.
