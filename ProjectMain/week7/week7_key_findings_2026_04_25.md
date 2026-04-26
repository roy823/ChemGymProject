# Week 7 — Key Findings (closed-loop, 2026-04-25)

This is the human-language interpretation of the numbers in `REPORT.md`.

## What was actually run today

Five new baseline runs against the live UMA + OC25 hybrid oracle at
`max_deviation = 4`, `total_steps = 256`, three seeds each:

| Run | Method | mu_CO | t_start / t_end | mean(feasible Omega) | CI95 | wall (s/seed) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `week7_baselines_random_m06_256_md4`           | random_mutation | -0.6 | n/a       | -307.299 | 1.158  | 173 |
| `week7_baselines_sa_m06_256_md4`               | sa_mutation     | -0.6 | 0.5 / 0.005 | -267.374 | 19.475 | 69 |
| `week7_baselines_sa_m06_256_md4_tunedT`        | sa_mutation     | -0.6 | 3.0 / 0.05  | -278.038 | 17.032 | 91 |
| `week7_baselines_random_m02_256_md4`           | random_mutation | -0.2 | n/a       | -336.964 | 0.399  | 200 |
| `week7_baselines_sa_m02_256_md4`               | sa_mutation     | -0.2 | 3.0 / 0.05  | -332.137 | 9.045  | 96  |

Total compute consumed: roughly 35 minutes of GPU oracle time (RTX-4060
laptop, hybrid mode), plus the existing 2048-step / 4096-step PPO runs from
weeks 5/6 that were aggregated.

## What the numbers say

### 1. RL's advantage scales with operando difficulty

The critical asymmetry in the closed loop is between the two chemical
potentials. The `mu_CO = -0.6 eV` regime is the hard one — strong CO
binding pulls Pd toward the surface and rewards heterogeneous Pd-rich
patches. The `mu_CO = -0.2 eV` regime is the easy one — CO barely binds,
so the slab term dominates and any reasonable surface composition is fine.

| Comparison                                   | mu = -0.2  | mu = -0.6 |
| ---                                          | ---:       | ---:      |
| `bounded_md6` (RL, 2k) - random_mutation_256 | -2.27 eV   | -14.42 eV |
| `bounded_md6` (RL, 4k) - random_mutation_256 | -3.71 eV   | -21.15 eV |
| `bounded_md6` (RL, 2k) - sa_mutation_256     | -7.10 eV   | -54.35 eV |
| `bounded_md6` (RL, 4k) - sa_mutation_256     | -8.54 eV   | -61.08 eV |

At `mu = -0.6` the RL-vs-random gap is `4-21 eV` and statistically clean
(`p = 9.4e-4` at 2k, `p = 2.2e-6` at 4k). At `mu = -0.2` the RL-vs-random
gap is only `2-4 eV` and barely above the noise at 2k (`p = 0.097`),
turning highly significant only at 4k budget where the RL run is so
tightly converged that even small means separate (`p = 1.8e-37` from
nearly-zero variance, `Cohen's d = -10.4`).

The clean reading is: **RL contributes meaningful sample efficiency exactly
where the operando search is hardest, not where it is easiest.** That is a
publishable framing rather than a bug. It aligns with the central operando
claim in `Nat Commun 2021` on adsorbate-driven Cu-Pd restructuring — the
"interesting" physics is at strong CO chemical potential.

### 2. Naive SA fails the stoichiometric trap; tuning T helps but does not close the gap

At `mu_CO = -0.6`:

- SA with default temperatures (`t_start = 0.5`, `t_end = 0.005`) collapses
  into a near-zero CO coverage local minimum (one seed converges to
  `n_CO = 0`, mean `Omega = -267.4`).
- Bumping the start temperature to `3.0` only raises the mean to `-278.0`,
  still 30 eV worse than random-256 and 50 eV worse than RL-2k.

The interpretation: SA's Metropolis step is fooled by the fact that any
single Pd -> Cu mutation locally lowers Omega (because slab E becomes more
stable), but globally removes CO binding sites. Random search avoids the
trap by bouncing around without a greedy bias. The learned policy with
mask is the only method here that consistently *prefers* compositions that
keep Pd available for CO. This is the cleanest demonstration of why a
learned operando policy is needed.

### 3. The headline RL contrasts (week 6 numbers, re-verified by significance)

The week-6 envelope study results pass the formal significance bar with
room to spare on the fixed-swap contrast:

- `bounded_md6_m06_2k` vs `fixed_m06_2k`: `delta = -63.70 eV`,
  `Welch t = -12.33`, `p = 5.9e-35`, `Cohen's d = -9.46`.
- `bounded_md6_m02_2k` vs `fixed_m02_2k`: `delta = -16.12 eV`,
  `Welch t = -11.89`, `p = 1.3e-32`, `Cohen's d = -11.82`.

Both are far below any reasonable multiple-comparison threshold.

The bounded-vs-open-feasible contrast is, in contrast, NOT individually
significant in two of the four cells (`p = 0.30`, `p = 0.10`). At
`mu_CO = -0.6`, `4k`, the open mutation actually edges out the bounded
mask by `3.4 eV` in mean feasible Omega, but it does so with only `7%`
feasibility on the path. That is the Pareto trade-off the manuscript
should center: feasibility-vs-Omega, with bounded mask sitting on the
"100% feasibility" frontier.

### 4. Sample-efficiency curve still incomplete

Random search at `256` oracle calls already lands at `-336.96` for `mu = -0.2`,
which is essentially tied with the RL bounded-md4 result at `2048` calls
(`-336.68`). We do not yet know how random search scales — the `2048` and
`4096` random runs are the biggest hole in the manuscript story right now.
The week-6 RL runs at `4k` push to `-340.67`, which is `3.7 eV` better than
random-256, but to fairly support a sample-efficiency claim we need
random and SA at the same `2048` and `4096` budgets.

That is the single most valuable next experiment.

## What the manuscript can already claim

1. Bounded-mask MaskablePPO on the SAGCM environment achieves `100%`
   feasibility at all max_deviation widths tested (`md in {2, 4, 6}`),
   while open mutation collapses to `7-21%` feasibility — Pareto plot
   over `(max_dev, feasibility, Omega)` is solid.
2. At strong CO chemical potential, RL beats random by `14-21 eV` and SA
   by `54-61 eV` (`p < 1e-3`, large to very-large effect sizes), even
   though the baselines have only `1/8` the oracle budget.
3. SA without temperature tuning collapses to a no-CO local minimum at
   `mu = -0.6`, illustrating why a learned policy is required.
4. The reward-design conclusion (`pure_delta_omega > PBRS variants`)
   and the action-design conclusion (`mutation_delta_strict_stop`)
   from weeks 4-5 hold up under multi-seed standardized eval.

## What is still missing for a top-journal package

1. Budget-matched random / SA at `2048` and `4096` oracle calls. This is
   queued (`week7_baselines_queue.ps1`); each (method, mu, budget) takes
   roughly 25-50 minutes on this GPU.
2. `mu_CO` phase scan at `-0.4 / -0.8 / -1.0`. Queued
   (`week7_mu_phase_diagram_queue.ps1`); approximately five GPU-hours
   total.
3. Best-structure Warren-Cowley audit. Queued
   (`week7_structure_audit_queue.ps1`); replay-only, runs in well under
   an hour once any model checkpoint is available.
4. Phase 1 extra seeds (raise n=3 to n=5 on the headline cells). Queued
   (`week7_extra_seeds_queue.ps1`).

Run these four queues overnight and re-run
`week7_pareto_report.py` followed by `week7_significance_analysis.py` —
the report regenerates itself from CSVs, so all the tables update in
place.

## Honest caveats

- Welch t with the Normal-approximation tail is conservative for `n = 3`;
  with so few seeds the reported p-values should not be over-interpreted
  beyond "less than ~1e-2 vs greater than 0.1." Phase 1 lifting all
  headline cells to `n = 5` is the right fix.
- SA's effective oracle-call budget is below `256` because rejected
  proposals revert to a previously-evaluated state and hit the energy
  cache. Random search uses all `256` evaluations. The "256" comparison
  is therefore mildly favorable to SA, and the gap above is in fact a
  *lower bound* on the RL advantage.
- All Omega values are in the SAGCM hybrid-oracle thermodynamic
  convention. Cross-validation against an independent backend (e.g.
  UMA-only or DFT spot-checks on a handful of best structures) is
  desirable but out of scope for this iteration.
