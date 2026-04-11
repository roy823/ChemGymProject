# Week 5 Top-Journal Gap Analysis

Date: 2026-04-11

## Literature-derived standards

The recent materials/catalysis RL literature does not evaluate models by `explained_variance` alone.
The stronger papers instead report a combination of:

1. Objective quality under a fixed compute or oracle budget.
2. Constraint legality or physical validity.
3. Multi-seed robustness.
4. Convergence speed and throughput.
5. Downstream physics or experimental validation.

Relevant primary-source signals:

- AIMATDESIGN (`npj Comput Mater`, 2026) reports legality, task success, and end-to-end success under a fixed prediction budget:
  `SR_legal`, `SR_cls`, `SR80`, `SR_done`, with a shared budget of 128,000 ML predictions.
- Deep RL for inverse inorganic materials design (`npj Comput Mater`, 2024) explicitly compares validity, diversity, objective quality, sample efficiency, and the bias-variance tradeoff, then recommends downstream DFT/first-principles validation rather than relying on surrogate rewards alone.
- HDRL-FP (`Nat Commun`, 2024) averages over five independent runs, reports convergence speed in wall-clock time, and emphasizes generalizable state/action design plus stable training under large parallel throughput.
- Invalid action masking (Huang and Ontanon, 2022) gives theoretical and empirical support for masking invalid actions instead of learning legality only through reward penalties.
- Constrained Policy Optimization (Achiam et al., 2017) argues that many RL problems are better expressed as reward plus explicit constraints instead of trying to encode all behavior in the reward.

Implication for this project:
paper-grade evaluation should prioritize `best_omega`, validity/constraint satisfaction, oracle budget, and seed robustness.
`explained_variance` remains a critic-health diagnostic, not the main scientific metric.

## Current closed-loop evidence

Trusted post-fix mainline:

- Action design: `mutation_delta_strict_stop`
- Reward design: `pure_delta_omega`

Reason:

- At `mu_CO = -0.2 eV`, 512-step standardized evaluation gave
  `mean_best_omega = -337.33`
  with `mean_noop_ratio = 0.0148`
  from `checkpoints/week5_pirpfix_m02_compare_v1/standard_eval_profile_summary.csv`.
- At `mu_CO = -0.6 eV`, 512-step standardized evaluation gave
  `mean_best_omega = -299.54`
  with `mean_noop_ratio = 0.0`
  from `checkpoints/week5_pirpfix_m06_compare_v1/standard_eval_profile_summary.csv`.
- Competing post-fix profiles with `noop` remained worse, and PBRS still has no post-fix evidence of beating pure `delta_omega`.

New long-budget validation completed in this iteration:

- At `mu_CO = -0.2 eV`, 2048-step standardized evaluation gave
  `mean_best_omega = -345.70`
  and `best_omega_global = -353.05`
  from `checkpoints/week5_longrun_m02_strictstop_2k_s2/standard_eval_profile_summary.csv`.
- At `mu_CO = -0.6 eV`, 2048-step standardized evaluation gave
  `mean_best_omega = -339.42`
  and `best_omega_global = -346.51`
  from `checkpoints/week5_longrun_m06_strictstop_2k_s2/standard_eval_profile_summary.csv`.

Budget scaling result:

- `mu_CO = -0.2 eV`: 2048-step training improved `mean_best_omega` by about `8.37 eV` versus the 512-step strict-stop baseline.
- `mu_CO = -0.6 eV`: 2048-step training improved `mean_best_omega` by about `39.88 eV` versus the 512-step strict-stop baseline.

New fixed-composition control:

- A fair `swap_delta_strict_stop` baseline was added so that fixed-composition local search uses the same strict-stop protocol as the mutation mainline instead of carrying an extra noop path.
- At `mu_CO = -0.2 eV`, the completed fixed-composition train seeds currently give
  `mean_best_omega = -323.25 eV` for `seed_11`,
  `-323.27 eV` for `seed_22`,
  and `-323.25 eV` for `seed_33`,
  from
  `checkpoints/week5_swapctrl_m02_strictstop_2k_s3/swap_delta_strict_stop/seed_11/standard_eval_summary.csv`
  and
  `checkpoints/week5_swapctrl_m02_strictstop_2k_s3/swap_delta_strict_stop/seed_22/standard_eval_summary.csv`
  and
  `checkpoints/week5_swapctrl_m02_strictstop_2k_s3/swap_delta_strict_stop/seed_33/standard_eval_summary.csv`.
- The current 3-seed fixed-composition summary is
  `mean_best_omega = -323.2575 eV`
  with `mean_constraint_valid_frac = 1.0`
  from `checkpoints/week5_swapctrl_m02_strictstop_2k_s3/standard_eval_profile_summary.csv`.
- The matched open-composition mainline at the same budget gave
  `mean_best_omega = -345.01 eV` for `seed_11`
  and `-346.39 eV` for `seed_22`
  from `checkpoints/week5_longrun_m02_strictstop_2k_s2/standard_eval_by_train_seed.csv`.
- Over the matched completed seeds, the mutation mainline is better by about `22.44 eV` in `mean_best_omega`.
- Over the currently available root summaries, the open-composition mean remains better by about `22.45 eV`.

Interpretation:

- This gap is too large to explain as a critic artifact.
- The gain is coming from allowing local composition reconfiguration under the grand-potential objective, not just from giving PPO a cleaner stop action.
- That is a materially stronger scientific claim because it aligns with operando adsorbate-induced segregation and restructuring physics rather than a purely algorithmic tuning story.

Critic-health signal:

- Recent 2048-step runs ended with substantially higher `train/explained_variance` than the short-budget runs:
  `0.621` and `0.822` for the two `mu=-0.2` runs
  `0.905` and `0.926` for the two `mu=-0.6` runs
  from TensorBoard runs `MaskablePPO_Experiment_100` to `103`.

Interpretation:

- The main bottleneck was not that the action or reward design was fundamentally wrong.
- The short 512-step budget was undertraining the critic and truncating policy improvement.
- Longer budget materially improved both the target objective and critic fit.

## What is already strong enough for a paper

- Clear post-fix action-design conclusion: strict stop is better than leaving a noop-like continuation path.
- Clear post-fix reward-design conclusion so far: pure `delta_omega` is a stronger mainline than the current PBRS variant.
- Real hybrid-oracle closed-loop evidence at two chemical potentials.
- Standardized multi-seed evaluation exists and is no longer based on a single lucky run.

## What is still below top-journal standard

1. Seed count is still thin.
   Most of the strongest long-budget evidence currently uses 2 train seeds, while stronger papers often average 5 runs or more.

2. Constraint reporting is not yet front-and-center.
   We track several diagnostics internally, but the paper-grade table should explicitly include legality/constraint success metrics, not only `omega`.

3. Oracle-budget accounting is not yet a headline metric.
   Top papers usually normalize performance by expensive calls, prediction budget, or wall-clock cost.

4. Fixed-composition rigor is missing from the main comparison.
   The current mutation setting allows composition drift. That is valid for a grand-canonical search claim, but weaker if the manuscript claim is about optimizing within a fixed alloy loading.
   The new strict-stop swap control is the right direction and should now be completed at both chemical potentials.

5. Post-fix PBRS is not fully sealed.
   The strongest negative result against PBRS is still short of a full long-budget multi-seed closure.

## Next optimization direction

The next step should not be "invent a new reward" or "rewrite the policy network".
The data now support a more disciplined sequence:

1. Promote `mutation_delta_strict_stop + pure_delta_omega` to the formal mainline.

2. Run the mainline at paper-grade budget:
   `2048-5000` train steps, `5` train seeds, at both `mu_CO = -0.2` and `-0.6 eV`.

3. Add a paper-facing constraint table:
   report `best_omega`, `best_omega_global`, `noop_ratio`, composition deviation, coverage validity, and thermodynamic-consistency checks.

4. Add a fixed-composition control:
   rerun a `swap`-style or otherwise composition-constrained baseline so the manuscript can distinguish
   grand-canonical discovery from fixed-loading optimization.

5. Only after step 2 and step 3, revisit critic surgery if needed.
   If long-budget `explained_variance` still fails to stabilize above roughly `0.8` across seeds, then test critic-side interventions such as normalization/tuning or a more decoupled policy-value update schedule.

## Immediate experiment queue

Recommended order:

1. `mutation_delta_strict_stop`, `mu_CO=-0.2`, `5` seeds, `2048-5000` steps.
2. `mutation_delta_strict_stop`, `mu_CO=-0.6`, `5` seeds, `2048-5000` steps.
3. Fixed-composition control under matched budget.
4. Long-budget post-fix `mutation_pbrs_strict_stop` only as a reward ablation, not as the mainline.

## Source links

- AIMATDESIGN: https://www.nature.com/articles/s41524-025-01894-x
- Deep RL for inverse inorganic materials design: https://www.nature.com/articles/s41524-024-01474-5
- HDRL-FP: https://www.nature.com/articles/s41467-024-50531-6
- Adsorbate-driven alloy surface restructuring example: https://www.nature.com/articles/s41467-021-21555-z
- Invalid action masking: https://arxiv.org/abs/2006.14171
- Constrained Policy Optimization: https://proceedings.mlr.press/v70/achiam17a.html
- PPO in cooperative games: https://arxiv.org/abs/2103.01955
