# Tier-2 Locked Experimental Plan v1.2

Date locked: 2026-04-25 (final)
Supersedes: `week8_locked_plan_v1_1.md` (which itself superseded v1.0)
Author of this revision: Xu Chengrui + Claude
Compute target: single RTX-4060 laptop GPU, ~2 weeks, no DFT
Status: **CONTRACT**. Any further change requires v1.3.

---

## 0. Revision scope

v1.2 finalizes the twelve open issues identified in the v1.1 audit. Decisions
are locked here with rationale; no more "yes/no" rounds. Anything not
overridden below is inherited from v1.1.

---

## 1. The twelve finalized decisions, with rationale

### D1. TOST equivalence margin: `epsilon = 3.0 eV`

**Rationale**:

The TOST margin must be (a) physically meaningful, (b) larger than the
unavoidable measurement noise, and (c) smaller than the smallest scientifically
relevant inter-method gap.

- DFT chemical accuracy is conventionally `0.05 eV/atom` (Norskov et al.,
  Nature Mater. 2009; Reuter & Scheffler, PRB 2002).
- Active sites in our slab: `N_active = 4 layers × 4 × 4 = 64`.
- Therefore `epsilon_phys = 0.05 eV/atom × 64 atoms = 3.2 eV`, rounded to
  **`epsilon = 3.0 eV`**.
- Sanity check upper: smallest scientifically meaningful inter-method gap
  reported in the Cu-Pd-CO MC literature is ~5-10 eV per `(4 × 4)` slab
  unit. `epsilon = 3 eV` is comfortably below this.
- Sanity check lower: typical seed-to-seed CI95 on `feasible_best_omega`
  in week6 data is `2-9 eV` for bounded mask experiments. With `n=5` and
  `CI95 ~ 5 eV`, `epsilon = 3 eV` requires PPO and canonical MC means to
  be within `~3 eV` to declare equivalence; in practice this needs the
  data to actively support equivalence (not just fail to reject).

**Locked**: `epsilon = 3.0 eV` for all paired-TOST decisions in Exp-1
static parity. Reported with two one-sided p-values (lower and upper).

### D2. Dynamic statistical structure: `n_run_seeds = 7`, paired Wilcoxon on per-seed mean as main; LME as SI

**Rationale**:

v1.1 set `n_run_seeds = 5` and proposed paired Wilcoxon on `(seed × schedule) = 25`
pairs. This is pseudoreplication: the 5 schedules within one seed share the
same trained PPO model, so observations are not independent across schedule.

Two cleaner options:

1. Aggregate per `run_seed` (mean over 5 schedules) → paired Wilcoxon on `n` pairs.
   Min two-sided p at `n=5` is `2/2^5 = 0.0625` — never reaches 0.05.
   Min two-sided p at `n=7` is `2/2^7 = 0.0156` — passes 0.05 with 7/7
   directional consistency.
2. Linear mixed-effects model with random intercept per `run_seed` and fixed
   effect for method (Henderson et al., AAAI 2018; Agarwal et al., NeurIPS
   2021 - "Deep Reinforcement Learning at the Edge of the Statistical
   Precipice"). Properly handles clustering.

**Locked**:

- Main test: `n_run_seeds = 7` for Exp-2 only. Paired Wilcoxon signed-rank
  on the per-`run_seed` mean cumulative omega, one-sided alternative
  `H1: PPO < canonical_MC`.
- Train seeds for Exp-2 = `{11, 22, 33, 44, 55, 121, 144}`.
  (n=5 is preserved for Exp-1 / Exp-3 / Exp-4 since TOST and bootstrap CI
  on n=5 are well-behaved; only Exp-2 needs more power because Wilcoxon at
  n=5 cannot reject at alpha = 0.05.)
- SI test: `statsmodels.formula.api.mixedlm` with formula
  `cumulative_omega ~ method + (1 | run_seed)`, treating
  `(run_seed, schedule_id)` as the unit. Reported as sensitivity
  analysis.

**Compute cost**: +2 PPO training runs at 4096 × 0.7s ≈ +1.6 GPU-h. Negligible.

### D3. Replica-Exchange MC budget: main is fair (4 × 1024 = 4096 pooled), SI is stress test (4 × 4096 = 16384 pooled)

**Rationale**:

Two competing fairness conventions:

- Wall-clock parity → 4096 pooled distinct evaluations across replicas
  (each replica gets 1024). Fair to PPO and canonical_MC, but RE-MC's
  replicas are short and may not equilibrate.
- Per-replica parity → each replica gets 4096 calls (16384 pooled). RE-MC
  has 4× the compute, but each chain is properly equilibrated. Unfair to
  PPO at 4096.

The fair option produces an "RE-MC at small budget" data point. The stress
test option produces an "RE-MC at extended budget" data point. Both are
informative, neither alone is conclusive.

**Locked**:

- Main table: RE-MC at 4 replicas × 1024 = 4096 pooled. Reported alongside
  PPO and canonical_MC.
- SI table: RE-MC at 4 replicas × 4096 = 16384 pooled (4× compute). Shows
  what happens when the MC baseline is given more search budget than RL.

**Compute cost (SI)**: 4 × 4096 × 0.7s = ~3.2 GPU-h per `(mu_CO, seed)` cell.
With 2 mu × 5 seeds (n=5 for SI is sufficient since this is sensitivity, not
the main claim) = 10 cells × ~3.2 = ~6 GPU-h.

### D4. Adaptation time: method-specific equilibrium anchor

**Rationale**:

v1.1 used "equilibrium reached during sustained operation" as anchor — but
the 50-step segments in our schedule do not reliably reach equilibrium, so
this is self-referential.

Alternative considered: anchor to canonical_MC's static equilibrium for both
methods. But this measures "how fast does each method reach MC's equilibrium"
which is biased: PPO might converge to a state below MC's equilibrium and
report `adaptation_time = 0`, masking the actual response speed.

**Locked**: each method has its own equilibrium reference, computed from
**static experiments at the same `mu_CO`**:

```
For each method m in {PPO, canonical_MC, RE-MC, random_swap}:
  For each mu in {-0.2, -0.4, -0.6, -0.8} eV:
    Omega_eq[m, mu] = mean Omega over the last 20 steps of the static run
                     of method m at chemical potential mu (Exp-1 + Exp-3,
                     n=5 seeds, take cross-seed mean)

For a dynamic rollout where segment t_seg starts at step t_start with
chemical potential mu_seg:
  adaptation_time[t_seg] = min{ k in {1, ..., 50} :
                                 |mean(Omega[t_start..t_start+k]) - Omega_eq[m, mu_seg]| < 1.0 eV }
                          or 50 if no such k exists.
```

Each method tracks its own optimum. The cross-method comparison reads:
"under a `mu_CO` step, PPO reaches its own asymptote in `X` steps;
canonical_MC reaches its own asymptote in `Y` steps". This is a clean
comparison of *response speed*, not of *absolute final value* (which is
captured separately by `cumulative_omega`).

The `1.0 eV` tolerance is chosen as roughly `0.015 eV/atom × 64 = 1 eV`,
~3x tighter than the TOST `epsilon` because adaptation is a within-method
metric not requiring inter-method equivalence margin.

### D5. PIRP auxiliary track: removed; reuse week6 data

**Rationale**:

v1.1's `5.4 Auxiliary mutation-only SI` adds 12 new training runs purely to
demonstrate PIRP under the mutation regime. We already have n=3 mutation+PIRP
data at md=4, 4096 budget for both `mu_CO ∈ {-0.2, -0.6}` from week6
(`week6_boundedopen_m02_md4_4k_s3`, `week6_boundedopen_m06_md4_4k_s3`).
Rerunning is wasteful.

**Locked**: SI Section "PIRP under mutation regime" cites week6 results
directly. No new runs. Saves ~10 GPU-h.

### D6. Five dynamic test schedules: explicitly enumerated and frozen

**Rationale**:

v1.1 specified `seeds {7001..7005}` to generate 5 random staircases. This
relies on RNG behavior of `numpy.random.default_rng` which is reproducible
but opaque. Listing the 5 schedules explicitly maximizes reproducibility
and lets reviewers see the test set without running code.

The 4 chemical potentials `{-0.2, -0.4, -0.6, -0.8} eV` give `4! = 24` distinct
permutations. The 5 schedules below were chosen by hand to maximize coverage of
qualitatively distinct transition patterns:

| Schedule | Segment 1 (t=0..49) | Segment 2 (t=50..99) | Segment 3 (t=100..149) | Segment 4 (t=150..199) | Pattern |
| --- | ---: | ---: | ---: | ---: | --- |
| **A_inc** | -0.2 | -0.4 | -0.6 | -0.8 | monotone increasing in `\|mu\|` |
| **B_dec** | -0.8 | -0.6 | -0.4 | -0.2 | monotone decreasing in `\|mu\|` |
| **C_zig1** | -0.4 | -0.2 | -0.8 | -0.6 | mid-low-high-mid |
| **D_zig2** | -0.6 | -0.8 | -0.2 | -0.4 | mid-high-low-mid |
| **E_swing** | -0.2 | -0.8 | -0.4 | -0.6 | large-amplitude alternation |

A and B test monotone trends. C and D test moderate zigzags going through
opposite mid points. E tests the largest single-step amplitude transition
(`-0.2 → -0.8`, which is a 0.6 eV jump in `mu_CO`).

**Locked**: schedules A through E above, deterministic and identical for all
methods. Each method × each `run_seed` × each schedule = one rollout.

### D7. Headline metric naming for swap-only protocols: `best_omega`, not `feasible_best_omega`

**Rationale**:

Under swap-only action mode + fixed composition target, every visited state
satisfies the composition constraint by construction. The
`feasible_best_omega` versus `best_omega` distinction is meaningful only for
mutation-mode experiments where the mask can be violated. Keeping
`feasible_` prefix in main-table headers wastes column space and confuses
the central narrative.

**Locked**: main table T1 column header is `best_omega`. A footnote states
"under fixed-composition swap-only protocols, `feasible_best_omega` and
`best_omega` are identical by construction; we retain the `feasible_*`
schema in CSVs for cross-experiment compatibility." SI mutation-PIRP track
keeps the `feasible_*` distinction.

### D8. Bootstrap-CI as the descriptive companion to mean (replacing parametric Welch CI)

**Rationale**:

For `n = 5` or `n = 7`, parametric Welch t-CIs assume normality of the per-seed
sampling distribution, which is rarely justifiable for highly-multimodal
search problems. Modern ML benchmark literature (Agarwal et al., NeurIPS 2021)
recommends bootstrap-CIs and IQM (Inter-Quartile Mean). For our small `n`
the IQM degenerates to the median, but bootstrap-CIs remain meaningful.

**Locked**: every main-table mean is reported as
`mean ± [bootstrap 95% CI]` from `B = 10000` resamples (BCa method). Median
is reported as a secondary descriptive in T1 / T3 / T4. Parametric Welch
intervals are reported in SI for direct comparability with v1.1.

### D9. Mainline ablation B (no_strict_stop_swap) semantics: explicit

**Rationale**:

v1.1 §5.4 wrote "swap + noop / soft-stop continuation" without parameters,
leaving multiple interpretations.

**Locked**: B = `swap` action mode with the following config delta versus A:

| parameter | A (full) | B (no_strict_stop) |
| --- | --- | --- |
| `enable_noop_action` | False | True |
| `stop_terminates` | True | False |
| `min_stop_steps` | 8 | 0 (irrelevant since stop disabled) |

Under B, episodes run to `max_steps_per_episode = 32` and the agent can
interleave noop with swap, but cannot terminate the episode early. Reward
during noop steps is zero (no `delta_omega`). All other reward and oracle
parameters are identical to A.

### D10. Resume verification: pilot-based, with quantitative tolerance

**Rationale**:

Bit-exact resume is infeasible with SB3 + VecNormalize + non-deterministic
GPU kernels. The acceptance criterion must be statistical.

**Locked**: pre-launch pilot test:

1. Train PPO_swap on `mu_CO = -0.6`, `run_seed = 11`, total budget = 4096,
   uninterrupted. Record final `best_omega` (call it `Omega_uninterrupted`).
2. Train the same configuration but interrupt at step 2048; load checkpoint;
   resume training to step 4096. Record final `best_omega` (call it
   `Omega_resumed`).
3. Resume is verified iff `|Omega_uninterrupted - Omega_resumed| < 5 eV`.

The 5 eV tolerance is conservative (similar to the seed-to-seed CI95 in week6
data). Failing this gate blocks the long matrix until the resume bug is fixed.

### D11. SGCMC: relegated to a sensitivity SI section, not a baseline

**Rationale**:

SGCMC violates the fixed-composition story (composition fluctuates, even if
biased to a target mean). Putting it in the main "baseline" list creates
a tension between the central claim and the comparator set. Yet SGCMC is
useful as a sensitivity analysis: "if we relax composition strictness by
allowing ±2-atom fluctuation around the target, how much does Omega
improve, and does the dynamic gap survive?"

**Locked**:

- SGCMC removed from main Exp-1 method table.
- New SI Section "Sensitivity to compositional flexibility" reports SGCMC
  at `mu_CO = -0.6, T = 0.10 eV, target Pd_frac = 0.08, n = 5` for both
  static and dynamic protocols, comparing absolute `omega` and observed
  `Pd_frac` distribution width to the strict-fixed canonical MC.
- This is a 2 cell × 5 seeds × 4096 = ~6 GPU-h budget item; included.

### D12. RE-MC ladder: pilot-tunable top temperature only

**Rationale**:

The temperature ladder `T ∈ {0.05, 0.10, 0.20, 0.35} eV` is from textbook
spacing (factor ~2 between adjacent temperatures). Whether this gives a
healthy ~30% swap acceptance for our specific `Omega` landscape needs
empirical confirmation.

**Locked**:

- Default ladder: `T ∈ {0.05, 0.10, 0.20, 0.35} eV`.
- One pilot run per `mu_CO` at `n = 1 seed`; observe inter-replica swap
  acceptance.
- If acceptance for any adjacent pair `< 10%` or `> 60%`, ONE adjustment
  is allowed: change only the highest temperature within
  `{0.30, 0.35, 0.40, 0.50} eV`. All other ladder temperatures fixed.
- After adjustment, the rest of Exp-1 RE-MC runs use the locked ladder. No
  further tuning.

This is a pre-registered single-knob calibration, not a fishing expedition.

---

## 2. Locked statistical protocol

The full main-text statistical reporting protocol, in one place:

### 2.1 Descriptive statistics

For every cell `(method, mu_CO)`:

- Mean ± bootstrap 95% CI (BCa, B = 10000)
- Median (secondary)
- Best run (lowest omega across the seeds at that cell)
- Worst run

### 2.2 Static parity (Exp-1)

Decision test: paired TOST on differences `d_i = omega_PPO_i - omega_canonical_MC_i`,
where `i` indexes `run_seed` (`n = 5`, paired by seed).

- Two one-sided tests:
  - lower: `H0: d <= -epsilon` vs `H1: d > -epsilon`, `epsilon = 3.0 eV`
  - upper: `H0: d >= epsilon` vs `H1: d < epsilon`
- Both p < `alpha_local` → declare equivalence
- `alpha_local = 0.05 / 2` (one-sided each, total Type-I = 0.05)

Multiplicity: BH-FDR over the four cells in the main family (PPO vs
canonical_MC × 2 mu) ∪ (PPO vs RE-MC × 2 mu) = 4 TOST decisions × 2 sides
= 8 p-values controlled together.

If TOST fails (cannot declare equivalence) but PPO is also not significantly
worse via paired Wilcoxon (one-sided), report as "indeterminate within
margin" — neither equivalent nor inferior — and discuss in text.

### 2.3 Dynamic superiority (Exp-2)

Decision test: paired Wilcoxon signed-rank on differences
`d_i = cumulative_omega_PPO_i - cumulative_omega_canonical_MC_i` where
`i` indexes `run_seed` after averaging over the 5 test schedules within
each seed (`n = 7`).

- One-sided alternative: `H1: PPO < canonical_MC`
- p < 0.05 → declare superiority

Effect size: Hodges-Lehmann delta + matched-pairs rank-biserial correlation.

Sensitivity (SI): linear mixed-effects model
`cumulative_omega ~ method + (1 | run_seed)` over `7 × 5 = 35` rollouts per
method, LR test for fixed-effect significance.

### 2.4 Ablation (Exp-4)

Paired t-test (or paired Wilcoxon for n=5 robustness) on differences
`d_i = omega_full_i - omega_ablation_i` per `mu_CO` cell. BH-FDR over 2 mu
× 1 ablation = 2 p-values.

Effect size: Hedges' g_z (paired version, less biased than Cohen's d at small n).

### 2.5 Phase diagram (Exp-3)

No formal hypothesis test for monotonicity at this scale. Reported with
mean ± bootstrap CI and visual inspection. Spearman rank correlation
between `mu_CO` and `theta_Pd_surface` reported as descriptive (n=4
mu points × 5 seeds = 20).

### 2.6 Multiplicity bookkeeping

Three orthogonal correction families:

1. **Exp-1 main family**: 4 TOST decisions (PPO vs each of canonical_MC,
   RE-MC) at 2 mu, BH-FDR.
2. **Exp-2 main family**: 1 paired Wilcoxon (PPO vs canonical_MC). No
   correction needed (n_test = 1).
3. **Exp-4 main family**: 2 paired tests (1 ablation × 2 mu). BH-FDR.

SI families (SGCMC, RE-MC extended, mutation+PIRP, random_swap, ...) are
each their own correction family, reported separately.

### 2.7 Reporting hygiene

Every reported p-value carries:

- the raw p
- the BH-FDR adjusted p (within its declared family)
- the effect size (HL delta or Hedges' g_z)
- the bootstrap 95% CI of the effect size
- the test type and laterality

---

## 3. Locked test schedules (already in §1 D6, summary here)

| Schedule | mu_CO sequence (eV, 50 steps each) |
| --- | --- |
| A_inc | -0.2, -0.4, -0.6, -0.8 |
| B_dec | -0.8, -0.6, -0.4, -0.2 |
| C_zig1 | -0.4, -0.2, -0.8, -0.6 |
| D_zig2 | -0.6, -0.8, -0.2, -0.4 |
| E_swing | -0.2, -0.8, -0.4, -0.6 |

Each schedule = 200 oracle calls (no stop, no reset, fixed horizon).

---

## 4. Locked method definitions

### 4.1 PPO_swap (mainline)

```
action_mode = "swap"
use_masking = True
use_pirp = False               # forced; PIRP not yet implemented for swap
profile = "swap_delta_strict_stop"
gamma = 0.97, lam = 0.95
learning_rate = 2e-4
ppo_n_steps = 256
ppo_ent_coef = 5e-4
max_steps_per_episode (static) = 32, with strict_stop after 8 steps
horizon (dynamic) = 200, no stop, no reset
observation: standard graph features + (current mu_CO scalar appended in dynamic only)
```

### 4.2 canonical_MC (mainline)

```
action_mode = "swap" via swap proposals (i, j) with i < j, state[i] != state[j]
T = 0.10 eV (fixed; same in static and dynamic; not annealed)
proposal: uniform over valid swap pairs
acceptance: P = min(1, exp(-Delta_Omega / kT))
no learned model, no schedule awareness
budget counted as distinct oracle evaluations (cache-aware)
static: stop+reset allowed at chosen iteration count, max 32 steps before forced reset
dynamic: 200-step continuous chain, no stop, mu_CO updates each step
```

### 4.3 replica_exchange_MC (mainline)

```
4 replicas at T = {0.05, 0.10, 0.20, 0.35} eV (default; adjustable per D12)
each replica: independent canonical_MC chain
exchange schedule: every 64 proposals per replica, attempt swaps in alternating pattern:
  even rounds (0,1) and (2,3)
  odd rounds (1,2)
swap acceptance: P = min(1, exp((beta_i - beta_j)(E_i - E_j)))
budget: 4096 pooled distinct oracle evaluations across replicas (each ~1024)
SI extended: 16384 pooled (each ~4096)
report: best omega across all replicas
```

### 4.4 random_swap (SI sanity)

```
uniform sampling from valid swap pairs
no acceptance criterion
budget = same as other methods
purely a lower-bound calibration
```

### 4.5 SGCMC (SI sensitivity)

```
action_mode = "mutation" with semi-grand-canonical bias
T = 0.10 eV, target Pd_frac = 0.08
acceptance: P = min(1, exp(-(Delta_Omega + Delta_mu_CuPd × Delta_N_Pd) / kT))
Delta_mu_CuPd is solved offline so equilibrium mean Pd_frac ≈ 0.08
report: omega and observed Pd_frac variance
```

---

## 5. Experiment overrides from v1.1 (delta only)

### 5.1 Exp-1 — Static parity benchmark

| Item | v1.1 | v1.2 |
| --- | --- | --- |
| Headline column | `feasible_best_omega` | `best_omega` (per D7) |
| Statistical decision | paired TOST `epsilon = 5 eV` | paired TOST `epsilon = 3 eV` (per D1) |
| Descriptive CI | parametric CI95 | bootstrap 95% BCa CI (per D8) |
| RE-MC budget | 4096 pooled | 4096 pooled main + 16384 pooled SI extended (per D3) |
| SGCMC role | SI | SI (sensitivity, per D11) |
| Run seeds | `{11, 22, 33, 44, 55}` | unchanged (n=5) |
| Eval seeds | `{66, 77, 88, 99, 111}` | unchanged (n=5) |

### 5.2 Exp-2 — Dynamic operando benchmark

| Item | v1.1 | v1.2 |
| --- | --- | --- |
| Run seeds | `{11, 22, 33, 44, 55}` (n=5) | `{11, 22, 33, 44, 55, 121, 144}` (n=7, per D2) |
| Statistical unit | `(seed, schedule)` = 25 | per-seed mean = 7 (Wilcoxon main) + LME on 35 (SI sensitivity) |
| Test schedules | RNG seeds 7001-7005 (random) | hand-picked A through E (frozen, per D6) |
| Adaptation time anchor | "sustained operation equilibrium" | method-specific static `Omega_eq[m, mu]` from Exp-1+Exp-3 (per D4) |
| Training protocol | underspecified | `episode = 200 steps`, new staircase per episode, `~20 episodes` per training run, no stop/reset, total 4096 distinct oracle calls |

### 5.3 Exp-3 — Phase diagram

Unchanged at the experiment level. New runs at `mu_CO ∈ {-0.4, -0.8} eV`,
PPO_swap × 5 seeds × 4096 budget. Reused for `Omega_eq[PPO, mu]` anchors in
adaptation-time computation. Plus SI canonical_MC at all 4 mu × 3 seeds for
phase-diagram visual reference.

### 5.4 Exp-4 — Ablation

| Item | v1.1 | v1.2 |
| --- | --- | --- |
| Mainline configs | A_full_swap + B_no_strict_stop_swap | unchanged |
| B definition | underspecified | explicit (per D9): `enable_noop=True, stop_terminates=False` |
| Auxiliary mutation+PIRP track | new runs (n=3) | removed; week6 data reused (per D5) |
| Statistical test | paired t | paired t with paired Hedges' g_z (per D2 / D8) |

### 5.5 Exp-5 — Structure validation

Unchanged. Replay over all final structures from Exp-1, Exp-2, Exp-3, Exp-4
mainline + week6 reused mutation+PIRP. Computes Warren-Cowley α + per-layer
Pd fraction + new connected-component cluster analysis + CO-Pd coordination.

---

## 6. Compute budget reconciliation

| Item | Runtime | Note |
| --- | ---: | --- |
| Exp-1 main (PPO + canonical_MC + RE-MC, 4096 budget) | 36 h | n=5, 4 cells = (PPO+canonical+REMC) × 2 mu, +random_swap SI |
| Exp-1 SI: RE-MC extended (16384 pooled) | 6 h | per D3 |
| Exp-1 SI: SGCMC | 6 h | per D11 |
| Exp-2 main (PPO_swap_dyn n=7 + canonical_MC_dyn + random_swap_dyn) | 14 h | per D2 (+1.6 h vs v1.1) |
| Exp-3 phase diagram (PPO at new mu + canonical_MC SI) | 12 h | unchanged |
| Exp-4 mainline ablation (A reused, B new, n=5 × 2 mu) | 16 h | PIRP SI dropped per D5 |
| Exp-5 replay + cluster + CO-Pd CN | 4 h | unchanged |
| Resume pilot verification | 1 h | per D10 |
| RE-MC ladder calibration pilot | 1 h | per D12 |
| Buffer / failures / weather | 8 h | reduced from 10 h |
| **Total** | **104 h** | within 14-day target |

At 7-8 productive GPU-hours per day (Win11 + dev work alongside) this is
13-15 days. The buffer absorbs one major rerun.

If buffer is exhausted, cuts in order:

1. RE-MC extended SI (saves 6 h, lose stress test only)
2. SGCMC SI (saves 6 h, lose compositional flexibility sensitivity)
3. canonical_MC SI in Exp-3 (saves 4 h, lose phase-diagram visual reference)

---

## 7. Pre-submission checklist (replaces v1.1 §6)

All eleven conditions must be met before manuscript writing begins.

- [ ] **Static parity** (Exp-1): paired TOST with `epsilon = 3 eV` declares
      equivalence between PPO and canonical_MC in at least one mu cell;
      PPO is not significantly worse than either MC baseline in either cell
      after BH-FDR correction over the 4-cell main family
- [ ] **Dynamic gap** (Exp-2): paired Wilcoxon (n=7, one-sided) gives
      `p_PPO < canonical_MC < 0.05`; HL-delta `<= -5 eV`; rank-biserial
      effect size in [-1.0, -0.5]
- [ ] **Dynamic robustness** (Exp-2 SI): mixed-effects LR test for
      `method` fixed effect gives `p < 0.05`
- [ ] **Phase diagram** (Exp-3): Spearman rho between `mu_CO` and
      `theta_Pd_surface` is positive (i.e. monotone in the expected
      direction) with magnitude > 0.3
- [ ] **Mainline ablation** (Exp-4): no_strict_stop_swap is either
      worse by `>= 3 eV` (paired t with BH-FDR) or honestly reported as
      neutral
- [ ] **Structure physics** (Exp-5): paired test of `alpha_PdPd(mu=-0.6)`
      vs `alpha_PdPd(mu=-0.2)` shows significant difference
- [ ] **Protocol split enforced**: code review confirms static uses
      stop+reset, dynamic uses fixed-horizon
- [ ] **Disjoint seed sets**: train ∈ {11..55,121,144}, eval ∈
      {66,77,88,99,111}, dynamic test schedules A-E (deterministic)
- [ ] **Statistical hygiene**: bootstrap CIs, paired tests, BH-FDR per
      family, effect sizes with CI
- [ ] **Resume verified**: pilot interruption-and-resume passes the 5 eV
      tolerance
- [ ] **RE-MC ladder healthy**: pilot inter-replica swap acceptance is
      in `[10%, 60%]` for all adjacent pairs, after at most one
      top-temperature adjustment

---

## 8. Implementation order (replaces v1.1 §9)

Ordered by dependency. Each step produces one mergeable PR. No long
matrix runs until step 5 has passed its pilot.

1. **`week8_protocol.py`**: stop+reset symmetry layer (static), fixed-horizon
   wrapper (dynamic), distinct-oracle-call accounting utility, frozen
   schedule definitions A-E. Smoke: each utility produces correct
   trajectories on the 32-step EMT-fallback path.
2. **`week8_baselines_mc.py`**: canonical_MC, RE-MC, random_swap, SGCMC.
   Output schema: `running_min_omega_trace.csv`,
   `static_budget_summary.csv` (or `dynamic_rollout_summary.csv` under
   dynamic protocol). Smoke: all four methods produce non-NaN summaries
   on EMT.
3. **`week8_dynamic_env.py`**: random staircase wrapper (training),
   schedule lookup wrapper (eval, schedules A-E), `current mu_CO`
   observation hook for PPO. Smoke: dynamic env returns the correct
   `mu_CO(t)` for each schedule.
4. **`week8_significance.py`**: paired TOST, paired Wilcoxon (one-sided),
   paired t with Hedges' g_z, mixed-effects via `statsmodels`, BH-FDR.
   Bootstrap CI utility (BCa, B=10000). Smoke: TOST on synthetic data
   returns expected p-values within 5% of analytical.
5. **`trainer.py` resume support** (or `week8_resume.py` wrapper):
   load checkpoint, restore optimizer state, reset_num_timesteps=False,
   continue training. Smoke: D10 pilot test passes.
6. **`week8_pareto_v2_report.py`**: aggregator producing T1-T6 main
   tables, F1-F5 main figures, and the SI table family. No new
   computation, only CSV manipulation.

After step 6, the long matrix begins under section 7's checklist gating.

---

## 9. New risk register entries (append to v1.1 §5)

| ID | Trigger | Mitigation |
| --- | --- | --- |
| R8 | TOST never declares equivalence in any cell because PPO is consistently better than canonical_MC by `> epsilon` | Reframe paper: instead of "static parity + dynamic gap", report "static superiority + dynamic gap". This is a stronger claim, not a failure. Update narrative; no plan version bump required. |
| R9 | RE-MC ladder pilot fails in spite of D12 single-knob adjustment | Drop RE-MC from main table; keep canonical_MC as the only main MC baseline. Report failure as "RE-MC at our compute budget is not properly thermalized in this system" in SI. |
| R10 | n=7 still does not give Wilcoxon p < 0.05 in Exp-2 because the dynamic gap is small but consistent | Switch primary test to LME (already in SI); LR test on 35 paired rollouts has more power than Wilcoxon on 7 pairs. Pre-register this fallback before unblinding. |
| R11 | Bootstrap CI computation produces undefined intervals (e.g. all 5 seeds identical) | Fallback to parametric Welch CI for that cell; flag in the paper footnote. |

---

## 10. Why this plan is final

I considered three alternative directions and rejected each:

1. **Add Bayesian Optimization baseline**. BoTorch on a 64-dim binary state
   space requires a custom kernel and surrogate. Engineering cost > 1 week,
   well outside the 2-week target. Documented as future work in §8 of
   v1.0 (carried forward).

2. **Pre-train PPO on schedules with lookahead, then deploy without it**.
   Would create a stronger dynamic claim ("RL transfers anticipation
   from training to inference"), but conflates knowledge transfer with
   the parity narrative. Saved for a follow-up paper.

3. **Add Allegro / MACE as oracle**. Currently using FAIRChem UMA + OC25
   ensemble. Adding a separate ML potential for cross-validation would
   address the "no DFT validation" limitation, but requires retraining
   or downloading new checkpoints (we already have UMA-S, eSEN-sm,
   eSEN-md and have used the GPU memory budget). Documented as future
   work.

Within the 2-week + RTX-4060 + no-DFT envelope, this plan is the
maximum-information design that I can construct. It is internally
consistent (all twelve audit points addressed), statistically
defensible (TOST + paired Wilcoxon + LME + BH-FDR with bootstrap CIs),
and pre-registered (schedules and parameters frozen before any data
is unblinded).
