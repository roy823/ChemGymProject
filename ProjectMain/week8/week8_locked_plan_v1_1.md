# Tier-2 Locked Experimental Plan v1.1

Date locked: 2026-04-25
Supersedes: `week8_locked_plan_v1_0.md`
Author of this revision: Xu Chengrui + Codex
Compute target: single RTX-4060 laptop GPU, about 2 weeks, no DFT
Status: **CONTRACT** - all v1.0 content remains active unless explicitly replaced below.

---

## 0. Revision scope

This document is a decision-level replacement for v1.0, not a fresh plan from
zero. Any item not overridden here is inherited from
`week8_locked_plan_v1_0.md`.

The purpose of v1.1 is to close the eight unresolved execution gaps discovered
after the v1.0 audit:

1. `swap + PIRP` was not an implementable mainline under the current code path.
2. Static and dynamic evaluation semantics were not aligned with the paper
   claims.
3. The parity statistics were not using the correct decision framework.
4. `SGCMC` diluted the fixed-composition story.
5. The dynamic MC framing was overstated.
6. The spare compute budget was not allocated intentionally.
7. The BH-FDR correction family was not defined coherently.
8. The replica-exchange MC baseline needed a concrete, reproducible protocol.

---

## 1. Resolved decisions

These eight decisions are now locked for v1.1.

1. **Mainline method**
   - Mainline is now `vanilla MaskablePPO + swap + strict_stop`.
   - `PIRP` is removed from the fixed-composition mainline.
   - `PIRP` moves to a mutation-only auxiliary ablation / SI track.

2. **Headline metrics**
   - Static benchmark uses `running-min feasible Omega` over `4096` distinct
     oracle evaluations.
   - Dynamic benchmark uses `200` continuous steps with
     `cumulative Omega = sum_t Omega_t`.
   - Dynamic episodes use **no stop/reset**.

3. **Statistical decisions**
   - Static parity is decided by **paired TOST** with `epsilon = 5 eV`.
   - Dynamic superiority is decided by **paired Wilcoxon signed-rank**.
   - Main-table effect sizes are `Hodges-Lehmann delta` plus
     `matched-pairs rank-biserial correlation`.
   - `paired Hedges' g_z` is reported in SI only.

4. **Baseline scope**
   - `SGCMC` moves to SI.
   - Static main table uses `PPO_swap`, `canonical_MC`, `replica_exchange_MC`.
   - `random_swap` is a sanity baseline and moves to SI.

5. **Dynamic wording**
   - The dynamic story is now:
     `per-regime re-equilibration without amortization`.
   - We no longer say MC "starts from scratch" after each regime change.

6. **Extra 10 GPU-h**
   - Spare compute is allocated to:
     - adding `replica_exchange_MC` to the static main table
     - implementing true resume / restart logic before long runs

7. **Multiplicity control**
   - BH-FDR is applied per experiment on the **main hypothesis family only**.
   - SI-only comparisons are corrected in separate SI families.

8. **Replica-exchange MC protocol**
   - Default ladder is `T in {0.05, 0.10, 0.20, 0.35} eV`.
   - Swap attempts are sweep-based, not ambiguous "every 50 calls".
   - One exchange round is attempted after each replica completes `64`
     proposals.

---

## 2. Replacement central claim

Replace Section 0 of v1.0 with the following:

> Under the industrially realistic constraint of a fixed Pd composition, a
> swap-based MaskablePPO learns an amortized state-conditioned rearrangement
> policy that reacts to a time-varying CO chemical potential `mu_CO(t)`.
> At static `mu_CO`, the policy is performance-parity with equilibrium MC
> baselines within a pre-registered margin at the same oracle budget. At
> dynamic `mu_CO`, the policy achieves lower cumulative Omega than MC methods
> that must re-equilibrate under each regime change without amortized policy
> knowledge.

Interpretation:

- **Static parity** establishes physical / algorithmic credibility.
- **Dynamic cumulative-Omega gap** is the unique paper contribution.

---

## 3. Locked design replacements

Replace Section 1 of v1.0 with the following.

### 3.1 Mainline algorithm choice

Mainline method:

- `action_mode = "swap"`
- `use_masking = True`
- `use_pirp = False`
- `experiment profile = swap_delta_strict_stop`

Auxiliary mutation-only SI track:

- `action_mode = "mutation"`
- `use_masking = True`
- `use_pirp in {False, True}`
- This track is **not** part of the fixed-composition main claim.

### 3.2 Static and dynamic protocol split

Static experiments (`Exp-1`, static portions of `Exp-3`, static ablations):

- allow `strict_stop + reset`
- `max_steps_per_episode = 32`
- headline metric is `running-min feasible Omega` vs distinct oracle calls

Dynamic experiment (`Exp-2` only):

- fixed horizon = `200` steps
- `stop_terminates = False`
- no reset inside the rollout
- headline metric is `cumulative Omega`

### 3.3 Observation discipline

Dynamic PPO observation appends one scalar `current_mu_CO` to the per-node
feature vector.

- broadcast to all nodes
- no schedule lookahead
- no transition flag
- no history window

MC baselines receive the same information only through the current environment
state and current `mu_CO`.

### 3.4 Seed discipline

Unchanged from v1.0 and now explicitly binding:

- run / train seeds: `{11, 22, 33, 44, 55}`
- eval seeds: `{66, 77, 88, 99, 111}`
- dynamic test schedule seeds: `{7001, 7002, 7003, 7004, 7005}`

### 3.5 Statistics discipline

Static parity:

- paired unit: `run_seed`
- per-seed statistic: aggregate over the shared eval seeds
- decision test: paired `TOST` with `epsilon = 5 eV`

Dynamic superiority:

- paired unit: `(run_seed, schedule_id)`
- decision test: paired `Wilcoxon signed-rank`

Main-table effect sizes:

- `Hodges-Lehmann delta`
- `matched-pairs rank-biserial correlation`

SI-only effect sizes:

- `paired Hedges' g_z`

---

## 4. Paper structure replacement

Replace Section 2 of v1.0 with the following table.

| Section | Main figure / table | Source experiment | Story beat |
| --- | --- | --- | --- |
| 3.1 Static parity | F1, T1, T2 | Exp-1 | PPO is parity-level with equilibrium MC within a pre-registered margin |
| 3.2 Dynamic operando | F2, T3 | Exp-2 | PPO beats non-amortized re-equilibration MC under time-varying `mu_CO` |
| 3.3 Phase diagram | F3, T4 | Exp-3 | Fixed-composition operando trends remain physically sensible |
| 3.4 Discovered structures | F4, T5 | Exp-5 | Layer-resolved segregation and SRO descriptors support the physics story |
| 3.5 Ablation | F5, T6 | Exp-4 | `strict_stop` matters in the mainline; PIRP is mutation-only auxiliary evidence |

SI allocation:

- `SGCMC` static tables
- `random_swap` sanity baselines
- dynamic `RE-MC` stress test if completed
- mutation `PIRP` auxiliary ablation tables
- full trajectory grid
- `g_z` and additional parametric effect-size tables

---

## 5. Experiment overrides

This section replaces only the changed parts of Section 3 in v1.0.

### 5.1 Exp-1 - Static parity benchmark

#### Main methods

| Code name | Role | Description |
| --- | --- | --- |
| `PPO_swap` | main | MaskablePPO, swap-only, strict_stop, no PIRP |
| `canonical_MC` | main | single-temperature Metropolis under fixed composition |
| `replica_exchange_MC` | main | fixed-composition replica-exchange Metropolis |
| `random_swap` | SI | random valid swap proposals |
| `SGCMC` | SI | mutation-based semi-grand-canonical control |

#### Replacement protocol

- Main-table static comparison is performed on `4096` **distinct** oracle
  evaluations.
- Cache hits do **not** consume the budget.
- Paper headline no longer depends on a separate `100-step deterministic greedy`
  evaluation.
- Existing `standard_eval_summary.csv` can still be emitted for compatibility,
  but it is no longer the primary scientific metric.

#### Replica-exchange MC default

- 4 replicas
- `T in {0.05, 0.10, 0.20, 0.35} eV`
- each replica performs `64` proposals per exchange block
- after each block, attempt adjacent swaps in alternating pattern:
  - round A: `(0,1)` and `(2,3)`
  - round B: `(1,2)`
- total budget counts the pooled distinct oracle evaluations across replicas

#### Main outputs

Per run:

- `running_min_feasible_omega_trace.csv`
- `static_budget_summary.csv`
- `best_feasible_atoms.cif`
- `best_feasible_atoms.xyz`
- `n_oracle_calls_actual`

#### Main aggregates

- **F1**: `distinct_oracle_calls vs running_min_feasible_omega`
- **T1**: method x `mu_CO` summary table
- **T2**: paired static-parity table with:
  - mean +/- CI95
  - median
  - `HL delta`
  - TOST lower / upper p-values
  - TOST decision
  - paired Wilcoxon p-value
  - rank-biserial correlation

#### Static multiplicity family

Main BH-FDR family for Exp-1:

- `PPO_swap vs canonical_MC` at `mu = -0.2`
- `PPO_swap vs canonical_MC` at `mu = -0.6`
- `PPO_swap vs replica_exchange_MC` at `mu = -0.2`
- `PPO_swap vs replica_exchange_MC` at `mu = -0.6`

`random_swap` and `SGCMC` are corrected in separate SI families only.

#### Replacement acceptance gate

Static parity survives if:

- PPO passes paired TOST with `epsilon = 5 eV` against `canonical_MC` in at
  least one `mu_CO` cell, and
- PPO is not significantly worse than `canonical_MC` in the other cell under
  paired Wilcoxon after Exp-1 BH-FDR correction, and
- PPO is not significantly worse than **both** MC baselines in **both** cells

If PPO loses to both `canonical_MC` and `replica_exchange_MC` in both cells,
trigger `R1`.

### 5.2 Exp-2 - Dynamic operando benchmark

#### Main methods

| Code name | Role | Description |
| --- | --- | --- |
| `PPO_swap_dyn` | main | swap-based PPO trained on random staircase schedules |
| `canonical_MC_dyn` | main | single-temperature MC reacting to current `mu_CO` |
| `replica_exchange_MC_dyn` | SI | dynamic replica-exchange stress test if budget allows |
| `random_swap_dyn` | SI | random valid swap sanity baseline |

#### Replacement protocol

- fixed horizon = `200` steps
- no stop action
- no episode reset inside the rollout
- primary endpoint = `cumulative_omega = sum_t Omega_t`
- secondary endpoint = `segment adaptation time`

The dynamic claim is now explicitly:

`PPO learns an amortized response policy, whereas MC methods perform per-regime re-equilibration without amortization.`

#### Main outputs

Per `(method, run_seed, schedule_id)` rollout:

- `dynamic_trajectory.csv`
- `dynamic_rollout_summary.csv`
  with:
  - `cumulative_omega`
  - `mean_segment_adaptation_time`
  - `final_omega`
  - `mean_constraint_violation`

#### Main aggregates

- **F2**: representative schedule with mean trajectory bands
- **T3**: PPO vs `canonical_MC_dyn`
  - paired Wilcoxon p-value
  - `HL delta`
  - rank-biserial correlation
  - descriptive mean +/- CI95

#### Dynamic multiplicity family

Main family for Exp-2:

- primary endpoint only: `cumulative_omega`, PPO vs `canonical_MC_dyn`

`adaptation_time`, `RE-MC_dyn`, and `random_swap_dyn` are secondary or SI.

#### Replacement acceptance gate

Dynamic claim survives if:

- PPO has lower `cumulative_omega` than `canonical_MC_dyn`
- `HL delta <= -5 eV`
- paired Wilcoxon p-value `< 0.05`

`adaptation_time` is supporting evidence, not the sole survival gate.

### 5.3 Exp-3 - Static phase diagram

This experiment remains conceptually unchanged, but the following overrides
apply:

- PPO method is `PPO_swap`, not `PPO_swap_PIRP`
- the primary metric reported at each `mu_CO` is
  `feasible_best_omega` from the static budget protocol
- `canonical_MC` remains an SI visual-reference control
- `SGCMC` is removed from the main narrative

### 5.4 Exp-4 - Ablation

Exp-4 is split into **mainline attribution** and **auxiliary mutation-only SI**.

#### Mainline attribution

| Code | Role | Description |
| --- | --- | --- |
| `A_full_swap` | main | swap + strict_stop |
| `B_no_strict_stop_swap` | main | swap + noop / soft-stop continuation |

Grid:

- `2 mu_CO` points: `{-0.2, -0.6}`
- `5` run seeds
- `4096` distinct oracle evaluations

Main question:

- does `strict_stop` matter for the actual fixed-composition mainline?

#### Auxiliary mutation-only SI

| Code | Role | Description |
| --- | --- | --- |
| `C_mutation_PIRP` | SI | mutation + PIRP + strict_stop |
| `D_mutation_noPIRP` | SI | mutation + no PIRP + strict_stop |

Grid:

- `2 mu_CO` points
- `3` run seeds: `{11, 22, 33}`
- `4096` distinct oracle evaluations

Interpretation:

- this is supportive methodology evidence only
- it does **not** define the fixed-composition mainline

#### Exp-4 acceptance gate

Main ablation survives if either:

- `B_no_strict_stop_swap` is worse by at least `3 eV` in one `mu_CO` cell, or
- the result is neutral and honestly reported as "strict_stop not individually
  dominant"

PIRP auxiliary SI is never a paper-killer under v1.1.

### 5.5 Exp-5 - Structure validation

No conceptual change, but the headline comparison is now:

- `PPO_swap` vs `canonical_MC`

`replica_exchange_MC`, `random_swap`, and mutation-PIRP structures are SI-only
unless they show a qualitatively important failure mode.

---

## 6. Compute reallocation

Replace the relevant parts of Section 4 in v1.0 with the following envelope.

| Item | Runtime envelope |
| --- | ---: |
| Exp-1 static main table with `RE-MC` | 36 h |
| Exp-2 dynamic main table | 10 h |
| Exp-3 phase diagram | 12 h |
| Exp-4 ablation (main + mutation-SI) | 18 h |
| Exp-5 replay / structure audit | 4 h |
| Retry / RE-MC overhead / buffer | 10 h |
| **Total** | **90 h** |

Interpretation:

- The added `10 h` is intentionally consumed by `RE-MC` overhead and rerun
  tolerance.
- Resume logic is an engineering prerequisite, not a separate GPU-h line item.

---

## 7. Risk-register replacements

Append the following updates to Section 5 of v1.0.

| ID | Updated trigger | Updated mitigation |
| --- | --- | --- |
| R1 | PPO fails parity vs `canonical_MC` and is also clearly worse than `RE-MC` in both `mu_CO` cells | First audit the static protocol and budget accounting; second increase PPO rollout size; only after that consider a version bump |
| R2 | Dynamic cumulative-Omega gap vs `canonical_MC_dyn` is not significant | Do **not** silently retune. Open `v1.2` and only then test shorter segment lengths such as `50 -> 25` |
| R6 | Resume logic is missing or untested before the long matrix begins | Implementation of long experiments is blocked until checkpoint restart is verified on one interrupted pilot run |
| R8 | `RE-MC` acceptance ladder is pathological | Tune only the top temperature within `{0.35, 0.50}` on one pilot cell; all other parameters remain fixed unless a version bump is issued |

---

## 8. Pre-submission checklist replacement

Replace the relevant checklist items in Section 6 of v1.0 with the following:

- [ ] **Static parity**: PPO passes paired TOST with `epsilon = 5 eV` against
      `canonical_MC` in at least one `mu_CO` cell
- [ ] **Static robustness**: PPO is not significantly worse than both MC
      baselines in both `mu_CO` cells after Exp-1 main-family BH-FDR
- [ ] **Dynamic gap**: PPO has lower `cumulative_omega` than
      `canonical_MC_dyn` with `HL delta <= -5 eV` and paired Wilcoxon
      `p < 0.05`
- [ ] **Phase diagram trends**: `theta_Pd_surface(mu_CO)` follows the sign
      convention consistently
- [ ] **Mainline ablation**: `strict_stop` is either beneficial or honestly
      neutral under the swap mainline
- [ ] **Structure physics**: SRO / layer-resolved Pd descriptors differ
      meaningfully across weak and strong CO conditions
- [ ] **Protocol split enforced**: static uses stop/reset + budget traces;
      dynamic uses fixed 200-step continuous trajectories
- [ ] **Resume verified**: at least one interrupted training run has been
      resumed successfully before the full matrix begins

---

## 9. Implementation order replacement

Replace Section 10 of v1.0 with the following order of work.

1. `week8_protocol.py`
   - static stop/reset symmetry
   - dynamic no-stop fixed-horizon protocol
   - distinct-oracle-call accounting utility

2. `week8_baselines_mc.py`
   - `canonical_MC`
   - `replica_exchange_MC`
   - optional `random_swap`
   - output schema aligned to the new static / dynamic summaries

3. `week8_dynamic_env.py`
   - random staircase wrapper
   - `current_mu_CO` observation hook
   - dynamic fixed-horizon evaluation path

4. `week8_significance.py`
   - paired TOST
   - paired Wilcoxon
   - `Hodges-Lehmann delta`
   - matched-pairs rank-biserial
   - SI-only `paired Hedges' g_z`
   - BH-FDR per experiment family

5. `trainer.py` resume upgrade or `week8_resume.py`
   - restart from checkpoint
   - resume optimizer state
   - resume VecNormalize state
   - pilot interruption test before matrix launch

6. `week8_pareto_v2_report.py`
   - static tables / figures
   - dynamic tables / figures
   - SI families handled separately

No long matrix execution starts until steps 1-5 are complete.

---

## 10. Effective status of v1.1

This is the controlling revision for Week 8 execution.

Any further change requires:

- `v1.2` if the dynamic schedule is changed
- `v1.2` if the RE-MC ladder changes beyond `{0.35, 0.50}` top-temperature
  pilot tuning
- `v1.2` if a baseline moves between main text and SI
- `v1.2` if static parity changes from TOST-based to another framework

