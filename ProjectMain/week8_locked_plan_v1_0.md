# Tier-2 Locked Experimental Plan v1.0

Date locked: 2026-04-25
Author of this draft: Xu Chengrui + Claude
Compute target: single RTX-4060 laptop GPU, ~2 weeks, no DFT
Status: **CONTRACT** — any deviation requires a new versioned plan.

---

## 0. Central claim (the one sentence the paper sells)

> Under the industrially realistic constraint of a fixed Pd composition, a
> constrained MaskablePPO learns a site-rearrangement policy that adapts to a
> time-varying CO chemical potential `mu_CO(t)`. At static `mu_CO`, the policy
> is statistically indistinguishable from canonical / semi-grand-canonical
> Monte Carlo at the same oracle budget. At dynamic `mu_CO`, the policy
> attains a lower cumulative Omega than re-equilibrating MC because it has
> learned a state-conditioned reaction policy that classical MC, lacking
> training, must rediscover from scratch each time `mu_CO` changes. This
> dynamic operando regime is fundamentally outside the scope of equilibrium
> MD + MC.

**Static parity** establishes baseline credibility. **Dynamic gap** is the
unique value-add. The paper requires both.

---

## 1. Locked design choices (the three yes/yes/yes)

1. **`mu_CO` schedule (Exp-2 dynamic)**: random staircase per training
   episode. Four segments of 50 steps each, values drawn without replacement
   from `{-0.2, -0.4, -0.6, -0.8} eV`. Episode length = 200 steps.
2. **Policy observation**: `current mu_CO` is appended to the GNN feature
   vector. **The next-transition step is NOT provided** — RL must learn a
   reactive policy on the same information MC has, which is the honest
   comparison.
3. **Seed discipline**: train seeds = `{11, 22, 33, 44, 55}`,
   eval seeds = `{66, 77, 88, 99, 111}`, **disjoint**. Used identically for
   all methods, all experiments, all cells.

These are non-negotiable for the duration of v1.0. If a result requires a
different schedule or observation, write v1.1 first.

---

## 2. Paper structure (locked figure / table allocation)

| Section | Main figure / table | Source experiment | Story beat |
|---|---|---|---|
| 3.1 Static parity | F1, T1, T2 | Exp-1 | RL is not worse than MC on MC's home turf |
| 3.2 Dynamic operando | **F2, T3** | **Exp-2** | RL beats MC under time-varying `mu_CO` |
| 3.3 Phase diagram | F3, T4 | Exp-3 | Operando trends consistent with literature |
| 3.4 Discovered structures | F4, T5 | Exp-5 | Warren-Cowley + per-layer Pd validates segregation |
| 3.5 Ablation | F5, T6 | Exp-4 | PIRP / strict_stop attribution |

SI: T7 (generalization gap), T8 (protocol sanity), F6 (mu-conditioned action
distribution heatmap), F7 (full dynamic trajectories on all five test schedules).

---

## 3. Five experiments — full specification

### 3.1 Exp-1 — Static parity benchmark (32 GPU-h)

#### Goal

Demonstrate that on a fixed Pd composition with fixed `mu_CO`, the trained
PPO is statistically equivalent to canonical MC and SGCMC at matched oracle
budget. Establishes baseline credibility for the dynamic claim.

#### Methods

| Code name | Description | Key parameters |
|---|---|---|
| `PPO_swap_PIRP` | `swap_delta_strict_stop` profile + PIRP enabled | gamma=0.97, lr=2e-4, ppo_n_steps=256, ent_coef=5e-4, pirp_scale=0.02 with cosine anneal |
| `canonical_MC` | swap-only Metropolis at fixed temperature | T = 0.10 eV (preliminary; sweep `{0.05, 0.10, 0.20}` once at one cell to pick best) |
| `SGCMC` | mutation Metropolis with `Delta_mu_CuPd` bias | T = 0.10 eV, `Delta_mu_CuPd` solved so equilibrium `Pd_frac == 0.08` |
| `random_swap` | uniform sampling from valid swap actions | none |

#### Protocol

- ChemGymEnv with `action_mode="swap"` for `PPO_swap_PIRP`,
  `canonical_MC`, `random_swap`. SGCMC uses `action_mode="mutation"`.
- Same hybrid oracle (UMA-S + OC25 ensemble), same `GreedyCOPlacer`.
- All methods support `stop+reset` with `max_steps_per_episode = 32`.
- Standard eval = 100 deterministic greedy steps, repeated for each
  `eval_seed`.
- **True oracle-call accounting**: cache hits do NOT count against the
  budget. The trajectory of distinct oracle evaluations is what gets capped
  at the cell's budget.

#### Grid

- `mu_CO ∈ {-0.2, -0.6} eV` (2 points).
- Budget B = 4096 oracle calls (single budget; sample-efficiency curve is
  reconstructed from the in-run running-min trajectory rather than a B sweep).
- Train seeds = `{11, 22, 33, 44, 55}`.
- Eval seeds = `{66, 77, 88, 99, 111}`.

Total cells: 4 methods × 2 mu × 5 train seeds = **40 runs**.

#### Compute estimate

40 runs × 4096 calls × 0.7 s ≈ 31.8 GPU-h, plus eval (100 × 5 eval seeds × 0.7s
× 40 runs ≈ 4 GPU-h with cache reuse). Budget envelope: **32 GPU-h**.

#### Required outputs

Per run:
- `running_min_feasible_omega_trace.csv` — length 4096 trajectory of
  `(t, omega_t, feasible_t, best_omega_so_far, n_oracle_calls_so_far)`.
- `standard_eval_summary.csv` row in the existing schema.
- `best_feasible_atoms.cif` and `best_feasible_atoms.xyz`.
- `n_oracle_calls_actual` excluding cache hits.

#### Required aggregates

- **F1** main: `oracle_calls (log) vs feasible_best_omega`. Four curves per
  panel, two panels (mu = -0.2, -0.6). n=5 shaded CI band per curve.
- **T1**: rows = method × mu, columns = `mean ± CI95`, `median`, `q95`.
- **T2**: pairwise Welch t between PPO and canonical_MC, between PPO and
  SGCMC, between canonical_MC and SGCMC. Use Student-t survival function
  with `df_eff` from Welch-Satterthwaite. Report Hedges' g and BH-FDR
  corrected p-value across the 4 × 2 = 8 comparisons in this experiment.

#### Acceptance thresholds

- |Omega gap PPO vs canonical_MC| < 5 eV with BH-FDR p > 0.05 in at least
  one mu cell **OR**
- PPO ≤ canonical_MC in mean for both mu cells, with neither comparison
  showing PPO significantly worse (BH-FDR p < 0.05 for "PPO worse")

If PPO is significantly worse than canonical_MC in BOTH cells, paper story
collapses; trigger R1 mitigation in section 5.

---

### 3.2 Exp-2 — Dynamic operando benchmark (22 GPU-h) — **the unique claim**

#### Goal

Demonstrate that with time-varying `mu_CO` and fixed composition, the
trained PPO attains lower cumulative Omega than reactive MC, because the
policy has learned a state-conditioned response to the current `mu_CO`,
which MC cannot.

#### Schedule generator

```
def random_staircase_schedule(rng, episode_steps=200, segment_steps=50,
                              values=(-0.2, -0.4, -0.6, -0.8)):
    n_seg = episode_steps // segment_steps          # 4
    perm  = rng.permutation(values)                 # random order, no replacement
    return np.repeat(perm, segment_steps)           # length 200
```

- During training, this is called once per episode with a fresh RNG.
- For evaluation, the same RNG seed produces a deterministic schedule. We
  pre-generate **5 fixed test schedules** from `rng = np.random.default_rng(s)`
  for `s ∈ {7001, 7002, 7003, 7004, 7005}` (these are reserved seeds, not
  in the train or eval pool).

#### Policy observation extension

The graph feature vector gets one extra scalar appended at the per-node
level: `current_mu_CO` (eV, broadcast to all nodes). No history, no
schedule lookahead. PIRP residual is unchanged.

For MC and `random_swap`, the env still updates `config.mu_co` each step.
MC's Metropolis acceptance is computed against the current `mu_CO`. No
information about future `mu_CO` is leaked to any method.

#### Methods

Same four as Exp-1:
- `PPO_swap_PIRP_dyn` — trained on random schedules
- `canonical_MC_dyn` — Metropolis at fixed T, reacts to current mu_CO
- `SGCMC_dyn` — mutation Metropolis with bias and reacts to current mu_CO
- `random_swap_dyn`

#### Grid

- 1 schedule type (random staircase, parameters fixed above)
- Train: 5 train seeds × 4096 oracle calls each (only PPO trains)
- Eval: 5 train seeds × 5 test schedules = 25 rollouts per method,
  episode length 200 steps each

Total cells: 4 methods + 1 PPO training set = effectively
- 5 PPO training runs at 4096 calls
- 4 methods × 25 evaluation rollouts × 200 steps = 100 + 75 = ... see
  compute below

#### Compute estimate

- PPO training: 5 × 4096 × 0.7 s ≈ 4 h × 5 = 20 GPU-h
- All eval rollouts: 4 methods × 25 rollouts × 200 steps × 0.7 s ≈ 3.9 GPU-h
  (cache reuse will reduce this somewhat)
- Total envelope: **22 GPU-h**

#### Required outputs

Per (method, train_seed, test_schedule) rollout:
- `dynamic_trajectory.csv` — 200 rows of
  `(t, mu_CO_t, action, omega_t, n_co_t, theta_pd_surface_t,
   constraint_violation_t)`.
- `cumulative_omega = sum_t omega_t` and `mean_adaptation_time` after each
  staircase transition (defined as the number of steps to bring `omega_t`
  within 1 eV of the equilibrium reached during sustained operation at
  that segment).

#### Required aggregates

- **F2** main: example trajectory plot for one test schedule. x = t,
  y_left = `mu_CO(t)` step plot, y_right = four `omega(t)` curves
  (one per method, n=5 shaded). Annotate transition points; show that
  PPO recovers faster.
- **T3**: method × {mean cumulative omega, mean adaptation time, paired
  Welch t vs PPO, Hedges' g, BH-FDR p}. Pair across (train_seed, schedule)
  to give n = 25.
- **F7** SI: 5 × 4 grid of trajectories — one panel per (test_schedule,
  method).

#### Acceptance thresholds (this is the paper's survival gate)

- PPO cumulative Omega lower than canonical_MC by >= 5 eV in mean
- BH-FDR p < 0.05 across the 3 PPO-vs-baseline comparisons
- Hedges' g <= -0.8 vs canonical_MC
- PPO mean adaptation time lower by >= 5 steps vs canonical_MC

If any of these fails, the dynamic claim is unsupported and we revisit the
schedule design (R2 in section 5).

---

### 3.3 Exp-3 — Static phase diagram (12 GPU-h)

#### Goal

Map the operando trends `theta_Pd_surface(mu_CO)`, `n_CO(mu_CO)`,
`feasible_best_omega(mu_CO)` under fixed composition to validate that the
search is finding physically meaningful structures.

#### Grid

- Method: `PPO_swap_PIRP` only (other methods get one panel in SI for
  visual reference)
- `mu_CO ∈ {-0.4, -0.8} eV` — new (the `-0.2` and `-0.6` points are
  reused from Exp-1)
- 5 train seeds × 5 eval seeds × 4096 budget

Total new cells: 2 mu × 5 train seeds = 10 runs.
Plus a small SI control: `canonical_MC` at the same 4 mu_CO points,
n = 3 train seeds, single eval seed (cheap, ~5 GPU-h).

#### Compute estimate

- 10 PPO new runs × 0.8 h = 8 GPU-h
- 4 mu × 3 MC seeds × 4k × 0.7 s = ~3.5 GPU-h
- Total envelope: **12 GPU-h**

#### Required outputs

Per cell, take the best feasible state:
- `theta_Pd_surface`, `n_CO`, `layer_Pd[0..3]`,
- `alpha_PdPd`, `alpha_CuPd`, `alpha_CuCu`,
- surface Pd cluster size distribution (connected components, cutoff 3.0 Å)
- mean CO-Pd coordination number

#### Required aggregates

- **F3** main: 3 panels (theta_Pd_surface / n_CO / Omega) vs mu_CO. n=5
  shaded CI band per curve. If a literature curve can be identified
  (e.g. Tan & Sholl 2010 type) it gets overlaid on panel (a).
- **T4**: mu × {theta_Pd, n_CO, omega, alpha_PdPd, alpha_CuPd}, n=5
  mean ± CI95.

#### Acceptance thresholds

- `theta_Pd_surface(mu_CO)` is monotone non-decreasing in `-mu_CO` (more
  negative mu means weaker CO and weaker Pd-surface drive — actually our
  convention is the opposite, double-check sign in writing). The trend
  must align with the operando expectation; if it inverts, oracle
  systematic error is suspected.
- At least one visible inflection point (kink) in
  `theta_Pd_surface(mu_CO)`.
- All `valid_frac == 1.0` (composition is fixed by construction).

---

### 3.4 Exp-4 — Ablation (16 GPU-h)

#### Goal

Attribute the RL advantage. Under the swap framework, the deviation mask is
inapplicable (composition is fixed by construction). The two ablations
that matter are PIRP and the strict_stop episode design.

#### Configurations

| Code | use_pirp | episode | reused from |
|---|---|---|---|
| **A — full** | yes | strict_stop | Exp-1 (no new run) |
| **B — no_PIRP** | no | strict_stop | new |
| **C — no_strict_stop** | yes | noop + soft_stop | new |

#### Grid

- 2 new configs × 2 mu (-0.2, -0.6) × 5 train seeds × 4096 budget = 20 runs
- Eval reuses Exp-1 protocol on the same 5 eval seeds

#### Compute estimate

20 × 0.8 h = **16 GPU-h**.

#### Required outputs

Standard `standard_eval_summary` rows.

#### Required aggregates

- **T6**: config × mu × {mean feasible omega, ΔΩ vs full, p, Hedges' g}
- **F5** main: bar chart of ΔΩ vs full, two color groups (mu = -0.2, -0.6)

#### Acceptance thresholds

- At least one of (no_PIRP, no_strict_stop) shows ΔΩ ≥ 3 eV with
  BH-FDR p < 0.05
- If neither does, the paper honestly reports "neither component
  individually dominates the gain" and points to mask + reward shape +
  policy class as collectively responsible. Not a failure.

---

### 3.5 Exp-5 — Structure validation, replay only (3 GPU-h)

#### Goal

Compute physical descriptors on every best feasible structure and produce
publication-grade visualizations.

#### Method

- Reload best feasible atoms from every run produced by Exp-1 / 2 / 3 / 4.
- Compute, for each:
  - Warren-Cowley `alpha_PdPd, alpha_CuPd, alpha_CuCu`
  - Per-layer Pd fraction (4 active layers + bulk)
  - Surface Pd cluster connected components (cutoff 3.0 Å): max cluster
    size, count of isolated Pd
  - CO-Pd coordination number per CO atom (cutoff 3.0 Å): mean and max

#### Compute estimate

Pure replay + ASE/numpy. **3 GPU-h** including a single reevaluation pass
through the oracle to log per-component energies.

#### Required outputs

`structure_audit.csv` indexed by `(experiment, method, mu_CO, schedule_id,
train_seed, eval_seed)`.

#### Required aggregates

- **T5**: (method, mu_CO) × n_seeds × {alpha_PdPd, alpha_CuPd,
  surface_Pd_frac, surf_to_bulk_ratio, max_cluster_size,
  mean_CO_Pd_CN}
- **F4** main: 4 panels — PPO and canonical_MC at mu = -0.2 and -0.6 each.
  Each panel: 3D rendered slab (top + side view) plus a layer-resolved Pd
  fraction histogram.

#### Acceptance thresholds

- `alpha_PdPd(mu = -0.6)` and `alpha_PdPd(mu = -0.2)` differ significantly
  (paired t over the 5 seeds, p < 0.05)
- `alpha_PdPd(mu = -0.6)` is positive (Pd clustering, a non-trivial
  signal away from the random baseline)

---

## 4. Compute budget summary

| Experiment | Runtime envelope | Share |
|---|---|---|
| Exp-1 Static | 32 h | 38% |
| Exp-2 Dynamic | 22 h | 26% |
| Exp-3 Phase diagram | 12 h | 14% |
| Exp-4 Ablation | 16 h | 19% |
| Exp-5 Replay | 3 h | 3% |
| **Total** | **85 h** | 100% |

10% time-out / retry buffer is implicit; if buffer is consumed, the cuts
are made in this order: SI MC control in Exp-3 → second mu in Exp-3
→ no_strict_stop ablation in Exp-4.

---

## 5. Risk register and triggers

| ID | Risk | Trigger | Mitigation |
|---|---|---|---|
| R1 | Static parity fails (PPO loses MC by > 5 eV) | Exp-1 T2 shows BH-FDR p < 0.05 with PPO mean higher in both mu | Audit PIRP alignment for swap actions ([pirp_policy.py](D:\学习相关文档\科研训练\ChemGymProject\ProjectMain\chem_gym\agent\pirp_policy.py)); rerun with `use_pirp=False` to isolate; if still loses, increase ppo_n_steps and re-budget |
| R2 | Dynamic gap not significant | Exp-2 T3 shows BH-FDR p > 0.05 PPO vs canonical_MC | Increase schedule transition frequency (segment_steps from 50 → 25 → 10), retest. If still flat, the static parity result becomes the paper's primary contribution and dynamic is reframed as future work. |
| R3 | PIRP ablation shows PIRP unhelpful | Exp-4 T6 shows ΔΩ < 1 eV for `no_PIRP` | Honest reporting: "PIRP does not contribute on the 4×4 swap system; future work will test its contribution on larger or compositionally heterogeneous systems." Not a paper-killer. |
| R4 | Oracle noise dominates Ω | All cells in Exp-1 show CI95 > 10 eV | Increase `oracle_max_steps` from 100 to 200 and rerun a single cell to compare; if effect is real, accept the higher LBFGS cost (about 2× wall-clock) for the rest of the matrix. |
| R5 | Time overruns, > 24 h on a single day's task | Cumulative GPU-h > 1.2× planned by W1 D5 | Cuts in order: SI canonical_MC control (Exp-3), second new mu in Exp-3 (drop -0.8), then C config in Exp-4 (drop no_strict_stop). |
| R6 | Process / OS interruption | Training crashes mid-run | All training runs save checkpoints every `ppo_n_steps`. Resume logic exists ([trainer.py]). MC runs are stateless: rerun the cell. |
| R7 | mu_CO sign convention bug surfaces in Exp-3 | `theta_Pd_surface(mu_CO)` is monotone in the *wrong* direction | Sanity check the convention by inspecting the limit `mu_CO → -infinity`: `theta_CO → 0`, hence `theta_Pd_surface → equilibrium without CO drive`. If signs disagree, fix the env config and rerun affected cells. |

---

## 6. Pre-submission acceptance checklist (run this before writing)

All eight conditions must be met:

- [ ] **Static parity**: PPO vs canonical_MC differ by < 5 eV in mean with
      BH-FDR p > 0.05 in at least one mu cell (Exp-1)
- [ ] **Dynamic gap**: PPO cumulative Omega < canonical_MC by >= 5 eV
      with BH-FDR p < 0.05 and Hedges' g <= -0.8 (Exp-2)
- [ ] **Phase diagram trends**: theta_Pd_surface monotone in mu_CO direction
      consistent with the convention; at least one inflection (Exp-3)
- [ ] **Ablation story**: at least one ablation produces ΔΩ >= 3 eV with
      BH-FDR p < 0.05 (Exp-4)
- [ ] **Structure physics**: alpha_PdPd at strong CO is significantly
      different from alpha_PdPd at weak CO (Exp-5)
- [ ] **Episode protocol synchronized**: all methods use stop+reset with
      max_steps=32 (verifiable from code)
- [ ] **Disjoint seed sets**: train ∈ {11,22,33,44,55}, eval ∈
      {66,77,88,99,111}, dynamic test schedule seeds ∈ {7001..7005}
- [ ] **Statistical hygiene**: all comparisons use Welch t with Student-t
      survival function on Welch-Satterthwaite df, Hedges' g instead of
      Cohen's d, BH-FDR over the 16+ comparisons in T2 + T3 + T6

---

## 7. Two-week schedule

| Day | Task | Cumulative GPU-h | Output milestone |
|---|---|---|---|
| W1 D1 | Exp-1: PPO_swap_PIRP × 2 mu × 5 seeds × 4k | 8 | PPO baseline trajectories saved |
| W1 D2 | Exp-1: canonical_MC × 2 mu × 5 seeds × 4k + T sweep at one cell | 8 | canonical_MC done |
| W1 D3 | Exp-1: SGCMC × 2 mu × 5 seeds × 4k | 8 | SGCMC done |
| W1 D4 | Exp-1: random_swap × 2 mu × 5 seeds × 4k + Exp-1 aggregate | 5 | F1, T1, T2 locked |
| W1 D5 | Exp-3: PPO at mu in {-0.4, -0.8} × 5 seeds × 4k | 8 | phase diagram main data done |
| W1 D6 | Exp-2 PPO training: 5 train seeds × 4k each on random staircases | 12 | dynamic policy training |
| W1 D7 | Exp-2 PPO training continued + Exp-2 eval rollouts begin | 8 | training done, eval begins |
| W2 D8 | Exp-2 eval rollouts: all 4 methods × 5 train seeds × 5 schedules | 5 | dynamic data complete |
| W2 D9 | Exp-4: no_PIRP × 2 mu × 5 seeds × 4k | 8 | PIRP ablation |
| W2 D10 | Exp-4: no_strict_stop × 2 mu × 5 seeds × 4k | 8 | strict_stop ablation |
| W2 D11 | Exp-5 replay: structure descriptors over all runs + cluster CN code | 3 | T5, F4 |
| W2 D12 | Statistical pass: BH-FDR, Student-t, Hedges' g; tables/figures lock | 0 | submission-ready snapshot |
| W2 D13 | Buffer / outlier rerun / SI fillers | 2 | buffered |
| W2 D14 | Writing kickoff, cross-table consistency check | 0 | manuscript started |

---

## 8. Known limitations to be reported in Discussion

1. **Single oracle budget** (4096) — sample-efficiency curve is reconstructed
   from the in-run running-min trajectory, not from a true budget sweep.
2. **System size**: 4×4 slab × 4 active layers, 64 active sites.
   Finite-size effects are not quantified.
3. **Bulk Pd composition fixed at 0.08** — generalization to higher Pd
   loadings (e.g., 0.20 / 0.40) is not tested.
4. **Oracle cross-validation**: the hybrid UMA + OC25 oracle is not
   benchmarked against DFT in this work. Absolute Omega numbers depend on
   the oracle's training distribution.
5. **Single schedule type for Exp-2**: random staircase. Sinusoidal,
   step-function, and noise-driven schedules are out of scope.
6. **No Bayesian optimization baseline** — MC is the gold standard in this
   domain; BO is documented as future work.

Each limitation is one sentence in Discussion, with a one-sentence
follow-up ("future work will...").

---

## 9. Versioning rules

- This is **v1.0** — locked.
- Changes during execution are logged as v1.1, v1.2, ... with a one-line
  rationale at the top.
- A "version bump" is required when:
  - Compute envelope is exceeded by > 25%
  - An acceptance threshold cannot be met and a downscope is needed
  - The schedule design (Exp-2) needs revision
- A simple rerun of a cell does NOT require a version bump; the version is
  for plan changes, not data changes.

---

## 10. Implementation entry point

The next message should request implementation of this plan, in the
following order:

1. Stop+reset symmetry layer applied to all baselines (single utility
   function in `chem_gym` or in a new `week8_protocol.py`).
2. Canonical MC and SGCMC drivers in `week8_baselines_mc.py` (CPU/GPU
   agnostic, output schema matches existing `standard_eval_summary`).
3. Dynamic schedule env wrapper in `week8_dynamic_env.py` plus the
   observation extension hook for `current mu_CO`.
4. Significance pipeline upgrade: Welch t with Student-t SF, Hedges' g,
   BH-FDR (`week8_significance.py` replacing the Normal-approx version).
5. Aggregator (`week8_pareto_v2_report.py`) producing all eight tables
   and seven figures from the three CSV families.

No implementation begins until v1.0 is committed and pushed.
