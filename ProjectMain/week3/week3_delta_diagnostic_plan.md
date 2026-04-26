# Delta Drift Diagnostic Plan (No Code Change)

## 1. Objective
Determine whether the current hybrid objective

\[
\Omega_{\text{hybrid}}(\sigma,\mathrm{CO}) =
E_{\text{UMA}}^{\text{slab}}(\sigma)
+
\Big(E_{\text{OC}}(\sigma+\mathrm{CO})-E_{\text{OC}}(\sigma)-N_{\mathrm{CO}}E_{\mathrm{ref}}\Big)
-\mu_{\mathrm{CO}}N_{\mathrm{CO}}
\]

contains a configuration-dependent mismatch term

\[
\delta(\sigma)=E_{\text{UMA}}^{\text{slab}}(\sigma)-E_{\text{OC}}^{\text{slab}}(\sigma)
\]

large enough to distort RL optimization.

## 2. Why this diagnostic is needed
- `e_cu_co/e_pd_co` synchronization fixed a real bug (prior/environment inconsistency).
- That fix does **not** prove thermodynamic consistency of the hybrid objective.
- If \(\delta(\sigma)\) is not near-constant, RL may optimize model mismatch instead of physical grand potential.

## 3. Diagnostic definitions

For each trajectory step \(t\):
- \(\sigma_t\): metal configuration
- \(\Omega_t^{\text{hybrid}}\): current training objective value
- \(E^{\text{UMA}}_{\text{slab}}(\sigma_t)\)
- \(E^{\text{OC}}_{\text{slab}}(\sigma_t)\) (same OC backend used in adsorbate term)
- \(\delta_t = E^{\text{UMA}}_{\text{slab}}(\sigma_t)-E^{\text{OC}}_{\text{slab}}(\sigma_t)\)
- \(\Delta\delta_t = \delta_t-\delta_{t-1}\)
- \(\Delta\Omega_t = \Omega_t^{\text{hybrid}}-\Omega_{t-1}^{\text{hybrid}}\)

Derived ratios:
- Drift ratio:
  \[
  R_{\delta/\Omega}=\frac{\mathrm{std}(\Delta\delta_t)}{\mathrm{std}(\Delta\Omega_t)}
  \]
- Coverage correlation:
  \[
  \rho_{\delta,\theta}=\mathrm{corr}(\delta_t,\theta_{Pd,t})
  \]
- Adsorption coupling correlation:
  \[
  \rho_{\Delta\delta,\Delta N_{CO}}=\mathrm{corr}(\Delta\delta_t,\Delta N_{CO,t})
  \]

## 4. Experimental matrix (minimal but publishable)
- Chemical potential points: \(\mu_{CO}\in\{-0.6,\,-0.2\}\) eV
- Seeds: 5 fixed seeds
- Per-seed rollout length: 1000 steps (deterministic eval policy)
- Total samples: 10 trajectories

Reason:
- This directly tests the already observed trend issue.
- 5 seeds gives basic statistical confidence.

## 5. Decision thresholds

### Pass (hybrid objective acceptable)
All conditions met:
- \(\mathrm{std}(\delta_t) < 0.05\) eV/supercell
- \(R_{\delta/\Omega} < 0.20\)
- \(|\rho_{\delta,\theta}| < 0.10\)
- \(|\rho_{\Delta\delta,\Delta N_{CO}}| < 0.10\)

### Warning (needs caution, not final-paper quality)
Any one condition:
- \(0.05 \le \mathrm{std}(\delta_t) < 0.10\)
- \(0.20 \le R_{\delta/\Omega} < 0.35\)
- \(0.10 \le |\rho| < 0.25\)

### Fail (must modify objective for scientific claims)
Any one condition:
- \(\mathrm{std}(\delta_t)\ge 0.10\) eV/supercell
- \(R_{\delta/\Omega}\ge 0.35\)
- \(|\rho_{\delta,\theta}|\ge 0.25\) or \(|\rho_{\Delta\delta,\Delta N_{CO}}|\ge 0.25\)

## 6. Required plots/tables
- Table T1: per-\(\mu\), per-seed summary of mean/std/range of \(\delta_t\)
- Plot P1: \(\delta_t\) vs step (10 trajectories)
- Plot P2: \(\Delta\delta_t\) vs \(\Delta\Omega_t\) scatter
- Plot P3: \(\delta_t\) vs \(\theta_{Pd,t}\) scatter with fitted slope
- Table T2: pass/warning/fail verdict by threshold

## 7. Interpretation logic

If **Pass**:
- Keep hybrid objective for now.
- Continue to 2x5000 and then full \(\mu\) scan.

If **Warning**:
- Keep hybrid only for engineering iteration.
- Paper-quality claims require a consistency ablation.

If **Fail**:
- Move to thermodynamically consistent single-backend objective for final \(\Omega\):
  \[
  \Omega_B=E_B(\sigma+\mathrm{CO})-E_B(\sigma)-\tilde{\mu}_{CO}N_{CO}
  \]
- Use UMA as prior/proposal/reranker only (not in reward).

## 8. Scope constraints for this phase
- No environment logic change.
- No policy architecture change.
- No long retraining.
- Only diagnostic measurement and objective-consistency decision.

## 9. What this does and does not prove

This diagnostic proves whether hybrid reward contamination is negligible or not.

It does **not** alone prove:
- Final phase diagram correctness
- PIRP superiority
- Experimental agreement

Those are next phases after objective consistency is settled.
