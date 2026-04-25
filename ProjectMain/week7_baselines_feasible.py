"""Week 7 — feasibility-aware Random / Simulated-Annealing baselines.

Outputs CSVs that share the schema written by
``week4_action_reward_ablation.standard_eval_*`` so the existing
``week6_envelope_report.py`` can ingest them without changes.

The script intentionally avoids any RL agent. For each rollout seed we:

1. Build a ChemGymEnv with `action_mode="mutation"`, optional
   `use_deviation_mask` + `max_deviation` to mirror the bounded-mask RL profile.
2. Run either a uniformly-random action policy or a Metropolis-style SA loop
   over mutation actions. SA reverts a rejected attempt by re-mutating the
   modified site back to its previous element.
3. Track ``best_omega``, ``feasible_best_omega`` (omega filtered by
   ``constraint_violation <= 0``), per-step ``constraint_d_frac``,
   ``noop_ratio = 0`` (no noop sampled), etc.

Per-seed rows go to ``<save-root>/standard_eval_per_seed.csv``,
the per-train-seed aggregate goes to
``<save-root>/standard_eval_by_train_seed.csv`` and
``<save-root>/standard_eval_profile_summary.csv``. The "train_seed" axis is
identity-mapped to the rollout seed because there is no learning step.
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from argparse import Namespace
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from chem_gym.config import (
    COAdsorptionConfig,
    ConstraintConfig,
    EnvConfig,
    RewardConfig,
    UMAPBRSConfig,
)
from chem_gym.envs.chem_env import ChemGymEnv


METHOD_CHOICES = ("random_mutation", "sa_mutation")


def parse_int_list(value: str) -> List[int]:
    return [int(x.strip()) for x in str(value).split(",") if x.strip()]


def build_env_config(
    mu_co: float,
    seed: int,
    total_steps: int,
    max_deviation: Optional[int],
    n_active_layers: int = 4,
    bulk_pd_fraction: float = 0.08,
) -> EnvConfig:
    reward_cfg = RewardConfig(
        mu_co=float(mu_co),
        delta_omega_scale=1.0,
        reward_profile="pure_delta_omega",
    )
    constraint_cfg = ConstraintConfig(
        constraint_update_mode="frozen",
        constraint_weight=0.0,
        constraint_lambda_init=0.0,
        constraint_lambda_min=0.0,
        constraint_lambda_max=0.0,
    )
    co_cfg = COAdsorptionConfig(co_max_coverage=1.0)
    uma_cfg = UMAPBRSConfig(use_uma_pbrs=False)

    cfg = EnvConfig(
        mode="graph",
        init_seed=int(seed),
        max_steps=int(total_steps + 64),
        bulk_pd_fraction=float(bulk_pd_fraction),
        n_active_layers=int(n_active_layers),
        action_mode="mutation",
        enable_noop_action=False,
        stop_terminates=False,
        min_stop_steps=0,
        use_deviation_mask=bool(max_deviation is not None),
        reward=reward_cfg,
        constraint=constraint_cfg,
        co_adsorption=co_cfg,
        uma_pbrs=uma_cfg,
    )
    if max_deviation is not None:
        cfg.max_deviation = int(max_deviation)
    return cfg


def maybe_load_oracle_lazy(args):
    """Best-effort hybrid oracle loader. Returns None on EMT-fallback (--oracle-mode none)."""
    if str(args.oracle_mode).lower() == "none":
        return None
    from main import maybe_load_oracle as _legacy_loader

    oracle_args = Namespace(
        oracle_ckpt=None,
        oracle_mode=str(args.oracle_mode),
        ads_task=args.ads_task,
        disable_ads_ensemble=False,
        ads_sm_ckpt=args.ads_sm_ckpt,
        ads_md_ckpt=args.ads_md_ckpt,
        eq2_ckpt=args.eq2_ckpt,
        uma_ckpt=args.uma_ckpt,
        oracle_fmax=0.05,
        oracle_max_steps=100,
        require_ads_oracle=False,
    )
    return _legacy_loader(oracle_args)


def _decode_action(env: ChemGymEnv, action: int) -> tuple[int, int]:
    site_idx, elem_idx = env.action_spec.to_indices(int(action))
    return int(site_idx), int(elem_idx)


def _safe_omega(info: Dict) -> float:
    val = info.get("omega", float("nan"))
    return float(val) if val is not None else float("nan")


def _sample_valid_action(env: ChemGymEnv, rng: np.random.Generator) -> int:
    masks = env.action_masks().astype(bool)
    valid = np.flatnonzero(masks)
    if valid.size == 0:
        # Should not happen in a well-formed mutation env, but guard anyway.
        return int(rng.integers(0, env.action_spec.action_dim))
    return int(rng.choice(valid))


def run_random_mutation(
    env: ChemGymEnv,
    total_steps: int,
    rng: np.random.Generator,
) -> Dict[str, float]:
    obs, info = env.reset()
    omega_trace: List[float] = []
    violation_trace: List[float] = []
    dfrac_trace: List[float] = []
    best_omega = float("inf")
    feasible_best_omega = float("inf")
    best_theta = float("nan")
    best_nco = -1
    feasible_best_theta = float("nan")
    feasible_best_nco = -1
    best_violation = float("nan")

    for _ in range(int(total_steps)):
        action = _sample_valid_action(env, rng)
        obs, _, terminated, truncated, info = env.step(action)
        omega = _safe_omega(info)
        violation = float(info.get("constraint_violation", float("nan")))
        d_frac = float(info.get("constraint_d_frac", float("nan")))
        theta = float(info.get("pd_surface_coverage", float("nan")))
        n_co = int(info.get("n_co", -1))

        if np.isfinite(omega):
            omega_trace.append(omega)
            if omega < best_omega:
                best_omega = omega
                best_theta = theta
                best_nco = n_co
                best_violation = violation
            if np.isfinite(violation) and violation <= 1e-12 and omega < feasible_best_omega:
                feasible_best_omega = omega
                feasible_best_theta = theta
                feasible_best_nco = n_co
        if np.isfinite(violation):
            violation_trace.append(violation)
        if np.isfinite(d_frac):
            dfrac_trace.append(d_frac)
        if terminated or truncated:
            obs, info = env.reset()

    return _summarize_rollout(
        omega_trace=omega_trace,
        violation_trace=violation_trace,
        dfrac_trace=dfrac_trace,
        best_omega=best_omega,
        feasible_best_omega=feasible_best_omega,
        best_theta=best_theta,
        feasible_best_theta=feasible_best_theta,
        best_nco=best_nco,
        feasible_best_nco=feasible_best_nco,
        best_violation=best_violation,
        total_steps=total_steps,
    )


def run_sa_mutation(
    env: ChemGymEnv,
    total_steps: int,
    rng: np.random.Generator,
    t_start: float,
    t_end: float,
) -> Dict[str, float]:
    obs, info = env.reset()
    cur_omega = _safe_omega(info)
    if not np.isfinite(cur_omega):
        cur_omega = float("inf")

    omega_trace: List[float] = []
    violation_trace: List[float] = []
    dfrac_trace: List[float] = []
    best_omega = cur_omega
    feasible_best_omega = float("inf")
    best_theta = float(info.get("pd_surface_coverage", float("nan")))
    best_nco = int(info.get("n_co", -1))
    feasible_best_theta = float("nan")
    feasible_best_nco = -1
    best_violation = float(info.get("constraint_violation", float("nan")))
    if np.isfinite(best_violation) and best_violation <= 1e-12 and np.isfinite(cur_omega):
        feasible_best_omega = cur_omega
        feasible_best_theta = best_theta
        feasible_best_nco = best_nco

    steps_used = 0
    while steps_used < int(total_steps):
        site_idx, elem_idx = _decode_action(env, _sample_valid_action(env, rng))
        prev_elem = int(env.state[site_idx])
        if prev_elem == elem_idx:
            continue
        forward_action = env.action_spec.to_action(site_idx, elem_idx)
        obs, _, terminated, truncated, info = env.step(int(forward_action))
        steps_used += 1
        new_omega = _safe_omega(info)
        violation = float(info.get("constraint_violation", float("nan")))
        d_frac = float(info.get("constraint_d_frac", float("nan")))
        theta = float(info.get("pd_surface_coverage", float("nan")))
        n_co = int(info.get("n_co", -1))

        if np.isfinite(new_omega):
            omega_trace.append(new_omega)
            if new_omega < best_omega:
                best_omega = new_omega
                best_theta = theta
                best_nco = n_co
                best_violation = violation
            if np.isfinite(violation) and violation <= 1e-12 and new_omega < feasible_best_omega:
                feasible_best_omega = new_omega
                feasible_best_theta = theta
                feasible_best_nco = n_co
        if np.isfinite(violation):
            violation_trace.append(violation)
        if np.isfinite(d_frac):
            dfrac_trace.append(d_frac)

        if terminated or truncated:
            obs, info = env.reset()
            cur_omega = _safe_omega(info)
            continue

        delta = (new_omega - cur_omega) if np.isfinite(new_omega) and np.isfinite(cur_omega) else 0.0
        progress = steps_used / max(1, int(total_steps))
        temperature = max(1e-8, t_start * (t_end / t_start) ** progress)
        accept = bool(delta < 0.0) or bool(rng.random() < math.exp(-max(delta, 0.0) / temperature))
        if accept:
            cur_omega = new_omega
            continue
        if steps_used >= int(total_steps):
            break
        revert_action = env.action_spec.to_action(site_idx, prev_elem)
        masks = env.action_masks().astype(bool)
        if not bool(masks[int(revert_action)]):
            continue
        obs, _, terminated, truncated, info = env.step(int(revert_action))
        steps_used += 1
        cur_omega = _safe_omega(info)
        if terminated or truncated:
            obs, info = env.reset()
            cur_omega = _safe_omega(info)

    return _summarize_rollout(
        omega_trace=omega_trace,
        violation_trace=violation_trace,
        dfrac_trace=dfrac_trace,
        best_omega=best_omega,
        feasible_best_omega=feasible_best_omega,
        best_theta=best_theta,
        feasible_best_theta=feasible_best_theta,
        best_nco=best_nco,
        feasible_best_nco=feasible_best_nco,
        best_violation=best_violation,
        total_steps=total_steps,
    )


def _summarize_rollout(
    *,
    omega_trace: Sequence[float],
    violation_trace: Sequence[float],
    dfrac_trace: Sequence[float],
    best_omega: float,
    feasible_best_omega: float,
    best_theta: float,
    feasible_best_theta: float,
    best_nco: int,
    feasible_best_nco: int,
    best_violation: float,
    total_steps: int,
) -> Dict[str, float]:
    legal_steps = int(np.sum(np.asarray(violation_trace, dtype=float) <= 1e-12)) if violation_trace else 0
    return {
        "best_omega": float(best_omega) if np.isfinite(best_omega) else float("nan"),
        "best_theta_pd": float(best_theta),
        "best_n_co": int(best_nco),
        "best_omega_is_feasible": float(np.isfinite(best_violation) and best_violation <= 1e-12),
        "feasible_best_omega": float(feasible_best_omega) if np.isfinite(feasible_best_omega) else float("nan"),
        "feasible_best_theta_pd": float(feasible_best_theta),
        "feasible_best_n_co": int(feasible_best_nco),
        "best_omega_feasibility_gap": (
            float(feasible_best_omega - best_omega)
            if np.isfinite(best_omega) and np.isfinite(feasible_best_omega)
            else float("nan")
        ),
        "omega_spearman": float("nan"),
        "dfrac_autocorr_lag2": float("nan"),
        "noop_ratio": 0.0,
        "mutation_ratio": 1.0,
        "lambda_last": float("nan"),
        "constraint_valid_frac": float(legal_steps / max(1, len(violation_trace))) if violation_trace else float("nan"),
        "mean_constraint_violation": float(np.nanmean(violation_trace)) if violation_trace else float("nan"),
        "max_constraint_violation": float(np.nanmax(violation_trace)) if violation_trace else float("nan"),
        "mean_constraint_d_frac": float(np.nanmean(dfrac_trace)) if dfrac_trace else float("nan"),
        "max_constraint_d_frac": float(np.nanmax(dfrac_trace)) if dfrac_trace else float("nan"),
        "mean_uncertainty": float("nan"),
        "n_oracle_calls": int(total_steps),
    }


def write_csv(path: Path, rows: Sequence[Dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_per_train_seed(rows: Sequence[Dict]) -> List[Dict]:
    grouped: Dict[int, List[Dict]] = {}
    for row in rows:
        grouped.setdefault(int(row["train_seed"]), []).append(row)
    out: List[Dict] = []
    for train_seed, sub in sorted(grouped.items()):
        out.append(
            {
                "profile": sub[0]["profile"],
                "train_seed": int(train_seed),
                "mu_co": float(sub[0]["mu_co"]),
                "mean_best_omega": float(np.nanmean([r["best_omega"] for r in sub])),
                "best_omega_global": float(np.nanmin([r["best_omega"] for r in sub])),
                "mean_best_omega_is_feasible": float(np.nanmean([r["best_omega_is_feasible"] for r in sub])),
                "mean_feasible_best_omega": float(np.nanmean([r["feasible_best_omega"] for r in sub])),
                "feasible_best_omega_global": float(np.nanmin([r["feasible_best_omega"] for r in sub])),
                "mean_best_omega_feasibility_gap": float(np.nanmean([r["best_omega_feasibility_gap"] for r in sub])),
                "mean_best_theta_pd": float(np.nanmean([r["best_theta_pd"] for r in sub])),
                "mean_best_n_co": float(np.nanmean([r["best_n_co"] for r in sub])),
                "mean_feasible_best_theta_pd": float(np.nanmean([r["feasible_best_theta_pd"] for r in sub])),
                "mean_feasible_best_n_co": float(np.nanmean([r["feasible_best_n_co"] for r in sub])),
                "mean_omega_spearman": float("nan"),
                "mean_dfrac_autocorr_lag2": float("nan"),
                "mean_noop_ratio": 0.0,
                "mean_lambda_last": float("nan"),
                "mean_mutation_ratio": 1.0,
                "mean_constraint_valid_frac": float(np.nanmean([r["constraint_valid_frac"] for r in sub])),
                "mean_constraint_violation": float(np.nanmean([r["mean_constraint_violation"] for r in sub])),
                "max_constraint_violation": float(np.nanmax([r["max_constraint_violation"] for r in sub])),
                "mean_constraint_d_frac": float(np.nanmean([r["mean_constraint_d_frac"] for r in sub])),
                "max_constraint_d_frac": float(np.nanmax([r["max_constraint_d_frac"] for r in sub])),
                "mean_uncertainty": float("nan"),
            }
        )
    return out


def aggregate_profile(rows: Sequence[Dict]) -> Dict[str, float]:
    if not rows:
        return {}
    keys_mean = [
        "mean_best_omega",
        "mean_feasible_best_omega",
        "mean_best_omega_is_feasible",
        "mean_best_omega_feasibility_gap",
        "mean_best_theta_pd",
        "mean_best_n_co",
        "mean_feasible_best_theta_pd",
        "mean_feasible_best_n_co",
        "mean_constraint_valid_frac",
        "mean_constraint_violation",
        "mean_constraint_d_frac",
    ]
    out = {
        "profile": rows[0]["profile"],
        "mu_co": float(rows[0]["mu_co"]),
        "n_train_seeds": len(rows),
        "best_omega_global": float(np.nanmin([r["best_omega_global"] for r in rows])),
        "feasible_best_omega_global": float(np.nanmin([r["feasible_best_omega_global"] for r in rows])),
        "mean_noop_ratio": 0.0,
        "mean_mutation_ratio": 1.0,
        "max_constraint_violation": float(np.nanmax([r["max_constraint_violation"] for r in rows])),
        "max_constraint_d_frac": float(np.nanmax([r["max_constraint_d_frac"] for r in rows])),
        "mean_lambda_last": float("nan"),
        "mean_omega_spearman": float("nan"),
        "mean_dfrac_autocorr_lag2": float("nan"),
        "mean_uncertainty": float("nan"),
    }
    for key in keys_mean:
        out[key] = float(np.nanmean([r[key] for r in rows]))
    if len(rows) > 1:
        std_v = float(np.nanstd([r["mean_feasible_best_omega"] for r in rows], ddof=1))
        out["std_feasible_best_omega"] = std_v
        out["ci95_feasible_best_omega"] = float(1.96 * std_v / math.sqrt(len(rows)))
    else:
        out["std_feasible_best_omega"] = float("nan")
        out["ci95_feasible_best_omega"] = float("nan")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Week-7 feasibility-aware random / SA baselines")
    parser.add_argument("--method", choices=METHOD_CHOICES, required=True)
    parser.add_argument("--mu-co", type=float, required=True)
    parser.add_argument("--total-steps", type=int, required=True, help="Oracle calls per rollout (matches RL training budget)")
    parser.add_argument("--seeds", type=str, default="11,22,33,44,55")
    parser.add_argument("--max-deviation", type=int, default=None, help="Bounded mask radius. Omit for open mutation.")
    parser.add_argument("--n-active-layers", type=int, default=4)
    parser.add_argument("--bulk-pd-fraction", type=float, default=0.08)
    parser.add_argument("--save-root", type=str, required=True)
    parser.add_argument("--oracle-mode", choices=["hybrid", "none"], default="hybrid",
                        help="`none` falls back to ASE EMT (smoke-only).")
    parser.add_argument("--ads-task", type=str, default="oc25")
    parser.add_argument("--ads-sm-ckpt", type=str, default="ProjectMain/checkpoints/esen_sm_conserve.pt")
    parser.add_argument("--ads-md-ckpt", type=str, default="ProjectMain/checkpoints/esen_md_direct.pt")
    parser.add_argument("--eq2-ckpt", type=str, default="ProjectMain/checkpoints/eq2_83M_2M.pt")
    parser.add_argument("--uma-ckpt", type=str, default="ProjectMain/checkpoints/uma-s-1p1.pt")
    parser.add_argument("--sa-t-start", type=float, default=0.5)
    parser.add_argument("--sa-t-end", type=float, default=0.005)
    parser.add_argument("--smoke", action="store_true", help="Tiny rollout for orchestration validation.")
    args = parser.parse_args()

    if args.smoke:
        args.total_steps = min(int(args.total_steps), 32)
        first_seed = parse_int_list(args.seeds)[:1]
        args.seeds = ",".join(str(s) for s in (first_seed or [11]))

    seeds = parse_int_list(args.seeds)
    if not seeds:
        raise ValueError("Need at least one seed")

    save_root = Path(args.save_root)
    save_root.mkdir(parents=True, exist_ok=True)

    md_label = "open" if args.max_deviation is None else f"md{int(args.max_deviation)}"
    profile_label = f"{args.method}_{md_label}"

    oracle = maybe_load_oracle_lazy(args)
    if oracle is None and not args.smoke:
        raise RuntimeError(
            "No oracle loaded for non-smoke run. Re-launch with --oracle-mode hybrid "
            "and valid checkpoints, or pass --smoke for an EMT sanity check."
        )

    per_seed_rows: List[Dict] = []
    for seed in seeds:
        env_cfg = build_env_config(
            mu_co=float(args.mu_co),
            seed=int(seed),
            total_steps=int(args.total_steps),
            max_deviation=args.max_deviation,
            n_active_layers=int(args.n_active_layers),
            bulk_pd_fraction=float(args.bulk_pd_fraction),
        )
        env = ChemGymEnv(env_cfg, oracle=oracle)
        rng = np.random.default_rng(int(seed))
        t0 = time.perf_counter()
        if args.method == "random_mutation":
            metrics = run_random_mutation(env, total_steps=int(args.total_steps), rng=rng)
        elif args.method == "sa_mutation":
            metrics = run_sa_mutation(
                env,
                total_steps=int(args.total_steps),
                rng=rng,
                t_start=float(args.sa_t_start),
                t_end=float(args.sa_t_end),
            )
        else:
            raise ValueError(f"Unsupported method: {args.method}")
        elapsed = float(time.perf_counter() - t0)
        row = {
            "profile": profile_label,
            "train_seed": int(seed),
            "eval_seed": int(seed),
            "mu_co": float(args.mu_co),
            **metrics,
            "wall_clock_seconds": elapsed,
        }
        per_seed_rows.append(row)
        print(
            f"[Week7-Baseline] {profile_label} seed={seed} "
            f"best_omega={row['best_omega']:.4f} "
            f"feas_best={row['feasible_best_omega']:.4f} "
            f"valid={row['constraint_valid_frac']:.3f} "
            f"d_frac={row['mean_constraint_d_frac']:.3f} "
            f"wall={elapsed:.1f}s"
        )

    train_seed_rows = aggregate_per_train_seed(per_seed_rows)
    profile_summary = aggregate_profile(train_seed_rows)

    write_csv(save_root / "standard_eval_per_seed.csv", per_seed_rows)
    write_csv(save_root / "standard_eval_by_train_seed.csv", train_seed_rows)
    if profile_summary:
        write_csv(save_root / "standard_eval_profile_summary.csv", [profile_summary])

    print(f"[Week7-Baseline] wrote {len(per_seed_rows)} per-seed rows to {save_root}")


if __name__ == "__main__":
    main()
