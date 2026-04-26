from __future__ import annotations

import argparse
import csv
from argparse import Namespace
from pathlib import Path
from typing import Dict, List

import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Path shim: put ProjectMain/ on sys.path so 'chem_gym' / 'main' resolve regardless of cwd.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

from chem_gym.config import (
    COAdsorptionConfig,
    ConstraintConfig,
    EnvConfig,
    RewardConfig,
    UMAPBRSConfig,
)
from chem_gym.envs.chem_env import ChemGymEnv
from main import maybe_load_oracle


def spearman_corr(values: List[float]) -> float:
    y = np.asarray(values, dtype=float)
    if y.size < 3 or not np.all(np.isfinite(y)):
        return float("nan")
    x = np.arange(y.size, dtype=float)
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = np.sqrt(np.sum(rx * rx) * np.sum(ry * ry))
    if denom <= 1e-12:
        return float("nan")
    return float(np.sum(rx * ry) / denom)


def autocorr_lag(values: List[float], lag: int = 2) -> float:
    y = np.asarray(values, dtype=float)
    if y.size <= lag:
        return float("nan")
    y0 = y[:-lag]
    y1 = y[lag:]
    y0 = y0 - y0.mean()
    y1 = y1 - y1.mean()
    denom = np.sqrt(np.sum(y0 * y0) * np.sum(y1 * y1))
    if denom <= 1e-12:
        return float("nan")
    return float(np.sum(y0 * y1) / denom)


def build_oracle(args) -> object:
    oracle_args = Namespace(
        oracle_ckpt=None,
        oracle_mode="hybrid",
        ads_task=args.ads_task,
        disable_ads_ensemble=False,
        ads_sm_ckpt=args.ads_sm_ckpt,
        ads_md_ckpt=args.ads_md_ckpt,
        eq2_ckpt=args.eq2_ckpt,
        uma_ckpt=args.uma_ckpt,
        oracle_fmax=0.05,
        oracle_max_steps=100,
        require_ads_oracle=True,
    )
    oracle = maybe_load_oracle(oracle_args)
    if oracle is None:
        raise RuntimeError("Hybrid oracle unavailable")
    return oracle


def eval_one_seed(
    run_dir: Path,
    oracle: object,
    mu_co: float,
    seed: int,
    eval_steps: int,
    e_cu_co: float,
    e_pd_co: float,
    enable_noop_action: bool,
    action_mode: str,
    stop_terminates: bool,
    min_stop_steps: int,
    reward_profile: str,
    use_deviation_mask: bool,
    use_uma_pbrs: bool,
    deterministic: bool,
    experiment_profile: str,
) -> Dict[str, float]:
    profile = str(experiment_profile).lower()
    use_route_a = profile == "route_a"
    if use_route_a:
        action_mode = "mutation"
        enable_noop_action = True
        stop_terminates = False
        min_stop_steps = 0
        reward_profile = "legacy"
        use_deviation_mask = True
        use_uma_pbrs = False

    cfg = EnvConfig(
        mode="graph",
        init_seed=seed,
        max_steps=max(96, eval_steps + 16),
        bulk_pd_fraction=0.08,
        action_mode=str(action_mode),
        enable_noop_action=bool(enable_noop_action),
        stop_terminates=bool(stop_terminates),
        min_stop_steps=int(min_stop_steps),
        use_deviation_mask=bool(use_deviation_mask),
        reward=RewardConfig(
            mu_co=float(mu_co),
            delta_omega_scale=20.0,
            reward_profile=str(reward_profile),
        ),
        constraint=ConstraintConfig(
            constraint_update_mode="frozen" if use_route_a else "rollout",
            constraint_weight=0.0 if use_route_a else 1.0,
            constraint_lambda_init=0.0 if use_route_a else 1.0,
            constraint_lambda_min=0.0,
            constraint_lambda_max=0.0 if use_route_a else 10.0,
        ),
        co_adsorption=COAdsorptionConfig(
            e_cu_co=float(e_cu_co),
            e_pd_co=float(e_pd_co),
        ),
        uma_pbrs=UMAPBRSConfig(use_uma_pbrs=bool(use_uma_pbrs)),
    )
    cfg.physics_prior = dict(cfg.physics_prior)
    cfg.physics_prior["e_cu_co"] = float(e_cu_co)
    cfg.physics_prior["e_pd_co"] = float(e_pd_co)

    def _make_env():
        return ChemGymEnv(cfg, oracle=oracle)

    base = DummyVecEnv([_make_env])
    venv = VecNormalize.load(str(run_dir / "vec_normalize.pkl"), base)
    venv.training = False
    venv.norm_reward = False
    model = MaskablePPO.load(str(run_dir / "model.zip"), env=venv)

    obs = venv.reset()
    omega_trace: List[float] = []
    dfrac_trace: List[float] = []
    noop_steps = 0
    best_omega = float("inf")
    best_theta = float("nan")
    best_nco = -1
    last_lambda = float("nan")

    for _ in range(eval_steps):
        masks = get_action_masks(venv)
        action, _ = model.predict(obs, action_masks=masks, deterministic=bool(deterministic))
        obs, _, dones, infos = venv.step(action)
        info = infos[0]

        omega = float(info.get("omega", np.nan))
        theta = float(info.get("pd_surface_coverage", np.nan))
        nco = int(info.get("n_co", -1))
        d_frac = float(info.get("constraint_d_frac", np.nan))
        lam = float(info.get("constraint_lambda", np.nan))
        action_type = str(info.get("action_type", ""))

        if action_type in {"no_op", "stop"}:
            noop_steps += 1
        if np.isfinite(omega):
            omega_trace.append(omega)
            if omega < best_omega:
                best_omega = omega
                best_theta = theta
                best_nco = nco
        if np.isfinite(d_frac):
            dfrac_trace.append(d_frac)
        if np.isfinite(lam):
            last_lambda = lam

        if dones[0]:
            obs = venv.reset()

    return {
        "seed": int(seed),
        "best_omega": float(best_omega),
        "best_theta_pd": float(best_theta),
        "best_n_co": int(best_nco),
        "omega_spearman": float(spearman_corr(omega_trace)),
        "dfrac_autocorr_lag2": float(autocorr_lag(dfrac_trace, lag=2)),
        "noop_ratio": float(noop_steps / max(1, eval_steps)),
        "lambda_last": float(last_lambda),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate reward-v2 models on two mu points")
    parser.add_argument("--run-dir", type=str, required=True, help="Path to run directory with model.zip")
    parser.add_argument("--mu-co", type=float, required=True)
    parser.add_argument("--seeds", type=str, default="11,22,33,44,55")
    parser.add_argument("--eval-steps", type=int, default=120)
    parser.add_argument("--e-cu-co", type=float, default=-0.55)
    parser.add_argument("--e-pd-co", type=float, default=-1.35)
    parser.add_argument("--action-mode", choices=["mutation", "swap"], default="mutation")
    parser.add_argument("--disable-noop-action", action="store_true")
    parser.add_argument("--stop-terminates", action="store_true")
    parser.add_argument("--min-stop-steps", type=int, default=0)
    parser.add_argument(
        "--reward-profile",
        choices=["legacy", "pure_delta_omega", "delta_omega_plus_pbrs"],
        default="legacy",
    )
    parser.add_argument("--enable-deviation-mask", action="store_true")
    parser.add_argument("--disable-uma-pbrs", action="store_true")
    parser.add_argument("--experiment-profile", choices=["default", "route_a"], default="default")
    parser.add_argument("--stochastic-eval", action="store_true", help="Sample actions instead of deterministic argmax.")
    parser.add_argument("--save-csv", type=str, default="")
    parser.add_argument("--ads-task", type=str, default="oc25")
    parser.add_argument("--ads-sm-ckpt", type=str, default="ProjectMain/checkpoints/esen_sm_conserve.pt")
    parser.add_argument("--ads-md-ckpt", type=str, default="ProjectMain/checkpoints/esen_md_direct.pt")
    parser.add_argument("--eq2-ckpt", type=str, default="ProjectMain/checkpoints/eq2_83M_2M.pt")
    parser.add_argument("--uma-ckpt", type=str, default="ProjectMain/checkpoints/uma-m-1p1.pt")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    run_dir = Path(args.run_dir)
    oracle = build_oracle(args)

    rows = []
    for seed in seeds:
        row = eval_one_seed(
            run_dir=run_dir,
            oracle=oracle,
            mu_co=float(args.mu_co),
            seed=seed,
            eval_steps=int(args.eval_steps),
            e_cu_co=float(args.e_cu_co),
            e_pd_co=float(args.e_pd_co),
            enable_noop_action=not bool(args.disable_noop_action),
            action_mode=str(args.action_mode),
            stop_terminates=bool(args.stop_terminates),
            min_stop_steps=int(args.min_stop_steps),
            reward_profile=str(args.reward_profile),
            use_deviation_mask=bool(args.enable_deviation_mask),
            use_uma_pbrs=not bool(args.disable_uma_pbrs),
            deterministic=not bool(args.stochastic_eval),
            experiment_profile=str(args.experiment_profile),
        )
        rows.append(row)
        print(row)

    print("mean_best_theta", float(np.mean([r["best_theta_pd"] for r in rows])))
    print("mean_best_nco", float(np.mean([r["best_n_co"] for r in rows])))
    print("mean_omega_spearman", float(np.nanmean([r["omega_spearman"] for r in rows])))
    print("mean_dfrac_autocorr_lag2", float(np.nanmean([r["dfrac_autocorr_lag2"] for r in rows])))
    print("mean_noop_ratio", float(np.mean([r["noop_ratio"] for r in rows])))
    print("mean_lambda_last", float(np.nanmean([r["lambda_last"] for r in rows])))
    print("best_omega_global", float(np.min([r["best_omega"] for r in rows])))

    if args.save_csv:
        save_path = Path(args.save_csv)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with save_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print("saved_csv", str(save_path))


if __name__ == "__main__":
    main()
