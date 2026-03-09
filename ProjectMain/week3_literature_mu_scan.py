from __future__ import annotations

import argparse
import csv
import itertools
from argparse import Namespace
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from chem_gym.agent.trainer import train_agent
from chem_gym.config import EnvConfig, TrainConfig
from chem_gym.envs.chem_env import ChemGymEnv
from main import maybe_load_oracle


def _parse_float_list(value: str) -> List[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def _parse_int_list(value: str) -> List[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def _build_hybrid_oracle(args) -> object:
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
        require_ads_oracle=False,
    )
    oracle = maybe_load_oracle(oracle_args)
    if oracle is None:
        raise RuntimeError("Hybrid oracle unavailable. Check FAIRChem install and checkpoint paths.")
    return oracle


def _make_env_config(
    seed: int,
    mu_co: float,
    e_cu_co: float,
    e_pd_co: float,
    max_steps: int,
    delta_omega_scale: float,
    reward_shift: float,
) -> EnvConfig:
    cfg = EnvConfig(
        mode="graph",
        init_seed=seed,
        max_steps=max_steps,
        mu_co=float(mu_co),
        bulk_pd_fraction=0.08,
        co_max_coverage=1.0,
        delta_omega_scale=float(delta_omega_scale),
        reward_shift=float(reward_shift),
        e_cu_co=float(e_cu_co),
        e_pd_co=float(e_pd_co),
    )
    cfg.physics_prior = dict(cfg.physics_prior)
    cfg.physics_prior["e_cu_co"] = float(cfg.e_cu_co)
    cfg.physics_prior["e_pd_co"] = float(cfg.e_pd_co)
    return cfg


def _train_once(
    save_dir: Path,
    env_cfg: EnvConfig,
    oracle,
    total_steps: int,
    ppo_n_steps: int,
    ppo_batch_size: int,
    pirp_scale: float,
    enable_vis: bool,
    device: str,
) -> None:
    train_cfg = TrainConfig(
        total_timesteps=int(total_steps),
        n_envs=1,
        device=device,
        gamma=0.97,
        lam=0.95,
        learning_rate=2e-4,
        ppo_n_steps=int(ppo_n_steps),
        ppo_batch_size=int(ppo_batch_size),
        ppo_ent_coef=5e-4,
        use_pirp=True,
        pirp_scale=float(pirp_scale),
        enable_visualization=bool(enable_vis),
    )
    train_agent(
        env_config=env_cfg,
        surrogate=None,
        train_config=train_cfg,
        oracle=oracle,
        oracle_energy_fn=oracle.compute_energy if hasattr(oracle, "compute_energy") else None,
        save_dir=str(save_dir),
        use_masking=True,
    )


def _eval_latest_model(
    save_dir: Path,
    oracle,
    eval_seeds: Sequence[int],
    mu_co: float,
    e_cu_co: float,
    e_pd_co: float,
    eval_steps: int,
    delta_omega_scale: float,
    reward_shift: float,
) -> List[Dict[str, float]]:
    model_path = save_dir / "latest_model"
    stats_path = save_dir / "latest_vec_normalize.pkl"
    if not model_path.with_suffix(".zip").exists():
        raise FileNotFoundError(f"Missing model file: {model_path.with_suffix('.zip')}")
    if not stats_path.exists():
        raise FileNotFoundError(f"Missing vec normalize file: {stats_path}")

    rows: List[Dict[str, float]] = []
    for seed in eval_seeds:
        env_cfg = _make_env_config(
            seed=int(seed),
            mu_co=float(mu_co),
            e_cu_co=float(e_cu_co),
            e_pd_co=float(e_pd_co),
            max_steps=max(eval_steps + 8, 80),
            delta_omega_scale=delta_omega_scale,
            reward_shift=reward_shift,
        )

        def _make_single():
            return ChemGymEnv(env_cfg, oracle=oracle)

        base_venv = DummyVecEnv([_make_single])
        venv = VecNormalize.load(str(stats_path), base_venv)
        venv.training = False
        venv.norm_reward = False
        model = MaskablePPO.load(str(model_path), env=venv)

        obs = venv.reset()
        best_omega = float("inf")
        best_theta = float("nan")
        best_n_co = -1
        last_omega = float("nan")
        last_theta = float("nan")
        last_n_co = -1

        for _ in range(eval_steps):
            masks = get_action_masks(venv)
            action, _ = model.predict(obs, action_masks=masks, deterministic=True)
            obs, _, dones, infos = venv.step(action)
            info = infos[0]

            omega = float(info.get("omega", np.nan))
            theta = float(info.get("pd_surface_coverage", np.nan))
            n_co = int(info.get("n_co", -1))

            last_omega = omega
            last_theta = theta
            last_n_co = n_co

            if np.isfinite(omega) and omega < best_omega:
                best_omega = omega
                best_theta = theta
                best_n_co = n_co

            if dones[0]:
                obs = venv.reset()

        rows.append(
            {
                "seed": int(seed),
                "best_omega": float(best_omega),
                "best_theta_pd": float(best_theta),
                "best_n_co": int(best_n_co),
                "final_omega": float(last_omega),
                "final_theta_pd": float(last_theta),
                "final_n_co": int(last_n_co),
            }
        )

    return rows


def _write_csv(path: Path, rows: Sequence[Dict], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _pair_candidates(cu_vals: Sequence[float], pd_vals: Sequence[float], min_gap: float) -> List[Tuple[float, float]]:
    pairs: List[Tuple[float, float]] = []
    for e_cu, e_pd in itertools.product(cu_vals, pd_vals):
        if e_pd <= (e_cu - float(min_gap)):
            pairs.append((float(e_cu), float(e_pd)))
    # Sort by adsorption contrast (weak -> strong) for stable subsampling.
    pairs.sort(key=lambda x: (x[1] - x[0]))
    return pairs


def _aggregate_pair_metrics(rows: Sequence[Dict], mu_low: float, mu_high: float) -> Dict[Tuple[float, float], Dict[str, float]]:
    out: Dict[Tuple[float, float], Dict[str, float]] = {}
    for row in rows:
        key = (float(row["e_cu_co"]), float(row["e_pd_co"]))
        out.setdefault(key, {})

    for key in list(out.keys()):
        e_cu, e_pd = key
        low_rows = [r for r in rows if r["e_cu_co"] == e_cu and r["e_pd_co"] == e_pd and r["mu_co"] == mu_low]
        high_rows = [r for r in rows if r["e_cu_co"] == e_cu and r["e_pd_co"] == e_pd and r["mu_co"] == mu_high]
        if not low_rows or not high_rows:
            out[key] = {"score": -1e9}
            continue

        theta_low = float(np.mean([r["best_theta_pd"] for r in low_rows]))
        theta_high = float(np.mean([r["best_theta_pd"] for r in high_rows]))
        nco_low = float(np.mean([r["best_n_co"] for r in low_rows]))
        nco_high = float(np.mean([r["best_n_co"] for r in high_rows]))
        omega_low = float(np.mean([r["best_omega"] for r in low_rows]))
        omega_high = float(np.mean([r["best_omega"] for r in high_rows]))

        theta_slope = theta_high - theta_low
        nco_slope = nco_high - nco_low
        score = theta_slope + 0.05 * nco_slope
        out[key] = {
            "theta_low": theta_low,
            "theta_high": theta_high,
            "nco_low": nco_low,
            "nco_high": nco_high,
            "omega_low": omega_low,
            "omega_high": omega_high,
            "theta_slope": theta_slope,
            "nco_slope": nco_slope,
            "score": score,
        }
    return out


def main():
    parser = argparse.ArgumentParser(description="Literature-range adsorption parameter scan + mu_CO scan")
    parser.add_argument("--mu-points", type=str, default="-1.2,-0.9,-0.6,-0.3,-0.2")
    parser.add_argument("--seeds", type=str, default="11,22,33")
    parser.add_argument("--e-cu-values", type=str, default="-0.45,-0.55,-0.65")
    parser.add_argument("--e-pd-values", type=str, default="-1.15,-1.30,-1.45")
    parser.add_argument("--min-gap", type=float, default=0.30)
    parser.add_argument("--pair-limit", type=int, default=3)
    parser.add_argument("--range-train-steps", type=int, default=512)
    parser.add_argument("--mu-train-steps", type=int, default=1024)
    parser.add_argument("--eval-steps", type=int, default=80)
    parser.add_argument("--ppo-n-steps", type=int, default=64)
    parser.add_argument("--ppo-batch-size", type=int, default=64)
    parser.add_argument("--pirp-scale", type=float, default=0.02)
    parser.add_argument("--delta-omega-scale", type=float, default=20.0)
    parser.add_argument("--reward-shift", type=float, default=0.1)
    parser.add_argument("--save-root", type=str, default="ProjectMain/checkpoints/week3_literature_mu_scan")
    parser.add_argument("--ads-task", type=str, default="oc25")
    parser.add_argument("--ads-sm-ckpt", type=str, default="ProjectMain/checkpoints/esen_sm_conserve.pt")
    parser.add_argument("--ads-md-ckpt", type=str, default="ProjectMain/checkpoints/esen_md_direct.pt")
    parser.add_argument("--eq2-ckpt", type=str, default="ProjectMain/checkpoints/eq2_83M_2M.pt")
    parser.add_argument("--uma-ckpt", type=str, default="ProjectMain/checkpoints/uma-m-1p1.pt")
    parser.add_argument("--enable-vis", action="store_true")
    args = parser.parse_args()

    mu_points = _parse_float_list(args.mu_points)
    seeds = _parse_int_list(args.seeds)
    cu_vals = _parse_float_list(args.e_cu_values)
    pd_vals = _parse_float_list(args.e_pd_values)
    save_root = Path(args.save_root)
    save_root.mkdir(parents=True, exist_ok=True)

    if len(mu_points) < 5:
        raise ValueError("mu-points must include at least 5 points.")
    if len(seeds) < 2:
        raise ValueError("Please provide at least 2 seeds for robust fixed-seed checks.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("DEVICE", device)
    print("MU_POINTS", mu_points)
    print("SEEDS", seeds)
    print("CU_RANGE", cu_vals)
    print("PD_RANGE", pd_vals)

    oracle = _build_hybrid_oracle(args)

    pairs = _pair_candidates(cu_vals, pd_vals, min_gap=args.min_gap)
    if not pairs:
        raise ValueError("No (e_cu_co, e_pd_co) pairs satisfy min-gap constraint.")
    pair_limit = max(1, int(args.pair_limit))
    if len(pairs) > pair_limit:
        idx = np.linspace(0, len(pairs) - 1, pair_limit)
        idx = sorted(set(int(round(x)) for x in idx))
        pairs = [pairs[i] for i in idx]

    print("\n[Phase A] Literature pair pre-scan at mu endpoints")
    mu_low = float(min(mu_points))
    mu_high = float(max(mu_points))
    pair_scan_rows: List[Dict] = []
    for e_cu, e_pd in pairs:
        for mu in (mu_low, mu_high):
            print(f"[PairScan] e_cu={e_cu:.3f}, e_pd={e_pd:.3f}, mu={mu:+.2f}")
            env_cfg = _make_env_config(
                seed=int(seeds[0]),
                mu_co=mu,
                e_cu_co=e_cu,
                e_pd_co=e_pd,
                max_steps=max(args.range_train_steps + args.eval_steps + 16, 96),
                delta_omega_scale=args.delta_omega_scale,
                reward_shift=args.reward_shift,
            )
            _train_once(
                save_dir=save_root,
                env_cfg=env_cfg,
                oracle=oracle,
                total_steps=args.range_train_steps,
                ppo_n_steps=args.ppo_n_steps,
                ppo_batch_size=args.ppo_batch_size,
                pirp_scale=args.pirp_scale,
                enable_vis=args.enable_vis,
                device=device,
            )
            eval_rows = _eval_latest_model(
                save_dir=save_root,
                oracle=oracle,
                eval_seeds=seeds,
                mu_co=mu,
                e_cu_co=e_cu,
                e_pd_co=e_pd,
                eval_steps=args.eval_steps,
                delta_omega_scale=args.delta_omega_scale,
                reward_shift=args.reward_shift,
            )
            for r in eval_rows:
                row = {
                    "stage": "pair_scan",
                    "e_cu_co": e_cu,
                    "e_pd_co": e_pd,
                    "mu_co": mu,
                    **r,
                }
                pair_scan_rows.append(row)

    _write_csv(
        save_root / "literature_pair_scan.csv",
        pair_scan_rows,
        fieldnames=[
            "stage",
            "e_cu_co",
            "e_pd_co",
            "mu_co",
            "seed",
            "best_omega",
            "best_theta_pd",
            "best_n_co",
            "final_omega",
            "final_theta_pd",
            "final_n_co",
        ],
    )

    pair_metrics = _aggregate_pair_metrics(pair_scan_rows, mu_low=mu_low, mu_high=mu_high)
    metric_rows = []
    for (e_cu, e_pd), m in pair_metrics.items():
        metric_rows.append({"e_cu_co": e_cu, "e_pd_co": e_pd, **m})
    metric_rows.sort(key=lambda x: x["score"], reverse=True)
    _write_csv(
        save_root / "literature_pair_scores.csv",
        metric_rows,
        fieldnames=[
            "e_cu_co",
            "e_pd_co",
            "theta_low",
            "theta_high",
            "nco_low",
            "nco_high",
            "omega_low",
            "omega_high",
            "theta_slope",
            "nco_slope",
            "score",
        ],
    )

    best_pair = metric_rows[0]
    e_cu_best = float(best_pair["e_cu_co"])
    e_pd_best = float(best_pair["e_pd_co"])
    print(
        f"[PairScan] Selected best pair: e_cu={e_cu_best:.3f}, e_pd={e_pd_best:.3f}, "
        f"theta_slope={best_pair['theta_slope']:.4f}, nco_slope={best_pair['nco_slope']:.4f}"
    )

    print("\n[Phase B] Full mu scan with fixed seeds")
    mu_scan_rows: List[Dict] = []
    mu_mean_rows: List[Dict] = []
    for mu in mu_points:
        print(f"[MuScan] mu={mu:+.2f}, e_cu={e_cu_best:.3f}, e_pd={e_pd_best:.3f}")
        env_cfg = _make_env_config(
            seed=int(seeds[0]),
            mu_co=mu,
            e_cu_co=e_cu_best,
            e_pd_co=e_pd_best,
            max_steps=max(args.mu_train_steps + args.eval_steps + 16, 120),
            delta_omega_scale=args.delta_omega_scale,
            reward_shift=args.reward_shift,
        )
        _train_once(
            save_dir=save_root,
            env_cfg=env_cfg,
            oracle=oracle,
            total_steps=args.mu_train_steps,
            ppo_n_steps=args.ppo_n_steps,
            ppo_batch_size=args.ppo_batch_size,
            pirp_scale=args.pirp_scale,
            enable_vis=args.enable_vis,
            device=device,
        )
        eval_rows = _eval_latest_model(
            save_dir=save_root,
            oracle=oracle,
            eval_seeds=seeds,
            mu_co=mu,
            e_cu_co=e_cu_best,
            e_pd_co=e_pd_best,
            eval_steps=args.eval_steps,
            delta_omega_scale=args.delta_omega_scale,
            reward_shift=args.reward_shift,
        )
        for r in eval_rows:
            mu_scan_rows.append(
                {
                    "e_cu_co": e_cu_best,
                    "e_pd_co": e_pd_best,
                    "mu_co": mu,
                    **r,
                }
            )

        best_theta_arr = np.array([r["best_theta_pd"] for r in eval_rows], dtype=float)
        best_nco_arr = np.array([r["best_n_co"] for r in eval_rows], dtype=float)
        best_omega_arr = np.array([r["best_omega"] for r in eval_rows], dtype=float)
        mu_mean_rows.append(
            {
                "e_cu_co": e_cu_best,
                "e_pd_co": e_pd_best,
                "mu_co": mu,
                "mean_best_theta_pd": float(np.mean(best_theta_arr)),
                "mean_best_n_co": float(np.mean(best_nco_arr)),
                "mean_best_omega": float(np.mean(best_omega_arr)),
                "std_best_theta_pd": float(np.std(best_theta_arr)),
                "std_best_n_co": float(np.std(best_nco_arr)),
                "std_best_omega": float(np.std(best_omega_arr)),
            }
        )

    _write_csv(
        save_root / "mu_scan_triplet.csv",
        mu_scan_rows,
        fieldnames=[
            "e_cu_co",
            "e_pd_co",
            "mu_co",
            "seed",
            "best_omega",
            "best_theta_pd",
            "best_n_co",
            "final_omega",
            "final_theta_pd",
            "final_n_co",
        ],
    )
    _write_csv(
        save_root / "mu_scan_triplet_mean.csv",
        mu_mean_rows,
        fieldnames=[
            "e_cu_co",
            "e_pd_co",
            "mu_co",
            "mean_best_theta_pd",
            "mean_best_n_co",
            "mean_best_omega",
            "std_best_theta_pd",
            "std_best_n_co",
            "std_best_omega",
        ],
    )

    print("\n[MuScan] Triplet table (mean over fixed seeds):")
    for row in mu_mean_rows:
        print(
            f"mu={row['mu_co']:+.2f} | theta_Pd={row['mean_best_theta_pd']:.4f} | "
            f"n_CO={row['mean_best_n_co']:.3f} | omega={row['mean_best_omega']:.4f}"
        )
    print(f"\nSaved files under: {save_root}")
    print(" - literature_pair_scan.csv")
    print(" - literature_pair_scores.csv")
    print(" - mu_scan_triplet.csv")
    print(" - mu_scan_triplet_mean.csv")


if __name__ == "__main__":
    main()
