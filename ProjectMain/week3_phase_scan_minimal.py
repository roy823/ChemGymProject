import csv
import os
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import torch
from sb3_contrib.common.maskable.utils import get_action_masks

from chem_gym.agent.trainer import train_agent
from chem_gym.config import EnvConfig, TrainConfig
from main import maybe_load_oracle


@dataclass
class ScanResult:
    mu_co: float
    best_omega: float
    final_omega: float
    best_pd_surface_coverage: float
    final_pd_surface_coverage: float
    best_n_co: int
    final_n_co: int


def build_hybrid_oracle():
    args = Namespace(
        oracle_ckpt=None,
        oracle_mode="hybrid",
        ads_task="oc25",
        disable_ads_ensemble=False,
        ads_sm_ckpt="checkpoints/esen_sm_conserve.pt",
        ads_md_ckpt="checkpoints/esen_md_direct.pt",
        eq2_ckpt="checkpoints/eq2_83M_2M.pt",
        uma_ckpt="checkpoints/uma-m-1p1.pt",
        oracle_fmax=0.05,
        oracle_max_steps=100,
        require_ads_oracle=False,
    )
    oracle = maybe_load_oracle(args)
    if oracle is None:
        raise RuntimeError("Hybrid oracle is unavailable. Please check checkpoints and fairchem install.")
    return oracle


def run_one_mu(
    oracle,
    mu_co: float,
    seed: int,
    train_steps: int,
    eval_steps: int,
    save_root: Path,
) -> ScanResult:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_steps = min(32, max(8, train_steps))

    env_cfg = EnvConfig(
        mode="graph",
        init_seed=seed,
        max_steps=max(train_steps + eval_steps + 8, 64),
        mu_co=float(mu_co),
        bulk_pd_fraction=0.08,
        co_max_coverage=1.0,
        delta_omega_scale=20.0,
        reward_shift=0.1,
    )
    train_cfg = TrainConfig(
        total_timesteps=int(train_steps),
        n_envs=1,
        device=device,
        gamma=0.97,
        lam=0.95,
        learning_rate=2e-4,
        ppo_n_steps=n_steps,
        ppo_batch_size=n_steps,
        ppo_ent_coef=5e-4,
        use_pirp=True,
        pirp_scale=0.02,
        enable_visualization=False,
    )

    run_dir = save_root / f"mu_{mu_co:+.2f}".replace(".", "p").replace("+", "pos").replace("-", "neg")
    run_dir.mkdir(parents=True, exist_ok=True)

    model = train_agent(
        env_config=env_cfg,
        surrogate=None,
        train_config=train_cfg,
        oracle=oracle,
        oracle_energy_fn=oracle.compute_energy if hasattr(oracle, "compute_energy") else None,
        save_dir=str(run_dir),
        use_masking=True,
    )

    venv = model.get_env()
    obs = venv.reset()

    best_omega = float("inf")
    best_pd_cov = float("nan")
    best_n_co = -1

    final_omega = float("nan")
    final_pd_cov = float("nan")
    final_n_co = -1

    for _ in range(eval_steps):
        masks = get_action_masks(venv)
        action, _ = model.predict(obs, action_masks=masks, deterministic=True)
        obs, _, dones, infos = venv.step(action)
        info = infos[0]

        omega = float(info.get("omega", np.nan))
        pd_cov = float(info.get("pd_surface_coverage", np.nan))
        n_co = int(info.get("n_co", -1))

        final_omega = omega
        final_pd_cov = pd_cov
        final_n_co = n_co

        if np.isfinite(omega) and omega < best_omega:
            best_omega = omega
            best_pd_cov = pd_cov
            best_n_co = n_co

        if dones[0]:
            obs = venv.reset()

    return ScanResult(
        mu_co=float(mu_co),
        best_omega=float(best_omega),
        final_omega=float(final_omega),
        best_pd_surface_coverage=float(best_pd_cov),
        final_pd_surface_coverage=float(final_pd_cov),
        best_n_co=int(best_n_co),
        final_n_co=int(final_n_co),
    )


def main():
    seed = 17
    train_steps = 16
    eval_steps = 8
    mu_points: List[float] = [-1.4, -1.0, -0.6, -0.2, 0.2]

    save_root = Path("checkpoints/week3_phase_scan_minimal")
    save_root.mkdir(parents=True, exist_ok=True)

    print("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    print("TRAIN_STEPS_PER_MU", train_steps)
    print("EVAL_STEPS_PER_MU", eval_steps)
    print("MU_POINTS", mu_points)

    oracle = build_hybrid_oracle()

    rows: List[ScanResult] = []
    for mu in mu_points:
        print(f"\n[Scan] Running mu_CO={mu:+.2f} eV")
        row = run_one_mu(
            oracle=oracle,
            mu_co=mu,
            seed=seed,
            train_steps=train_steps,
            eval_steps=eval_steps,
            save_root=save_root,
        )
        rows.append(row)
        print(
            f"[Scan] mu={row.mu_co:+.2f} | best_omega={row.best_omega:.4f} | "
            f"best_theta_pd={row.best_pd_surface_coverage:.4f} | best_n_co={row.best_n_co}"
        )

    csv_path = save_root / "phase_scan_minimal.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "mu_co",
                "best_omega",
                "final_omega",
                "best_pd_surface_coverage",
                "final_pd_surface_coverage",
                "best_n_co",
                "final_n_co",
            ]
        )
        for r in rows:
            writer.writerow(
                [
                    r.mu_co,
                    r.best_omega,
                    r.final_omega,
                    r.best_pd_surface_coverage,
                    r.final_pd_surface_coverage,
                    r.best_n_co,
                    r.final_n_co,
                ]
            )

    print("\n[Scan] Summary")
    for r in rows:
        print(
            f"mu={r.mu_co:+.2f} | best_omega={r.best_omega:.4f} | "
            f"best_theta_pd={r.best_pd_surface_coverage:.4f} | best_n_co={r.best_n_co}"
        )
    print(f"[Scan] CSV saved to: {csv_path}")


if __name__ == "__main__":
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    main()
