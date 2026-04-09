import glob
import os
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from chem_gym.agent.trainer import train_agent
from chem_gym.config import COAdsorptionConfig, EnvConfig, RewardConfig, SurrogateConfig, TrainConfig
from chem_gym.surrogate.ensemble import SurrogateEnsemble
from main import maybe_load_oracle


@dataclass
class EVRunResult:
    seed: int
    ev_last: float
    ev_max: float
    ev_count: int


def _event_files() -> List[str]:
    return sorted(glob.glob("chem_gym_tensorboard/*/events.out.tfevents.*"), key=os.path.getmtime)


def _extract_ev(event_file: str):
    ea = EventAccumulator(event_file)
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    if "train/explained_variance" not in tags:
        return float("nan"), float("nan"), 0
    series = ea.Scalars("train/explained_variance")
    if not series:
        return float("nan"), float("nan"), 0
    vals = [float(x.value) for x in series]
    return float(vals[-1]), float(max(vals)), len(vals)


def build_hybrid_oracle_if_available():
    ckpt_dir = Path("checkpoints")
    args = Namespace(
        oracle_ckpt=None,
        oracle_mode="hybrid",
        ads_task="oc25",
        disable_ads_ensemble=False,
        ads_sm_ckpt=str(ckpt_dir / "esen_sm_conserve.pt"),
        ads_md_ckpt=str(ckpt_dir / "esen_md_direct.pt"),
        eq2_ckpt=str(ckpt_dir / "eq2_83M_2M.pt"),
        uma_ckpt=str(ckpt_dir / "uma-m-1p1.pt"),
        oracle_fmax=0.05,
        oracle_max_steps=100,
        require_ads_oracle=False,
    )
    try:
        return maybe_load_oracle(args)
    except Exception as exc:
        print(f"[EV] Oracle init failed, fallback to surrogate: {exc}")
        return None


def run_one(seed: int, total_steps: int, use_pirp: bool = True, oracle=None) -> EVRunResult:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    use_oracle = oracle is not None
    env_cfg = EnvConfig(
        mode="graph",
        init_seed=seed,
        max_steps=160,
        bulk_pd_fraction=0.08,
        reward=RewardConfig(
            mu_co=-1.0,
            delta_omega_scale=20.0 if use_oracle else 2.0,
            reward_shift=0.2,
        ),
        co_adsorption=COAdsorptionConfig(co_max_coverage=1.0),
    )

    train_cfg = TrainConfig(
        total_timesteps=total_steps,
        n_envs=1,
        device=device,
        gamma=0.97,
        lam=0.95,
        ppo_n_steps=512,
        ppo_batch_size=128,
        ppo_ent_coef=0.0005,
        learning_rate=2e-4,
        use_pirp=use_pirp,
        pirp_scale=0.02,
        enable_visualization=False,
    )

    surrogate = None
    if not use_oracle:
        surrogate = SurrogateEnsemble(
            SurrogateConfig(
                n_models=3,
                mean_energy=-1.0,
                # Keep a non-zero noise floor to avoid degenerate integer Omega levels.
                noise_scale=0.02,
            )
        )

    before = set(_event_files())

    train_agent(
        env_config=env_cfg,
        surrogate=surrogate,
        train_config=train_cfg,
        oracle=oracle,
        oracle_energy_fn=oracle.compute_energy if (oracle is not None and hasattr(oracle, "compute_energy")) else None,
        save_dir=os.path.join("checkpoints", "week2_ev_multirun"),
        use_masking=True,
    )

    after = set(_event_files())
    new_files = sorted(after - before, key=os.path.getmtime)
    if not new_files:
        # fallback to newest
        event_file = _event_files()[-1]
    else:
        event_file = new_files[-1]

    ev_last, ev_max, ev_count = _extract_ev(event_file)
    return EVRunResult(seed=seed, ev_last=ev_last, ev_max=ev_max, ev_count=ev_count)


def main():
    seeds = [11, 22, 33]
    total_steps = 4096

    use_oracle = os.environ.get("CHEMGYM_EV_USE_ORACLE", "0") == "1"
    oracle = build_hybrid_oracle_if_available() if use_oracle else None
    print("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    print("TOTAL_STEPS", total_steps)
    print("ENERGY_BACKEND", "hybrid_oracle" if oracle is not None else "surrogate_fallback")
    if use_oracle and oracle is None:
        print("[EV] Requested oracle mode but checkpoints/backend unavailable; using surrogate fallback.")

    rows = []
    for s in seeds:
        res = run_one(seed=s, total_steps=total_steps, use_pirp=True, oracle=oracle)
        rows.append(res)
        print(
            f"SEED {res.seed} | EV_LAST {res.ev_last:.4f} | "
            f"EV_MAX {res.ev_max:.4f} | EV_POINTS {res.ev_count}"
        )

    ev_last_arr = np.array([r.ev_last for r in rows], dtype=float)
    ev_max_arr = np.array([r.ev_max for r in rows], dtype=float)

    print("SUMMARY_EV_LAST_MEAN", float(np.nanmean(ev_last_arr)))
    print("SUMMARY_EV_LAST_MIN", float(np.nanmin(ev_last_arr)))
    print("SUMMARY_EV_MAX_MEAN", float(np.nanmean(ev_max_arr)))
    print("SUMMARY_EV_MAX_MIN", float(np.nanmin(ev_max_arr)))


if __name__ == "__main__":
    main()
