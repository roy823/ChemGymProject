import os
import numpy as np
import torch

from sb3_contrib.common.maskable.utils import get_action_masks

# Path shim: put ProjectMain/ on sys.path so 'chem_gym' / 'main' resolve regardless of cwd.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

from chem_gym.agent.trainer import train_agent
from chem_gym.config import COAdsorptionConfig, EnvConfig, RewardConfig, SurrogateConfig, TrainConfig
from chem_gym.surrogate.ensemble import SurrogateEnsemble


def run_case(name: str, use_pirp: bool, seed: int):
    env_cfg = EnvConfig(
        mode="graph",
        init_seed=seed,
        bulk_pd_fraction=0.08,
        reward=RewardConfig(
            mu_co=-1.0,
            delta_omega_scale=2.0,
            reward_shift=0.15,
        ),
        co_adsorption=COAdsorptionConfig(co_max_coverage=1.0),
    )
    train_cfg = TrainConfig(
        total_timesteps=1536,
        n_envs=1,
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_pirp=use_pirp,
        pirp_scale=0.02,
        ppo_n_steps=512,
        ppo_batch_size=128,
        enable_visualization=False,
    )
    surrogate = SurrogateEnsemble(SurrogateConfig(n_models=3, mean_energy=-1.0, noise_scale=0.05))

    save_dir = os.path.join("checkpoints", "week2_smoke_v3", name)
    os.makedirs(save_dir, exist_ok=True)

    model = train_agent(
        env_config=env_cfg,
        surrogate=surrogate,
        train_config=train_cfg,
        oracle=None,
        oracle_energy_fn=None,
        save_dir=save_dir,
        use_masking=True,
    )

    venv = model.get_env()
    obs = venv.reset()

    best_omega = float("inf")
    last_omega = float("inf")
    raw_rewards = []

    for _ in range(240):
        masks = get_action_masks(venv)
        action, _ = model.predict(obs, action_masks=masks, deterministic=True)
        obs, _, dones, infos = venv.step(action)
        omega = float(infos[0].get("omega", np.nan))
        raw_rewards.append(float(infos[0].get("reward_terms", {}).get("reward_total", np.nan)))
        if omega < best_omega:
            best_omega = omega
        last_omega = omega
        if dones[0]:
            obs = venv.reset()

    rr = np.asarray(raw_rewards, dtype=float)
    rr = rr[np.isfinite(rr)]
    pos_frac = float(np.mean(rr > 0.0)) if rr.size > 0 else float("nan")
    mean_rr = float(np.mean(rr)) if rr.size > 0 else float("nan")
    return best_omega, last_omega, mean_rr, pos_frac


def summarize(rows):
    arr = np.asarray(rows, dtype=float)
    return np.nanmean(arr, axis=0)


seeds = [11, 22, 33]
base_rows = [run_case("base", use_pirp=False, seed=s) for s in seeds]
pirp_rows = [run_case("pirp", use_pirp=True, seed=s) for s in seeds]

base_best, base_last, base_mean_r, base_pos = summarize(base_rows)
pirp_best, pirp_last, pirp_mean_r, pirp_pos = summarize(pirp_rows)

print("RESULT_BASE_BEST", base_best)
print("RESULT_BASE_LAST", base_last)
print("RESULT_BASE_MEAN_RAW_REWARD", base_mean_r)
print("RESULT_BASE_POS_FRAC", base_pos)
print("RESULT_PIRP_BEST", pirp_best)
print("RESULT_PIRP_LAST", pirp_last)
print("RESULT_PIRP_MEAN_RAW_REWARD", pirp_mean_r)
print("RESULT_PIRP_POS_FRAC", pirp_pos)
