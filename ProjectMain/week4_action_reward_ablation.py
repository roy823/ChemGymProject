from __future__ import annotations

import argparse
import csv
from argparse import Namespace
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from chem_gym.agent.trainer import train_agent
from chem_gym.config import (
    COAdsorptionConfig,
    ConstraintConfig,
    EnvConfig,
    RewardConfig,
    TrainConfig,
    UMAPBRSConfig,
)
from chem_gym.envs.chem_env import ChemGymEnv
from main import maybe_load_oracle


PROFILE_CHOICES = {
    "mutation_legacy",
    "mutation_delta_stop",
    "mutation_delta_stop_nobonus",
    "mutation_delta_strict_stop",
    "mutation_pbrs_stop",
    "swap_delta_stop",
}


def parse_list(value: str) -> List[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def parse_int_list(value: str) -> List[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def spearman_corr(values: Sequence[float]) -> float:
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


def autocorr_lag(values: Sequence[float], lag: int = 2) -> float:
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
    if str(getattr(args, "oracle_mode", "hybrid")).lower() == "none":
        return None
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
        raise RuntimeError("Hybrid oracle unavailable. Check checkpoint paths and FAIRChem installation.")
    return oracle


def build_env_config(profile: str, seed: int, mu_co: float, max_steps: int) -> EnvConfig:
    reward_cfg = RewardConfig(
        mu_co=float(mu_co),
        delta_omega_scale=20.0,
        reward_profile="pure_delta_omega",
    )
    constraint_cfg = ConstraintConfig()
    co_cfg = COAdsorptionConfig(co_max_coverage=1.0)
    uma_cfg = UMAPBRSConfig(use_uma_pbrs=False)

    cfg = EnvConfig(
        mode="graph",
        init_seed=int(seed),
        max_steps=int(max_steps),
        bulk_pd_fraction=0.08,
        reward=reward_cfg,
        constraint=constraint_cfg,
        co_adsorption=co_cfg,
        uma_pbrs=uma_cfg,
    )

    if profile == "mutation_legacy":
        cfg.action_mode = "mutation"
        cfg.enable_noop_action = True
        cfg.stop_terminates = False
        cfg.reward.reward_profile = "legacy"
        cfg.reward_profile = "legacy"
        cfg.uma_pbrs.use_uma_pbrs = True
        cfg.use_uma_pbrs = True
    elif profile == "mutation_delta_stop":
        cfg.action_mode = "mutation"
        cfg.enable_noop_action = True
        cfg.stop_terminates = True
        cfg.min_stop_steps = 8
        cfg.reward.reward_profile = "pure_delta_omega"
        cfg.reward_profile = "pure_delta_omega"
        cfg.uma_pbrs.use_uma_pbrs = False
        cfg.use_uma_pbrs = False
        cfg.constraint.constraint_update_mode = "frozen"
        cfg.constraint.constraint_weight = 0.0
        cfg.constraint.constraint_lambda_init = 0.0
        cfg.constraint.constraint_lambda_min = 0.0
        cfg.constraint.constraint_lambda_max = 0.0
        cfg.constraint_update_mode = "frozen"
        cfg.constraint_weight = 0.0
        cfg.constraint_lambda_init = 0.0
        cfg.constraint_lambda_min = 0.0
        cfg.constraint_lambda_max = 0.0
    elif profile == "mutation_delta_strict_stop":
        cfg.action_mode = "mutation"
        cfg.enable_noop_action = False
        cfg.stop_terminates = True
        cfg.min_stop_steps = 8
        cfg.reward.reward_profile = "pure_delta_omega"
        cfg.reward_profile = "pure_delta_omega"
        cfg.uma_pbrs.use_uma_pbrs = False
        cfg.use_uma_pbrs = False
        cfg.constraint.constraint_update_mode = "frozen"
        cfg.constraint.constraint_weight = 0.0
        cfg.constraint.constraint_lambda_init = 0.0
        cfg.constraint.constraint_lambda_min = 0.0
        cfg.constraint.constraint_lambda_max = 0.0
        cfg.constraint_update_mode = "frozen"
        cfg.constraint_weight = 0.0
        cfg.constraint_lambda_init = 0.0
        cfg.constraint_lambda_min = 0.0
        cfg.constraint_lambda_max = 0.0
    elif profile == "mutation_delta_stop_nobonus":
        cfg.action_mode = "mutation"
        cfg.enable_noop_action = True
        cfg.stop_terminates = True
        cfg.min_stop_steps = 8
        cfg.reward.reward_profile = "pure_delta_omega"
        cfg.reward_profile = "pure_delta_omega"
        cfg.uma_pbrs.use_uma_pbrs = False
        cfg.use_uma_pbrs = False
        cfg.constraint.constraint_update_mode = "frozen"
        cfg.constraint.constraint_weight = 0.0
        cfg.constraint.constraint_lambda_init = 0.0
        cfg.constraint.constraint_lambda_min = 0.0
        cfg.constraint.constraint_lambda_max = 0.0
        cfg.constraint_update_mode = "frozen"
        cfg.constraint_weight = 0.0
        cfg.constraint_lambda_init = 0.0
        cfg.constraint_lambda_min = 0.0
        cfg.constraint_lambda_max = 0.0
    elif profile == "mutation_pbrs_stop":
        cfg.action_mode = "mutation"
        cfg.enable_noop_action = True
        cfg.stop_terminates = True
        cfg.min_stop_steps = 8
        cfg.reward.reward_profile = "delta_omega_plus_pbrs"
        cfg.reward_profile = "delta_omega_plus_pbrs"
        cfg.uma_pbrs.use_uma_pbrs = True
        cfg.use_uma_pbrs = True
        cfg.constraint.constraint_update_mode = "frozen"
        cfg.constraint.constraint_weight = 0.0
        cfg.constraint.constraint_lambda_init = 0.0
        cfg.constraint.constraint_lambda_min = 0.0
        cfg.constraint.constraint_lambda_max = 0.0
        cfg.constraint_update_mode = "frozen"
        cfg.constraint_weight = 0.0
        cfg.constraint_lambda_init = 0.0
        cfg.constraint_lambda_min = 0.0
        cfg.constraint_lambda_max = 0.0
    elif profile == "swap_delta_stop":
        cfg.action_mode = "swap"
        cfg.enable_noop_action = True
        cfg.stop_terminates = True
        cfg.min_stop_steps = 8
        cfg.reward.reward_profile = "pure_delta_omega"
        cfg.reward_profile = "pure_delta_omega"
        cfg.uma_pbrs.use_uma_pbrs = False
        cfg.use_uma_pbrs = False
        cfg.constraint.constraint_update_mode = "frozen"
        cfg.constraint.constraint_weight = 0.0
        cfg.constraint.constraint_lambda_init = 0.0
        cfg.constraint.constraint_lambda_min = 0.0
        cfg.constraint.constraint_lambda_max = 0.0
        cfg.constraint_update_mode = "frozen"
        cfg.constraint_weight = 0.0
        cfg.constraint_lambda_init = 0.0
        cfg.constraint_lambda_min = 0.0
        cfg.constraint_lambda_max = 0.0
    else:
        raise ValueError(f"Unsupported profile: {profile}")

    return cfg


def build_train_config(profile: str, total_steps: int, device: str) -> TrainConfig:
    use_pirp = profile.startswith("mutation_")
    noop_bonus = 0.0 if profile in {"mutation_delta_strict_stop", "mutation_delta_stop_nobonus"} else 0.25
    return TrainConfig(
        total_timesteps=int(total_steps),
        n_envs=1,
        device=device,
        gamma=0.97,
        lam=0.95,
        learning_rate=2e-4,
        ppo_n_steps=min(256, max(64, int(total_steps // 4))),
        ppo_batch_size=min(128, max(32, int(total_steps // 8))),
        ppo_ent_coef=5e-4,
        use_pirp=bool(use_pirp),
        pirp_scale=0.02,
        noop_logit_bonus=noop_bonus,
        enable_visualization=False,
    )


def eval_model(model, eval_steps: int) -> Dict[str, float]:
    venv = model.get_env()
    obs = venv.reset()

    omega_trace: List[float] = []
    reward_trace: List[float] = []
    stop_steps = 0
    best_omega = float("inf")
    best_theta = float("nan")
    best_nco = -1
    final_omega = float("nan")
    final_theta = float("nan")
    final_nco = -1

    for _ in range(int(eval_steps)):
        masks = get_action_masks(venv)
        action, _ = model.predict(obs, action_masks=masks, deterministic=True)
        obs, _, dones, infos = venv.step(action)
        info = infos[0]

        action_type = str(info.get("action_type", ""))
        omega = float(info.get("omega", np.nan))
        theta = float(info.get("pd_surface_coverage", np.nan))
        nco = int(info.get("n_co", -1))
        reward_raw = float(info.get("reward_terms", {}).get("reward_total", np.nan))

        if action_type in {"no_op", "stop"}:
            stop_steps += 1
        if np.isfinite(omega):
            omega_trace.append(omega)
            if omega < best_omega:
                best_omega = omega
                best_theta = theta
                best_nco = nco
        if np.isfinite(reward_raw):
            reward_trace.append(reward_raw)

        final_omega = omega
        final_theta = theta
        final_nco = nco

        if dones[0]:
            obs = venv.reset()

    return {
        "best_omega": float(best_omega),
        "best_theta_pd": float(best_theta),
        "best_n_co": int(best_nco),
        "final_omega": float(final_omega),
        "final_theta_pd": float(final_theta),
        "final_n_co": int(final_nco),
        "stop_ratio": float(stop_steps / max(1, int(eval_steps))),
        "mean_eval_omega": float(np.nanmean(omega_trace)) if omega_trace else float("nan"),
        "mean_eval_reward": float(np.nanmean(reward_trace)) if reward_trace else float("nan"),
    }


def write_csv(path: Path, rows: Sequence[Dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _maybe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def _maybe_int(value):
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return value
    if np.isfinite(parsed) and abs(parsed - round(parsed)) <= 1e-9:
        return int(round(parsed))
    return parsed


def convert_case_row(row: Dict[str, str]) -> Dict:
    converted = dict(row)
    for key in ("profile", "action_mode", "reward_profile"):
        if key in converted:
            converted[key] = str(converted[key])
    for key in ("stop_terminates",):
        if key in converted:
            converted[key] = str(converted[key]).strip().lower() == "true"
    for key in ("seed", "best_n_co", "final_n_co"):
        if key in converted:
            converted[key] = _maybe_int(converted[key])
    if "mu_co" in converted:
        converted["mu_co"] = _maybe_float(converted["mu_co"])
    for key in (
        "best_omega",
        "best_theta_pd",
        "final_omega",
        "final_theta_pd",
        "stop_ratio",
        "mean_eval_omega",
        "mean_eval_reward",
    ):
        if key in converted:
            converted[key] = _maybe_float(converted[key])
    return converted


def convert_standard_seed_row(row: Dict[str, str]) -> Dict:
    converted = dict(row)
    for key in ("profile",):
        if key in converted:
            converted[key] = str(converted[key])
    for key in ("train_seed", "eval_seed", "best_n_co"):
        if key in converted:
            converted[key] = _maybe_int(converted[key])
    if "mu_co" in converted:
        converted["mu_co"] = _maybe_float(converted["mu_co"])
    for key in (
        "best_omega",
        "best_theta_pd",
        "omega_spearman",
        "dfrac_autocorr_lag2",
        "noop_ratio",
        "lambda_last",
    ):
        if key in converted:
            converted[key] = _maybe_float(converted[key])
    return converted


def convert_standard_train_row(row: Dict[str, str]) -> Dict:
    converted = dict(row)
    for key in ("profile",):
        if key in converted:
            converted[key] = str(converted[key])
    for key in ("train_seed",):
        if key in converted:
            converted[key] = _maybe_int(converted[key])
    if "mu_co" in converted:
        converted["mu_co"] = _maybe_float(converted["mu_co"])
    for key in (
        "mean_best_omega",
        "best_omega_global",
        "mean_best_theta_pd",
        "mean_best_n_co",
        "mean_omega_spearman",
        "mean_dfrac_autocorr_lag2",
        "mean_noop_ratio",
        "mean_lambda_last",
    ):
        if key in converted:
            converted[key] = _maybe_float(converted[key])
    return converted


def load_saved_model(
    profile: str,
    run_dir: Path,
    oracle,
    mu_co: float,
    seed: int,
    eval_steps: int,
):
    model_path = run_dir / "latest_model"
    stats_path = run_dir / "latest_vec_normalize.pkl"
    if not model_path.with_suffix(".zip").exists():
        raise FileNotFoundError(f"Missing model file: {model_path.with_suffix('.zip')}")
    if not stats_path.exists():
        raise FileNotFoundError(f"Missing vec normalize file: {stats_path}")

    env_cfg = build_env_config(
        profile=profile,
        seed=int(seed),
        mu_co=float(mu_co),
        max_steps=max(int(eval_steps) + 24, 128),
    )

    def _make_single():
        return ChemGymEnv(env_cfg, oracle=oracle)

    base_venv = DummyVecEnv([_make_single])
    venv = VecNormalize.load(str(stats_path), base_venv)
    venv.training = False
    venv.norm_reward = False
    return MaskablePPO.load(str(model_path), env=venv)


def standard_eval_saved_model(
    profile: str,
    run_dir: Path,
    oracle,
    mu_co: float,
    eval_seeds: Sequence[int],
    eval_steps: int,
) -> List[Dict[str, float]]:
    model_path = run_dir / "latest_model"
    stats_path = run_dir / "latest_vec_normalize.pkl"
    if not model_path.with_suffix(".zip").exists():
        raise FileNotFoundError(f"Missing model file: {model_path.with_suffix('.zip')}")
    if not stats_path.exists():
        raise FileNotFoundError(f"Missing vec normalize file: {stats_path}")

    rows: List[Dict[str, float]] = []
    for eval_seed in eval_seeds:
        env_cfg = build_env_config(
            profile=profile,
            seed=int(eval_seed),
            mu_co=float(mu_co),
            max_steps=max(int(eval_steps) + 16, 96),
        )

        def _make_single():
            return ChemGymEnv(env_cfg, oracle=oracle)

        base_venv = DummyVecEnv([_make_single])
        venv = VecNormalize.load(str(stats_path), base_venv)
        venv.training = False
        venv.norm_reward = False
        model = MaskablePPO.load(str(model_path), env=venv)

        obs = venv.reset()
        omega_trace: List[float] = []
        dfrac_trace: List[float] = []
        stop_steps = 0
        best_omega = float("inf")
        best_theta = float("nan")
        best_nco = -1
        last_lambda = float("nan")

        for _ in range(int(eval_steps)):
            masks = get_action_masks(venv)
            action, _ = model.predict(obs, action_masks=masks, deterministic=True)
            obs, _, dones, infos = venv.step(action)
            info = infos[0]

            omega = float(info.get("omega", np.nan))
            theta = float(info.get("pd_surface_coverage", np.nan))
            nco = int(info.get("n_co", -1))
            d_frac = float(info.get("constraint_d_frac", np.nan))
            lam = float(info.get("constraint_lambda", np.nan))
            action_type = str(info.get("action_type", ""))

            if action_type in {"no_op", "stop"}:
                stop_steps += 1
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

        rows.append(
            {
                "eval_seed": int(eval_seed),
                "best_omega": float(best_omega),
                "best_theta_pd": float(best_theta),
                "best_n_co": int(best_nco),
                "omega_spearman": float(spearman_corr(omega_trace)),
                "dfrac_autocorr_lag2": float(autocorr_lag(dfrac_trace, lag=2)),
                "noop_ratio": float(stop_steps / max(1, int(eval_steps))),
                "lambda_last": float(last_lambda),
            }
        )

    return rows


def aggregate_standard_eval(rows: Sequence[Dict[str, float]]) -> Dict[str, float]:
    return {
        "mean_best_omega": float(np.nanmean([r["best_omega"] for r in rows])),
        "best_omega_global": float(np.nanmin([r["best_omega"] for r in rows])),
        "mean_best_theta_pd": float(np.nanmean([r["best_theta_pd"] for r in rows])),
        "mean_best_n_co": float(np.nanmean([r["best_n_co"] for r in rows])),
        "mean_omega_spearman": float(np.nanmean([r["omega_spearman"] for r in rows])),
        "mean_dfrac_autocorr_lag2": float(np.nanmean([r["dfrac_autocorr_lag2"] for r in rows])),
        "mean_noop_ratio": float(np.nanmean([r["noop_ratio"] for r in rows])),
        "mean_lambda_last": float(np.nanmean([r["lambda_last"] for r in rows])),
    }


def write_markdown_summary(
    path: Path,
    mu_co: float,
    train_steps: int,
    eval_steps: int,
    standard_eval_steps: int,
    summary_rows: Sequence[Dict],
    standard_profile_rows: Sequence[Dict],
    standard_eval_train_rows: Sequence[Dict],
) -> None:
    lines: List[str] = []
    lines.append(f"## Week 4 Closed-Loop Summary at mu_CO = {float(mu_co):.3f} eV")
    lines.append("")
    lines.append("Protocol:")
    lines.append(f"- Train steps per run: {int(train_steps)}")
    lines.append(f"- Short eval steps after train: {int(eval_steps)}")
    lines.append(f"- Standard eval steps per seed: {int(standard_eval_steps)}")
    lines.append("")

    if standard_profile_rows:
        lines.append("### Standardized profile ranking")
        lines.append("")
        lines.append("| Profile | mean(best_omega) | best_omega_global | mean(theta_Pd) | mean(N_CO) | mean(noop_ratio) | mean(omega_spearman) |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for row in standard_profile_rows:
            lines.append(
                "| "
                f"{row['profile']} | "
                f"{row['mean_best_omega']:.6f} | "
                f"{row['best_omega_global']:.6f} | "
                f"{row['mean_best_theta_pd']:.6f} | "
                f"{row['mean_best_n_co']:.6f} | "
                f"{row['mean_noop_ratio']:.6f} | "
                f"{row['mean_omega_spearman']:.6f} |"
            )
        lines.append("")

    if standard_eval_train_rows:
        lines.append("### Per train-seed standardized evaluation")
        lines.append("")
        lines.append("| Profile | train_seed | mean(best_omega) | best_omega_global | mean(theta_Pd) | mean(N_CO) | mean(noop_ratio) |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for row in standard_eval_train_rows:
            lines.append(
                "| "
                f"{row['profile']} | "
                f"{int(row['train_seed'])} | "
                f"{row['mean_best_omega']:.6f} | "
                f"{row['best_omega_global']:.6f} | "
                f"{row['mean_best_theta_pd']:.6f} | "
                f"{row['mean_best_n_co']:.6f} | "
                f"{row['mean_noop_ratio']:.6f} |"
            )
        lines.append("")

    if summary_rows:
        lines.append("### Short post-train evaluation")
        lines.append("")
        lines.append("| Profile | mean(best_omega) | mean(final_omega) | mean(theta_Pd) | mean(N_CO) | mean(stop_ratio) |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
        for row in summary_rows:
            lines.append(
                "| "
                f"{row['profile']} | "
                f"{row['mean_best_omega']:.6f} | "
                f"{row['mean_final_omega']:.6f} | "
                f"{row['mean_best_theta_pd']:.6f} | "
                f"{row['mean_best_n_co']:.6f} | "
                f"{row['mean_stop_ratio']:.6f} |"
            )
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def collect_case_artifacts(save_root: Path, profiles: Sequence[str]) -> tuple[List[Dict], List[Dict], List[Dict]]:
    case_rows: List[Dict] = []
    standard_seed_rows: List[Dict] = []
    standard_train_rows: List[Dict] = []
    for profile in profiles:
        profile_dir = save_root / profile
        if not profile_dir.exists():
            continue
        for seed_dir in sorted(p for p in profile_dir.iterdir() if p.is_dir()):
            case_csv = seed_dir / "case_metrics.csv"
            standard_seed_csv = seed_dir / "standard_eval_per_seed.csv"
            standard_train_csv = seed_dir / "standard_eval_summary.csv"
            case_rows.extend(convert_case_row(row) for row in read_csv_rows(case_csv))
            standard_seed_rows.extend(convert_standard_seed_row(row) for row in read_csv_rows(standard_seed_csv))
            standard_train_rows.extend(convert_standard_train_row(row) for row in read_csv_rows(standard_train_csv))
    return case_rows, standard_seed_rows, standard_train_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Week-4 action/reward ablation runner")
    parser.add_argument(
        "--profiles",
        type=str,
        default="mutation_legacy,mutation_delta_stop,swap_delta_stop",
        help=f"Comma-separated subset of: {','.join(sorted(PROFILE_CHOICES))}",
    )
    parser.add_argument("--seeds", type=str, default="11,22,33")
    parser.add_argument("--mu-co", type=float, default=-0.6)
    parser.add_argument("--train-steps", type=int, default=2048)
    parser.add_argument("--eval-steps", type=int, default=120)
    parser.add_argument("--standard-eval-seeds", type=str, default="11,22,33")
    parser.add_argument("--standard-eval-steps", type=int, default=60)
    parser.add_argument("--disable-standard-eval", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--write-summary-md", action="store_true")
    parser.add_argument("--save-root", type=str, default="ProjectMain/checkpoints/week4_action_reward_ablation")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--oracle-mode", choices=["hybrid", "none"], default="hybrid")
    parser.add_argument("--ads-task", type=str, default="oc25")
    parser.add_argument("--ads-sm-ckpt", type=str, default="ProjectMain/checkpoints/esen_sm_conserve.pt")
    parser.add_argument("--ads-md-ckpt", type=str, default="ProjectMain/checkpoints/esen_md_direct.pt")
    parser.add_argument("--eq2-ckpt", type=str, default="ProjectMain/checkpoints/eq2_83M_2M.pt")
    parser.add_argument("--uma-ckpt", type=str, default="ProjectMain/checkpoints/uma-m-1p1.pt")
    args = parser.parse_args()

    profiles = parse_list(args.profiles)
    invalid = [p for p in profiles if p not in PROFILE_CHOICES]
    if invalid:
        raise ValueError(f"Unsupported profiles: {invalid}")

    seeds = parse_int_list(args.seeds)
    save_root = Path(args.save_root)
    save_root.mkdir(parents=True, exist_ok=True)

    print("DEVICE", args.device)
    print("PROFILES", profiles)
    print("SEEDS", seeds)
    print("MU_CO", float(args.mu_co))
    print("TRAIN_STEPS", int(args.train_steps))
    print("EVAL_STEPS", int(args.eval_steps))
    print("STANDARD_EVAL", not bool(args.disable_standard_eval))

    oracle = build_oracle(args)
    standard_eval_seeds = parse_int_list(args.standard_eval_seeds)

    if not bool(args.aggregate_only):
        for profile in profiles:
            for seed in seeds:
                env_cfg = build_env_config(
                    profile=profile,
                    seed=int(seed),
                    mu_co=float(args.mu_co),
                    max_steps=max(int(args.eval_steps) + 24, 128),
                )
                train_cfg = build_train_config(profile=profile, total_steps=int(args.train_steps), device=str(args.device))
                case_dir = save_root / profile / f"seed_{seed}"
                case_dir.mkdir(parents=True, exist_ok=True)

                print(
                    f"[Run] profile={profile} seed={seed} action_mode={env_cfg.action_mode} "
                    f"reward_profile={env_cfg.reward_profile} stop={env_cfg.stop_terminates}"
                )
                has_saved_model = (case_dir / "latest_model.zip").exists() and (case_dir / "latest_vec_normalize.pkl").exists()
                if bool(args.skip_existing) and has_saved_model:
                    print(f"[SkipExisting] Reusing saved model for profile={profile} seed={seed}")
                    model = load_saved_model(
                        profile=profile,
                        run_dir=case_dir,
                        oracle=oracle,
                        mu_co=float(args.mu_co),
                        seed=int(seed),
                        eval_steps=int(args.eval_steps),
                    )
                else:
                    model = train_agent(
                        env_config=env_cfg,
                        surrogate=None,
                        train_config=train_cfg,
                        oracle_energy_fn=oracle.compute_energy if hasattr(oracle, "compute_energy") else None,
                        oracle=oracle,
                        save_dir=str(case_dir),
                        use_masking=True,
                    )
                metrics = eval_model(model, eval_steps=int(args.eval_steps))
                row = {
                    "profile": profile,
                    "seed": int(seed),
                    "mu_co": float(args.mu_co),
                    "action_mode": env_cfg.action_mode,
                    "reward_profile": env_cfg.reward_profile,
                    "stop_terminates": bool(env_cfg.stop_terminates),
                    **metrics,
                }
                write_csv(case_dir / "case_metrics.csv", [row])
                print(row)

                if not bool(args.disable_standard_eval):
                    eval_rows = standard_eval_saved_model(
                        profile=profile,
                        run_dir=case_dir,
                        oracle=oracle,
                        mu_co=float(args.mu_co),
                        eval_seeds=standard_eval_seeds,
                        eval_steps=int(args.standard_eval_steps),
                    )
                    persisted_eval_rows: List[Dict] = []
                    for eval_row in eval_rows:
                        persisted_eval_rows.append(
                            {
                                "profile": profile,
                                "train_seed": int(seed),
                                "mu_co": float(args.mu_co),
                                **eval_row,
                            }
                        )
                    write_csv(case_dir / "standard_eval_per_seed.csv", persisted_eval_rows)
                    summary = {
                        "profile": profile,
                        "train_seed": int(seed),
                        "mu_co": float(args.mu_co),
                        **aggregate_standard_eval(eval_rows),
                    }
                    write_csv(case_dir / "standard_eval_summary.csv", [summary])
                    print(f"[StandardEval] profile={profile} train_seed={seed} summary={summary}")

    rows, standard_eval_rows, standard_eval_train_rows = collect_case_artifacts(save_root=save_root, profiles=profiles)

    write_csv(save_root / "per_seed_metrics.csv", rows)

    summary_rows: List[Dict] = []
    for profile in profiles:
        sub = [r for r in rows if r["profile"] == profile]
        if not sub:
            continue
        summary_rows.append(
            {
                "profile": profile,
                "mean_best_omega": float(np.nanmean([r["best_omega"] for r in sub])),
                "mean_best_theta_pd": float(np.nanmean([r["best_theta_pd"] for r in sub])),
                "mean_best_n_co": float(np.nanmean([r["best_n_co"] for r in sub])),
                "mean_final_omega": float(np.nanmean([r["final_omega"] for r in sub])),
                "mean_stop_ratio": float(np.nanmean([r["stop_ratio"] for r in sub])),
                "mean_eval_reward": float(np.nanmean([r["mean_eval_reward"] for r in sub])),
            }
        )
    write_csv(save_root / "summary_metrics.csv", summary_rows)

    if standard_eval_rows:
        write_csv(save_root / "standard_eval_per_seed.csv", standard_eval_rows)
        write_csv(save_root / "standard_eval_by_train_seed.csv", standard_eval_train_rows)

        standard_profile_rows: List[Dict[str, float]] = []
        for profile in profiles:
            sub = [r for r in standard_eval_train_rows if r["profile"] == profile]
            if not sub:
                continue
            standard_profile_rows.append(
                {
                    "profile": profile,
                    "mean_best_omega": float(np.nanmean([r["mean_best_omega"] for r in sub])),
                    "best_omega_global": float(np.nanmin([r["best_omega_global"] for r in sub])),
                    "mean_best_theta_pd": float(np.nanmean([r["mean_best_theta_pd"] for r in sub])),
                    "mean_best_n_co": float(np.nanmean([r["mean_best_n_co"] for r in sub])),
                    "mean_omega_spearman": float(np.nanmean([r["mean_omega_spearman"] for r in sub])),
                    "mean_dfrac_autocorr_lag2": float(np.nanmean([r["mean_dfrac_autocorr_lag2"] for r in sub])),
                    "mean_noop_ratio": float(np.nanmean([r["mean_noop_ratio"] for r in sub])),
                }
            )

        standard_profile_rows.sort(key=lambda r: (r["mean_best_omega"], r["best_omega_global"]))
        write_csv(save_root / "standard_eval_profile_summary.csv", standard_profile_rows)
        if bool(args.write_summary_md):
            write_markdown_summary(
                path=save_root / "RESULT_SUMMARY.md",
                mu_co=float(args.mu_co),
                train_steps=int(args.train_steps),
                eval_steps=int(args.eval_steps),
                standard_eval_steps=int(args.standard_eval_steps),
                summary_rows=summary_rows,
                standard_profile_rows=standard_profile_rows,
                standard_eval_train_rows=standard_eval_train_rows,
            )

    print("[Done] Saved metrics:")
    print(save_root / "per_seed_metrics.csv")
    print(save_root / "summary_metrics.csv")
    if standard_eval_rows:
        print(save_root / "standard_eval_per_seed.csv")
        print(save_root / "standard_eval_by_train_seed.csv")
        print(save_root / "standard_eval_profile_summary.csv")
        if bool(args.write_summary_md):
            print(save_root / "RESULT_SUMMARY.md")


if __name__ == "__main__":
    main()
