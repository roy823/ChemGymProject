import argparse
import dataclasses
import math
from pathlib import Path

import torch

from chem_gym.agent.trainer import train_agent
from chem_gym.baselines import random_search, simulated_annealing
from chem_gym.config import (
    COAdsorptionConfig, ConstraintConfig, EnvConfig,
    RewardConfig, SurrogateConfig, TrainConfig, UMAPBRSConfig,
)
from chem_gym.envs.chem_env import ChemGymEnv
from chem_gym.surrogate.ensemble import SurrogateEnsemble
from chem_gym.surrogate.hybrid_oracle import HybridGrandPotentialOracle
from chem_gym.surrogate.ocp_model import EquiformerV2Oracle, OC25EnsembleOracle, UMAOracle


def _fill_dataclass(cls, namespace, overrides=None):
    """Create a dataclass instance from an argparse Namespace.

    For each field in `cls`, look for an attribute with the same name
    (underscores matching hyphens) on `namespace`. Fields not found on
    the namespace are left at their dataclass defaults. Explicit
    `overrides` dict takes precedence.
    """
    kwargs = {}
    for f in dataclasses.fields(cls):
        name = f.name
        if overrides and name in overrides:
            kwargs[name] = overrides[name]
        elif hasattr(namespace, name):
            kwargs[name] = getattr(namespace, name)
        # Also try hyphen-to-underscore variants from argparse
        elif hasattr(namespace, name.replace("_", "-")):
            kwargs[name] = getattr(namespace, name.replace("_", "-"))
    return cls(**kwargs)


def parse_args():
    parser = argparse.ArgumentParser(description="Chem-Gym launcher")

    parser.add_argument("--mode", choices=["train", "baseline", "eval"], default="train")
    parser.add_argument(
        "--experiment-profile",
        choices=["default", "route_a", "mutation_delta_stop", "mutation_delta_strict_stop", "swap_stop"],
        default="default",
        help="Apply a curated parameter bundle. route_a = hard mask + debt shaping + no UMA PBRS; mutation_delta_stop = mutation + stop + pure DeltaOmega; mutation_delta_strict_stop = mutation + explicit stop only; swap_stop = swap-only + stop + pure DeltaOmega.",
    )
    parser.add_argument("--obs-mode", choices=["image", "graph"], default="graph")
    parser.add_argument("--total-steps", type=int, default=5000)
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=None)

    # Operando env knobs
    parser.add_argument("--n-active-layers", type=int, default=4)
    parser.add_argument("--mu-co", type=float, default=-1.0)
    parser.add_argument("--bulk-pd-fraction", type=float, default=0.08)
    parser.add_argument("--step-penalty", type=float, default=0.01)
    parser.add_argument("--omega-reward-scale", type=float, default=1.0)
    parser.add_argument("--delta-omega-scale", type=float, default=1.0)
    parser.add_argument("--debt-improvement-scale", type=float, default=0.1)
    parser.add_argument("--debt-abs-penalty", type=float, default=0.0)
    parser.add_argument("--reward-shift", type=float, default=0.1)
    parser.add_argument("--linear-reward-clip", type=float, default=3.0)
    parser.add_argument(
        "--disable-thermo-consistent-backend",
        action="store_true",
        help="Use legacy hybrid Omega with cross-backend slab term (for ablation only).",
    )
    parser.add_argument("--action-mode", choices=["mutation", "swap"], default="mutation")
    parser.add_argument("--disable-noop-action", action="store_true")
    parser.add_argument("--stop-terminates", action="store_true")
    parser.add_argument("--min-stop-steps", type=int, default=0)
    parser.add_argument("--enable-deviation-mask", action="store_true")
    parser.add_argument(
        "--reward-profile",
        choices=["legacy", "pure_delta_omega", "delta_omega_plus_pbrs"],
        default="pure_delta_omega",
    )
    parser.add_argument("--constraint-threshold-frac", type=float, default=0.12)
    parser.add_argument("--constraint-weight", type=float, default=1.0)
    parser.add_argument("--constraint-lambda-init", type=float, default=1.0)
    parser.add_argument("--constraint-lambda-min", type=float, default=0.0)
    parser.add_argument("--constraint-lambda-max", type=float, default=10.0)
    parser.add_argument("--constraint-pid-kp", type=float, default=0.10)
    parser.add_argument("--constraint-pid-ki", type=float, default=0.01)
    parser.add_argument("--constraint-pid-kd", type=float, default=0.01)
    parser.add_argument("--constraint-integral-clip", type=float, default=100.0)
    parser.add_argument("--constraint-update-mode", choices=["step", "rollout", "frozen"], default="rollout")
    parser.add_argument("--constraint-rollout-gain", type=float, default=1.0)
    parser.add_argument("--disable-uma-pbrs", action="store_true")
    parser.add_argument("--uma-pbrs-gamma", type=float, default=None)
    parser.add_argument("--uma-pbrs-scale", type=float, default=50.0)
    parser.add_argument("--uma-pbrs-weight", type=float, default=1.0)
    parser.add_argument("--co-max-coverage", type=float, default=1.0)
    parser.add_argument("--co-gas-ref-energy", type=float, default=0.0)
    parser.add_argument("--co-temperature-k", type=float, default=300.0)
    parser.add_argument("--co-ref-pressure-pa", type=float, default=101325.0)
    parser.add_argument("--co-partial-pressure-pa", type=float, default=None)
    parser.add_argument("--co-mu-ref-ev", type=float, default=-1.0)
    parser.add_argument("--use-effective-mu-co", action="store_true")
    parser.add_argument("--disable-co-langmuir-target", action="store_true")
    parser.add_argument("--co-repulsion-distance-a", type=float, default=3.0)
    parser.add_argument("--co-repulsion-strength-ev", type=float, default=0.15)
    parser.add_argument("--co-repulsion-sigma-a", type=float, default=2.0)
    parser.add_argument(
        "--e-cu-co",
        type=float,
        default=None,
        help="Analytical low-coverage CO adsorption energy on Cu-like site (eV).",
    )
    parser.add_argument(
        "--e-pd-co",
        type=float,
        default=None,
        help="Analytical low-coverage CO adsorption energy on Pd-like site (eV).",
    )
    parser.add_argument("--absolute-mu-co", action="store_true")
    parser.add_argument("--disable-co-adsorption", action="store_true")

    # PIRP knobs (week-2)
    parser.add_argument("--use-pirp", action="store_true")
    parser.add_argument("--pirp-scale", type=float, default=0.02)
    parser.add_argument("--noop-logit-bonus", type=float, default=0.0)
    parser.add_argument("--pirp-final-scale", type=float, default=0.005)
    parser.add_argument("--pirp-anneal-fraction", type=float, default=0.5)
    parser.add_argument("--pirp-anneal-schedule", choices=["cosine", "linear"], default="cosine")
    parser.add_argument("--ppo-n-steps", type=int, default=512)
    parser.add_argument("--ppo-batch-size", type=int, default=128)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--lr-schedule", choices=["constant", "linear"], default="constant")
    parser.add_argument("--eval-freq", type=int, default=0, help="Steps between eval (0=disable)")
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--early-stop-patience", type=int, default=0, help="Eval rounds without improvement before stop (0=disable)")

    # Uncertainty/oracle wrappers
    parser.add_argument("--uncertainty-penalty", type=float, default=0.0)
    parser.add_argument("--oracle-threshold", type=float, default=None)
    parser.add_argument("--oracle-fmax", type=float, default=0.05)
    parser.add_argument("--oracle-max-steps", type=int, default=100)
    parser.add_argument("--oracle-disable-amp", type=bool, default=True)

    # Surrogate params
    parser.add_argument("--surrogate-models", type=int, default=3)
    parser.add_argument("--surrogate-mean", type=float, default=-1.0)
    parser.add_argument("--surrogate-noise", type=float, default=0.1)

    # Oracle checkpoints and composition mode
    parser.add_argument("--oracle-mode", choices=["auto", "eq2", "uma", "hybrid"], default="auto")
    parser.add_argument("--ads-task", choices=["oc25", "oc20"], default="oc25")
    parser.add_argument("--ads-sm-ckpt", type=str, default="checkpoints/esen_sm_conserve.pt")
    parser.add_argument("--ads-md-ckpt", type=str, default="checkpoints/esen_md_direct.pt")
    parser.add_argument("--disable-ads-ensemble", action="store_true")
    parser.add_argument("--eq2-ckpt", type=str, default="checkpoints/eq2_83M_2M.pt")
    parser.add_argument("--uma-ckpt", type=str, default="checkpoints/uma-s-1p1.pt")
    parser.add_argument(
        "--require-ads-oracle",
        action="store_true",
        help="Fail fast if an adsorbate-capable oracle (EqV2 path) is unavailable.",
    )
    parser.add_argument(
        "--oracle-ckpt",
        type=str,
        default=None,
        help="Backward-compatible alias for --eq2-ckpt",
    )

    # RL switches
    parser.add_argument("--use-masking", action="store_true")
    parser.add_argument("--enable-vis", action="store_true")

    parser.add_argument("--load-dir", type=str, default=None)
    parser.add_argument("--save-dir", type=Path, default=Path("checkpoints"))
    return parser.parse_args()


def apply_experiment_profile(args) -> None:
    profile = str(getattr(args, "experiment_profile", "default")).lower()
    if profile != "route_a":
        if profile == "mutation_delta_stop":
            args.action_mode = "mutation"
            args.disable_noop_action = False
            args.stop_terminates = True
            args.min_stop_steps = max(int(getattr(args, "min_stop_steps", 0)), 8)
            args.reward_profile = "pure_delta_omega"
            args.disable_uma_pbrs = True
            args.constraint_update_mode = "frozen"
            args.constraint_weight = 0.0
            args.constraint_lambda_init = 0.0
            args.constraint_lambda_min = 0.0
            args.constraint_lambda_max = 0.0
            args.enable_deviation_mask = False
            if float(args.noop_logit_bonus) <= 0.0:
                args.noop_logit_bonus = 0.25
            return

        if profile == "mutation_delta_strict_stop":
            args.action_mode = "mutation"
            args.disable_noop_action = True
            args.stop_terminates = True
            args.min_stop_steps = max(int(getattr(args, "min_stop_steps", 0)), 8)
            args.reward_profile = "pure_delta_omega"
            args.disable_uma_pbrs = True
            args.constraint_update_mode = "frozen"
            args.constraint_weight = 0.0
            args.constraint_lambda_init = 0.0
            args.constraint_lambda_min = 0.0
            args.constraint_lambda_max = 0.0
            args.enable_deviation_mask = False
            args.noop_logit_bonus = 0.0
            return

        if profile != "swap_stop":
            return

        # Fixed-composition local-search baseline.
        args.action_mode = "swap"
        args.disable_noop_action = False
        args.stop_terminates = True
        args.min_stop_steps = max(int(getattr(args, "min_stop_steps", 0)), 8)
        args.reward_profile = "pure_delta_omega"
        args.disable_uma_pbrs = True
        args.constraint_update_mode = "frozen"
        args.constraint_weight = 0.0
        args.constraint_lambda_init = 0.0
        args.constraint_lambda_min = 0.0
        args.constraint_lambda_max = 0.0
        args.enable_deviation_mask = False
        if float(args.noop_logit_bonus) <= 0.0:
            args.noop_logit_bonus = 0.25
        return

    # Route A (SAGCM-like): hard boundary + dense debt guidance, avoid conflicting shaping.
    args.enable_deviation_mask = True
    args.disable_uma_pbrs = True

    # Keep PID controller present but neutralized.
    args.constraint_update_mode = "frozen"
    args.constraint_weight = 0.0
    args.constraint_lambda_init = 0.0
    args.constraint_lambda_min = 0.0
    args.constraint_lambda_max = 0.0

    # Reactivate debt-shaping as the primary composition guidance.
    if float(args.debt_improvement_scale) <= 0.0:
        args.debt_improvement_scale = 0.20
    if float(args.debt_abs_penalty) <= 0.0:
        args.debt_abs_penalty = 0.05

    # Keep no-op available and slightly easier to sample.
    args.disable_noop_action = False
    if float(args.noop_logit_bonus) <= 0.0:
        args.noop_logit_bonus = 0.5


def maybe_load_oracle(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if getattr(args, "oracle_ckpt", None):
        args.eq2_ckpt = args.oracle_ckpt
    base_dir = Path(__file__).resolve().parent
    ads_task = getattr(args, "ads_task", "oc25")
    disable_ads_ensemble = bool(getattr(args, "disable_ads_ensemble", False))
    oracle_fmax = float(getattr(args, "oracle_fmax", 0.05))
    oracle_max_steps = int(getattr(args, "oracle_max_steps", 100))

    def _resolve_ckpt(path_like):
        if not path_like:
            return None
        p = Path(path_like)
        if p.exists():
            return p
        p2 = base_dir / p
        if p2.exists():
            return p2
        return p

    ads_oracle = None
    uma_oracle = None

    eq2_path = _resolve_ckpt(getattr(args, "eq2_ckpt", None))
    uma_path = _resolve_ckpt(getattr(args, "uma_ckpt", None))
    ads_sm_path = _resolve_ckpt(getattr(args, "ads_sm_ckpt", None))
    ads_md_path = _resolve_ckpt(getattr(args, "ads_md_ckpt", None))

    oracle_mode = getattr(args, "oracle_mode", "auto")
    wants_eq2 = oracle_mode in {"auto", "eq2", "hybrid"}
    wants_uma = oracle_mode in {"auto", "uma", "hybrid"}

    if wants_eq2:
        if not disable_ads_ensemble:
            ads_ckpts = []
            if ads_sm_path is not None and ads_sm_path.exists():
                ads_ckpts.append(str(ads_sm_path))
            if ads_md_path is not None and ads_md_path.exists():
                ads_ckpts.append(str(ads_md_path))

            if ads_ckpts:
                try:
                    print(f"[Main] Loading OC25 ads ensemble ({len(ads_ckpts)} ckpt) ...")
                    ads_oracle = OC25EnsembleOracle(
                        checkpoint_paths=ads_ckpts,
                        task_name=ads_task,
                        device=device,
                    )
                except Exception as exc:
                    print(f"[Main] Failed to load OC25 ensemble oracle: {exc}")

        if ads_oracle is None and eq2_path is not None and eq2_path.exists():
            try:
                print(f"[Main] Loading legacy EqV2-style Oracle from {eq2_path} ...")
                ads_oracle = EquiformerV2Oracle(
                    str(eq2_path),
                    device=device,
                    fmax=oracle_fmax,
                    max_steps=oracle_max_steps,
                    task_name=ads_task,
                )
            except Exception as exc:
                print(f"[Main] Failed to load EqV2 oracle: {exc}")

    if wants_uma and uma_path is not None and uma_path.exists():
        try:
            print(f"[Main] Loading UMA Oracle from {uma_path} ...")
            uma_oracle = UMAOracle(str(uma_path), device=device)
        except Exception as exc:
            print(f"[Main] Failed to load UMA oracle: {exc}")

    if oracle_mode == "eq2":
        return ads_oracle
    if oracle_mode == "uma":
        return uma_oracle
    if oracle_mode == "hybrid":
        if uma_oracle is not None or ads_oracle is not None:
            if bool(getattr(args, "require_ads_oracle", False)) and ads_oracle is None:
                raise RuntimeError(
                    "Hybrid mode requested but EqV2/ads oracle is unavailable. "
                    "Install compatible OCP/OCPModels stack or disable --require-ads-oracle."
                )
            return HybridGrandPotentialOracle(slab_oracle=uma_oracle, ads_oracle=ads_oracle)
        return None

    # auto mode preference: hybrid > eq2 > uma
    if uma_oracle is not None and ads_oracle is not None:
        return HybridGrandPotentialOracle(slab_oracle=uma_oracle, ads_oracle=ads_oracle)
    if ads_oracle is not None:
        return ads_oracle
    if uma_oracle is not None:
        if bool(getattr(args, "require_ads_oracle", False)):
            raise RuntimeError(
                "Only UMA is available, but --require-ads-oracle is enabled. "
                "Please provide a working EqV2/ads oracle backend."
            )
        return uma_oracle

    print("[Main] No oracle available. Falling back to surrogate/EMT.")
    return None


def build_env_config(args) -> EnvConfig:
    kb_ev_per_k = 8.617333262145e-5
    use_effective_mu = bool(args.use_effective_mu_co or (args.co_partial_pressure_pa is not None))
    mu_co = float(args.mu_co)
    if use_effective_mu:
        t = max(1.0, float(args.co_temperature_k))
        p_ref = max(1e-30, float(args.co_ref_pressure_pa))
        p = float(args.co_partial_pressure_pa) if args.co_partial_pressure_pa is not None else p_ref
        p = max(1e-30, p)
        mu_co = float(args.co_mu_ref_ev + kb_ev_per_k * t * math.log(p / p_ref))

    default_cfg = EnvConfig()
    e_cu_co = float(args.e_cu_co) if args.e_cu_co is not None else float(default_cfg.e_cu_co)
    e_pd_co = float(args.e_pd_co) if args.e_pd_co is not None else float(default_cfg.e_pd_co)
    physics_prior = dict(default_cfg.physics_prior)
    physics_prior["e_cu_co"] = e_cu_co
    physics_prior["e_pd_co"] = e_pd_co

    uma_pbrs_gamma = float(args.gamma) if args.uma_pbrs_gamma is None else float(args.uma_pbrs_gamma)

    # Sub-configs: auto-fill matching fields, override special cases
    reward_cfg = _fill_dataclass(RewardConfig, args, overrides={
        "mu_co": mu_co,
        "mu_co_is_effective": use_effective_mu,
        "thermo_consistent_backend": not args.disable_thermo_consistent_backend,
    })
    constraint_cfg = _fill_dataclass(ConstraintConfig, args)
    co_cfg = _fill_dataclass(COAdsorptionConfig, args, overrides={
        "enable_co_adsorption": not args.disable_co_adsorption,
        "co_use_langmuir_target": not args.disable_co_langmuir_target,
        "use_relative_mu_co": not args.absolute_mu_co,
        "e_cu_co": e_cu_co,
        "e_pd_co": e_pd_co,
    })
    uma_cfg = _fill_dataclass(UMAPBRSConfig, args, overrides={
        "use_uma_pbrs": not args.disable_uma_pbrs,
        "uma_pbrs_gamma": uma_pbrs_gamma,
    })

    return EnvConfig(
        mode=args.obs_mode,
        init_seed=args.seed,
        n_active_layers=args.n_active_layers,
        bulk_pd_fraction=args.bulk_pd_fraction,
        action_mode=args.action_mode,
        enable_noop_action=not args.disable_noop_action,
        stop_terminates=args.stop_terminates,
        min_stop_steps=args.min_stop_steps,
        use_deviation_mask=args.enable_deviation_mask,
        reward=reward_cfg,
        constraint=constraint_cfg,
        co_adsorption=co_cfg,
        uma_pbrs=uma_cfg,
        physics_prior=physics_prior,
    )


def launch_train(args):
    oracle = maybe_load_oracle(args)
    env_config = build_env_config(args)
    mu_mode = "effective(T,p)" if env_config.mu_co_is_effective else "direct"
    print(
        f"[Main] profile={getattr(args, 'experiment_profile', 'default')}, "
        f"mu_CO={env_config.mu_co:.4f} eV ({mu_mode}), "
        f"T={env_config.co_temperature_k:.1f} K, "
        f"p_CO={env_config.co_partial_pressure_pa if env_config.co_partial_pressure_pa is not None else env_config.co_ref_pressure_pa} Pa, "
        f"Eads(Cu)={env_config.e_cu_co:.3f} eV, "
        f"Eads(Pd)={env_config.e_pd_co:.3f} eV, "
        f"thermo_consistent={env_config.thermo_consistent_backend}, "
        f"action_mode={env_config.action_mode}, "
        f"noop={env_config.enable_noop_action}, "
        f"stop_terminates={env_config.stop_terminates}, "
        f"min_stop_steps={env_config.min_stop_steps}, "
        f"deviation_mask={env_config.use_deviation_mask}, "
        f"reward_profile={env_config.reward_profile}, "
        f"uma_pbrs={env_config.use_uma_pbrs}, "
        f"gamma={args.gamma:.4f}, "
        f"uma_pbrs_gamma={env_config.uma_pbrs_gamma:.4f}, "
        f"constraint_update={env_config.constraint_update_mode}, "
        f"constraint_weight={env_config.constraint_weight:.3f}, "
        f"noop_bonus={args.noop_logit_bonus:.3f}"
    )

    surrogate = None
    if args.surrogate_models > 0:
        print(f"[Main] Initializing Surrogate Ensemble ({args.surrogate_models} models)")
        surrogate_cfg = SurrogateConfig(
            n_models=args.surrogate_models,
            mean_energy=args.surrogate_mean,
            noise_scale=args.surrogate_noise,
        )
        surrogate = SurrogateEnsemble(config=surrogate_cfg)
    else:
        print("[Main] Surrogate disabled.")

    # Safety check: warn when neither a real oracle nor a trained surrogate is available.
    if oracle is None and surrogate is not None:
        import warnings
        warnings.warn(
            "[Main] WARNING: No oracle loaded. The surrogate ensemble uses random "
            "noise models and does NOT depend on atomic structure. Energy evaluations "
            "will fall back to EMT, which is inaccurate for CO adsorption. Training "
            "results may be meaningless. Consider providing --eq2-ckpt or --uma-ckpt.",
            stacklevel=2,
        )
    elif oracle is None and surrogate is None:
        import warnings
        warnings.warn(
            "[Main] WARNING: Neither oracle nor surrogate is available. All energy "
            "evaluations will use EMT fallback. This is only suitable for smoke tests.",
            stacklevel=2,
        )

    train_config = TrainConfig(
        total_timesteps=args.total_steps,
        n_envs=args.n_envs,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        device=args.device,
        uncertainty_penalty=args.uncertainty_penalty,
        oracle_threshold=args.oracle_threshold,
        oracle_fmax=args.oracle_fmax,
        oracle_max_steps=args.oracle_max_steps,
        oracle_disable_amp=args.oracle_disable_amp,
        ppo_n_steps=args.ppo_n_steps,
        ppo_batch_size=args.ppo_batch_size,
        max_grad_norm=args.max_grad_norm,
        lr_schedule=args.lr_schedule,
        eval_freq=args.eval_freq,
        eval_episodes=args.eval_episodes,
        early_stop_patience=args.early_stop_patience,
        use_pirp=args.use_pirp,
        pirp_scale=args.pirp_scale,
        noop_logit_bonus=args.noop_logit_bonus,
        pirp_final_scale=args.pirp_final_scale,
        pirp_anneal_fraction=args.pirp_anneal_fraction,
        pirp_anneal_schedule=args.pirp_anneal_schedule,
        enable_visualization=args.enable_vis,
    )

    train_agent(
        env_config=env_config,
        train_config=train_config,
        surrogate=surrogate,
        oracle_energy_fn=oracle.compute_energy if hasattr(oracle, "compute_energy") else None,
        oracle=oracle,
        save_dir=args.save_dir,
        use_masking=args.use_masking,
    )


def launch_baselines(args):
    env_config = build_env_config(args)
    if str(getattr(env_config, "action_mode", "mutation")).lower() != "mutation":
        raise ValueError("Baseline search helpers currently support action_mode='mutation' only.")
    env = ChemGymEnv(env_config)

    print("Running random search...")
    rand_result = random_search(env, total_steps=args.total_steps)
    print(f"Random best omega: {rand_result['best_energy']}")

    print("Running simulated annealing...")
    sa_result = simulated_annealing(env, total_steps=args.total_steps)
    print(f"SA best omega: {sa_result['best_energy']}")


def launch_eval(args):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.maskable.utils import get_action_masks

    oracle = maybe_load_oracle(args)
    env_config = build_env_config(args)

    def _make_single():
        return ChemGymEnv(env_config, oracle=oracle)

    base_venv = DummyVecEnv([_make_single])

    if args.load_dir:
        run_dir = Path(args.load_dir)
        stats_path = run_dir / "vec_normalize.pkl"
        model_path = run_dir / "model"
    else:
        stats_path = args.save_dir / "latest_vec_normalize.pkl"
        model_path = args.save_dir / "latest_model"

    if stats_path.exists():
        venv = VecNormalize.load(str(stats_path), base_venv)
        venv.training = False
        venv.norm_reward = False
    else:
        venv = base_venv

    if not model_path.with_suffix(".zip").exists():
        raise FileNotFoundError(f"Model not found: {model_path.with_suffix('.zip')}")

    try:
        model = MaskablePPO.load(model_path, env=venv)
        is_maskable = True
        print("[Eval] Loaded MaskablePPO model.")
    except Exception:
        model = PPO.load(model_path, env=venv)
        is_maskable = False
        print("[Eval] Loaded standard PPO model.")

    obs = venv.reset()
    best_omega = float("inf")

    for step in range(1000):
        if is_maskable:
            action_masks = get_action_masks(venv)
            action, _ = model.predict(obs, action_masks=action_masks, deterministic=False)
        else:
            action, _ = model.predict(obs, deterministic=False)

        obs, rewards, dones, infos = venv.step(action)
        current_omega = float(infos[0]["omega"])

        if step % 50 == 0 or current_omega < best_omega:
            print(f"Step {step + 1:04d} | omega: {current_omega:.6f}")

        if current_omega < best_omega:
            best_omega = current_omega
            best_atoms = infos[0].get("atoms", None)
            if best_atoms is not None:
                best_atoms.write("best_optimized.xyz")

    print(f"[Eval] Final best omega: {best_omega:.6f}")


if __name__ == "__main__":
    cli_args = parse_args()
    apply_experiment_profile(cli_args)
    if cli_args.mode == "train":
        launch_train(cli_args)
    elif cli_args.mode == "baseline":
        launch_baselines(cli_args)
    else:
        launch_eval(cli_args)
