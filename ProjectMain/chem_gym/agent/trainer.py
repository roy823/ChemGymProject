from __future__ import annotations

import datetime
import math
import os
from typing import Callable, Optional

import gymnasium as gym
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback, EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from chem_gym.agent.graph_feature_extractor import CrystalGraphFeatureExtractor
from chem_gym.agent.pirp_policy import PIRPMaskableActorCriticPolicy
from chem_gym.analysis.vis_callback import VisualizationCallback
from chem_gym.config import EnvConfig, TrainConfig
from chem_gym.envs.chem_env import ChemGymEnv
from chem_gym.surrogate.ensemble import SurrogateEnsemble


class UncertaintyPenaltyWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, coefficient: float):
        super().__init__(env)
        self.coefficient = coefficient

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self.coefficient and "uncertainty" in info:
            reward -= self.coefficient * float(info.get("uncertainty", 0.0))
        return obs, reward, terminated, truncated, info


class OracleWrapper(gym.Wrapper):
    def __init__(
        self,
        env: ChemGymEnv,
        surrogate: SurrogateEnsemble,
        threshold: float,
        oracle_energy_fn: Callable[[Optional[object]], float],
    ):
        super().__init__(env)
        self.surrogate = surrogate
        self.threshold = threshold
        self.oracle_energy_fn = oracle_energy_fn

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        current_uncertainty = float(info.get("uncertainty", 0.0))

        if current_uncertainty > self.threshold:
            oracle_energy = self.oracle_energy_fn(info.get("atoms"))
            self.surrogate.update_with_oracle(info.get("atoms"), oracle_energy)
            info["oracle_energy"] = oracle_energy

        return obs, reward, terminated, truncated, info


class EnergyLoggerCallback(BaseCallback):
    def __init__(self, verbose=0, log_freq=10):
        super().__init__(verbose)
        self.log_freq = log_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq == 0:
            infos = self.locals.get("infos", [])
            if infos:
                omega = infos[0].get("omega", float("nan"))
                reward_norm = self.locals.get("rewards")[0]
                reward_raw = infos[0].get("reward_terms", {}).get("reward_total", float("nan"))
                n_co = infos[0].get("n_co", "N/A")
                debt = infos[0].get("stoich_debt", "N/A")
                lam = infos[0].get("constraint_lambda", float("nan"))
                viol = infos[0].get("constraint_violation", float("nan"))
                noop_ratio = infos[0].get("noop_ratio", float("nan"))
                b_slab = infos[0].get("energy_backend_slab", "unknown")
                b_ads = infos[0].get("energy_backend_ads", "unknown")
                print(
                    f"[Step {self.n_calls}] omega={omega:.4f} | reward_raw={reward_raw:.4f} | "
                    f"reward_norm={reward_norm:.4f} | n_co={n_co} | debt={debt} | "
                    f"lam={lam:.3f} | viol={viol:.3f} | noop={noop_ratio:.2f} | "
                    f"backend(slab={b_slab},ads={b_ads})"
                )
        return True


class PIRPScaleAnnealCallback(BaseCallback):
    def __init__(
        self,
        initial_scale: float,
        final_scale: float,
        anneal_steps: int,
        schedule: str = "cosine",
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.initial_scale = float(initial_scale)
        self.final_scale = float(final_scale)
        self.anneal_steps = int(max(1, anneal_steps))
        self.schedule = str(schedule).lower()

    def _on_step(self) -> bool:
        if not hasattr(self.model, "policy") or not hasattr(self.model.policy, "pirp_scale"):
            return True

        t = min(1.0, float(self.num_timesteps) / float(self.anneal_steps))
        if self.schedule == "linear":
            weight = 1.0 - t
        else:
            weight = 0.5 * (1.0 + math.cos(math.pi * t))

        current = self.final_scale + (self.initial_scale - self.final_scale) * weight
        self.model.policy.pirp_scale = float(current)
        self.logger.record("train/pirp_scale", float(current))
        return True


class ConstraintLambdaUpdateCallback(BaseCallback):
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._violations = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            v = info.get("constraint_violation", None)
            if v is not None:
                self._violations.append(float(v))
        return True

    def _on_rollout_end(self) -> None:
        if not self._violations:
            return
        mean_v = float(sum(self._violations) / max(1, len(self._violations)))
        self._violations.clear()
        try:
            self.training_env.env_method("update_constraint_lambda", mean_v)
            lambdas = self.training_env.env_method("get_constraint_lambda")
            if lambdas:
                self.logger.record("train/constraint_lambda", float(sum(lambdas) / len(lambdas)))
            self.logger.record("train/constraint_violation_rollout", mean_v)
        except Exception:
            # Keep training robust when wrappers do not expose methods.
            pass


class SelectiveVecNormalize(VecNormalize):
    """Normalize only node features for dict observations."""

    def _normalize_obs(self, obs, var_type):
        if isinstance(obs, dict):
            for key in obs.keys():
                if key == "node_features":
                    obs[key] = super()._normalize_obs(obs[key], var_type)
            return obs
        return super()._normalize_obs(obs, var_type)


def make_vec_env(
    env_config: EnvConfig,
    surrogate: Optional[SurrogateEnsemble],
    train_config: TrainConfig,
    oracle_energy_fn: Optional[Callable] = None,
    oracle=None,
    use_masking: bool = False,
):
    def _make_single():
        env = ChemGymEnv(env_config, surrogate=surrogate, oracle=oracle)

        if surrogate is not None:
            if train_config.uncertainty_penalty > 0:
                env = UncertaintyPenaltyWrapper(env, coefficient=train_config.uncertainty_penalty)

            if train_config.oracle_threshold is not None and oracle_energy_fn:
                env = OracleWrapper(env, surrogate, train_config.oracle_threshold, oracle_energy_fn)

        if use_masking:
            env = ActionMasker(env, lambda wrapped_env: wrapped_env.action_masks())

        return env

    env = DummyVecEnv([_make_single for _ in range(train_config.n_envs)])
    env = SelectiveVecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)
    return env


def train_agent(
    env_config: EnvConfig,
    surrogate: Optional[SurrogateEnsemble],
    train_config: TrainConfig,
    oracle_energy_fn: Optional[Callable] = None,
    oracle=None,
    save_dir: str = ".",
    use_masking: bool = False,
):
    vec_env = make_vec_env(
        env_config,
        surrogate,
        train_config,
        oracle_energy_fn,
        oracle=oracle,
        use_masking=use_masking,
    )

    policy_kwargs = {}

    if env_config.mode == "graph":
        policy_kwargs.update(
            dict(
                features_extractor_class=CrystalGraphFeatureExtractor,
                features_extractor_kwargs=dict(features_dim=256, hidden_dim=128, n_layers=3),
            )
        )

    if use_masking:
        model_class = MaskablePPO
        tb_log_name = "MaskablePPO_Experiment"

        can_use_pirp = (
            env_config.mode == "graph"
            and train_config.use_pirp
            and str(getattr(env_config, "action_mode", "mutation")).lower() == "mutation"
        )
        if can_use_pirp:
            policy = PIRPMaskableActorCriticPolicy
            policy_kwargs.update(
                dict(
                    pirp_scale=train_config.pirp_scale,
                    noop_logit_bonus=train_config.noop_logit_bonus,
                    has_noop_action=bool(getattr(env_config, "enable_noop_action", True)),
                    has_stop_action=bool(getattr(env_config, "stop_terminates", False)),
                    pirp_mu_co=env_config.mu_co,
                    pirp_n_elements=len(env_config.element_types),
                    prior_constants=env_config.physics_prior,
                )
            )
            print("[Trainer] Initializing MaskablePPO + PIRP policy")
        else:
            policy = "MultiInputPolicy" if env_config.mode == "graph" else "MlpPolicy"
            if train_config.use_pirp and env_config.mode != "graph":
                print("[Trainer] PIRP requested but obs-mode is not graph. Falling back to base policy.")
            elif train_config.use_pirp and str(getattr(env_config, "action_mode", "mutation")).lower() != "mutation":
                print("[Trainer] PIRP currently supports mutation action_mode only. Falling back to base policy.")
            print("[Trainer] Initializing MaskablePPO (without PIRP)")
    else:
        model_class = PPO
        tb_log_name = "StandardPPO_Baseline"
        policy = "MultiInputPolicy" if env_config.mode == "graph" else "MlpPolicy"
        if train_config.use_pirp:
            print("[Trainer] PIRP requires use_masking=True in this implementation. Falling back to PPO base policy.")
        print("[Trainer] Initializing Standard PPO")

    # Learning rate schedule
    lr = train_config.learning_rate
    if str(getattr(train_config, "lr_schedule", "constant")).lower() == "linear":
        lr = lambda progress: float(progress) * train_config.learning_rate
        print(f"[Trainer] Using linear LR schedule: {train_config.learning_rate} -> 0")

    model = model_class(
        policy,
        vec_env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        learning_rate=lr,
        gamma=train_config.gamma,
        gae_lambda=train_config.lam,
        device=train_config.device,
        n_steps=train_config.ppo_n_steps,
        batch_size=train_config.ppo_batch_size,
        clip_range=train_config.ppo_clip_range,
        ent_coef=train_config.ppo_ent_coef,
        max_grad_norm=float(getattr(train_config, "max_grad_norm", 0.5)),
        tensorboard_log="./chem_gym_tensorboard/",
    )

    timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
    run_type = "maskable" if use_masking else "standard"
    if train_config.use_pirp and use_masking:
        run_type = f"{run_type}_pirp"

    steps_k = train_config.total_timesteps // 1000
    run_id = f"{run_type}_{steps_k}k_{timestamp}"

    run_save_dir = os.path.join(save_dir, run_id)
    os.makedirs(run_save_dir, exist_ok=True)

    energy_callback = EnergyLoggerCallback(log_freq=5)
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=run_save_dir,
        name_prefix="rl_model",
    )
    callback_list = [energy_callback, checkpoint_callback]
    if str(getattr(env_config, "constraint_update_mode", "rollout")).lower() == "rollout":
        callback_list.append(ConstraintLambdaUpdateCallback())
    if train_config.use_pirp and use_masking and train_config.pirp_anneal_fraction > 0:
        anneal_steps = int(max(1, train_config.total_timesteps * float(train_config.pirp_anneal_fraction)))
        pirp_anneal = PIRPScaleAnnealCallback(
            initial_scale=float(train_config.pirp_scale),
            final_scale=float(train_config.pirp_final_scale),
            anneal_steps=anneal_steps,
            schedule=str(train_config.pirp_anneal_schedule),
        )
        callback_list.append(pirp_anneal)
    if train_config.enable_visualization:
        vis_callback = VisualizationCallback(save_freq=200, save_dir=os.path.join(run_save_dir, "vis"))
        callback_list.insert(0, vis_callback)

    # Evaluation callback with optional early stopping
    eval_freq = int(getattr(train_config, "eval_freq", 0))
    if eval_freq > 0:
        eval_env = make_vec_env(
            env_config, surrogate=None, train_config=train_config,
            oracle_energy_fn=None, oracle=oracle, use_masking=use_masking,
        )
        eval_kwargs = dict(
            eval_env=eval_env,
            n_eval_episodes=int(getattr(train_config, "eval_episodes", 5)),
            eval_freq=max(1, eval_freq // train_config.n_envs),
            best_model_save_path=os.path.join(run_save_dir, "best_model"),
            log_path=os.path.join(run_save_dir, "eval_logs"),
            deterministic=False,
        )
        patience = int(getattr(train_config, "early_stop_patience", 0))
        if patience > 0:
            stop_cb = StopTrainingOnNoModelImprovement(
                max_no_improvement_evals=patience, verbose=1,
            )
            eval_kwargs["callback_after_eval"] = stop_cb
            print(f"[Trainer] Early stopping enabled: patience={patience} evals")
        eval_callback = EvalCallback(**eval_kwargs)
        callback_list.append(eval_callback)
        print(f"[Trainer] Eval callback: every {eval_freq} steps, {eval_kwargs['n_eval_episodes']} episodes")

    callbacks = CallbackList(callback_list)

    print(f"[Trainer] Starting training run: {run_id}")

    model.learn(
        total_timesteps=train_config.total_timesteps,
        callback=callbacks,
        tb_log_name=tb_log_name,
    )

    model_path = os.path.join(run_save_dir, "model")
    stats_path = os.path.join(run_save_dir, "vec_normalize.pkl")

    model.save(model_path)
    vec_env.save(stats_path)

    model.save(os.path.join(save_dir, "latest_model"))
    vec_env.save(os.path.join(save_dir, "latest_vec_normalize.pkl"))

    print(f"[Trainer] Saved model to {run_save_dir}")
    return model
