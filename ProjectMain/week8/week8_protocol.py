"""Week-8 protocol primitives — implements v1.2 §8 step 1.

Four method-agnostic utilities that the rest of the week-8 infrastructure
(MC baselines, dynamic env wrapper, significance tests, resume support,
aggregator) is built on:

  1. Frozen test schedules A-E (per v1.2 §1 D6 / §3) and the random
     staircase generator used by Exp-2 training.
  2. Distinct-oracle-call accounting: cache-aware counter that survives
     ``env.reset()`` so a budget can be enforced across episodes.
  3. Static protocol driver (``run_static_protocol``): stop+reset
     symmetry layer with a per-episode step cap. Drives PPO_swap with
     ``stop`` action, canonical_MC / RE-MC / random_swap with forced reset
     at the episode boundary. Bounded by total distinct oracle calls.
  4. Dynamic protocol driver (``run_dynamic_protocol``): fixed-horizon
     wrapper with per-segment ``mu_CO`` schedule lookup. No stop, no
     reset across the horizon (per v1.2 §5.2).

Smoke test (``python week8_protocol.py``) exercises all four primitives
on the 32-step EMT-fallback path with no oracle and no surrogate.

The protocol layer is intentionally policy-agnostic: a policy is any
callable ``(env, obs, info) -> int`` returning a valid action index.
PPO models, MC samplers, and random baselines plug in identically.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

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


# =============================================================================
# Frozen test schedules (v1.2 §1 D6 / §3)
# =============================================================================

MU_CO_VALUES: List[float] = [-0.2, -0.4, -0.6, -0.8]
SEGMENT_LENGTH: int = 50
HORIZON_STEPS: int = 200  # = 4 segments x 50 steps

TEST_SCHEDULES_V1_2: Dict[str, List[float]] = {
    "A_inc":   [-0.2, -0.4, -0.6, -0.8],  # monotone increasing in |mu|
    "B_dec":   [-0.8, -0.6, -0.4, -0.2],  # monotone decreasing in |mu|
    "C_zig1":  [-0.4, -0.2, -0.8, -0.6],  # mid-low-high-mid
    "D_zig2":  [-0.6, -0.8, -0.2, -0.4],  # mid-high-low-mid
    "E_swing": [-0.2, -0.8, -0.4, -0.6],  # large-amplitude alternation
}


def mu_co_at_step(schedule: Any, t: int) -> float:
    """Return the chemical potential active at integer step ``t``.

    ``schedule`` may be either a schedule name (``"A_inc"`` etc.) or an
    explicit 4-element sequence. ``t`` is clamped into ``[0, HORIZON_STEPS-1]``
    for safety.
    """
    if isinstance(schedule, str):
        if schedule not in TEST_SCHEDULES_V1_2:
            raise KeyError(f"unknown schedule '{schedule}'; expected one of {list(TEST_SCHEDULES_V1_2)}")
        seq = TEST_SCHEDULES_V1_2[schedule]
    else:
        seq = list(schedule)
    if not seq:
        raise ValueError("schedule must have at least one segment")
    t_clamped = max(0, min(int(t), HORIZON_STEPS - 1))
    seg_idx = min(len(seq) - 1, t_clamped // SEGMENT_LENGTH)
    return float(seq[seg_idx])


def random_staircase(rng: np.random.Generator) -> List[float]:
    """Random permutation of the four mu_CO segment values, used for Exp-2 training.

    Each PPO training episode draws a fresh staircase by calling this with
    ``rng = np.random.default_rng(episode_seed)`` so the schedule is
    reproducible from the seed alone.
    """
    perm = np.asarray(rng.permutation(len(MU_CO_VALUES)))
    return [float(MU_CO_VALUES[int(i)]) for i in perm]


# =============================================================================
# Distinct-oracle-call accounting
# =============================================================================

_COUNTER_ATTR = "_w8_distinct_state_hashes"


def install_oracle_counter(env: ChemGymEnv) -> ChemGymEnv:
    """Attach a cumulative distinct-state counter to ``env``.

    The env clears its own ``energy_cache`` on every ``reset()``, so the
    naive cache size is not a faithful budget metric across episodes. We
    wrap ``env._evaluate_energy`` to also record the state hash into a
    set living on the env that survives resets. The counter is the size
    of this set.

    Idempotent: a second call on the same env is a no-op.
    """
    if getattr(env, _COUNTER_ATTR, None) is not None:
        return env

    setattr(env, _COUNTER_ATTR, set())
    original = env._evaluate_energy

    def counted(atoms, role):  # type: ignore[no-untyped-def]
        result = original(atoms, role)
        if atoms is not None:
            try:
                state_hash = env._hash_atoms(atoms)
                getattr(env, _COUNTER_ATTR).add(state_hash)
            except Exception:
                # Hashing should never fail in normal use, but a counter
                # failure must not corrupt the rollout.
                pass
        return result

    env._evaluate_energy = counted  # type: ignore[assignment]
    return env


def distinct_oracle_calls(env: ChemGymEnv) -> int:
    """Number of distinct slab states ever evaluated by the env's oracle.

    Requires :func:`install_oracle_counter` to have been called once.
    Each unique state contributes 1 to the count regardless of how many
    roles (slab + ads_system + ads_reference) were evaluated for it.
    """
    counter = getattr(env, _COUNTER_ATTR, None)
    if counter is None:
        raise RuntimeError(
            "distinct_oracle_calls(env) requires install_oracle_counter(env) "
            "to be called once after env construction."
        )
    return int(len(counter))


# =============================================================================
# mu_CO update helper
# =============================================================================

def set_env_mu_co(env: ChemGymEnv, mu_co: float) -> None:
    """Update the env's chemical potential mid-rollout.

    The env reads ``self.config.mu_co`` on every step (inside
    ``_decorate_with_co`` and ``_compute_omega``), so mutating the config
    field is sufficient to apply the new ``mu_CO`` to subsequent steps.
    Also updates the nested ``RewardConfig`` so a snapshot of the config
    remains internally consistent.
    """
    env.config.mu_co = float(mu_co)
    if getattr(env.config, "reward", None) is not None:
        env.config.reward.mu_co = float(mu_co)


# =============================================================================
# Protocol drivers
# =============================================================================

PolicyFn = Callable[[ChemGymEnv, Any, Dict], int]


def run_static_protocol(
    env: ChemGymEnv,
    policy: PolicyFn,
    *,
    total_oracle_budget: int,
    max_steps_per_episode: int = 32,
    record_trace: bool = True,
) -> Dict[str, Any]:
    """Stop+reset symmetry layer for the static protocol (v1.2 §4 / §5.1).

    Drives the env in episodic mode: each episode runs at most
    ``max_steps_per_episode`` steps. Episodes end when (a) the policy
    emits the explicit ``stop`` action and the env terminates, (b) the
    env truncates because ``env.steps >= env.config.max_steps``, or
    (c) the driver hits ``max_steps_per_episode`` independently. After
    any of these the driver calls ``env.reset()`` and starts a new
    episode. Total work is bounded by ``total_oracle_budget`` distinct
    state evaluations (cache-aware via :func:`distinct_oracle_calls`).

    The ``policy`` is method-agnostic: PPO models, MC samplers, random
    baselines, and SGCMC all conform to the same ``(env, obs, info) -> int``
    interface. Stop semantics (``min_stop_steps``, ``stop_terminates``)
    are configured on the env, not the driver.
    """
    install_oracle_counter(env)

    obs, info = env.reset()
    best_omega = float("inf")
    best_theta_pd = float("nan")
    best_n_co: int = -1
    omega_trace: List[float] = []
    steps_log: List[Dict[str, Any]] = []

    n_episodes = 0
    n_steps_total = 0
    n_stops_explicit = 0
    n_resets = 1  # initial reset
    episode_step = 0

    while distinct_oracle_calls(env) < int(total_oracle_budget):
        action = int(policy(env, obs, info))
        obs, _, terminated, truncated, info = env.step(action)
        n_steps_total += 1
        episode_step += 1

        omega = info.get("omega", float("nan"))
        omega = float(omega) if omega is not None else float("nan")
        if np.isfinite(omega):
            omega_trace.append(omega)
            if omega < best_omega:
                best_omega = omega
                best_theta_pd = float(info.get("pd_surface_coverage", float("nan")))
                best_n_co = int(info.get("n_co", -1))

        if record_trace:
            steps_log.append({
                "global_step": n_steps_total,
                "episode": n_episodes,
                "episode_step": episode_step,
                "action_type": str(info.get("action_type", "?")),
                "omega": omega,
                "best_omega_so_far": best_omega,
                "n_distinct_oracle_calls": distinct_oracle_calls(env),
            })

        if str(info.get("action_type", "")) == "stop":
            n_stops_explicit += 1

        end_episode = bool(terminated or truncated or episode_step >= int(max_steps_per_episode))
        if end_episode:
            n_episodes += 1
            if distinct_oracle_calls(env) >= int(total_oracle_budget):
                break
            obs, info = env.reset()
            episode_step = 0
            n_resets += 1

    return {
        "best_omega": float(best_omega) if np.isfinite(best_omega) else float("nan"),
        "best_theta_pd": float(best_theta_pd),
        "best_n_co": int(best_n_co),
        "omega_trace": np.asarray(omega_trace, dtype=float),
        "n_episodes": int(n_episodes),
        "n_steps_total": int(n_steps_total),
        "n_stops_explicit": int(n_stops_explicit),
        "n_resets": int(n_resets),
        "n_distinct_oracle_calls": int(distinct_oracle_calls(env)),
        "steps_log": steps_log if record_trace else None,
    }


def run_dynamic_protocol(
    env: ChemGymEnv,
    policy: PolicyFn,
    *,
    schedule: Sequence[float],
    horizon_steps: int = HORIZON_STEPS,
    segment_length: int = SEGMENT_LENGTH,
    record_trace: bool = True,
) -> Dict[str, Any]:
    """Fixed-horizon dynamic protocol driver (v1.2 §4 / §5.2).

    Updates ``env.config.mu_co`` to ``schedule[t // segment_length]``
    BEFORE each step. No stop, no reset across the horizon (the env's
    ``stop_terminates`` should be ``False`` and ``max_steps`` should be
    ``>= horizon_steps`` to avoid mid-horizon truncation).

    The product ``len(schedule) * segment_length`` must equal
    ``horizon_steps``. For the locked v1.2 evaluation this is
    ``4 * 50 = 200`` for schedules A-E. The smoke test uses a shorter
    ``4 * 8 = 32`` to fit the 32-step EMT-fallback path.

    Returns the per-step trace, the omega trace, the mu_CO trace, and
    the cumulative_omega / mean_omega summaries used for v1.2 §2.3
    paired-Wilcoxon dynamic-superiority tests.
    """
    if int(segment_length) * len(schedule) != int(horizon_steps):
        raise ValueError(
            f"segment_length * len(schedule) = {int(segment_length) * len(schedule)} "
            f"!= horizon_steps = {int(horizon_steps)}"
        )
    if int(env.config.max_steps) < int(horizon_steps):
        raise ValueError(
            f"env.config.max_steps = {env.config.max_steps} < horizon_steps = {horizon_steps}; "
            "the env would truncate inside the dynamic horizon."
        )

    install_oracle_counter(env)

    set_env_mu_co(env, float(schedule[0]))
    obs, info = env.reset()

    best_omega = float("inf")
    omega_trace: List[float] = []
    mu_trace: List[float] = []
    steps_log: List[Dict[str, Any]] = []

    for t in range(int(horizon_steps)):
        seg_idx = int(t) // int(segment_length)
        mu_t = float(schedule[seg_idx])
        set_env_mu_co(env, mu_t)
        mu_trace.append(mu_t)

        action = int(policy(env, obs, info))
        obs, _, terminated, truncated, info = env.step(action)

        omega = info.get("omega", float("nan"))
        omega = float(omega) if omega is not None else float("nan")
        if np.isfinite(omega):
            omega_trace.append(omega)
            if omega < best_omega:
                best_omega = omega

        if record_trace:
            steps_log.append({
                "step": int(t),
                "segment": int(seg_idx),
                "mu_co": float(mu_t),
                "action_type": str(info.get("action_type", "?")),
                "omega": omega,
                "best_omega_so_far": best_omega,
                "n_distinct_oracle_calls": distinct_oracle_calls(env),
            })

        if terminated:
            # v1.2 specifies no resets in the dynamic horizon. NaN-driven
            # env termination is rare under EMT/UMA but defensive: re-init
            # internal state, keep the same mu schedule, continue running
            # until horizon_steps total iterations are done.
            obs, info = env.reset()

    n_omega_finite = len(omega_trace)
    return {
        "best_omega": float(best_omega) if np.isfinite(best_omega) else float("nan"),
        "cumulative_omega": float(np.nansum(omega_trace)) if n_omega_finite else float("nan"),
        "mean_omega": float(np.nanmean(omega_trace)) if n_omega_finite else float("nan"),
        "omega_trace": np.asarray(omega_trace, dtype=float),
        "mu_trace": np.asarray(mu_trace, dtype=float),
        "n_steps_total": int(horizon_steps),
        "n_distinct_oracle_calls": int(distinct_oracle_calls(env)),
        "steps_log": steps_log if record_trace else None,
    }


# =============================================================================
# Smoke test
# =============================================================================

def _random_valid_action(env: ChemGymEnv, rng: np.random.Generator) -> int:
    masks = env.action_masks().astype(bool)
    valid = np.flatnonzero(masks)
    if valid.size == 0:
        return int(rng.integers(0, env.action_spec.action_dim))
    return int(rng.choice(valid))


def _build_smoke_env_config(
    *,
    mu_co: float,
    seed: int,
    max_steps: int,
    action_mode: str = "swap",
    enable_noop: bool = False,
    stop_terminates: bool = False,
    min_stop_steps: int = 0,
) -> EnvConfig:
    """EMT-fallback config: no oracle, no surrogate, minimal constraint shaping."""
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
    return EnvConfig(
        mode="graph",
        init_seed=int(seed),
        max_steps=int(max_steps),
        bulk_pd_fraction=0.08,
        n_active_layers=4,
        action_mode=str(action_mode),
        enable_noop_action=bool(enable_noop),
        stop_terminates=bool(stop_terminates),
        min_stop_steps=int(min_stop_steps),
        use_deviation_mask=False,
        reward=reward_cfg,
        constraint=constraint_cfg,
        co_adsorption=co_cfg,
        uma_pbrs=uma_cfg,
    )


def _smoke_test() -> int:
    """Run all four primitives on the 32-step EMT-fallback path. Returns 0 on success."""
    print("=" * 64)
    print("week8_protocol.py — smoke test under EMT-fallback (v1.2 §8 step 1)")
    print("=" * 64)

    # ----- Test 1: schedule lookup -------------------------------------
    print("\n[1/5] Frozen schedule lookup (mu_co_at_step)...")
    assert mu_co_at_step("A_inc", 0) == -0.2
    assert mu_co_at_step("A_inc", 49) == -0.2
    assert mu_co_at_step("A_inc", 50) == -0.4
    assert mu_co_at_step("A_inc", 199) == -0.8
    assert mu_co_at_step("E_swing", 0) == -0.2
    assert mu_co_at_step("E_swing", 50) == -0.8
    assert mu_co_at_step("E_swing", 100) == -0.4
    assert mu_co_at_step("E_swing", 150) == -0.6
    expected = {
        "A_inc":   (-0.2, -0.8),
        "B_dec":   (-0.8, -0.2),
        "C_zig1":  (-0.4, -0.6),
        "D_zig2":  (-0.6, -0.4),
        "E_swing": (-0.2, -0.6),
    }
    for name, sched in TEST_SCHEDULES_V1_2.items():
        assert len(sched) == 4, f"{name} must have 4 segments, got {len(sched)}"
        assert sched[0] == expected[name][0], f"{name} start: expected {expected[name][0]}, got {sched[0]}"
        assert sched[-1] == expected[name][1], f"{name} end: expected {expected[name][1]}, got {sched[-1]}"
    print("    OK: 5 schedules frozen and indexed correctly")

    # ----- Test 2: random_staircase reproducibility --------------------
    print("\n[2/5] random_staircase reproducibility...")
    s1 = random_staircase(np.random.default_rng(7001))
    s2 = random_staircase(np.random.default_rng(7001))
    s3 = random_staircase(np.random.default_rng(7002))
    assert s1 == s2, f"same seed must produce identical staircase: {s1} vs {s2}"
    assert sorted(s1) == sorted(MU_CO_VALUES), f"staircase must permute MU_CO_VALUES, got {s1}"
    assert sorted(s3) == sorted(MU_CO_VALUES)
    print(f"    OK: seed=7001 -> {s1}; seed=7002 -> {s3}")

    # ----- Test 3: distinct-oracle-call counter ------------------------
    print("\n[3/5] install_oracle_counter / distinct_oracle_calls...")
    cfg = _build_smoke_env_config(mu_co=-0.2, seed=11, max_steps=32, action_mode="swap")
    env = ChemGymEnv(cfg)
    install_oracle_counter(env)
    install_oracle_counter(env)  # idempotency check
    assert distinct_oracle_calls(env) == 0, "counter must start at 0"
    obs, info = env.reset()
    n0 = distinct_oracle_calls(env)
    assert n0 >= 1, f"reset must evaluate the initial state, got {n0}"
    rng = np.random.default_rng(0)
    for _ in range(8):
        action = _random_valid_action(env, rng)
        obs, _, term, trunc, info = env.step(action)
        if term or trunc:
            break
    n1 = distinct_oracle_calls(env)
    assert n1 >= n0, "counter must be monotone non-decreasing across steps"
    cache_size_before_reset = len(env.energy_cache)
    obs, info = env.reset()
    cache_size_after_reset = len(env.energy_cache)
    n2 = distinct_oracle_calls(env)
    assert cache_size_after_reset <= cache_size_before_reset + 2, "reset clears cache before re-evaluating initial state"
    assert n2 >= n1, f"counter must survive reset(): had {n1}, now {n2}"
    print(f"    OK: counter survives reset (n0={n0}, n1={n1}, n2={n2}); idempotent")

    # ----- Test 4: static protocol driver ------------------------------
    print("\n[4/5] run_static_protocol (budget=32, max_steps_per_episode=8)...")
    cfg2 = _build_smoke_env_config(
        mu_co=-0.2,
        seed=11,
        max_steps=32,
        action_mode="swap",
        enable_noop=False,
        stop_terminates=False,
        min_stop_steps=0,
    )
    env2 = ChemGymEnv(cfg2)
    rng2 = np.random.default_rng(42)

    def policy_random_swap(_env: ChemGymEnv, _obs: Any, _info: Dict) -> int:
        return _random_valid_action(_env, rng2)

    result = run_static_protocol(
        env2,
        policy_random_swap,
        total_oracle_budget=32,
        max_steps_per_episode=8,
        record_trace=True,
    )
    assert result["n_distinct_oracle_calls"] >= 1, "no oracle calls were made"
    assert result["n_distinct_oracle_calls"] <= 64, (
        f"distinct calls {result['n_distinct_oracle_calls']} far exceeds budget 32 "
        "(some overshoot is expected because the budget gate triggers AFTER a step)"
    )
    assert result["n_steps_total"] >= 1
    assert result["n_episodes"] >= 1
    assert result["n_resets"] >= 1
    assert np.isfinite(result["best_omega"]), "best_omega must be finite under EMT"
    if result["steps_log"]:
        oracle_calls_log = [s["n_distinct_oracle_calls"] for s in result["steps_log"]]
        assert all(b >= a for a, b in zip(oracle_calls_log, oracle_calls_log[1:])), (
            "steps_log distinct_oracle_calls must be monotone non-decreasing"
        )
    print(
        f"    OK: n_distinct_calls={result['n_distinct_oracle_calls']}, "
        f"n_steps={result['n_steps_total']}, n_episodes={result['n_episodes']}, "
        f"n_resets={result['n_resets']}, best_omega={result['best_omega']:.4f}"
    )

    # ----- Test 5: dynamic protocol driver -----------------------------
    print("\n[5/5] run_dynamic_protocol (E_swing, horizon=32, segment=8)...")
    cfg3 = _build_smoke_env_config(
        mu_co=-0.2,
        seed=11,
        max_steps=64,  # >= horizon
        action_mode="swap",
        enable_noop=False,
        stop_terminates=False,
        min_stop_steps=0,
    )
    env3 = ChemGymEnv(cfg3)
    rng3 = np.random.default_rng(43)

    def policy_random_swap_3(_env: ChemGymEnv, _obs: Any, _info: Dict) -> int:
        return _random_valid_action(_env, rng3)

    sched = TEST_SCHEDULES_V1_2["E_swing"]
    result_d = run_dynamic_protocol(
        env3,
        policy_random_swap_3,
        schedule=sched,
        horizon_steps=32,
        segment_length=8,
        record_trace=True,
    )
    assert result_d["n_steps_total"] == 32
    mu_trace = result_d["mu_trace"]
    assert len(mu_trace) == 32
    # Verify mu_CO schedule was applied at the right segment boundaries
    assert mu_trace[0] == -0.2 and mu_trace[7] == -0.2, "segment 0 must be -0.2"
    assert mu_trace[8] == -0.8 and mu_trace[15] == -0.8, "segment 1 must be -0.8"
    assert mu_trace[16] == -0.4 and mu_trace[23] == -0.4, "segment 2 must be -0.4"
    assert mu_trace[24] == -0.6 and mu_trace[31] == -0.6, "segment 3 must be -0.6"
    # Verify env.config.mu_co was actually updated
    assert env3.config.mu_co == -0.6, f"env mu_co should be -0.6 after E_swing, got {env3.config.mu_co}"
    assert np.isfinite(result_d["mean_omega"]), "mean_omega must be finite"
    assert np.isfinite(result_d["cumulative_omega"]), "cumulative_omega must be finite"
    print(
        f"    OK: mu_trace transitions correct, mean_omega={result_d['mean_omega']:.4f}, "
        f"cumulative_omega={result_d['cumulative_omega']:.4f}, "
        f"n_distinct_calls={result_d['n_distinct_oracle_calls']}"
    )

    # Edge case: schedule * segment != horizon must raise
    try:
        run_dynamic_protocol(
            env3, policy_random_swap_3,
            schedule=[-0.2, -0.4, -0.6],  # 3 segments
            horizon_steps=32,
            segment_length=8,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("dynamic driver must reject schedule/segment/horizon mismatch")

    # Edge case: env.config.max_steps < horizon_steps must raise
    cfg_short = _build_smoke_env_config(mu_co=-0.2, seed=11, max_steps=16, action_mode="swap")
    env_short = ChemGymEnv(cfg_short)
    try:
        run_dynamic_protocol(
            env_short, policy_random_swap_3,
            schedule=sched,
            horizon_steps=32,
            segment_length=8,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("dynamic driver must reject max_steps < horizon_steps")

    print("    OK: edge-case validation works")

    print("\n" + "=" * 64)
    print("ALL UTILITIES PASS.")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_smoke_test())
