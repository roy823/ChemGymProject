"""Week-8 MC baselines — implements v1.2 §8 step 2.

Four MC-family samplers, all consuming the same protocol primitives from
``week8_protocol.py``:

  1. ``RandomSwapPolicy`` — uniform sampling from valid swap pairs, no
     acceptance criterion. Lower-bound calibration (v1.2 §4.4).
  2. ``CanonicalMCSwapPolicy`` — Metropolis on swap proposals at fixed
     ``T`` (default 0.10 eV). Two-phase state machine: each MC iteration
     occupies one "propose" env-step and one "decide" env-step
     (decide = re-apply the proposal as its own inverse on reject, or
     emit a fresh proposal on accept). Cache-aware: a rejected swap
     consumes 1 distinct oracle call. Includes mu_CO drift correction
     (refresh ``omega_pre`` to the current mu_CO via
     ``omega + (mu_old - mu_new) * n_co``) so dynamic-protocol chains
     remain in detailed balance through the per-segment chemical-
     potential changes (v1.2 §4.2).
  3. ``SGCMCMutationPolicy`` — semi-grand canonical mutation MC with
     bias toward target Pd fraction (v1.2 §4.5). Same two-phase state
     machine as ``CanonicalMCSwapPolicy`` but the acceptance log-ratio
     adds ``-Delta_mu_CuPd * Delta_N_Pd / kT``. ``Delta_mu_CuPd`` is a
     CLI flag that must be calibrated offline so equilibrium mean
     Pd_frac approaches the target — calibration is out of scope for
     step 2 (defaults to 0; SI-only metric).
  4. ``run_replica_exchange_mc_static`` / ``run_replica_exchange_mc_dynamic``
     — orchestrates 4 ``CanonicalMCSwapPolicy`` chains at distinct
     temperatures with periodic exchange (v1.2 §4.3). Implemented as
     a multi-env driver because exchanging chain configurations
     between two envs requires direct slab-state surgery (clone
     ``state``, ``atoms``, ``atoms_with_co``, ``current_*`` between
     envs); the policy abstraction is insufficient.

Smoke test (``python ProjectMain/week8/week8_baselines_mc.py``)
exercises each baseline on the 32-step EMT-fallback path with a tiny
budget; confirms non-NaN summaries, monotone running-min-omega traces,
and (for RE-MC) at least one inter-replica exchange attempt.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

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

# week8_protocol lives next to this module
from week8.week8_protocol import (
    HORIZON_STEPS,
    SEGMENT_LENGTH,
    TEST_SCHEDULES_V1_2,
    distinct_oracle_calls,
    install_oracle_counter,
    run_dynamic_protocol,
    run_static_protocol,
    set_env_mu_co,
)


# =============================================================================
# Mask sampling helpers
# =============================================================================

def _sample_uniform_valid_action(env: ChemGymEnv, rng: np.random.Generator) -> int:
    """Pick a uniformly random valid action from ``env.action_masks()``.

    Falls back to a uniform random over the full action space if no
    action is masked-in (which should not happen in practice but is
    defensive against env edge cases).
    """
    masks = env.action_masks().astype(bool)
    valid = np.flatnonzero(masks)
    if valid.size == 0:
        return int(rng.integers(0, env.action_space.n))
    return int(rng.choice(valid))


def _refresh_omega_for_mu_change(
    omega_at_old_mu: float,
    mu_old: float,
    mu_new: float,
    n_co: int,
) -> float:
    """Recompute Omega at a new mu_CO using the env's Omega = E - mu * N convention.

    Omega(new) = (E_slab + E_ads) - mu_new * N_CO
               = Omega(old) + (mu_old - mu_new) * N_CO.

    When the dynamic-protocol driver crosses a segment boundary the
    chemical potential changes between two consecutive env.step() calls;
    the canonical-MC acceptance Delta_Omega = Omega_post - Omega_pre is
    only valid if both terms are evaluated at the same mu_CO. We refresh
    ``omega_pre`` (the saved pre-proposal value) to the current mu_CO
    before computing Delta.
    """
    return float(omega_at_old_mu) + (float(mu_old) - float(mu_new)) * float(n_co)


# =============================================================================
# Policies
# =============================================================================

class RandomSwapPolicy:
    """Uniform-random valid swap, no acceptance criterion.

    Stateless: every call samples a fresh valid swap. Plugs straight
    into ``run_static_protocol`` / ``run_dynamic_protocol``.
    """

    name: str = "random_swap"

    def __init__(self, seed: int):
        self.rng = np.random.default_rng(int(seed))

    def __call__(self, env: ChemGymEnv, obs: Any, info: Dict) -> int:
        return _sample_uniform_valid_action(env, self.rng)


class RandomMutationPolicy:
    """Uniform-random valid mutation, no acceptance criterion.

    Used as the SGCMC baseline-of-baselines comparator under mutation
    mode and as the SI random-mutation lower bound (v1.2 §4.4 in
    spirit). Stateless.
    """

    name: str = "random_mutation"

    def __init__(self, seed: int):
        self.rng = np.random.default_rng(int(seed))

    def __call__(self, env: ChemGymEnv, obs: Any, info: Dict) -> int:
        return _sample_uniform_valid_action(env, self.rng)


class CanonicalMCSwapPolicy:
    """Metropolis canonical-MC over swap moves at fixed T.

    State machine (per call):

      * On the very first call ``omega_pre`` is unset: just emit a
        proposal and remember it.
      * If ``just_did_revert`` is True: the env state has been reverted
        to ``omega_pre``; emit a fresh proposal.
      * Otherwise: ``info[omega]`` is the omega *after* the previous
        proposal. Compute Delta vs ``omega_pre`` (refreshed to the
        current mu_CO if it has changed). Accept with probability
        min(1, exp(-Delta / kT)):

          - Accept: ``omega_pre = omega_now``, emit a fresh proposal.
          - Reject: re-emit the previous proposal action (swap is its
            own inverse, so re-applying it returns the env to
            ``omega_pre``); set ``just_did_revert``.

    The acceptance ratio at iteration ``k`` therefore consumes either
    one env.step (accept) or two env.step calls (reject) under the
    protocol driver. Cache-aware budget accounting still works:
    re-evaluating the previously visited state hits ``energy_cache``,
    so a reject contributes only one *distinct* oracle call.
    """

    name: str = "canonical_mc"

    def __init__(self, seed: int, T: float = 0.10):
        self.T = float(T)
        self.rng = np.random.default_rng(int(seed))
        self.omega_pre: Optional[float] = None
        self.n_co_pre: int = -1
        self.mu_co_at_save: Optional[float] = None
        self.last_proposal: Optional[int] = None
        self.just_did_revert: bool = False
        self.n_proposals: int = 0
        self.n_accepts: int = 0
        self.n_rejects: int = 0

    def reset_internal_state(self) -> None:
        """Drop the saved omega_pre / last_proposal so the next call enters
        PROPOSE phase against whatever state the env now holds.

        Used by RE-MC after an accepted replica exchange: the swap
        replaces the env's slab configuration, so the cached
        ``(omega_pre, last_proposal)`` no longer match. Accept/reject
        statistics are preserved.
        """
        self.omega_pre = None
        self.n_co_pre = -1
        self.mu_co_at_save = None
        self.last_proposal = None
        self.just_did_revert = False

    def _propose(self, env: ChemGymEnv) -> int:
        action = _sample_uniform_valid_action(env, self.rng)
        self.last_proposal = action
        self.n_proposals += 1
        return action

    def _save_pre(self, env: ChemGymEnv, omega: float, n_co: int) -> None:
        self.omega_pre = float(omega)
        self.n_co_pre = int(n_co)
        self.mu_co_at_save = float(env.config.mu_co)

    def __call__(self, env: ChemGymEnv, obs: Any, info: Dict) -> int:
        # Detect a between-call env.reset() (max_steps_per_episode boundary in
        # the static protocol, or defensive reset on env termination in the
        # dynamic protocol). After reset, env.mutation_count is 0 and our
        # cached omega_pre / last_proposal refer to the previous episode's
        # state — drop them so this call enters PROPOSE phase fresh.
        if int(getattr(env, "mutation_count", 0)) == 0 and self.omega_pre is not None:
            self.reset_internal_state()

        omega_now = float(info.get("omega", float("nan")))
        n_co_now = int(info.get("n_co", -1))

        if self.just_did_revert:
            self.just_did_revert = False
            return self._propose(env)

        if self.omega_pre is None or self.last_proposal is None:
            self._save_pre(env, omega_now, n_co_now)
            return self._propose(env)

        mu_now = float(env.config.mu_co)
        omega_pre_now = (
            self.omega_pre if mu_now == self.mu_co_at_save
            else _refresh_omega_for_mu_change(
                self.omega_pre, float(self.mu_co_at_save), mu_now, self.n_co_pre,
            )
        )
        delta = omega_now - omega_pre_now

        if not np.isfinite(delta):
            accept = False
        elif delta <= 0.0:
            accept = True
        else:
            log_p = -delta / max(self.T, 1e-12)
            accept = bool(self.rng.uniform() < float(np.exp(log_p)))

        if accept:
            self.n_accepts += 1
            self._save_pre(env, omega_now, n_co_now)
            return self._propose(env)

        self.n_rejects += 1
        self.just_did_revert = True
        return int(self.last_proposal)


class SGCMCMutationPolicy:
    """Semi-grand canonical mutation MC (v1.2 §4.5).

    Same two-phase state machine as ``CanonicalMCSwapPolicy`` but:

      * action_mode is mutation (each action mutates one site to one
        element), not swap;
      * the acceptance log-ratio carries an extra
        ``- Delta_mu_CuPd * Delta_N_Pd / kT`` term to bias the chain
        toward a target Pd fraction;
      * ``Delta_mu_CuPd`` (eV per Pd atom added at the expense of a
        Cu atom) is a free parameter; it must be calibrated offline so
        the equilibrium mean Pd fraction matches the target. Calibration
        is out of scope for step 2; we ship the policy with
        ``delta_mu_cupd`` defaulting to 0 (the SGCMC baseline reduces
        to canonical-MC over mutation moves under that default).

    Reverting a mutation requires remembering the previous element at
    the mutated site; we hook into ``env.action_spec`` to decode the
    proposal action into ``(site_idx, new_elem_idx)``, snapshot the
    previous element before the env.step is taken, and on reject we
    propose ``(site_idx, prev_elem_idx)`` which restores the original
    state.

    Pd is element index 1 in the env (Cu=0, Pd=1) by convention.
    """

    name: str = "sgcmc"

    def __init__(
        self,
        seed: int,
        T: float = 0.10,
        target_pd_frac: float = 0.08,
        delta_mu_cupd: float = 0.0,
        pd_elem_idx: int = 1,
    ):
        self.T = float(T)
        self.target_pd_frac = float(target_pd_frac)
        self.delta_mu_cupd = float(delta_mu_cupd)
        self.pd_elem_idx = int(pd_elem_idx)
        self.rng = np.random.default_rng(int(seed))

        self.omega_pre: Optional[float] = None
        self.n_co_pre: int = -1
        self.n_pd_pre: int = -1
        self.mu_co_at_save: Optional[float] = None
        self.last_proposal: Optional[int] = None
        self.last_proposal_site: int = -1
        self.last_proposal_prev_elem: int = -1
        self.just_did_revert: bool = False
        self.revert_action: Optional[int] = None
        self.n_proposals: int = 0
        self.n_accepts: int = 0
        self.n_rejects: int = 0

    def reset_internal_state(self) -> None:
        """Drop cached state so the next call enters PROPOSE phase. Used by
        the protocol layer's episode boundary (env.reset) and by RE-MC when
        an exchange is accepted. Accept/reject statistics are preserved.
        """
        self.omega_pre = None
        self.n_co_pre = -1
        self.n_pd_pre = -1
        self.mu_co_at_save = None
        self.last_proposal = None
        self.last_proposal_site = -1
        self.last_proposal_prev_elem = -1
        self.just_did_revert = False
        self.revert_action = None

    def _count_pd(self, env: ChemGymEnv) -> int:
        return int(np.sum(env.state == self.pd_elem_idx))

    def _propose(self, env: ChemGymEnv) -> int:
        action = _sample_uniform_valid_action(env, self.rng)
        site_idx, new_elem = env.action_spec.to_indices(action)
        self.last_proposal = action
        self.last_proposal_site = int(site_idx)
        self.last_proposal_prev_elem = int(env.state[site_idx]) if site_idx >= 0 else -1
        self.n_proposals += 1
        return action

    def _save_pre(self, env: ChemGymEnv, omega: float, n_co: int) -> None:
        self.omega_pre = float(omega)
        self.n_co_pre = int(n_co)
        self.n_pd_pre = self._count_pd(env)
        self.mu_co_at_save = float(env.config.mu_co)

    def __call__(self, env: ChemGymEnv, obs: Any, info: Dict) -> int:
        # Reset detection — see CanonicalMCSwapPolicy.__call__ for rationale.
        if int(getattr(env, "mutation_count", 0)) == 0 and self.omega_pre is not None:
            self.reset_internal_state()

        omega_now = float(info.get("omega", float("nan")))
        n_co_now = int(info.get("n_co", -1))

        if self.just_did_revert:
            self.just_did_revert = False
            self.revert_action = None
            return self._propose(env)

        if self.omega_pre is None or self.last_proposal is None:
            self._save_pre(env, omega_now, n_co_now)
            return self._propose(env)

        mu_now = float(env.config.mu_co)
        omega_pre_now = (
            self.omega_pre if mu_now == self.mu_co_at_save
            else _refresh_omega_for_mu_change(
                self.omega_pre, float(self.mu_co_at_save), mu_now, self.n_co_pre,
            )
        )
        delta_omega = omega_now - omega_pre_now
        n_pd_now = self._count_pd(env)
        delta_n_pd = n_pd_now - self.n_pd_pre
        log_arg = -(delta_omega + self.delta_mu_cupd * float(delta_n_pd)) / max(self.T, 1e-12)

        if not np.isfinite(log_arg):
            accept = False
        elif log_arg >= 0.0:
            accept = True
        else:
            accept = bool(self.rng.uniform() < float(np.exp(log_arg)))

        if accept:
            self.n_accepts += 1
            self._save_pre(env, omega_now, n_co_now)
            return self._propose(env)

        self.n_rejects += 1
        # Build the inverse mutation: restore (site, prev_elem).
        revert = env.action_spec.to_action(self.last_proposal_site, self.last_proposal_prev_elem)
        self.revert_action = int(revert)
        self.just_did_revert = True
        return int(revert)


# =============================================================================
# Replica-Exchange MC
# =============================================================================

DEFAULT_RE_LADDER: Tuple[float, ...] = (0.05, 0.10, 0.20, 0.35)


def _adjacent_pair_keys(n_replicas: int) -> List[str]:
    return [f"{i}-{i + 1}" for i in range(max(int(n_replicas) - 1, 0))]


def _exchange_acceptance_by_pair(
    attempts: Dict[str, int],
    accepts: Dict[str, int],
) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {}
    for key in sorted(attempts.keys(), key=lambda s: tuple(int(x) for x in s.split("-"))):
        n_att = int(attempts.get(key, 0))
        n_acc = int(accepts.get(key, 0))
        out[key] = float(n_acc / n_att) if n_att > 0 else None
    return out


def _snapshot_env_state(env: ChemGymEnv) -> Dict[str, Any]:
    """Take a deep-enough snapshot of an env's slab+CO state to enable
    replica swaps. Hot path: every 64 proposals × 4 replicas.

    Returns a dict that ``_restore_env_state`` consumes. We do NOT
    snapshot caches, observation buffers, or RNG state — only the
    physical state and its derived quantities.
    """
    return {
        "state": np.array(env.state, copy=True),
        "atoms": env.atoms.copy() if env.atoms is not None else None,
        "atoms_with_co": env.atoms_with_co.copy() if env.atoms_with_co is not None else None,
        "current_n_co": int(env.current_n_co),
        "current_omega": float(env.current_omega),
        "current_energy_slab": float(env.current_energy_slab),
        "current_energy_with_co": float(env.current_energy_with_co),
        "current_ads_energy": float(env.current_ads_energy),
        "min_omega_so_far": float(env.min_omega_so_far),
    }


def _restore_env_state(env: ChemGymEnv, snap: Dict[str, Any]) -> None:
    env.state = np.array(snap["state"], copy=True)
    env.atoms = snap["atoms"].copy() if snap["atoms"] is not None else None
    env.atoms_with_co = snap["atoms_with_co"].copy() if snap["atoms_with_co"] is not None else None
    env.current_n_co = int(snap["current_n_co"])
    env.current_omega = float(snap["current_omega"])
    env.current_energy_slab = float(snap["current_energy_slab"])
    env.current_energy_with_co = float(snap["current_energy_with_co"])
    env.current_ads_energy = float(snap["current_ads_energy"])
    env.min_omega_so_far = min(env.min_omega_so_far, float(snap["min_omega_so_far"]))


def _post_exchange_info(env: ChemGymEnv) -> Dict[str, Any]:
    """Synthesize the info dict the next policy call will see after an
    accepted exchange. The env attributes were just clobbered by
    ``_restore_env_state``, so we read out the post-swap omega / n_co
    and tag the action_type as 'exchange' for traceability.
    """
    return {
        "omega": float(env.current_omega),
        "n_co": int(env.current_n_co),
        "action_type": "exchange",
        "pd_surface_coverage": float("nan"),
    }


def _attempt_replica_exchange(
    env_i: ChemGymEnv,
    env_j: ChemGymEnv,
    T_i: float,
    T_j: float,
    rng: np.random.Generator,
) -> bool:
    """Metropolis exchange of two replicas at temperatures T_i, T_j.

    Acceptance probability:
        P = min(1, exp((1/kT_i - 1/kT_j) * (E_i - E_j))).
    On accept the *physical state* (slab/CO/derived energies) is swapped
    between the two envs. The temperatures of the two replicas remain
    fixed; what flows is the chain configuration. Per v1.2 §4.3.
    """
    E_i = float(env_i.current_omega)
    E_j = float(env_j.current_omega)
    if not (np.isfinite(E_i) and np.isfinite(E_j)):
        return False

    inv_kT_i = 1.0 / max(T_i, 1e-12)
    inv_kT_j = 1.0 / max(T_j, 1e-12)
    log_arg = (inv_kT_i - inv_kT_j) * (E_i - E_j)
    if log_arg >= 0.0:
        accept = True
    else:
        accept = bool(rng.uniform() < float(np.exp(log_arg)))

    if not accept:
        return False

    snap_i = _snapshot_env_state(env_i)
    snap_j = _snapshot_env_state(env_j)
    _restore_env_state(env_i, snap_j)
    _restore_env_state(env_j, snap_i)
    return True


def _exchange_pairs(round_idx: int, n_replicas: int) -> List[Tuple[int, int]]:
    """Alternating-pair exchange schedule (v1.2 §4.3).

    Even rounds: (0,1), (2,3), ...
    Odd  rounds: (1,2), (3,4), ...
    """
    pairs: List[Tuple[int, int]] = []
    start = int(round_idx) % 2
    i = start
    while i + 1 < int(n_replicas):
        pairs.append((int(i), int(i + 1)))
        i += 2
    return pairs


def run_replica_exchange_mc_static(
    envs: Sequence[ChemGymEnv],
    *,
    temperatures: Sequence[float],
    total_pooled_oracle_budget: int,
    max_steps_per_episode: int = 32,
    proposals_per_replica_per_round: int = 64,
    seed: int = 0,
    record_trace: bool = True,
) -> Dict[str, Any]:
    """RE-MC under the static protocol (v1.2 §4.3 main).

    Each replica = one ``ChemGymEnv`` running its own
    ``CanonicalMCSwapPolicy`` chain at its own temperature. Every
    ``proposals_per_replica_per_round`` MC iterations we attempt
    Metropolis exchange of physical configurations between adjacent
    pairs (alternating odd/even rounds). The combined chain is bounded
    by ``total_pooled_oracle_budget`` distinct oracle calls summed
    across replicas.
    """
    if len(envs) != len(temperatures):
        raise ValueError(f"envs and temperatures must align: {len(envs)} vs {len(temperatures)}")
    n_repl = len(envs)
    rng = np.random.default_rng(int(seed))
    policies = [
        CanonicalMCSwapPolicy(seed=int(seed) + 1000 * (k + 1), T=float(temperatures[k]))
        for k in range(n_repl)
    ]
    for env in envs:
        install_oracle_counter(env)

    obs_list: List[Any] = []
    info_list: List[Dict] = []
    for env in envs:
        obs, info = env.reset()
        obs_list.append(obs)
        info_list.append(info)

    best_omega = float("inf")
    best_replica = -1
    best_trace: List[float] = []
    n_exchange_attempts = 0
    n_exchange_accepts = 0
    exchange_attempts_by_pair = {key: 0 for key in _adjacent_pair_keys(n_repl)}
    exchange_accepts_by_pair = {key: 0 for key in _adjacent_pair_keys(n_repl)}
    n_steps_per_replica = [0] * n_repl
    n_episodes_per_replica = [0] * n_repl
    n_resets_per_replica = [1] * n_repl
    episode_step_per_replica = [0] * n_repl

    proposals_since_round = [0] * n_repl
    round_idx = 0

    def _pooled_distinct() -> int:
        return int(sum(distinct_oracle_calls(env) for env in envs))

    while _pooled_distinct() < int(total_pooled_oracle_budget):
        for k in range(n_repl):
            action = int(policies[k](envs[k], obs_list[k], info_list[k]))
            obs, _, terminated, truncated, info = envs[k].step(action)
            obs_list[k] = obs
            info_list[k] = info
            n_steps_per_replica[k] += 1
            episode_step_per_replica[k] += 1
            proposals_since_round[k] += 1

            omega = info.get("omega", float("nan"))
            omega = float(omega) if omega is not None else float("nan")
            if np.isfinite(omega) and omega < best_omega:
                best_omega = omega
                best_replica = k
            best_trace.append(float(best_omega))

            end_episode = bool(
                terminated or truncated
                or episode_step_per_replica[k] >= int(max_steps_per_episode)
            )
            if end_episode:
                n_episodes_per_replica[k] += 1
                episode_step_per_replica[k] = 0
                if _pooled_distinct() < int(total_pooled_oracle_budget):
                    obs2, info2 = envs[k].reset()
                    obs_list[k] = obs2
                    info_list[k] = info2
                    n_resets_per_replica[k] += 1
                    # Policy self-detects the reset via env.mutation_count == 0
                    # at the next call and drops its cached omega_pre.

            if _pooled_distinct() >= int(total_pooled_oracle_budget):
                break

        # Exchange round?
        if min(proposals_since_round) >= int(proposals_per_replica_per_round):
            for (a, b) in _exchange_pairs(round_idx, n_repl):
                pair_key = f"{a}-{b}"
                n_exchange_attempts += 1
                exchange_attempts_by_pair[pair_key] = exchange_attempts_by_pair.get(pair_key, 0) + 1
                accepted = _attempt_replica_exchange(
                    envs[a], envs[b], float(temperatures[a]), float(temperatures[b]), rng,
                )
                if accepted:
                    n_exchange_accepts += 1
                    exchange_accepts_by_pair[pair_key] = exchange_accepts_by_pair.get(pair_key, 0) + 1
                    # Configurations were swapped between envs[a] and envs[b];
                    # the policies' saved omega_pre / last_proposal no longer
                    # match the env state. Drop them and refresh the info dict
                    # so the next policy call enters PROPOSE phase against the
                    # newly-arrived configuration.
                    policies[a].reset_internal_state()
                    policies[b].reset_internal_state()
                    info_list[a] = _post_exchange_info(envs[a])
                    info_list[b] = _post_exchange_info(envs[b])
            round_idx += 1
            proposals_since_round = [0] * n_repl

    return {
        "best_omega": float(best_omega) if np.isfinite(best_omega) else float("nan"),
        "best_replica": int(best_replica),
        "best_omega_trace": np.asarray(best_trace, dtype=float),
        "n_exchange_attempts": int(n_exchange_attempts),
        "n_exchange_accepts": int(n_exchange_accepts),
        "exchange_acceptance": (
            float(n_exchange_accepts / n_exchange_attempts)
            if n_exchange_attempts > 0 else float("nan")
        ),
        "exchange_attempts_by_pair": dict(exchange_attempts_by_pair),
        "exchange_accepts_by_pair": dict(exchange_accepts_by_pair),
        "exchange_acceptance_by_pair": _exchange_acceptance_by_pair(
            exchange_attempts_by_pair, exchange_accepts_by_pair,
        ),
        "n_steps_per_replica": list(n_steps_per_replica),
        "n_episodes_per_replica": list(n_episodes_per_replica),
        "n_resets_per_replica": list(n_resets_per_replica),
        "n_distinct_oracle_calls_per_replica": [int(distinct_oracle_calls(e)) for e in envs],
        "n_pooled_distinct_oracle_calls": int(_pooled_distinct()),
        "temperatures": [float(t) for t in temperatures],
        "per_replica_accept_rates": [
            float(p.n_accepts / max(p.n_proposals, 1)) for p in policies
        ],
    }


def run_replica_exchange_mc_dynamic(
    envs: Sequence[ChemGymEnv],
    *,
    temperatures: Sequence[float],
    schedule: Sequence[float],
    horizon_steps: int = HORIZON_STEPS,
    segment_length: int = SEGMENT_LENGTH,
    proposals_per_replica_per_round: int = 64,
    seed: int = 0,
    record_trace: bool = True,
) -> Dict[str, Any]:
    """RE-MC under the dynamic protocol (v1.2 §4.3 dynamic).

    All replicas track the same mu_CO schedule synchronously. Each
    replica runs ``horizon_steps`` env.step calls (one per t); exchanges
    are attempted every ``proposals_per_replica_per_round`` t-ticks.
    """
    if len(envs) != len(temperatures):
        raise ValueError(f"envs and temperatures must align: {len(envs)} vs {len(temperatures)}")
    if int(segment_length) * len(schedule) != int(horizon_steps):
        raise ValueError("segment_length * len(schedule) != horizon_steps")
    n_repl = len(envs)
    rng = np.random.default_rng(int(seed))
    policies = [
        CanonicalMCSwapPolicy(seed=int(seed) + 1000 * (k + 1), T=float(temperatures[k]))
        for k in range(n_repl)
    ]

    for env in envs:
        install_oracle_counter(env)
        set_env_mu_co(env, float(schedule[0]))

    obs_list: List[Any] = []
    info_list: List[Dict] = []
    for env in envs:
        obs, info = env.reset()
        obs_list.append(obs)
        info_list.append(info)

    best_omega_per_replica = [float("inf")] * n_repl
    omega_trace_per_replica: List[List[float]] = [[] for _ in range(n_repl)]
    cumulative_omega_per_replica = [0.0] * n_repl
    n_finite_per_replica = [0] * n_repl
    mu_trace: List[float] = []
    n_exchange_attempts = 0
    n_exchange_accepts = 0
    exchange_attempts_by_pair = {key: 0 for key in _adjacent_pair_keys(n_repl)}
    exchange_accepts_by_pair = {key: 0 for key in _adjacent_pair_keys(n_repl)}
    proposals_since_round = [0] * n_repl
    round_idx = 0

    for t in range(int(horizon_steps)):
        seg_idx = int(t) // int(segment_length)
        mu_t = float(schedule[seg_idx])
        for env in envs:
            set_env_mu_co(env, mu_t)
        mu_trace.append(mu_t)

        for k in range(n_repl):
            action = int(policies[k](envs[k], obs_list[k], info_list[k]))
            obs, _, terminated, truncated, info = envs[k].step(action)
            obs_list[k] = obs
            info_list[k] = info
            proposals_since_round[k] += 1

            omega = info.get("omega", float("nan"))
            omega = float(omega) if omega is not None else float("nan")
            if np.isfinite(omega):
                omega_trace_per_replica[k].append(omega)
                cumulative_omega_per_replica[k] += omega
                n_finite_per_replica[k] += 1
                if omega < best_omega_per_replica[k]:
                    best_omega_per_replica[k] = omega

            if terminated:
                # v1.2 §5.2: no resets within dynamic horizon. NaN/term
                # is rare under EMT/UMA; defensive reset preserves the
                # mu schedule for remaining ticks.
                obs2, info2 = envs[k].reset()
                obs_list[k] = obs2
                info_list[k] = info2

        if min(proposals_since_round) >= int(proposals_per_replica_per_round):
            for (a, b) in _exchange_pairs(round_idx, n_repl):
                pair_key = f"{a}-{b}"
                n_exchange_attempts += 1
                exchange_attempts_by_pair[pair_key] = exchange_attempts_by_pair.get(pair_key, 0) + 1
                accepted = _attempt_replica_exchange(
                    envs[a], envs[b], float(temperatures[a]), float(temperatures[b]), rng,
                )
                if accepted:
                    n_exchange_accepts += 1
                    exchange_accepts_by_pair[pair_key] = exchange_accepts_by_pair.get(pair_key, 0) + 1
                    policies[a].reset_internal_state()
                    policies[b].reset_internal_state()
                    info_list[a] = _post_exchange_info(envs[a])
                    info_list[b] = _post_exchange_info(envs[b])
            round_idx += 1
            proposals_since_round = [0] * n_repl

    overall_best = (
        float(min(best_omega_per_replica))
        if any(np.isfinite(b) for b in best_omega_per_replica) else float("nan")
    )
    overall_best_replica = int(np.argmin(best_omega_per_replica)) if np.isfinite(overall_best) else -1

    return {
        "best_omega": overall_best,
        "best_replica": overall_best_replica,
        "best_omega_per_replica": [float(b) for b in best_omega_per_replica],
        "cumulative_omega_per_replica": [float(c) for c in cumulative_omega_per_replica],
        "mean_omega_per_replica": [
            float(c / max(n, 1)) for c, n in zip(cumulative_omega_per_replica, n_finite_per_replica)
        ],
        "n_finite_per_replica": list(n_finite_per_replica),
        "mu_trace": np.asarray(mu_trace, dtype=float),
        "n_exchange_attempts": int(n_exchange_attempts),
        "n_exchange_accepts": int(n_exchange_accepts),
        "exchange_acceptance": (
            float(n_exchange_accepts / n_exchange_attempts)
            if n_exchange_attempts > 0 else float("nan")
        ),
        "exchange_attempts_by_pair": dict(exchange_attempts_by_pair),
        "exchange_accepts_by_pair": dict(exchange_accepts_by_pair),
        "exchange_acceptance_by_pair": _exchange_acceptance_by_pair(
            exchange_attempts_by_pair, exchange_accepts_by_pair,
        ),
        "n_distinct_oracle_calls_per_replica": [int(distinct_oracle_calls(e)) for e in envs],
        "n_pooled_distinct_oracle_calls": int(sum(distinct_oracle_calls(e) for e in envs)),
        "temperatures": [float(t) for t in temperatures],
        "per_replica_accept_rates": [
            float(p.n_accepts / max(p.n_proposals, 1)) for p in policies
        ],
    }


# =============================================================================
# Smoke test config builder (mirrors week8_protocol's helper, kept local so
# this module's smoke test does not depend on a private helper from another)
# =============================================================================

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


# =============================================================================
# Production env builder + oracle loader (used by the CLI for long runs)
# =============================================================================

def build_run_env_config(
    *,
    mu_co: float,
    seed: int,
    max_steps: int,
    action_mode: str,
    bulk_pd_fraction: float = 0.08,
    n_active_layers: int = 4,
    enable_noop: bool = False,
    stop_terminates: bool = False,
    min_stop_steps: int = 0,
) -> EnvConfig:
    """Production env config (mirrors ``_build_smoke_env_config`` but without
    constraint shaping forced off; matches week7 baselines schema).
    """
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
        bulk_pd_fraction=float(bulk_pd_fraction),
        n_active_layers=int(n_active_layers),
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


def maybe_load_oracle_lazy(args) -> Any:
    """Best-effort hybrid-oracle loader. Returns None on EMT fallback.

    Mirrors the API in ``ProjectMain/week7/week7_baselines_feasible.py``
    so the same checkpoint paths flow through.
    """
    if str(getattr(args, "oracle_mode", "hybrid")).lower() == "none":
        return None
    from main import maybe_load_oracle as _legacy_loader
    from argparse import Namespace as _NS
    ns = _NS(
        oracle_ckpt=None,
        oracle_mode=str(args.oracle_mode),
        ads_task=getattr(args, "ads_task", "oc25"),
        disable_ads_ensemble=False,
        ads_sm_ckpt=getattr(args, "ads_sm_ckpt", "ProjectMain/checkpoints/esen_sm_conserve.pt"),
        ads_md_ckpt=getattr(args, "ads_md_ckpt", "ProjectMain/checkpoints/esen_md_direct.pt"),
        eq2_ckpt=getattr(args, "eq2_ckpt", "ProjectMain/checkpoints/eq2_83M_2M.pt"),
        uma_ckpt=getattr(args, "uma_ckpt", "ProjectMain/checkpoints/uma-s-1p1.pt"),
        oracle_fmax=0.05,
        oracle_max_steps=100,
        require_ads_oracle=False,
    )
    return _legacy_loader(ns)


# =============================================================================
# Per-seed runners + CSV writers (used by the CLI)
# =============================================================================

import csv as _csv  # late import so the smoke test stays standalone
import json as _json
import time as _time


def _running_min_trace(omega_seq: Sequence[float]) -> List[float]:
    out: List[float] = []
    cur = float("inf")
    for w in omega_seq:
        if np.isfinite(w) and w < cur:
            cur = float(w)
        out.append(float(cur) if np.isfinite(cur) else float("nan"))
    return out


def _write_csv(path: _Path, rows: Sequence[Dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = _csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def _append_csv(path: _Path, row: Dict, fieldnames: Sequence[str]) -> None:
    """Append-only summary writer, so partial overnight runs survive
    interruption with rows already on disk for completed seeds.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = _csv.DictWriter(f, fieldnames=list(fieldnames))
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def _trace_rows_from_static(steps_log: Sequence[Dict], seed: int, schedule_id: str) -> List[Dict]:
    out: List[Dict] = []
    for s in steps_log:
        out.append({
            "seed": int(seed),
            "schedule": str(schedule_id),
            "global_step": int(s["global_step"]),
            "episode": int(s["episode"]),
            "episode_step": int(s["episode_step"]),
            "action_type": str(s["action_type"]),
            "omega": float(s["omega"]) if np.isfinite(s["omega"]) else float("nan"),
            "best_omega_so_far": float(s["best_omega_so_far"]) if np.isfinite(s["best_omega_so_far"]) else float("nan"),
            "n_distinct_oracle_calls": int(s["n_distinct_oracle_calls"]),
        })
    return out


def _trace_rows_from_dynamic(steps_log: Sequence[Dict], seed: int, schedule_id: str) -> List[Dict]:
    out: List[Dict] = []
    for s in steps_log:
        out.append({
            "seed": int(seed),
            "schedule": str(schedule_id),
            "step": int(s["step"]),
            "segment": int(s["segment"]),
            "mu_co": float(s["mu_co"]),
            "action_type": str(s["action_type"]),
            "omega": float(s["omega"]) if np.isfinite(s["omega"]) else float("nan"),
            "best_omega_so_far": float(s["best_omega_so_far"]) if np.isfinite(s["best_omega_so_far"]) else float("nan"),
            "n_distinct_oracle_calls": int(s["n_distinct_oracle_calls"]),
        })
    return out


# Static-protocol summary fields (one row per seed, plus per-seed for RE-MC pooled)
STATIC_SUMMARY_FIELDS = [
    "method", "mu_co", "T", "seed",
    "best_omega",
    "n_distinct_oracle_calls", "n_steps_total",
    "n_episodes", "n_resets",
    "n_proposals", "n_accepts", "n_rejects", "accept_rate",
    "n_exchange_attempts", "n_exchange_accepts", "exchange_acceptance",
    "exchange_attempts_by_pair", "exchange_accepts_by_pair", "exchange_acceptance_by_pair",
    "wall_clock_seconds",
]

# Dynamic-protocol summary fields (one row per (seed, schedule))
DYNAMIC_SUMMARY_FIELDS = [
    "method", "mu_co_segment_0", "T", "seed", "schedule",
    "best_omega", "mean_omega", "cumulative_omega",
    "n_distinct_oracle_calls", "n_steps_total",
    "n_proposals", "n_accepts", "n_rejects", "accept_rate",
    "n_exchange_attempts", "n_exchange_accepts", "exchange_acceptance",
    "exchange_attempts_by_pair", "exchange_accepts_by_pair", "exchange_acceptance_by_pair",
    "wall_clock_seconds",
]


# -----------------------------------------------------------------------------
# Single-seed dispatch — returns a summary row + steps_log list
# -----------------------------------------------------------------------------

def _build_policy_for_seed(args, seed: int):
    method = str(args.method)
    if method == "random_swap":
        return RandomSwapPolicy(seed=int(seed))
    if method == "random_mutation":
        return RandomMutationPolicy(seed=int(seed))
    if method == "canonical_mc":
        return CanonicalMCSwapPolicy(seed=int(seed), T=float(args.T))
    if method == "sgcmc":
        return SGCMCMutationPolicy(
            seed=int(seed),
            T=float(args.T),
            target_pd_frac=float(args.sgcmc_target_pd_frac),
            delta_mu_cupd=float(args.sgcmc_delta_mu_cupd),
            pd_elem_idx=int(args.pd_elem_idx),
        )
    raise ValueError(f"unsupported method for single-policy run: {method}")


def _action_mode_for(method: str) -> str:
    """v1.2 §4 method → env action_mode mapping."""
    if method in {"random_swap", "canonical_mc", "replica_exchange_mc"}:
        return "swap"
    if method in {"random_mutation", "sgcmc"}:
        return "mutation"
    raise ValueError(f"unknown method {method}")


def run_one_seed_static(args, seed: int, oracle: Any) -> Tuple[Dict, List[Dict]]:
    method = str(args.method)
    action_mode = _action_mode_for(method)
    cfg = build_run_env_config(
        mu_co=float(args.mu_co),
        seed=int(seed),
        max_steps=int(args.max_steps_per_episode) + 8,  # +8 cushion vs strict ceiling
        action_mode=action_mode,
        bulk_pd_fraction=float(args.bulk_pd_fraction),
        n_active_layers=int(args.n_active_layers),
    )

    t0 = _time.perf_counter()
    if method == "replica_exchange_mc":
        temps = [float(t.strip()) for t in str(args.re_temperatures).split(",") if t.strip()]
        envs = [ChemGymEnv(cfg, oracle=oracle) for _ in range(len(temps))]
        re_res = run_replica_exchange_mc_static(
            envs,
            temperatures=temps,
            total_pooled_oracle_budget=int(args.total_oracle_budget),
            max_steps_per_episode=int(args.max_steps_per_episode),
            proposals_per_replica_per_round=int(args.re_proposals_per_replica_per_round),
            seed=int(seed),
            record_trace=False,
        )
        elapsed = float(_time.perf_counter() - t0)
        row = {
            "method": method,
            "mu_co": float(args.mu_co),
            "T": "ladder=" + ",".join(f"{t:g}" for t in temps),
            "seed": int(seed),
            "best_omega": re_res["best_omega"],
            "n_distinct_oracle_calls": int(re_res["n_pooled_distinct_oracle_calls"]),
            "n_steps_total": int(sum(re_res["n_steps_per_replica"])),
            "n_episodes": int(sum(re_res["n_episodes_per_replica"])),
            "n_resets": int(sum(re_res["n_resets_per_replica"])),
            "n_proposals": -1,  # per-replica, not pooled
            "n_accepts": -1,
            "n_rejects": -1,
            "accept_rate": float("nan"),
            "n_exchange_attempts": int(re_res["n_exchange_attempts"]),
            "n_exchange_accepts": int(re_res["n_exchange_accepts"]),
            "exchange_acceptance": float(re_res["exchange_acceptance"]) if np.isfinite(re_res["exchange_acceptance"]) else float("nan"),
            "exchange_attempts_by_pair": _json.dumps(re_res["exchange_attempts_by_pair"], sort_keys=True),
            "exchange_accepts_by_pair": _json.dumps(re_res["exchange_accepts_by_pair"], sort_keys=True),
            "exchange_acceptance_by_pair": _json.dumps(re_res["exchange_acceptance_by_pair"], sort_keys=True),
            "wall_clock_seconds": elapsed,
        }
        return row, []  # RE-MC trace logging is per-replica; skip pooled trace for now

    env = ChemGymEnv(cfg, oracle=oracle)
    policy = _build_policy_for_seed(args, seed)
    res = run_static_protocol(
        env,
        policy,
        total_oracle_budget=int(args.total_oracle_budget),
        max_steps_per_episode=int(args.max_steps_per_episode),
        record_trace=bool(args.record_trace),
    )
    elapsed = float(_time.perf_counter() - t0)
    n_props = int(getattr(policy, "n_proposals", -1))
    n_accs = int(getattr(policy, "n_accepts", -1))
    n_rejs = int(getattr(policy, "n_rejects", -1))
    accept_rate = float(n_accs) / max(n_props, 1) if n_props >= 0 else float("nan")
    row = {
        "method": method,
        "mu_co": float(args.mu_co),
        "T": float(args.T) if method in {"canonical_mc", "sgcmc"} else float("nan"),
        "seed": int(seed),
        "best_omega": float(res["best_omega"]),
        "n_distinct_oracle_calls": int(res["n_distinct_oracle_calls"]),
        "n_steps_total": int(res["n_steps_total"]),
        "n_episodes": int(res["n_episodes"]),
        "n_resets": int(res["n_resets"]),
        "n_proposals": n_props,
        "n_accepts": n_accs,
        "n_rejects": n_rejs,
        "accept_rate": accept_rate,
        "n_exchange_attempts": -1,
        "n_exchange_accepts": -1,
        "exchange_acceptance": float("nan"),
        "exchange_attempts_by_pair": "{}",
        "exchange_accepts_by_pair": "{}",
        "exchange_acceptance_by_pair": "{}",
        "wall_clock_seconds": elapsed,
    }
    trace_rows: List[Dict] = []
    if args.record_trace and res.get("steps_log"):
        trace_rows = _trace_rows_from_static(res["steps_log"], seed=int(seed), schedule_id="static")
    return row, trace_rows


def run_one_seed_dynamic(args, seed: int, oracle: Any) -> Tuple[List[Dict], List[Dict]]:
    """Run all 5 schedules A-E for the given seed. Returns (summary rows, trace rows)."""
    method = str(args.method)
    if method == "replica_exchange_mc":
        raise NotImplementedError(
            "RE-MC dynamic CLI runner is not exposed in step 2 — "
            "use canonical_mc / random_swap / sgcmc for now."
        )
    action_mode = _action_mode_for(method)
    schedules = list(TEST_SCHEDULES_V1_2.keys())  # A_inc..E_swing, deterministic order
    summary_rows: List[Dict] = []
    trace_rows: List[Dict] = []
    for sched_id in schedules:
        cfg = build_run_env_config(
            mu_co=float(TEST_SCHEDULES_V1_2[sched_id][0]),
            seed=int(seed),
            max_steps=int(args.horizon_steps) + 16,
            action_mode=action_mode,
            bulk_pd_fraction=float(args.bulk_pd_fraction),
            n_active_layers=int(args.n_active_layers),
        )
        env = ChemGymEnv(cfg, oracle=oracle)
        policy = _build_policy_for_seed(args, seed)
        t0 = _time.perf_counter()
        res = run_dynamic_protocol(
            env,
            policy,
            schedule=TEST_SCHEDULES_V1_2[sched_id],
            horizon_steps=int(args.horizon_steps),
            segment_length=int(args.segment_length),
            record_trace=bool(args.record_trace),
        )
        elapsed = float(_time.perf_counter() - t0)
        n_props = int(getattr(policy, "n_proposals", -1))
        n_accs = int(getattr(policy, "n_accepts", -1))
        accept_rate = float(n_accs) / max(n_props, 1) if n_props >= 0 else float("nan")
        summary_rows.append({
            "method": method,
            "mu_co_segment_0": float(TEST_SCHEDULES_V1_2[sched_id][0]),
            "T": float(args.T) if method in {"canonical_mc", "sgcmc"} else float("nan"),
            "seed": int(seed),
            "schedule": str(sched_id),
            "best_omega": float(res["best_omega"]),
            "mean_omega": float(res["mean_omega"]),
            "cumulative_omega": float(res["cumulative_omega"]),
            "n_distinct_oracle_calls": int(res["n_distinct_oracle_calls"]),
            "n_steps_total": int(res["n_steps_total"]),
            "n_proposals": n_props,
            "n_accepts": n_accs,
            "n_rejects": int(getattr(policy, "n_rejects", -1)),
            "accept_rate": accept_rate,
            "n_exchange_attempts": -1,
            "n_exchange_accepts": -1,
            "exchange_acceptance": float("nan"),
            "exchange_attempts_by_pair": "{}",
            "exchange_accepts_by_pair": "{}",
            "exchange_acceptance_by_pair": "{}",
            "wall_clock_seconds": elapsed,
        })
        if args.record_trace and res.get("steps_log"):
            trace_rows.extend(_trace_rows_from_dynamic(res["steps_log"], seed=int(seed), schedule_id=sched_id))
    return summary_rows, trace_rows


# =============================================================================
# CLI main()
# =============================================================================

def _parse_int_list(value: str) -> List[int]:
    return [int(x.strip()) for x in str(value).split(",") if x.strip()]


def _build_argparser():
    import argparse
    p = argparse.ArgumentParser(description="Week-8 MC baselines CLI (v1.2 §8 step 2)")
    p.add_argument("--method", required=True, choices=[
        "random_swap", "random_mutation", "canonical_mc", "replica_exchange_mc", "sgcmc",
    ])
    p.add_argument("--protocol", choices=["static", "dynamic"], default="static")
    p.add_argument("--mu-co", type=float, default=-0.6, help="Static mu_CO; ignored for dynamic.")
    p.add_argument("--T", type=float, default=0.10, help="Temperature for canonical_mc / sgcmc, eV.")
    p.add_argument("--total-oracle-budget", type=int, default=4096, help="Static-protocol budget; pooled for RE-MC.")
    p.add_argument("--max-steps-per-episode", type=int, default=32)
    p.add_argument("--horizon-steps", type=int, default=HORIZON_STEPS)
    p.add_argument("--segment-length", type=int, default=SEGMENT_LENGTH)
    p.add_argument("--seeds", default="11,22,33,44,55")
    p.add_argument("--re-temperatures", default="0.05,0.10,0.20,0.35")
    p.add_argument("--re-proposals-per-replica-per-round", type=int, default=64)
    p.add_argument("--sgcmc-target-pd-frac", type=float, default=0.08)
    p.add_argument("--sgcmc-delta-mu-cupd", type=float, default=0.0)
    p.add_argument("--pd-elem-idx", type=int, default=1)
    p.add_argument("--bulk-pd-fraction", type=float, default=0.08)
    p.add_argument("--n-active-layers", type=int, default=4)
    p.add_argument("--save-root", required=True)
    p.add_argument("--record-trace", action="store_true",
                   help="Persist per-step CSV trace (large file under long budgets).")
    p.add_argument("--oracle-mode", choices=["hybrid", "none"], default="hybrid")
    p.add_argument("--ads-task", default="oc25")
    p.add_argument("--ads-sm-ckpt", default="ProjectMain/checkpoints/esen_sm_conserve.pt")
    p.add_argument("--ads-md-ckpt", default="ProjectMain/checkpoints/esen_md_direct.pt")
    p.add_argument("--eq2-ckpt", default="ProjectMain/checkpoints/eq2_83M_2M.pt")
    p.add_argument("--uma-ckpt", default="ProjectMain/checkpoints/uma-s-1p1.pt")
    p.add_argument("--smoke", action="store_true", help="Run module smoke test instead of CLI.")
    return p


def _cli_main() -> int:
    args = _build_argparser().parse_args()
    if args.smoke:
        return _smoke_test()

    seeds = _parse_int_list(args.seeds)
    if not seeds:
        raise ValueError("Need at least one seed")

    save_root = _Path(args.save_root)
    save_root.mkdir(parents=True, exist_ok=True)

    meta = {
        "method": args.method,
        "protocol": args.protocol,
        "mu_co": float(args.mu_co),
        "T": float(args.T) if args.method in {"canonical_mc", "sgcmc"} else None,
        "total_oracle_budget": int(args.total_oracle_budget),
        "max_steps_per_episode": int(args.max_steps_per_episode),
        "horizon_steps": int(args.horizon_steps),
        "segment_length": int(args.segment_length),
        "seeds": seeds,
        "re_temperatures": (
            [float(t) for t in args.re_temperatures.split(",")]
            if args.method == "replica_exchange_mc" else None
        ),
        "re_proposals_per_replica_per_round": int(args.re_proposals_per_replica_per_round),
        "sgcmc_target_pd_frac": float(args.sgcmc_target_pd_frac) if args.method == "sgcmc" else None,
        "sgcmc_delta_mu_cupd": float(args.sgcmc_delta_mu_cupd) if args.method == "sgcmc" else None,
        "bulk_pd_fraction": float(args.bulk_pd_fraction),
        "n_active_layers": int(args.n_active_layers),
        "oracle_mode": str(args.oracle_mode),
        "v1_2_locked_plan": "ProjectMain/week8/week8_locked_plan_v1_2.md",
    }
    (save_root / "meta.json").write_text(_json.dumps(meta, indent=2), encoding="utf-8")

    oracle = maybe_load_oracle_lazy(args)
    if oracle is None and not args.smoke:
        print("[Week8-MC] WARNING: --oracle-mode none — running on EMT fallback. "
              "Production runs MUST use --oracle-mode hybrid.", flush=True)

    summary_path = save_root / (
        "static_budget_summary.csv" if args.protocol == "static" else "dynamic_rollout_summary.csv"
    )
    fields = STATIC_SUMMARY_FIELDS if args.protocol == "static" else DYNAMIC_SUMMARY_FIELDS

    # Wipe stale summary so an interrupted prior run does not leave inconsistent rows.
    if summary_path.exists():
        summary_path.unlink()

    n_done = 0
    for seed in seeds:
        try:
            if args.protocol == "static":
                row, trace_rows = run_one_seed_static(args, seed=int(seed), oracle=oracle)
                _append_csv(summary_path, row, fields)
                if trace_rows:
                    _write_csv(save_root / "seeds" / f"{int(seed)}" / "running_omega_trace.csv", trace_rows)
                print(
                    f"[Week8-MC] {args.method} mu={args.mu_co} seed={seed} "
                    f"best_omega={row['best_omega']:.3f} "
                    f"calls={row['n_distinct_oracle_calls']} "
                    f"acc={row['accept_rate'] if not np.isnan(row['accept_rate']) else 'n/a'} "
                    f"wall={row['wall_clock_seconds']:.1f}s",
                    flush=True,
                )
            else:
                summary_rows, trace_rows = run_one_seed_dynamic(args, seed=int(seed), oracle=oracle)
                for r in summary_rows:
                    _append_csv(summary_path, r, fields)
                if trace_rows:
                    by_sched: Dict[str, List[Dict]] = {}
                    for tr in trace_rows:
                        by_sched.setdefault(tr["schedule"], []).append(tr)
                    for sched_id, srows in by_sched.items():
                        _write_csv(
                            save_root / "seeds" / f"{int(seed)}" / sched_id / "running_omega_trace.csv",
                            srows,
                        )
                for r in summary_rows:
                    print(
                        f"[Week8-MC] {args.method} schedule={r['schedule']} seed={seed} "
                        f"best_omega={r['best_omega']:.3f} mean={r['mean_omega']:.3f} "
                        f"calls={r['n_distinct_oracle_calls']} "
                        f"acc={r['accept_rate'] if not np.isnan(r['accept_rate']) else 'n/a'} "
                        f"wall={r['wall_clock_seconds']:.1f}s",
                        flush=True,
                    )
            n_done += 1
        except Exception as exc:  # pragma: no cover - best-effort logging
            print(f"[Week8-MC] ERROR seed={seed}: {type(exc).__name__}: {exc}", flush=True)
            import traceback as _tb
            _tb.print_exc()

    print(f"[Week8-MC] DONE: {n_done}/{len(seeds)} seeds finished. Summary: {summary_path}", flush=True)
    return 0


# =============================================================================
# Smoke test
# =============================================================================

def _smoke_test() -> int:
    print("=" * 64)
    print("week8_baselines_mc.py — smoke test (v1.2 §8 step 2)")
    print("=" * 64)

    # ---- 1. RandomSwapPolicy on static protocol --------------------
    print("\n[1/4] RandomSwapPolicy via run_static_protocol...")
    cfg = _build_smoke_env_config(mu_co=-0.6, seed=11, max_steps=32, action_mode="swap")
    env = ChemGymEnv(cfg)
    policy = RandomSwapPolicy(seed=42)
    res_rs = run_static_protocol(
        env, policy, total_oracle_budget=24, max_steps_per_episode=8, record_trace=False,
    )
    assert np.isfinite(res_rs["best_omega"]), "random_swap best_omega must be finite"
    assert res_rs["n_distinct_oracle_calls"] >= 8, (
        f"random_swap consumed too few oracle calls: {res_rs['n_distinct_oracle_calls']}"
    )
    print(
        f"    OK: best_omega={res_rs['best_omega']:.3f}, "
        f"n_distinct_calls={res_rs['n_distinct_oracle_calls']}, "
        f"n_episodes={res_rs['n_episodes']}"
    )

    # ---- 2. CanonicalMCSwapPolicy on static protocol ---------------
    print("\n[2/4] CanonicalMCSwapPolicy on static protocol...")
    cfg = _build_smoke_env_config(mu_co=-0.6, seed=11, max_steps=32, action_mode="swap")
    env = ChemGymEnv(cfg)
    mc_policy = CanonicalMCSwapPolicy(seed=42, T=0.10)
    res_mc = run_static_protocol(
        env, mc_policy, total_oracle_budget=24, max_steps_per_episode=8, record_trace=False,
    )
    assert np.isfinite(res_mc["best_omega"]), "canonical_mc best_omega must be finite"
    assert mc_policy.n_proposals > 0, "MC policy must have proposed at least once"
    accept_rate = mc_policy.n_accepts / max(mc_policy.n_proposals, 1)
    print(
        f"    OK: best_omega={res_mc['best_omega']:.3f}, "
        f"n_distinct_calls={res_mc['n_distinct_oracle_calls']}, "
        f"n_proposals={mc_policy.n_proposals}, accept_rate={accept_rate:.2f}"
    )

    # ---- 3. CanonicalMCSwapPolicy on dynamic protocol --------------
    print("\n[3/4] CanonicalMCSwapPolicy on dynamic protocol (E_swing, h=32 s=8)...")
    cfg_dyn = _build_smoke_env_config(mu_co=-0.2, seed=11, max_steps=64, action_mode="swap")
    env_dyn = ChemGymEnv(cfg_dyn)
    mc_policy_d = CanonicalMCSwapPolicy(seed=42, T=0.10)
    res_dyn = run_dynamic_protocol(
        env_dyn, mc_policy_d, schedule=TEST_SCHEDULES_V1_2["E_swing"],
        horizon_steps=32, segment_length=8, record_trace=False,
    )
    assert np.isfinite(res_dyn["mean_omega"]), "dynamic mean_omega must be finite"
    assert res_dyn["n_steps_total"] == 32
    assert mc_policy_d.n_proposals > 0
    print(
        f"    OK: mean_omega={res_dyn['mean_omega']:.3f}, "
        f"cumulative_omega={res_dyn['cumulative_omega']:.1f}, "
        f"n_proposals={mc_policy_d.n_proposals}, accept_rate="
        f"{mc_policy_d.n_accepts / max(mc_policy_d.n_proposals, 1):.2f}"
    )

    # ---- 4. RE-MC static + dynamic, 4 replicas -----------------------
    print("\n[4/4] run_replica_exchange_mc_static (4 replicas, default ladder, budget=24)...")
    envs = [
        ChemGymEnv(_build_smoke_env_config(mu_co=-0.6, seed=11 + k, max_steps=32, action_mode="swap"))
        for k in range(4)
    ]
    re_static = run_replica_exchange_mc_static(
        envs, temperatures=DEFAULT_RE_LADDER,
        total_pooled_oracle_budget=24,  # tiny budget to keep smoke fast
        max_steps_per_episode=8,
        proposals_per_replica_per_round=2,  # frequent exchange to force at least one
        seed=42,
    )
    assert np.isfinite(re_static["best_omega"]), "RE-MC best_omega must be finite"
    assert re_static["n_exchange_attempts"] >= 1, (
        f"RE-MC must attempt at least one exchange (got {re_static['n_exchange_attempts']})"
    )
    print(
        f"    OK [static]: best_omega={re_static['best_omega']:.3f} (replica {re_static['best_replica']}), "
        f"pooled_calls={re_static['n_pooled_distinct_oracle_calls']}, "
        f"exch={re_static['n_exchange_attempts']} attempts / "
        f"{re_static['n_exchange_accepts']} accepts"
    )

    print("\n[4b/4] run_replica_exchange_mc_dynamic (4 replicas, A_inc, h=32 s=8)...")
    envs_dyn = [
        ChemGymEnv(_build_smoke_env_config(mu_co=-0.2, seed=11 + k, max_steps=64, action_mode="swap"))
        for k in range(4)
    ]
    re_dyn = run_replica_exchange_mc_dynamic(
        envs_dyn, temperatures=DEFAULT_RE_LADDER,
        schedule=TEST_SCHEDULES_V1_2["A_inc"],
        horizon_steps=32, segment_length=8,
        proposals_per_replica_per_round=4,
        seed=42,
    )
    assert np.isfinite(re_dyn["best_omega"]), "RE-MC dynamic best_omega must be finite"
    assert len(re_dyn["best_omega_per_replica"]) == 4
    print(
        f"    OK [dynamic]: best_omega={re_dyn['best_omega']:.3f} (replica {re_dyn['best_replica']}), "
        f"per_replica_best={[f'{b:.1f}' for b in re_dyn['best_omega_per_replica']]}, "
        f"exch={re_dyn['n_exchange_attempts']}/{re_dyn['n_exchange_accepts']}"
    )

    # ---- 5. SGCMCMutationPolicy on static protocol -------------------
    print("\n[5/5] SGCMCMutationPolicy on static protocol (mutation mode)...")
    cfg_mut = _build_smoke_env_config(mu_co=-0.6, seed=11, max_steps=32, action_mode="mutation")
    env_mut = ChemGymEnv(cfg_mut)
    sgcmc = SGCMCMutationPolicy(
        seed=42, T=0.10, target_pd_frac=0.08, delta_mu_cupd=0.0, pd_elem_idx=1,
    )
    res_sg = run_static_protocol(
        env_mut, sgcmc, total_oracle_budget=24, max_steps_per_episode=8, record_trace=False,
    )
    assert np.isfinite(res_sg["best_omega"]), "SGCMC best_omega must be finite"
    assert sgcmc.n_proposals > 0
    n_pd_final = int(np.sum(env_mut.state == 1))
    print(
        f"    OK: best_omega={res_sg['best_omega']:.3f}, "
        f"n_proposals={sgcmc.n_proposals}, "
        f"accept_rate={sgcmc.n_accepts / max(sgcmc.n_proposals, 1):.2f}, "
        f"final_n_pd={n_pd_final}"
    )

    print("\n" + "=" * 64)
    print("ALL BASELINES PASS.")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    import sys
    # If any non-smoke CLI flag is present, dispatch to the production CLI.
    if any(a.startswith("--") and a not in {"--smoke"} for a in sys.argv[1:]):
        sys.exit(_cli_main())
    sys.exit(_smoke_test())
