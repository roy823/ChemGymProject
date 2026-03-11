from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import hashlib
import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:  # pragma: no cover
    raise ImportError("gymnasium is required for ChemGymEnv") from exc

try:
    from ase import Atoms
    from ase.build import fcc111
    from ase.calculators.emt import EMT
    from ase.constraints import FixAtoms
    from ase.visualize.plot import plot_atoms
except ImportError:  # pragma: no cover
    Atoms = None
    fcc111 = None
    EMT = None
    FixAtoms = None
    plot_atoms = None

from chem_gym.config import EnvConfig
from chem_gym.physics.co_placer import GreedyCOPlacer
from chem_gym.physics.analytical_prior import build_prior_constants
from chem_gym.physics.pid_lagrangian import PIDLagrangianConstraint
from chem_gym.physics.uma_shaping import UMAPotentialShaper


@dataclass(frozen=True)
class ActionSpec:
    """Compact helper for mutation/no-op action indexing."""

    n_sites: int
    n_elements: int
    enable_noop: bool

    @property
    def n_mutation_actions(self) -> int:
        return int(self.n_sites * self.n_elements)

    @property
    def noop_action_idx(self) -> Optional[int]:
        return int(self.n_mutation_actions) if self.enable_noop else None

    @property
    def action_dim(self) -> int:
        return int(self.n_mutation_actions + (1 if self.enable_noop else 0))

    def to_action(self, site_idx: int, elem_idx: int) -> int:
        return int(site_idx * self.n_elements + elem_idx)

    def to_indices(self, action: int) -> Tuple[int, int]:
        if int(action) < 0 or int(action) >= self.n_mutation_actions:
            return -1, -1
        return int(action // self.n_elements), int(action % self.n_elements)

    def is_explicit_noop(self, action: int) -> bool:
        return bool(self.enable_noop and int(action) == int(self.noop_action_idx))


class ChemGymEnv(gym.Env):
    """
    Operando Cu-Pd alloy environment with lightweight CO adsorption handling.

    First-week scope:
    - fixed Cu-Pd 4x4x4 (default config)
    - grand-potential reward
    - greedy CO placer with coverage and lateral repulsion constraints
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    LATTICE_CONSTANTS = {
        "Cu": 3.615,
        "Pd": 3.889,
        "Pt": 3.923,
    }

    def __init__(self, config: EnvConfig, surrogate=None, oracle=None):
        super().__init__()
        self.config = config
        self.surrogate = surrogate
        self.oracle = oracle

        self.rng = np.random.default_rng(config.init_seed)

        self.element_types = list(config.element_types)
        self.n_elements = len(self.element_types)
        if self.n_elements != 2:
            raise ValueError("This environment currently supports exactly 2 elements (Cu, Pd).")

        self.n_sites_per_layer = int(config.slab_size[0] * config.slab_size[1])
        self.n_total_atoms = int(self.n_sites_per_layer * config.n_layers)
        self.n_active_layers = int(config.n_active_layers)
        self.n_active_atoms = int(self.n_sites_per_layer * self.n_active_layers)
        self.active_start_idx = self.n_total_atoms - self.n_active_atoms

        self.max_deviation = (
            int(config.max_deviation)
            if config.max_deviation is not None
            else max(2, int(round(0.10 * self.n_active_atoms)))
        )
        self.target_pd_fraction = (
            float(config.bulk_pd_fraction)
            if config.bulk_pd_fraction is not None
            else float(np.mean(config.bulk_pd_fraction_range))
        )

        self.render_mode = config.render_mode
        self.e_cu_co = float(getattr(config, "e_cu_co", -0.55))
        self.e_pd_co = float(getattr(config, "e_pd_co", -1.35))
        prior_cfg = build_prior_constants(getattr(config, "physics_prior", {}) or {})
        # Enforce single-source adsorption constants from config fields.
        prior_cfg["e_cu_co"] = self.e_cu_co
        prior_cfg["e_pd_co"] = self.e_pd_co
        adsorption_energies = {
            "Cu": self.e_cu_co,
            "Pd": self.e_pd_co,
        }
        self.physics_prior = prior_cfg
        self.co_placer = GreedyCOPlacer(
            site_height=config.co_site_height,
            bond_length=config.co_bond_length,
            repulsion_distance=config.co_repulsion_distance_a,
            max_coverage=config.co_max_coverage,
            repulsion_strength_ev=config.co_repulsion_strength_ev,
            repulsion_sigma_a=config.co_repulsion_sigma_a,
            use_langmuir_target=config.co_use_langmuir_target,
            e_cu_co=self.e_cu_co,
            e_pd_co=self.e_pd_co,
            adsorption_energies_ev=adsorption_energies,
        )

        self.action_spec = ActionSpec(
            n_sites=self.n_active_atoms,
            n_elements=self.n_elements,
            enable_noop=bool(config.enable_noop_action),
        )
        # Backward-compatible public attributes for existing scripts.
        self.n_mutation_actions = self.action_spec.n_mutation_actions
        self.noop_action_idx = self.action_spec.noop_action_idx
        self.action_space = spaces.Discrete(self.action_spec.action_dim)

        if self.config.mode == "image":
            self.observation_space = spaces.Box(
                low=0.0,
                high=1.0,
                shape=(config.slab_size[0], config.slab_size[1], self.n_elements),
                dtype=np.float32,
            )
        elif self.config.mode == "graph":
            # one-hot(n_elements) + rel_pos(3) + layer + coord + avg_bond + debt(2)
            # + is_surface + co_load
            self.node_feat_dim = self.n_elements + 3 + 1 + 1 + 1 + self.n_elements + 1 + 1
            self.observation_space = spaces.Dict(
                {
                    "node_features": spaces.Box(
                        low=-100.0,
                        high=100.0,
                        shape=(self.n_total_atoms, self.node_feat_dim),
                        dtype=np.float32,
                    ),
                    "adjacency": spaces.Box(
                        low=0.0,
                        high=1.0,
                        shape=(self.n_total_atoms, self.n_total_atoms),
                        dtype=np.float32,
                    ),
                    "node_mask": spaces.Box(
                        low=0.0,
                        high=1.0,
                        shape=(self.n_total_atoms,),
                        dtype=np.float32,
                    ),
                }
            )
        else:
            raise ValueError(f"Unsupported mode: {self.config.mode}")

        self.state: Optional[np.ndarray] = None
        self.target_counts: Optional[np.ndarray] = None

        self.base_slab: Optional["Atoms"] = None
        self.atoms: Optional["Atoms"] = None
        self.atoms_with_co: Optional["Atoms"] = None

        self.current_energy_slab = 0.0
        self.current_energy_with_co = 0.0
        self.current_ads_energy = 0.0
        self.current_omega = 0.0
        self.prev_omega = 0.0
        self.min_omega_so_far = np.inf

        self.current_n_co = 0
        self.current_co_target_n = 0
        self.current_co_target_coverage = 0.0
        self.current_co_mean_delta_omega0 = 0.0
        self.current_debt = 0.0
        self.current_uncertainty = 0.0
        self.current_lambda = float(config.constraint_lambda_init)
        self.current_constraint_violation = 0.0
        self.current_d_frac = 0.0
        self.noop_count = 0
        self.mutation_count = 0
        self.last_reward_terms: Dict[str, float] = {}
        self.energy_backend_slab = "uninitialized"
        self.energy_backend_with_co = "uninitialized"
        self.energy_backend_ads = "uninitialized"

        self.steps = 0
        self.energy_cache: Dict[str, Tuple[float, float, str]] = {}
        self.composition_constraint = PIDLagrangianConstraint(
            x_target=self.target_pd_fraction,
            d_threshold=float(config.constraint_threshold_frac),
            lambda_init=float(config.constraint_lambda_init),
            kp=float(config.constraint_pid_kp),
            ki=float(config.constraint_pid_ki),
            kd=float(config.constraint_pid_kd),
            lambda_min=float(config.constraint_lambda_min),
            lambda_max=float(config.constraint_lambda_max),
            integral_clip=float(config.constraint_integral_clip),
        )
        self.uma_shaper = UMAPotentialShaper(
            energy_fn=self._get_uma_energy,
            gamma=float(config.uma_pbrs_gamma),
            scale=float(config.uma_pbrs_scale),
            weight=float(config.uma_pbrs_weight),
        )

        self._build_base_slab_once()

    # ---------------------- Public RL API ----------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.steps = 0

        self.target_counts = self._sample_target_counts()
        self.state = self._sample_state_from_target(self.target_counts)

        self.atoms = self._build_atoms_from_state()
        (
            self.atoms_with_co,
            self.current_n_co,
            _,
            self.current_co_target_n,
            self.current_co_target_coverage,
            self.current_co_mean_delta_omega0,
        ) = self._decorate_with_co(self.atoms)

        self.current_energy_slab, unc_slab = self._evaluate_slab_energy(self.atoms)
        self.current_energy_with_co, unc_with_co = self._evaluate_ads_system_energy(self.atoms_with_co)
        self.current_uncertainty = float(max(unc_slab, unc_with_co))

        self.current_ads_energy = self._evaluate_adsorption_energy(
            slab_atoms=self.atoms,
            atoms_with_co=self.atoms_with_co,
            n_co=self.current_n_co,
            precomputed_ads_system_energy=self.current_energy_with_co,
        )
        self.current_omega = self._compute_omega(
            self.current_energy_slab,
            self.current_ads_energy,
            self.current_n_co,
        )
        self.prev_omega = self.current_omega
        self.min_omega_so_far = self.current_omega

        self.current_debt = self._debt_l2(self.state)
        self.composition_constraint.reset_episode()
        self.current_lambda = float(self.composition_constraint.lam)
        self.current_constraint_violation, self.current_d_frac = self._compute_constraint_stats(self.state)
        self.noop_count = 0
        self.mutation_count = 0
        if bool(getattr(self.config, "use_uma_pbrs", True)):
            self.uma_shaper.reset(self.atoms)
        else:
            self.uma_shaper.reset(None)
        self.last_reward_terms = {
            "reward_total": 0.0,
            "delta_omega_term": 0.0,
            "delta_omega_raw": 0.0,
            "delta_omega_normalized": 0.0,
            "debt_delta_norm": 0.0,
            "debt_improve_term": 0.0,
            "debt_abs_term": 0.0,
            "constraint_penalty_term": 0.0,
            "constraint_penalty_raw": 0.0,
            "constraint_violation": float(self.current_constraint_violation),
            "constraint_lambda": float(self.current_lambda),
            "uma_shaping_term": 0.0,
            "step_penalty_term": float(-float(self.config.step_penalty)),
        }

        info = self._build_info(action_type="reset", delta_omega=0.0)
        return self._state_to_observation(), info

    def step(self, action: int):
        self.steps += 1
        debt_before = float(self.current_debt)
        is_noop = self.action_spec.is_explicit_noop(action)
        site_idx = -1
        target_elem_idx = -1

        if not is_noop:
            site_idx, target_elem_idx = self._action_to_indices(action)
            if site_idx < 0 or site_idx >= self.n_active_atoms:
                is_noop = True
            else:
                current_elem_idx = int(self.state[site_idx])
                if current_elem_idx == target_elem_idx:
                    is_noop = True

        if is_noop:
            delta_omega = 0.0
            reward, reward_terms = self._compose_reward_terms(
                delta_omega=delta_omega,
                debt_before=debt_before,
                debt_after=debt_before,
            )
            self.current_debt = debt_before
            self.noop_count += 1
            self.last_reward_terms = reward_terms

            terminated = not np.isfinite(reward)
            if terminated:
                reward = -100.0
            truncated = self.steps >= self.config.max_steps
            info = self._build_info(action_type="no_op", delta_omega=delta_omega)
            return self._state_to_observation(), float(reward), terminated, truncated, info

        self.state[site_idx] = target_elem_idx

        self.atoms = self._build_atoms_from_state()
        (
            self.atoms_with_co,
            self.current_n_co,
            _,
            self.current_co_target_n,
            self.current_co_target_coverage,
            self.current_co_mean_delta_omega0,
        ) = self._decorate_with_co(self.atoms)

        self.current_energy_slab, unc_slab = self._evaluate_slab_energy(self.atoms)
        self.current_energy_with_co, unc_with_co = self._evaluate_ads_system_energy(self.atoms_with_co)
        self.current_uncertainty = float(max(unc_slab, unc_with_co))

        self.current_ads_energy = self._evaluate_adsorption_energy(
            slab_atoms=self.atoms,
            atoms_with_co=self.atoms_with_co,
            n_co=self.current_n_co,
            precomputed_ads_system_energy=self.current_energy_with_co,
        )
        self.current_omega = self._compute_omega(
            self.current_energy_slab,
            self.current_ads_energy,
            self.current_n_co,
        )

        delta_omega = float(self.current_omega - self.prev_omega)
        current_debt = self._debt_l2(self.state)
        reward, reward_terms = self._compose_reward_terms(
            delta_omega=delta_omega,
            debt_before=debt_before,
            debt_after=float(current_debt),
        )
        self.last_reward_terms = reward_terms

        self.prev_omega = self.current_omega
        self.current_debt = current_debt
        self.mutation_count += 1
        if self.current_omega < self.min_omega_so_far:
            self.min_omega_so_far = self.current_omega

        terminated = not np.isfinite(reward)
        if terminated:
            reward = -100.0

        truncated = self.steps >= self.config.max_steps

        info = self._build_info(action_type="mutation", delta_omega=delta_omega)
        return self._state_to_observation(), float(reward), terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        """Mask only invalid actions by default; optional deviation envelope for ablation."""
        mask = np.ones(self.action_space.n, dtype=bool)
        for site_idx in range(self.n_active_atoms):
            cur_elem = int(self.state[site_idx])
            mask[self.action_spec.to_action(site_idx, cur_elem)] = False

        if self.noop_action_idx is not None:
            mask[self.noop_action_idx] = True

        if bool(getattr(self.config, "use_deviation_mask", False)):
            current_counts = np.bincount(self.state, minlength=self.n_elements)
            diffs = current_counts - self.target_counts

            for elem_idx in range(self.n_elements):
                diff = int(diffs[elem_idx])

                if diff <= -self.max_deviation:
                    affected_sites = np.where(self.state == elem_idx)[0]
                    for site_idx in affected_sites:
                        start = int(site_idx * self.n_elements)
                        mask[start : start + self.n_elements] = False

                if diff >= self.max_deviation:
                    mask[elem_idx : self.n_mutation_actions : self.n_elements] = False

        mutation_mask = mask[: self.n_mutation_actions]
        if not mutation_mask.any():
            mask[: self.n_mutation_actions] = True
            for site_idx in range(self.n_active_atoms):
                cur_elem = int(self.state[site_idx])
                mask[self.action_spec.to_action(site_idx, cur_elem)] = False
            if self.noop_action_idx is not None:
                mask[self.noop_action_idx] = True

        return mask

    def _normalized_debt(self, debt_value: float) -> float:
        denom = float(max(1, self.n_active_atoms * self.n_active_atoms))
        return float(debt_value / denom)

    def _constraint_penalty(self) -> Tuple[float, float, float, float]:
        n_pd = int(np.sum(self.state == self.element_types.index("Pd")))
        penalty_raw, lam, violation, d_frac = self.composition_constraint.compute_penalty(
            n_pd=n_pd,
            n_active=self.n_active_atoms,
            update_lambda=bool(getattr(self.config, "constraint_update_mode", "rollout") == "step"),
            gain=float(getattr(self.config, "constraint_rollout_gain", 1.0)),
        )
        penalty_weighted = float(getattr(self.config, "constraint_weight", 1.0)) * float(penalty_raw)
        self.current_lambda = float(lam)
        self.current_constraint_violation = float(violation)
        self.current_d_frac = float(d_frac)
        return float(penalty_weighted), float(penalty_raw), float(lam), float(violation)

    def _compose_reward_terms(
        self,
        delta_omega: float,
        debt_before: float,
        debt_after: float,
    ) -> Tuple[float, Dict[str, float]]:
        delta_omega_norm = float(delta_omega / max(1e-6, float(self.config.delta_omega_scale)))
        delta_omega_linear = -float(self.config.omega_reward_scale) * delta_omega_norm
        reward_clip = float(max(1e-6, getattr(self.config, "linear_reward_clip", 3.0)))
        delta_omega_term = float(np.clip(delta_omega_linear, -reward_clip, reward_clip))

        debt_before_norm = self._normalized_debt(float(debt_before))
        debt_after_norm = self._normalized_debt(float(debt_after))
        debt_delta_norm = float(debt_before_norm - debt_after_norm)
        debt_improve_term = float(self.config.debt_improvement_scale) * debt_delta_norm
        debt_abs_term = -float(self.config.debt_abs_penalty) * debt_after_norm

        penalty_weighted, penalty_raw, lam, violation = self._constraint_penalty()

        uma_shaping = (
            float(self.uma_shaper.transition(self.atoms))
            if bool(getattr(self.config, "use_uma_pbrs", True))
            else 0.0
        )
        step_penalty_term = -float(self.config.step_penalty)
        reward = float(
            delta_omega_term
            + debt_improve_term
            + debt_abs_term
            - penalty_weighted
            + uma_shaping
            + step_penalty_term
        )

        terms = {
            "reward_total": float(reward),
            "delta_omega_term": float(delta_omega_term),
            "delta_omega_raw": float(delta_omega),
            "delta_omega_normalized": float(delta_omega_norm),
            "debt_delta_norm": float(debt_delta_norm),
            "debt_improve_term": float(debt_improve_term),
            "debt_abs_term": float(debt_abs_term),
            "constraint_penalty_term": float(-penalty_weighted),
            "constraint_penalty_raw": float(-penalty_raw),
            "constraint_violation": float(violation),
            "constraint_lambda": float(lam),
            "uma_shaping_term": float(uma_shaping),
            "step_penalty_term": float(step_penalty_term),
        }
        return float(reward), terms

    def render(self):
        if self.render_mode is None:
            return None
        if plot_atoms is None or self.atoms_with_co is None:
            return None
        fig = plot_atoms(self.atoms_with_co, show_unit_cell=0, rotation="-90x")
        if self.render_mode == "human":
            return fig
        if self.render_mode == "rgb_array":
            fig.canvas.draw()
            data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            w, h = fig.canvas.get_width_height()
            return data.reshape((h, w, 3))
        return None

    # ---------------------- Core physics ----------------------
    def _build_base_slab_once(self) -> None:
        if fcc111 is None:
            self.base_slab = None
            return

        substrate = str(self.config.substrate_element)
        if substrate not in self.LATTICE_CONSTANTS:
            substrate = "Cu"

        a0 = float(self.LATTICE_CONSTANTS[substrate])
        slab = fcc111(
            substrate,
            size=(self.config.slab_size[0], self.config.slab_size[1], self.config.n_layers),
            a=a0,
            vacuum=10.0,
        )
        slab.set_pbc(True)

        n_fixed = self.n_total_atoms - self.n_active_atoms
        if FixAtoms is not None and n_fixed > 0:
            slab.set_constraint(FixAtoms(indices=list(range(n_fixed))))

        self.base_slab = slab

    def _build_atoms_from_state(self) -> Optional["Atoms"]:
        if self.base_slab is None or self.state is None:
            return None

        atoms = self.base_slab.copy()
        symbols = atoms.get_chemical_symbols()

        for k, elem_idx in enumerate(self.state):
            symbols[self.active_start_idx + k] = self.element_types[int(elem_idx)]

        atoms.set_chemical_symbols(symbols)
        return atoms

    def _decorate_with_co(
        self,
        slab_atoms: Optional["Atoms"],
    ) -> Tuple[Optional["Atoms"], int, List[int], int, float, float]:
        if slab_atoms is None:
            return None, 0, [], 0, 0.0, 0.0

        if not self.config.enable_co_adsorption:
            return slab_atoms.copy(), 0, [], 0, 0.0, 0.0

        surface_indices = self._get_surface_indices(slab_atoms)
        placement = self.co_placer.place(
            slab_atoms,
            surface_indices,
            mu_co=float(self.config.mu_co),
            temperature_k=float(self.config.co_temperature_k),
        )
        return (
            placement.atoms_with_co,
            placement.n_co,
            placement.occupied_site_indices,
            placement.target_n_co,
            placement.target_coverage,
            placement.mean_initial_delta_omega,
        )

    def _get_surface_indices(self, atoms: "Atoms") -> List[int]:
        positions = atoms.get_positions()
        active_indices = np.arange(self.active_start_idx, self.n_total_atoms)
        z_active = positions[active_indices, 2]
        z_max = float(np.max(z_active))
        tol = float(self.config.co_surface_z_tol)

        surface = [
            int(idx)
            for idx in active_indices
            if positions[idx, 2] >= (z_max - tol)
        ]
        return surface

    def _evaluate_slab_energy(self, atoms: Optional["Atoms"]) -> Tuple[float, float]:
        if bool(getattr(self.config, "thermo_consistent_backend", False)):
            # Thermodynamically consistent mode:
            # evaluate slab reference in the same backend family used for adsorbate systems.
            energy, unc = self._evaluate_energy(atoms, role="ads_reference")
        else:
            energy, unc = self._evaluate_energy(atoms, role="slab")
        return energy, unc

    def _evaluate_ads_system_energy(self, atoms: Optional["Atoms"]) -> Tuple[float, float]:
        energy, unc = self._evaluate_energy(atoms, role="ads_system")
        return energy, unc

    def _evaluate_adsorption_energy(
        self,
        slab_atoms: Optional["Atoms"],
        atoms_with_co: Optional["Atoms"],
        n_co: int,
        precomputed_ads_system_energy: Optional[float] = None,
    ) -> float:
        if slab_atoms is None or atoms_with_co is None:
            self.energy_backend_ads = "none"
            return 0.0

        co_ref = self._co_reference_energy()
        n_co = int(max(0, n_co))

        # Preferred path: oracle-provided adsorption-energy API.
        if self.oracle is not None and hasattr(self.oracle, "compute_adsorption_energy"):
            try:
                e_ads = float(
                    self.oracle.compute_adsorption_energy(
                        slab_atoms=slab_atoms,
                        ads_atoms=atoms_with_co,
                        n_co=n_co,
                        co_reference_energy=co_ref,
                        relax=False,
                        precomputed_ads_system_energy=precomputed_ads_system_energy,
                    )
                )
                self.energy_backend_ads = "oracle:compute_adsorption_energy"
                return e_ads
            except Exception:
                pass

        # Fallback: same-backend subtraction for consistency.
        if precomputed_ads_system_energy is None:
            e_ads_sys, _ = self._evaluate_energy(atoms_with_co, role="ads_system")
        else:
            e_ads_sys = float(precomputed_ads_system_energy)
        e_slab_ads_ref, _ = self._evaluate_energy(slab_atoms, role="ads_reference")
        self.energy_backend_ads = f"derived:{self.energy_backend_with_co}-minus-ads_reference"
        return float(e_ads_sys - e_slab_ads_ref - n_co * co_ref)

    def _evaluate_energy(self, atoms: Optional["Atoms"], role: str = "generic") -> Tuple[float, float]:
        if atoms is None:
            return 0.0, 0.0

        key = f"{role}:{self._hash_atoms(atoms)}"
        if key in self.energy_cache:
            energy, unc, backend = self.energy_cache[key]
            self._record_backend(role, backend)
            return energy, unc

        # 1) High-fidelity oracle
        if self.oracle is not None:
            if role == "slab" and hasattr(self.oracle, "compute_slab_energy"):
                try:
                    energy = float(self.oracle.compute_slab_energy(atoms, relax=False))
                    result = (energy, 0.0, "oracle:compute_slab_energy")
                    self.energy_cache[key] = result
                    self._record_backend(role, result[2])
                    return result[0], result[1]
                except Exception:
                    pass

            if role in {"ads_system", "ads_reference"} and hasattr(self.oracle, "evaluate_adsorbate_system_energy"):
                try:
                    mean_e, std_e = self.oracle.evaluate_adsorbate_system_energy(atoms, relax=False)
                    result = (float(mean_e), float(std_e), "oracle:evaluate_adsorbate_system_energy")
                    self.energy_cache[key] = result
                    self._record_backend(role, result[2])
                    return result[0], result[1]
                except Exception:
                    pass

            if role in {"ads_system", "ads_reference"} and hasattr(self.oracle, "compute_adsorbate_system_energy"):
                try:
                    energy = float(self.oracle.compute_adsorbate_system_energy(atoms, relax=False))
                    result = (energy, 0.0, "oracle:compute_adsorbate_system_energy")
                    self.energy_cache[key] = result
                    self._record_backend(role, result[2])
                    return result[0], result[1]
                except Exception:
                    pass

            if hasattr(self.oracle, "evaluate_energy"):
                try:
                    mean_e, std_e = self.oracle.evaluate_energy(atoms, relax=False)
                    result = (float(mean_e), float(std_e), "oracle:evaluate_energy")
                    self.energy_cache[key] = result
                    self._record_backend(role, result[2])
                    return result[0], result[1]
                except Exception:
                    pass

            if hasattr(self.oracle, "compute_total_energy"):
                try:
                    energy = float(self.oracle.compute_total_energy(atoms, relax=False))
                    result = (energy, 0.0, "oracle:compute_total_energy")
                    self.energy_cache[key] = result
                    self._record_backend(role, result[2])
                    return result[0], result[1]
                except Exception:
                    pass

            if hasattr(self.oracle, "compute_energy"):
                try:
                    energy = float(self.oracle.compute_energy(atoms, relax=False))
                    result = (energy, 0.0, "oracle:compute_energy")
                    self.energy_cache[key] = result
                    self._record_backend(role, result[2])
                    return result[0], result[1]
                except Exception:
                    pass

        # 2) Surrogate ensemble
        if self.surrogate is not None and hasattr(self.surrogate, "evaluate"):
            try:
                mean_e, std_e = self.surrogate.evaluate(atoms)
                result = (float(mean_e), float(std_e), "surrogate:ensemble")
                self.energy_cache[key] = result
                self._record_backend(role, result[2])
                return result[0], result[1]
            except Exception:
                pass

        # 3) EMT fallback (cheap but not physically faithful for adsorbates)
        if EMT is not None:
            try:
                atoms_calc = atoms.copy()
                atoms_calc.calc = EMT()
                energy = float(atoms_calc.get_potential_energy())
                result = (energy, 0.0, "emt")
                self.energy_cache[key] = result
                self._record_backend(role, result[2])
                return result[0], result[1]
            except Exception:
                pass

        result = (0.0, 0.0, "zero")
        self.energy_cache[key] = result
        self._record_backend(role, result[2])
        return result[0], result[1]

    def _compute_omega(self, slab_energy: float, ads_energy: float, n_co: int) -> float:
        # Omega = E_slab + E_ads - mu_CO * N_CO
        return float(slab_energy + ads_energy - float(self.config.mu_co) * float(n_co))

    def _co_reference_energy(self) -> float:
        if bool(self.config.use_relative_mu_co):
            # Relative chemical potential convention:
            # E_ads = E(slab+CO) - E(slab), and mu_co absorbs gas-phase reference.
            return 0.0
        return float(self.config.co_gas_ref_energy)

    def _record_backend(self, role: str, backend: str) -> None:
        if role in {"slab", "ads_reference"}:
            self.energy_backend_slab = backend
            return
        if role == "ads_system":
            self.energy_backend_with_co = backend

    # ---------------------- Observation helpers ----------------------
    def _state_to_observation(self):
        if self.config.mode == "image":
            top_layer_state = self.state[-self.n_sites_per_layer :]
            flat_one_hot = np.eye(self.n_elements, dtype=np.float32)[top_layer_state]
            return flat_one_hot.reshape(self.config.slab_size[0], self.config.slab_size[1], self.n_elements)

        if self.atoms is None:
            return {
                "node_features": np.zeros((self.n_total_atoms, self.node_feat_dim), dtype=np.float32),
                "adjacency": np.zeros((self.n_total_atoms, self.n_total_atoms), dtype=np.float32),
                "node_mask": np.zeros((self.n_total_atoms,), dtype=np.float32),
            }

        positions = self.atoms.get_positions().astype(np.float32)
        symbols = self.atoms.get_chemical_symbols()

        dist_matrix = self.atoms.get_all_distances(mic=True).astype(np.float32)
        cutoff = float(self.config.graph_cutoff)
        sigma = max(float(self.config.graph_sigma), 1e-6)

        adjacency = np.zeros_like(dist_matrix, dtype=np.float32)
        edge_mask = (dist_matrix > 0.0) & (dist_matrix <= cutoff)
        adjacency[edge_mask] = np.exp(-np.square(dist_matrix[edge_mask] / sigma))
        np.fill_diagonal(adjacency, 1.0)

        coord_num = np.sum(edge_mask, axis=1, dtype=np.float32)

        avg_bond = np.zeros(self.n_total_atoms, dtype=np.float32)
        for i in range(self.n_total_atoms):
            neigh = dist_matrix[i][edge_mask[i]]
            avg_bond[i] = float(np.mean(neigh)) if len(neigh) > 0 else 0.0

        current_counts = np.bincount(self.state, minlength=self.n_elements)
        debt_vec = (self.target_counts - current_counts).astype(np.float32)

        z_all = positions[:, 2]
        z_active_max = float(np.max(z_all[self.active_start_idx :]))
        is_surface = (z_all >= (z_active_max - float(self.config.co_surface_z_tol))).astype(np.float32)

        co_load = 0.0
        max_surface = max(1, len(self._get_surface_indices(self.atoms)))
        co_load = float(self.current_n_co) / float(max_surface)

        center = positions.mean(axis=0, keepdims=True)
        rel_pos = positions - center

        node_features = np.zeros((self.n_total_atoms, self.node_feat_dim), dtype=np.float32)
        node_mask = np.ones((self.n_total_atoms,), dtype=np.float32)

        for i in range(self.n_total_atoms):
            one_hot = np.zeros((self.n_elements,), dtype=np.float32)
            symbol = symbols[i]
            if symbol in self.element_types:
                one_hot[self.element_types.index(symbol)] = 1.0

            layer_idx = i // self.n_sites_per_layer
            layer_norm = float(layer_idx) / max(1.0, float(self.config.n_layers - 1))

            node_features[i] = np.concatenate(
                [
                    one_hot,
                    rel_pos[i],
                    np.array([layer_norm], dtype=np.float32),
                    np.array([coord_num[i] / 12.0], dtype=np.float32),
                    np.array([avg_bond[i]], dtype=np.float32),
                    debt_vec,
                    np.array([is_surface[i]], dtype=np.float32),
                    np.array([co_load], dtype=np.float32),
                ]
            )

        return {
            "node_features": node_features,
            "adjacency": adjacency,
            "node_mask": node_mask,
        }

    # ---------------------- Utility ----------------------
    def _sample_target_counts(self) -> np.ndarray:
        if self.config.bulk_pd_fraction is not None:
            frac_pd = float(self.config.bulk_pd_fraction)
        else:
            lo, hi = self.config.bulk_pd_fraction_range
            frac_pd = float(self.rng.uniform(lo, hi))

        frac_pd = float(np.clip(frac_pd, 0.0, 1.0))
        pd_count = int(round(frac_pd * self.n_active_atoms))
        pd_count = int(np.clip(pd_count, 0, self.n_active_atoms))
        cu_count = self.n_active_atoms - pd_count

        # assume element_types contains Cu and Pd
        counts = np.zeros(self.n_elements, dtype=np.int32)
        counts[self.element_types.index("Cu")] = cu_count
        counts[self.element_types.index("Pd")] = pd_count
        return counts

    def _sample_state_from_target(self, target_counts: np.ndarray) -> np.ndarray:
        state = []
        for elem_idx in range(self.n_elements):
            state.extend([elem_idx] * int(target_counts[elem_idx]))
        state_arr = np.array(state, dtype=np.int32)
        self.rng.shuffle(state_arr)
        return state_arr

    def _action_to_indices(self, action: int) -> Tuple[int, int]:
        return self.action_spec.to_indices(action)

    def _compute_constraint_stats(self, state: np.ndarray) -> Tuple[float, float]:
        pd_idx = self.element_types.index("Pd")
        n_pd = int(np.sum(state == pd_idx))
        violation, d_frac = self.composition_constraint.violation(
            n_pd=n_pd,
            n_active=self.n_active_atoms,
        )
        return float(violation), float(d_frac)

    def update_constraint_lambda(self, mean_violation: float) -> float:
        gain = float(getattr(self.config, "constraint_rollout_gain", 1.0))
        lam = self.composition_constraint.update_from_violation(float(mean_violation), gain=gain)
        self.current_lambda = float(lam)
        return float(lam)

    def get_constraint_lambda(self) -> float:
        return float(self.composition_constraint.lam)

    def _get_uma_energy(self, atoms: Optional["Atoms"]) -> float:
        if not bool(getattr(self.config, "use_uma_pbrs", True)):
            return 0.0
        if atoms is None or self.oracle is None:
            return 0.0
        if not hasattr(self.oracle, "compute_slab_energy"):
            return 0.0
        try:
            return float(self.oracle.compute_slab_energy(atoms, relax=False))
        except Exception:
            return 0.0

    def _debt_l2(self, state: np.ndarray) -> float:
        current_counts = np.bincount(state, minlength=self.n_elements)
        diff = current_counts - self.target_counts
        return float(np.sum(np.square(diff)))

    def _hash_atoms(self, atoms: "Atoms") -> str:
        symbols = ",".join(atoms.get_chemical_symbols())
        positions = np.round(atoms.get_positions(), 3)
        payload = f"{symbols}|{positions.tobytes()}"
        return hashlib.md5(payload.encode("latin1", errors="ignore")).hexdigest()

    def _surface_pd_coverage(self) -> float:
        if self.atoms is None:
            return 0.0

        surface_indices = self._get_surface_indices(self.atoms)
        if len(surface_indices) == 0:
            return 0.0

        pd_symbol = "Pd"
        symbols = self.atoms.get_chemical_symbols()
        n_pd = sum(1 for idx in surface_indices if symbols[idx] == pd_symbol)
        return float(n_pd) / float(len(surface_indices))

    def _build_info(self, action_type: str, delta_omega: float) -> Dict:
        return {
            "energy": float(self.current_omega),
            "omega": float(self.current_omega),
            "delta_omega": float(delta_omega),
            "energy_slab": float(self.current_energy_slab),
            "energy_with_co": float(self.current_energy_with_co),
            "energy_ads": float(self.current_ads_energy),
            "energy_backend_slab": self.energy_backend_slab,
            "energy_backend_with_co": self.energy_backend_with_co,
            "energy_backend_ads": self.energy_backend_ads,
            "n_co": int(self.current_n_co),
            "co_target_n": int(self.current_co_target_n),
            "co_target_coverage": float(self.current_co_target_coverage),
            "co_mean_initial_delta_omega": float(self.current_co_mean_delta_omega0),
            "mu_co": float(self.config.mu_co),
            "mu_co_is_effective": bool(self.config.mu_co_is_effective),
            "thermo_consistent_backend": bool(getattr(self.config, "thermo_consistent_backend", False)),
            "co_temperature_k": float(self.config.co_temperature_k),
            "co_partial_pressure_pa": None
            if self.config.co_partial_pressure_pa is None
            else float(self.config.co_partial_pressure_pa),
            "e_cu_co": float(self.e_cu_co),
            "e_pd_co": float(self.e_pd_co),
            "stoich_debt": float(self.current_debt),
            "constraint_lambda": float(self.current_lambda),
            "constraint_violation": float(self.current_constraint_violation),
            "constraint_d_frac": float(self.current_d_frac),
            "pd_surface_coverage": float(self._surface_pd_coverage()),
            "uncertainty": float(self.current_uncertainty),
            "action_type": action_type,
            "noop_count": int(self.noop_count),
            "mutation_count": int(self.mutation_count),
            "noop_ratio": float(self.noop_count / max(1, self.noop_count + self.mutation_count)),
            "atoms": self.atoms_with_co,
            "metal_atoms": self.atoms,
            "reward_terms": dict(self.last_reward_terms),
        }
