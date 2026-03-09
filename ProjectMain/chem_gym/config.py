"""
Centralized configuration dataclasses used across the Chem-Gym scaffold.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class EnvConfig:
    mode: str = "graph"  # "image" or "graph"
    element_types: List[str] = field(default_factory=lambda: ["Cu", "Pd"])
    slab_size: Tuple[int, int] = (4, 4)
    n_layers: int = 4
    n_active_layers: int = 4
    graph_cutoff: float = 6.0
    graph_sigma: float = 2.0
    max_steps: int = 800
    step_penalty: float = 0.0
    init_seed: Optional[int] = None
    render_mode: Optional[str] = None  # "human" or "rgb_array"
    use_edge_features: bool = True
    n_rbf: int = 10
    rbf_cutoff: float = 6.0

    # Cu-Pd composition control
    substrate_element: str = "Cu"
    bulk_pd_fraction: Optional[float] = 0.08
    bulk_pd_fraction_range: Tuple[float, float] = (0.05, 0.10)
    max_deviation: Optional[int] = None

    # Reward core terms.
    mu_co: float = -1.0
    # If provided by higher-level launcher from (T, p), mu_co should be this effective value.
    mu_co_is_effective: bool = False
    omega_reward_scale: float = 1.0
    delta_omega_scale: float = 1.0
    # Debt shaping terms (normalized by n_active_atoms^2 inside env).
    debt_improvement_scale: float = 0.1
    debt_abs_penalty: float = 0.0
    # Legacy field kept for compatibility; currently unused in reward assembly.
    reward_shift: float = 0.1
    # Reward v2: linear physical main term with clipping.
    linear_reward_clip: float = 3.0
    # If True, evaluate slab reference energy in the same adsorbate-capable backend
    # as E(slab+CO), avoiding cross-backend mismatch in Omega.
    thermo_consistent_backend: bool = True

    # Action-space and masking behavior
    enable_noop_action: bool = True
    use_deviation_mask: bool = False

    # PID-Lagrangian soft composition constraint
    constraint_threshold_frac: float = 0.12
    constraint_weight: float = 1.0
    constraint_lambda_init: float = 1.0
    constraint_lambda_min: float = 0.0
    constraint_lambda_max: float = 10.0
    constraint_pid_kp: float = 0.10
    constraint_pid_ki: float = 0.01
    constraint_pid_kd: float = 0.01
    constraint_integral_clip: float = 100.0
    # Lambda update schedule:
    # - "step": update lambda at every env step
    # - "rollout": update lambda once per rollout in trainer callback
    # - "frozen": keep lambda fixed
    constraint_update_mode: str = "rollout"
    constraint_rollout_gain: float = 1.0

    # UMA potential-based reward shaping (PBRS)
    use_uma_pbrs: bool = True
    uma_pbrs_gamma: float = 0.97
    uma_pbrs_scale: float = 50.0
    uma_pbrs_weight: float = 1.0

    # Lightweight CO adsorption heuristic
    enable_co_adsorption: bool = True
    co_max_coverage: float = 1.0
    co_site_height: float = 1.85
    co_bond_length: float = 1.15
    co_gas_ref_energy: float = 0.0
    co_surface_z_tol: float = 0.35
    use_relative_mu_co: bool = True
    co_temperature_k: float = 300.0
    co_ref_pressure_pa: float = 101325.0
    co_partial_pressure_pa: Optional[float] = None
    co_mu_ref_ev: float = -1.0
    co_use_langmuir_target: bool = True
    co_repulsion_distance_a: float = 3.0
    co_repulsion_strength_ev: float = 0.15
    co_repulsion_sigma_a: float = 2.0

    # Single source of truth for site-level CO adsorption constants (eV).
    e_cu_co: float = -0.55
    e_pd_co: float = -1.35

    # Analytical physical prior constants for PIRP
    physics_prior: Dict[str, float] = field(
        default_factory=lambda: {
            "gamma_cu": 1.0,
            "gamma_pd": 1.3,
            "strain_coeff": 1.0,
            # These are synchronized with e_cu_co/e_pd_co in __post_init__.
            "e_cu_co": -0.55,
            "e_pd_co": -1.35,
        }
    )

    def __post_init__(self) -> None:
        # Keep physics_prior adsorption constants synchronized with explicit fields.
        prior = dict(self.physics_prior or {})
        if "e_cu_co" in prior and prior["e_cu_co"] is not None:
            self.e_cu_co = float(prior["e_cu_co"])
        if "e_pd_co" in prior and prior["e_pd_co"] is not None:
            self.e_pd_co = float(prior["e_pd_co"])
        prior["e_cu_co"] = float(self.e_cu_co)
        prior["e_pd_co"] = float(self.e_pd_co)
        self.physics_prior = prior


@dataclass
class TrainConfig:
    total_timesteps: int = 200_000
    n_envs: int = 1
    learning_rate: float = 3e-4
    gamma: float = 0.995
    lam: float = 0.95
    device: str = "cuda"
    uncertainty_penalty: float = 0.0
    oracle_threshold: Optional[float] = None
    oracle_fmax: float = 0.05
    oracle_max_steps: int = 100
    oracle_disable_amp: bool = True

    # PPO optimization knobs
    ppo_n_steps: int = 512
    ppo_batch_size: int = 128
    ppo_clip_range: float = 0.2
    ppo_ent_coef: float = 0.001

    # PIRP knobs
    use_pirp: bool = False
    pirp_scale: float = 0.02
    noop_logit_bonus: float = 0.0
    pirp_final_scale: float = 0.005
    pirp_anneal_fraction: float = 0.5
    pirp_anneal_schedule: str = "cosine"

    # Diagnostics / visualization
    enable_visualization: bool = False


@dataclass
class SurrogateConfig:
    n_models: int = 3
    seeds: Optional[List[int]] = None  # None -> range(n_models)
    mean_energy: float = -1.0
    noise_scale: float = 0.1
