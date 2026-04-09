"""
Centralized configuration dataclasses used across the Chem-Gym scaffold.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class GraphFeatureLayout:
    """Index map for the per-node feature vector built in ChemGymEnv.

    Feature layout (concatenated in this order):
        one_hot      : n_elements       (element identity)
        rel_pos      : 3                (relative position to slab center)
        layer_norm   : 1                (normalized layer index)
        coord_num    : 1                (coordination number / 12)
        avg_bond     : 1                (mean neighbor distance)
        debt_vec     : n_elements       (target - current element counts)
        is_surface   : 1                (binary surface flag)
        co_load      : 1                (CO coverage fraction)

    This is the single source of truth shared by chem_env and analytical_prior.
    """
    n_elements: int = 2

    @property
    def one_hot_start(self) -> int:
        return 0

    @property
    def rel_pos_start(self) -> int:
        return self.n_elements

    @property
    def layer_norm_idx(self) -> int:
        return self.n_elements + 3

    @property
    def coord_num_idx(self) -> int:
        return self.n_elements + 4

    @property
    def avg_bond_idx(self) -> int:
        return self.n_elements + 5

    @property
    def debt_vec_start(self) -> int:
        return self.n_elements + 6

    @property
    def is_surface_idx(self) -> int:
        return 2 * self.n_elements + 6

    @property
    def co_load_idx(self) -> int:
        return 2 * self.n_elements + 7

    @property
    def total_dim(self) -> int:
        return 2 * self.n_elements + 8


@dataclass
class RewardConfig:
    """Reward assembly parameters."""
    mu_co: float = -1.0
    mu_co_is_effective: bool = False
    omega_reward_scale: float = 1.0
    delta_omega_scale: float = 1.0
    debt_improvement_scale: float = 0.1
    debt_abs_penalty: float = 0.0
    reward_shift: float = 0.1
    linear_reward_clip: float = 3.0
    thermo_consistent_backend: bool = True
    step_penalty: float = 0.0
    reward_profile: str = "delta_omega_plus_pbrs"


@dataclass
class ConstraintConfig:
    """PID-Lagrangian soft composition constraint."""
    constraint_threshold_frac: float = 0.12
    constraint_weight: float = 1.0
    constraint_lambda_init: float = 1.0
    constraint_lambda_min: float = 0.0
    constraint_lambda_max: float = 10.0
    constraint_pid_kp: float = 0.10
    constraint_pid_ki: float = 0.01
    constraint_pid_kd: float = 0.01
    constraint_integral_clip: float = 100.0
    constraint_update_mode: str = "rollout"
    constraint_rollout_gain: float = 1.0


@dataclass
class COAdsorptionConfig:
    """Lightweight CO adsorption heuristic parameters."""
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
    e_cu_co: float = -0.55
    e_pd_co: float = -1.35


@dataclass
class UMAPBRSConfig:
    """UMA potential-based reward shaping."""
    use_uma_pbrs: bool = True
    uma_pbrs_gamma: float = 0.97
    uma_pbrs_scale: float = 50.0
    uma_pbrs_weight: float = 1.0


@dataclass
class EnvConfig:
    # --- Slab / graph topology ---
    mode: str = "graph"  # "image" or "graph"
    element_types: List[str] = field(default_factory=lambda: ["Cu", "Pd"])
    slab_size: Tuple[int, int] = (4, 4)
    n_layers: int = 4
    n_active_layers: int = 4
    graph_cutoff: float = 6.0
    graph_sigma: float = 2.0
    max_steps: int = 800
    init_seed: Optional[int] = None
    render_mode: Optional[str] = None
    use_edge_features: bool = True
    n_rbf: int = 10
    rbf_cutoff: float = 6.0
    substrate_element: str = "Cu"
    bulk_pd_fraction: Optional[float] = 0.08
    bulk_pd_fraction_range: Tuple[float, float] = (0.05, 0.10)
    max_deviation: Optional[int] = None

    # --- Action space ---
    action_mode: str = "mutation"
    enable_noop_action: bool = True
    stop_terminates: bool = False
    min_stop_steps: int = 0
    use_deviation_mask: bool = False

    # --- Sub-configs (fields exposed for direct access via __post_init__) ---
    reward: RewardConfig = field(default_factory=RewardConfig)
    constraint: ConstraintConfig = field(default_factory=ConstraintConfig)
    co_adsorption: COAdsorptionConfig = field(default_factory=COAdsorptionConfig)
    uma_pbrs: UMAPBRSConfig = field(default_factory=UMAPBRSConfig)

    # Analytical physical prior constants for PIRP
    physics_prior: Dict[str, float] = field(
        default_factory=lambda: {
            "gamma_cu": 1.0,
            "gamma_pd": 1.3,
            "strain_coeff": 1.0,
            "e_cu_co": -0.55,
            "e_pd_co": -1.35,
        }
    )

    def __post_init__(self) -> None:
        # Flatten sub-config fields onto self for backward compatibility.
        # All existing code using config.mu_co, config.constraint_weight, etc.
        # continues to work without changes.
        for sub in (self.reward, self.constraint, self.co_adsorption, self.uma_pbrs):
            for f_name in sub.__dataclass_fields__:
                if not hasattr(self, f_name):
                    object.__setattr__(self, f_name, getattr(sub, f_name))

        # Keep physics_prior adsorption constants synchronized with CO config.
        prior = dict(self.physics_prior or {})
        e_cu_co = float(self.co_adsorption.e_cu_co)
        e_pd_co = float(self.co_adsorption.e_pd_co)
        if "e_cu_co" in prior and prior["e_cu_co"] is not None:
            e_cu_co = float(prior["e_cu_co"])
        if "e_pd_co" in prior and prior["e_pd_co"] is not None:
            e_pd_co = float(prior["e_pd_co"])
        prior["e_cu_co"] = e_cu_co
        prior["e_pd_co"] = e_pd_co
        self.physics_prior = prior
        # Expose as top-level for convenience
        object.__setattr__(self, "e_cu_co", e_cu_co)
        object.__setattr__(self, "e_pd_co", e_pd_co)


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
