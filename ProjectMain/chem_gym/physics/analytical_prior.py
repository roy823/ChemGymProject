from __future__ import annotations

from typing import Dict, Optional

import torch

from chem_gym.config import EnvConfig, GraphFeatureLayout

_CFG_DEFAULTS = EnvConfig()

DEFAULT_PRIOR_CONSTANTS: Dict[str, float] = {
    "gamma_cu": 1.0,
    "gamma_pd": 1.3,
    "strain_coeff": 1.0,
    # Single-source defaults from EnvConfig.
    "e_cu_co": float(_CFG_DEFAULTS.e_cu_co),
    "e_pd_co": float(_CFG_DEFAULTS.e_pd_co),
}


def build_prior_constants(overrides: Optional[Dict[str, float]]) -> Dict[str, float]:
    cfg = dict(DEFAULT_PRIOR_CONSTANTS)
    if overrides:
        for key, value in overrides.items():
            if key in cfg and value is not None:
                cfg[key] = float(value)
    return cfg


def compute_potential_adsorption_drive(
    active_node_features: torch.Tensor,
    n_elements: int,
    mu_co: float,
    prior_constants: Optional[Dict[str, float]] = None,
) -> torch.Tensor:
    """
    Compute analytical potential adsorption drive for Pd preference.

    Uses GraphFeatureLayout to locate feature indices rather than
    hardcoded magic numbers.

    Returns:
        g_pd: (B, N_active)
    """
    if active_node_features.ndim != 3:
        raise ValueError("active_node_features must have shape (B, N, F)")

    bsz, n_nodes, feat_dim = active_node_features.shape
    if n_nodes == 0:
        return active_node_features.new_zeros((bsz, 0))

    cfg = build_prior_constants(prior_constants)
    layout = GraphFeatureLayout(n_elements=n_elements)

    if feat_dim <= max(layout.avg_bond_idx, layout.is_surface_idx):
        return active_node_features.new_zeros((bsz, n_nodes))

    avg_bond = active_node_features[..., layout.avg_bond_idx]
    is_surface = active_node_features[..., layout.is_surface_idx].clamp(min=0.0, max=1.0)

    if feat_dim > layout.co_load_idx:
        co_load = active_node_features[..., layout.co_load_idx].clamp(min=0.0)
    else:
        co_load = active_node_features.new_zeros((bsz, n_nodes))

    strain_ref = avg_bond.mean(dim=1, keepdim=True)
    strain_term = cfg["strain_coeff"] * (avg_bond - strain_ref) * is_surface

    surface_term = (cfg["gamma_cu"] - cfg["gamma_pd"]) * is_surface
    ads_term = (cfg["e_pd_co"] - cfg["e_cu_co"] - float(mu_co)) * is_surface

    # Higher CO-loaded condition strengthens adsorption-related drive.
    ads_term = ads_term * (1.0 + 0.5 * co_load)

    g_pd = surface_term + strain_term + ads_term
    g_pd = torch.tanh(g_pd)
    return g_pd
