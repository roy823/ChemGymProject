from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch as th
import torch.nn as nn
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

from chem_gym.physics.analytical_prior import build_prior_constants, compute_potential_adsorption_drive


class PIRPMaskableActorCriticPolicy(MaskableActorCriticPolicy):
    """
    Maskable actor-critic policy with PIRP logits injection:
        logits = base_logits + pirp_scale * (v_i * g_{i->e})
    """

    def __init__(
        self,
        *args,
        pirp_scale: float = 1.0,
        noop_logit_bonus: float = 0.0,
        pirp_mu_co: float = -1.0,
        pirp_n_elements: int = 2,
        prior_constants: Optional[Dict[str, float]] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if not self.share_features_extractor:
            raise ValueError("PIRPMaskableActorCriticPolicy requires share_features_extractor=True")

        self.pirp_scale = float(pirp_scale)
        self.noop_logit_bonus = float(noop_logit_bonus)
        self.pirp_mu_co = float(pirp_mu_co)
        self.pirp_n_elements = int(pirp_n_elements)
        self.prior_constants = build_prior_constants(prior_constants)

        self.base_action_dim = (self.action_space.n // self.pirp_n_elements) * self.pirp_n_elements
        extra_actions = int(self.action_space.n - self.base_action_dim)
        if extra_actions not in {0, 1}:
            raise ValueError(
                "PIRP policy supports action spaces with optional single extra action (e.g. no-op)."
            )
        self.n_sites = self.base_action_dim // self.pirp_n_elements

        hidden_dim = getattr(self.features_extractor, "hidden_dim", 128)
        gate_hidden = max(16, hidden_dim // 2)
        self.site_gate = nn.Sequential(
            nn.Linear(hidden_dim, gate_hidden),
            nn.Tanh(),
            nn.Linear(gate_hidden, 1),
            nn.Tanh(),
        )

        # Start with a mild prior preference, not a hard bias.
        final_linear = self.site_gate[2]
        nn.init.zeros_(final_linear.weight)
        nn.init.constant_(final_linear.bias, 0.2)

    def _forward_with_node_context(self, obs: th.Tensor | Dict[str, th.Tensor]):
        if hasattr(self.features_extractor, "encode_nodes") and hasattr(self.features_extractor, "pool_nodes"):
            node_hidden, node_mask = self.features_extractor.encode_nodes(obs)
            pooled = self.features_extractor.pool_nodes(node_hidden, node_mask)
            features = self.features_extractor.project_pooled(pooled)
        else:
            node_hidden = None
            features = self.extract_features(obs)

        latent_pi, latent_vf = self.mlp_extractor(features)
        return latent_pi, latent_vf, node_hidden

    def _compute_prior_logits(
        self,
        obs: th.Tensor | Dict[str, th.Tensor],
        node_hidden: Optional[th.Tensor],
        action_dim: int,
    ) -> th.Tensor:
        if node_hidden is None or not isinstance(obs, dict):
            batch_size = node_hidden.shape[0] if node_hidden is not None else 1
            return th.zeros((batch_size, action_dim), device=self.device)

        node_features = obs.get("node_features", None)
        if node_features is None:
            return th.zeros((node_hidden.shape[0], action_dim), device=node_hidden.device, dtype=node_hidden.dtype)

        bsz = node_hidden.shape[0]
        if node_hidden.shape[1] < self.n_sites:
            return th.zeros((bsz, action_dim), device=node_hidden.device, dtype=node_hidden.dtype)

        active_hidden = node_hidden[:, -self.n_sites :, :]
        active_feats = node_features[:, -self.n_sites :, :]

        v = self.site_gate(active_hidden).squeeze(-1)
        g_pd = compute_potential_adsorption_drive(
            active_node_features=active_feats,
            n_elements=self.pirp_n_elements,
            mu_co=self.pirp_mu_co,
            prior_constants=self.prior_constants,
        )

        if self.pirp_n_elements != 2:
            g = th.zeros((bsz, self.n_sites, self.pirp_n_elements), device=active_hidden.device, dtype=active_hidden.dtype)
            g[..., 0] = -g_pd
            g[..., 1] = g_pd
        else:
            g = th.stack([-g_pd, g_pd], dim=-1)

        prior_site = v.unsqueeze(-1) * g
        prior_logits = prior_site.reshape(bsz, -1)

        if prior_logits.shape[1] < action_dim:
            pad = th.zeros((bsz, action_dim - prior_logits.shape[1]), device=prior_logits.device, dtype=prior_logits.dtype)
            prior_logits = th.cat([prior_logits, pad], dim=1)
        elif prior_logits.shape[1] > action_dim:
            prior_logits = prior_logits[:, :action_dim]

        return prior_logits

    def _distribution_from_latent_and_prior(
        self,
        latent_pi: th.Tensor,
        prior_logits: Optional[th.Tensor] = None,
    ):
        action_logits = self.action_net(latent_pi)
        if prior_logits is not None:
            action_logits = action_logits + self.pirp_scale * prior_logits
        # Keep no-op encouragement independent from PIRP scale so it remains effective.
        if action_logits.shape[1] > self.base_action_dim and self.noop_logit_bonus != 0.0:
            action_logits[:, self.base_action_dim] = (
                action_logits[:, self.base_action_dim] + float(self.noop_logit_bonus)
            )
        return self.action_dist.proba_distribution(action_logits=action_logits)

    def forward(
        self,
        obs: th.Tensor,
        deterministic: bool = False,
        action_masks: Optional[np.ndarray] = None,
    ):
        latent_pi, latent_vf, node_hidden = self._forward_with_node_context(obs)
        prior_logits = self._compute_prior_logits(obs, node_hidden, self.action_space.n)

        values = self.value_net(latent_vf)
        distribution = self._distribution_from_latent_and_prior(latent_pi, prior_logits)

        if action_masks is not None:
            distribution.apply_masking(action_masks)

        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob

    def evaluate_actions(
        self,
        obs: th.Tensor,
        actions: th.Tensor,
        action_masks: Optional[th.Tensor] = None,
    ):
        latent_pi, latent_vf, node_hidden = self._forward_with_node_context(obs)
        prior_logits = self._compute_prior_logits(obs, node_hidden, self.action_space.n)

        distribution = self._distribution_from_latent_and_prior(latent_pi, prior_logits)
        if action_masks is not None:
            distribution.apply_masking(action_masks)

        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        return values, log_prob, distribution.entropy()

    def get_distribution(
        self,
        obs: th.Tensor | Dict[str, th.Tensor],
        action_masks: Optional[np.ndarray] = None,
    ):
        latent_pi, _, node_hidden = self._forward_with_node_context(obs)
        prior_logits = self._compute_prior_logits(obs, node_hidden, self.action_space.n)

        distribution = self._distribution_from_latent_and_prior(latent_pi, prior_logits)
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        return distribution
