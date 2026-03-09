from __future__ import annotations

import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class GraphResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.lin_msg = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.lin_upd = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h: torch.Tensor, a_norm: torch.Tensor) -> torch.Tensor:
        msg = torch.bmm(a_norm, self.lin_msg(h))
        out = self.lin_upd(msg)
        return self.norm(h + out)


class CrystalGraphFeatureExtractor(BaseFeaturesExtractor):
    """
    Graph encoder for SB3 MultiInputPolicy.

    Extra methods (`encode_nodes`, `pool_nodes`) are exposed for PIRP logits injection.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        features_dim: int = 256,
        hidden_dim: int = 128,
        n_layers: int = 3,
    ):
        super().__init__(observation_space, features_dim)

        node_f = observation_space["node_features"].shape[1]
        self.hidden_dim = hidden_dim

        self.node_in = nn.Sequential(
            nn.Linear(node_f, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        self.blocks = nn.ModuleList([GraphResidualBlock(hidden_dim) for _ in range(n_layers)])

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, features_dim),
        )

    def normalize_adjacency(self, adjacency: torch.Tensor) -> torch.Tensor:
        deg = adjacency.sum(dim=-1).clamp(min=1e-6)
        deg_inv_sqrt = deg.rsqrt()
        return adjacency * deg_inv_sqrt.unsqueeze(-1) * deg_inv_sqrt.unsqueeze(-2)

    def encode_nodes(self, observations: dict) -> tuple[torch.Tensor, torch.Tensor]:
        x = observations["node_features"]
        a = observations["adjacency"]

        h = self.node_in(x)
        a_norm = self.normalize_adjacency(a)

        for blk in self.blocks:
            h = blk(h, a_norm)

        return h, observations.get("node_mask", None)

    def pool_nodes(self, node_hidden: torch.Tensor, node_mask: torch.Tensor | None) -> torch.Tensor:
        if node_mask is None:
            return node_hidden.mean(dim=1)

        m = node_mask.unsqueeze(-1)
        masked_hidden = node_hidden * m
        denom = m.sum(dim=1).clamp(min=1e-6)
        return masked_hidden.sum(dim=1) / denom

    def project_pooled(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.head(pooled)

    def forward(self, observations: dict) -> torch.Tensor:
        node_hidden, node_mask = self.encode_nodes(observations)
        pooled = self.pool_nodes(node_hidden, node_mask)
        return self.project_pooled(pooled)
