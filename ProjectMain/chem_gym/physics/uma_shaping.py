from __future__ import annotations

from typing import Callable, Optional

import numpy as np


class UMAPotentialShaper:
    """
    Potential-based reward shaping:
        F(s, s') = gamma * Phi(s') - Phi(s)
    with Phi(s) = -E_UMA(s) / scale.
    """

    def __init__(
        self,
        energy_fn: Optional[Callable[[object], float]],
        gamma: float = 0.97,
        scale: float = 50.0,
        weight: float = 1.0,
    ) -> None:
        self.energy_fn = energy_fn
        self.gamma = float(gamma)
        self.scale = float(max(1e-6, scale))
        self.weight = float(weight)
        self.prev_phi: Optional[float] = None

    def _phi(self, atoms: object) -> float:
        if self.energy_fn is None or atoms is None:
            return 0.0
        try:
            e = float(self.energy_fn(atoms))
        except Exception:
            return 0.0
        return float(-e / self.scale)

    def reset(self, initial_atoms: object) -> None:
        self.prev_phi = self._phi(initial_atoms)

    def transition(self, new_atoms: object) -> float:
        if self.prev_phi is None:
            self.reset(new_atoms)
            return 0.0
        phi_new = self._phi(new_atoms)
        shaping = float(self.weight * (self.gamma * phi_new - self.prev_phi))
        if not np.isfinite(shaping):
            shaping = 0.0
        self.prev_phi = phi_new
        return shaping
