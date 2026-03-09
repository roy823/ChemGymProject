from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class PIDLagrangianConstraint:
    """
    PID-Lagrangian composition constraint controller.

    The penalty is:
        penalty = lambda * max(0, |x_pd - x_target| - threshold)^2
    where lambda is updated by a PID-like rule.
    """

    x_target: float
    d_threshold: float
    lambda_init: float = 1.0
    kp: float = 0.10
    ki: float = 0.01
    kd: float = 0.01
    lambda_min: float = 0.0
    lambda_max: float = 10.0
    integral_clip: float = 100.0

    def __post_init__(self) -> None:
        self.lam = float(np.clip(self.lambda_init, self.lambda_min, self.lambda_max))
        self._integral = 0.0
        self._prev_error = 0.0

    def reset_episode(self) -> None:
        self._integral = 0.0
        self._prev_error = 0.0

    def violation(self, n_pd: int, n_active: int) -> Tuple[float, float]:
        n_active = max(1, int(n_active))
        x_pd = float(n_pd) / float(n_active)
        d_frac = abs(x_pd - float(self.x_target))
        v = max(0.0, d_frac - float(self.d_threshold))
        return float(v), float(d_frac)

    def update_from_violation(self, violation: float, gain: float = 1.0) -> float:
        error = float(max(0.0, violation))
        self._integral = float(np.clip(self._integral + error, -self.integral_clip, self.integral_clip))
        derivative = float(error - self._prev_error)
        self._prev_error = error

        delta_lambda = float(gain) * float(self.kp * error + self.ki * self._integral + self.kd * derivative)
        self.lam = float(np.clip(self.lam + delta_lambda, self.lambda_min, self.lambda_max))
        return float(self.lam)

    def penalty_from_violation(self, violation: float) -> float:
        v = float(max(0.0, violation))
        return float(self.lam * (v ** 2))

    def compute_penalty(
        self,
        n_pd: int,
        n_active: int,
        update_lambda: bool = False,
        gain: float = 1.0,
    ) -> Tuple[float, float, float, float]:
        v, d_frac = self.violation(n_pd=n_pd, n_active=n_active)

        if update_lambda:
            self.update_from_violation(v, gain=gain)

        penalty = self.penalty_from_violation(v)
        return penalty, float(self.lam), float(v), float(d_frac)
