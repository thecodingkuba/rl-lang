"""
Thin wrappers to give SB3 models the same .select_action() interface
as the custom baselines, used during evaluation.
"""

import numpy as np
from numpy.typing import NDArray


class SB3PolicyWrapper:
    """Wraps a trained Stable-Baselines3 model for uniform evaluation."""

    def __init__(self, model):
        self.model = model

    def select_action(self, obs: NDArray[np.float64]) -> int:
        action, _ = self.model.predict(obs, deterministic=True)
        return int(action)
