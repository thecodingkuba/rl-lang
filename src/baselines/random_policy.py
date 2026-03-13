import numpy as np
from numpy.typing import NDArray


class RandomPolicy:
    def __init__(self, n_actions: int, seed: int = 42):
        self.n_actions = n_actions
        self.rng = np.random.default_rng(seed)

    def select_action(self, obs: NDArray[np.float64]) -> int:
        return int(self.rng.integers(self.n_actions))

    def update(self, obs, action, reward, next_obs) -> None:
        pass
