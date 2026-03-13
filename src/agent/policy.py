import numpy as np
from numpy.typing import NDArray


class SB3PolicyWrapper:
    def __init__(self, model):
        self.model = model

    def select_action(self, obs: NDArray[np.float64]) -> int:
        action, _ = self.model.predict(obs, deterministic=True)
        return int(action)
