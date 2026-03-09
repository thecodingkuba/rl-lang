import numpy as np
from numpy.typing import NDArray


class LinUCB:
    """
    LinUCB contextual bandit baseline.

    Maintains per-arm linear models and selects the arm with the highest
    upper confidence bound on expected immediate reward. This is the myopic
    baseline — it does not plan over future timesteps.
    """

    def __init__(self, n_actions: int, obs_dim: int, alpha: float = 1.0):
        self.n_actions = n_actions
        self.d = obs_dim
        self.alpha = alpha

        # per-arm parameters
        self.A = [np.eye(self.d) for _ in range(n_actions)]
        self.b = [np.zeros(self.d) for _ in range(n_actions)]

    def select_action(self, obs: NDArray[np.float64]) -> int:
        x = obs.astype(np.float64)
        ucbs = np.zeros(self.n_actions)

        for a in range(self.n_actions):
            A_inv = np.linalg.inv(self.A[a])
            theta = A_inv @ self.b[a]
            ucbs[a] = theta @ x + self.alpha * np.sqrt(x @ A_inv @ x)

        return int(np.argmax(ucbs))

    def update(
        self,
        obs: NDArray[np.float64],
        action: int,
        reward: float,
        next_obs: NDArray[np.float64],
    ) -> None:
        x = obs.astype(np.float64)
        self.A[action] += np.outer(x, x)
        self.b[action] += reward * x
