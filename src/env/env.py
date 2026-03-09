from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from src.config.env_config import EnvConfig
from src.env.core.learner import Learner, _sigmoid
from src.env.core.skill_map import SkillMap
from src.env.core.solver_router import build_solver_list


class LanguageTutoringEnv(gym.Env):
    """
    Gymnasium environment for adaptive language tutoring.

    Observation: belief state [mastery_estimates (K,), normalized_counts (K,)]
    Action:      Discrete(5) — one per solver module
    Reward:      delta_mean_proficiency - lambda * solver_cost
    """

    metadata = {"render_modes": []}

    def __init__(self, config: Optional[EnvConfig] = None):
        super().__init__()
        self.config = config or EnvConfig()
        self.skill_map = SkillMap(self.config.skills)
        self.K = len(self.skill_map)
        self.solvers = build_solver_list()

        self.observation_space = spaces.Box(
            low=0.0, high=np.inf, shape=(2 * self.K,), dtype=np.float64
        )
        self.action_space = spaces.Discrete(len(self.solvers))

        self._rng: np.random.Generator = np.random.default_rng()
        self.learner: Optional[Learner] = None
        self._belief_mastery: Optional[NDArray] = None
        self._belief_counts: Optional[NDArray] = None
        self._step_count: int = 0
        self._prev_mean_mastery: float = 0.0

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[NDArray[np.float64], Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.learner = Learner(self.config, self._rng)

        # belief state starts at the prior (init_mastery, zero counts)
        self._belief_mastery = np.full(self.K, self.config.init_mastery, dtype=np.float64)
        self._belief_counts = np.zeros(self.K, dtype=np.float64)
        self._step_count = 0
        self._prev_mean_mastery = float(self._belief_mastery.mean())

        return self._get_obs(), self._get_info()

    def step(
        self, action: int
    ) -> Tuple[NDArray[np.float64], float, bool, bool, Dict[str, Any]]:
        solver = self.solvers[action]

        # 1. solver generates a question using current belief
        question = solver.generate_question(self._belief_mastery, self.skill_map, self._rng)

        # 2. learner responds (hidden ground-truth model)
        correct = self.learner.respond(question)

        # 3. update hidden learner mastery
        self.learner.transition(question, correct)

        # 4. apply forgetting to un-practiced skills
        practiced = set(question.skill_indices)
        self.learner.decay(practiced)

        # 5. update belief state (same prediction-error rule, but on our estimates)
        for k in question.skill_indices:
            predicted = _sigmoid(5.0 * (self._belief_mastery[k] - question.difficulty))
            alpha_k = self.config.learning_rate / np.sqrt(1.0 + self._belief_counts[k])
            self._belief_mastery[k] += alpha_k * (float(correct) - predicted)
            self._belief_mastery[k] = np.clip(
                self._belief_mastery[k], self.config.min_mastery, self.config.max_mastery
            )
            self._belief_counts[k] += 1

        # 6. reward = delta belief mastery - cost penalty
        new_mean = float(self._belief_mastery.mean())
        reward = (new_mean - self._prev_mean_mastery) - self.config.cost_penalty_lambda * question.cost
        self._prev_mean_mastery = new_mean

        # 7. episode termination
        self._step_count += 1
        terminated = self._step_count >= self.config.episode_length
        truncated = False

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def _get_obs(self) -> NDArray[np.float64]:
        normalized_counts = self._belief_counts / max(1.0, self._belief_counts.max())
        return np.concatenate([self._belief_mastery, normalized_counts])

    def _get_info(self) -> Dict[str, Any]:
        return {
            "true_mastery": self.learner.true_mastery.copy() if self.learner else None,
            "belief_mastery": self._belief_mastery.copy() if self._belief_mastery is not None else None,
            "step": self._step_count,
        }
