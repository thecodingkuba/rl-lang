import numpy as np
from numpy.typing import NDArray

from src.config.env_config import EnvConfig


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


class Learner:
    def __init__(self, config: EnvConfig, rng: np.random.Generator):
        self.config = config
        self.K = len(config.skills)
        self.rng = rng
        self.true_mastery: NDArray[np.float64] = np.clip(
            config.init_mastery + rng.normal(0, config.mastery_noise_std, self.K),
            config.min_mastery,
            config.max_mastery,
        )
        self.interaction_counts: NDArray[np.float64] = np.zeros(self.K)

    def respond(self, question) -> bool:
        relevant_mastery = self.true_mastery[question.skill_indices].mean()
        p_correct = _sigmoid(self.config.sigmoid_beta * (relevant_mastery - question.difficulty))
        return bool(self.rng.random() < p_correct)

    def transition(self, question, correct: bool) -> None:
        for k in question.skill_indices:
            predicted = _sigmoid(self.config.sigmoid_beta * (self.true_mastery[k] - question.difficulty))
            alpha_k = self.config.learning_rate / np.sqrt(1.0 + self.interaction_counts[k])
            self.true_mastery[k] += question.effectiveness * alpha_k * (float(correct) - predicted)
            self.true_mastery[k] = np.clip(
                self.true_mastery[k], self.config.min_mastery, self.config.max_mastery
            )
            self.interaction_counts[k] += 1

    def decay(self, practiced_skills: set) -> None:
        for k in range(self.K):
            if k not in practiced_skills:
                self.true_mastery[k] *= (1.0 - self.config.decay_rate)
                self.true_mastery[k] = max(self.true_mastery[k], self.config.min_mastery)

    def mean_mastery(self) -> float:
        return float(self.true_mastery.mean())

    def reset(self) -> None:
        self.true_mastery = np.clip(
            self.config.init_mastery
            + self.rng.normal(0, self.config.mastery_noise_std, self.K),
            self.config.min_mastery,
            self.config.max_mastery,
        )
        self.interaction_counts = np.zeros(self.K)
