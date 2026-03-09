import numpy as np
from numpy.typing import NDArray

from src.config.env_config import EnvConfig


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


class Learner:
    """
    Simulated language learner with hidden true mastery state.

    The agent never observes `true_mastery` directly — it can only infer
    the learner's knowledge through observed correct/incorrect responses.
    """

    def __init__(self, config: EnvConfig, rng: np.random.Generator):
        self.config = config
        self.K = len(config.skills)
        self.rng = rng

        # hidden ground-truth mastery (not visible to the agent)
        self.true_mastery: NDArray[np.float64] = np.clip(
            config.init_mastery + rng.normal(0, config.mastery_noise_std, self.K),
            config.min_mastery,
            config.max_mastery,
        )

        # per-skill interaction counts (used for decaying learning rate)
        self.interaction_counts: NDArray[np.float64] = np.zeros(self.K)

    def respond(self, question) -> bool:
        """
        Simulate learner answering a question using sigmoid IRT model.
        P(correct) = sigma(mean_mastery_on_tested_skills - difficulty)
        """
        relevant_mastery = self.true_mastery[question.skill_indices].mean()
        p_correct = _sigmoid(5.0 * (relevant_mastery - question.difficulty))
        return bool(self.rng.random() < p_correct)

    def transition(self, question, correct: bool) -> None:
        """
        Update true mastery using prediction-error rule from the writeup:
          m_{t+1,k} = m_k + alpha_k * (c - sigma(m_k - d))
          alpha_k   = alpha_0 / sqrt(1 + count_k)
        """
        for k in question.skill_indices:
            predicted = _sigmoid(5.0 * (self.true_mastery[k] - question.difficulty))
            alpha_k = self.config.learning_rate / np.sqrt(1.0 + self.interaction_counts[k])
            self.true_mastery[k] += alpha_k * (float(correct) - predicted)
            self.true_mastery[k] = np.clip(
                self.true_mastery[k], self.config.min_mastery, self.config.max_mastery
            )
            self.interaction_counts[k] += 1

    def decay(self, practiced_skills: set) -> None:
        """
        Apply forgetting to skills NOT practiced this timestep.
        m_k *= (1 - decay_rate) for un-practiced skills.
        """
        for k in range(self.K):
            if k not in practiced_skills:
                self.true_mastery[k] *= (1.0 - self.config.decay_rate)
                self.true_mastery[k] = max(self.true_mastery[k], self.config.min_mastery)

    def mean_mastery(self) -> float:
        return float(self.true_mastery.mean())

    def reset(self) -> None:
        """Re-randomize learner for a new episode."""
        self.true_mastery = np.clip(
            self.config.init_mastery
            + self.rng.normal(0, self.config.mastery_noise_std, self.K),
            self.config.min_mastery,
            self.config.max_mastery,
        )
        self.interaction_counts = np.zeros(self.K)
