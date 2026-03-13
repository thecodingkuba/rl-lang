import numpy as np
from numpy.typing import NDArray

from src.env.core.question import Question
from src.env.core.skill_map import SkillMap
from src.env.solvers.base import BaseSolver


class SpacedRepetition(BaseSolver):
    """Targets the weakest skill by belief estimate — optimal challenge point."""

    name = "spaced_repetition"
    cost = 0.2
    effectiveness = 2.0  # targeting weak spots is pedagogically most efficient

    def generate_question(
        self,
        mastery_estimate: NDArray[np.float64],
        skill_map: SkillMap,
        rng: np.random.Generator,
    ) -> Question:
        weakest = int(np.argmin(mastery_estimate))
        difficulty = np.clip(mastery_estimate[weakest] + rng.normal(0, 0.05), 0.05, 0.95)
        return Question(
            skill_indices=[weakest], difficulty=float(difficulty),
            cost=self.cost, effectiveness=self.effectiveness,
        )
