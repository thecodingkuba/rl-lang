import numpy as np
from numpy.typing import NDArray

from src.env.core.question import Question
from src.env.core.skill_map import SkillMap
from src.env.solvers.base import BaseSolver


class FreeForm(BaseSolver):
    name = "free_form"
    cost = 0.50
    effectiveness = 2.0

    def generate_question(
        self,
        mastery_estimate: NDArray[np.float64],
        skill_map: SkillMap,
        rng: np.random.Generator,
    ) -> Question:
        indices = skill_map.all_indices
        mean_diff = skill_map.skill_difficulties[indices].mean()
        difficulty = np.clip(mean_diff + rng.normal(0, 0.15), 0.05, 0.95)
        return Question(
            skill_indices=indices, difficulty=float(difficulty),
            cost=self.cost, effectiveness=self.effectiveness,
        )
