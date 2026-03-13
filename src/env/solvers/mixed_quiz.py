import numpy as np
from numpy.typing import NDArray

from src.env.core.question import Question
from src.env.core.skill_map import SkillMap
from src.env.solvers.base import BaseSolver


class MixedQuiz(BaseSolver):
    name = "mixed_quiz"
    cost = 0.43
    effectiveness = 1.4

    def generate_question(
        self,
        mastery_estimate: NDArray[np.float64],
        skill_map: SkillMap,
        rng: np.random.Generator,
    ) -> Question:
        n_skills = rng.integers(2, 4)
        indices = rng.choice(skill_map.all_indices, size=n_skills, replace=False).tolist()
        mean_diff = skill_map.skill_difficulties[indices].mean()
        difficulty = np.clip(mean_diff + rng.normal(0, 0.1), 0.05, 0.95)
        return Question(
            skill_indices=indices, difficulty=float(difficulty),
            cost=self.cost, effectiveness=self.effectiveness,
        )
