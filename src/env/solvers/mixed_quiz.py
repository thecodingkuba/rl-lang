import numpy as np
from numpy.typing import NDArray

from src.env.core.question import Question
from src.env.core.skill_map import SkillMap
from src.env.solvers.base import BaseSolver


class MixedQuiz(BaseSolver):
    """Medium-cost quiz testing 2-3 skills from any category."""

    name = "mixed_quiz"
    cost = 0.43                # SLAM: listen median 19s, scaled
    effectiveness = 1.4        # Roediger & Karpicke (2006): retrieval practice across skills

    def generate_question(
        self,
        mastery_estimate: NDArray[np.float64],
        skill_map: SkillMap,
        rng: np.random.Generator,
    ) -> Question:
        n_skills = rng.integers(2, 4)  # 2 or 3
        indices = rng.choice(skill_map.all_indices, size=n_skills, replace=False).tolist()
        mean_diff = skill_map.skill_difficulties[indices].mean()
        difficulty = np.clip(mean_diff + rng.normal(0, 0.1), 0.05, 0.95)
        return Question(
            skill_indices=indices, difficulty=float(difficulty),
            cost=self.cost, effectiveness=self.effectiveness,
        )
