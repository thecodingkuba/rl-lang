import numpy as np
from numpy.typing import NDArray

from src.env.core.question import Question
from src.env.core.skill_map import SkillMap
from src.env.solvers.base import BaseSolver


class GrammarExplanation(BaseSolver):
    """Medium-cost explanation targeting a random grammar skill."""

    name = "grammar_explanation"
    cost = 0.47                # SLAM: reverse_translate median 17s, scaled
    effectiveness = 1.2        # DeKeyser (2003): explicit instruction < practice-based methods

    def generate_question(
        self,
        mastery_estimate: NDArray[np.float64],
        skill_map: SkillMap,
        rng: np.random.Generator,
    ) -> Question:
        idx = rng.choice(skill_map.grammar_indices)
        difficulty = np.clip(skill_map.skill_difficulties[idx] + rng.normal(0, 0.1), 0.05, 0.95)
        return Question(
            skill_indices=[idx], difficulty=float(difficulty),
            cost=self.cost, effectiveness=self.effectiveness,
        )
