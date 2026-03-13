"""
Curriculum baseline: teaches skills in order from easiest to hardest
based on fitted per-skill difficulties, then reviews weak spots.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Dict, List


# Solver action indices (must match solver_router.build_solver_list order)
ACTION_VOCAB_DRILL = 0
ACTION_GRAMMAR_EXPLANATION = 1
ACTION_MIXED_QUIZ = 2
ACTION_SPACED_REPETITION = 3
ACTION_FREE_FORM = 4


class CurriculumPolicy:
    """
    Hand-designed curriculum that progresses from easy to hard skills.

    Phase 1 (first 80% of episode): iterate through skills sorted by
    difficulty (easiest first), spending equal time on each. Picks the
    solver matching the skill category (VocabDrill for vocab,
    GrammarExplanation for grammar).

    Phase 2 (final 20%): switch to SpacedRepetition to review weak spots.
    """

    def __init__(
        self,
        skill_names: List[str],
        skill_difficulties: Dict[str, float],
        episode_length: int = 200,
        n_actions: int = 5,
    ):
        self.n_actions = n_actions
        self.episode_length = episode_length
        self._step = 0

        sorted_skills = sorted(skill_difficulties.items(), key=lambda x: x[1])
        self._skill_order = [name for name, _ in sorted_skills]

        self._skill_to_idx = {name: i for i, name in enumerate(skill_names)}

        self._review_start = int(0.8 * episode_length)
        n_skills = len(self._skill_order)
        self._steps_per_skill = self._review_start // n_skills

    def _solver_for_skill(self, skill_name: str) -> int:
        if skill_name.startswith("grammar:"):
            return ACTION_GRAMMAR_EXPLANATION
        return ACTION_VOCAB_DRILL

    def select_action(self, obs: NDArray[np.float64]) -> int:
        step = self._step
        self._step += 1

        if step >= self._review_start:
            return ACTION_SPACED_REPETITION

        phase_idx = min(step // max(1, self._steps_per_skill), len(self._skill_order) - 1)
        current_skill = self._skill_order[phase_idx]
        return self._solver_for_skill(current_skill)

    def update(self, obs, action, reward, next_obs) -> None:
        pass

    def reset(self) -> None:
        self._step = 0
