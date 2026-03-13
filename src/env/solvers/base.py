from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from src.env.core.question import Question
from src.env.core.skill_map import SkillMap


class BaseSolver(ABC):
    """Interface that every solver module must implement."""

    name: str
    cost: float
    effectiveness: float  # learning rate multiplier (>1 means solver teaches better)

    @abstractmethod
    def generate_question(
        self,
        mastery_estimate: NDArray[np.float64],
        skill_map: SkillMap,
        rng: np.random.Generator,
    ) -> Question:
        ...
