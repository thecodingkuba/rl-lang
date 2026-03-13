from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from src.env.core.question import Question
from src.env.core.skill_map import SkillMap


class BaseSolver(ABC):
    name: str
    cost: float
    effectiveness: float

    @abstractmethod
    def generate_question(
        self,
        mastery_estimate: NDArray[np.float64],
        skill_map: SkillMap,
        rng: np.random.Generator,
    ) -> Question:
        ...
