from dataclasses import dataclass
from typing import List


@dataclass
class Question:
    """A concrete question instance presented to the learner."""

    skill_indices: List[int]
    difficulty: float       # in (0, 1): 0 = trivial, 1 = requires perfect mastery
    cost: float             # computational cost of the solver that generated this
    effectiveness: float = 1.0  # learning rate multiplier (>1 = solver teaches better)
