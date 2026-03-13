from dataclasses import dataclass
from typing import List


@dataclass
class Question:
    skill_indices: List[int]
    difficulty: float
    cost: float
    effectiveness: float = 1.0
