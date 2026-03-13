from typing import Dict, List

import numpy as np
from numpy.typing import NDArray


class SkillMap:
    """Immutable bidirectional mapping between skill names and integer indices."""

    def __init__(self, skills: List[str], skill_difficulties: Dict[str, float] = None):
        if len(skills) != len(set(skills)):
            raise ValueError("Duplicate skills detected.")
        if not skills:
            raise ValueError("Skills list must not be empty.")

        self._index_to_skill: tuple = tuple(skills)
        self._skill_to_index: dict = {name: i for i, name in enumerate(skills)}

        defaults = {s: 0.15 for s in skills}
        if skill_difficulties:
            defaults.update(skill_difficulties)
        self._skill_difficulties: NDArray[np.float64] = np.array(
            [defaults[s] for s in skills], dtype=np.float64
        )

    def __len__(self) -> int:
        return len(self._index_to_skill)

    def __getitem__(self, key):
        """Look up by int index or string name."""
        if isinstance(key, int):
            return self._index_to_skill[key]
        return self._skill_to_index[key]

    def index(self, skill_name: str) -> int:
        return self._skill_to_index[skill_name]

    def name(self, idx: int) -> str:
        return self._index_to_skill[idx]

    @property
    def grammar_indices(self) -> List[int]:
        return [i for i, s in enumerate(self._index_to_skill) if s.startswith("grammar:")]

    @property
    def vocab_indices(self) -> List[int]:
        return [i for i, s in enumerate(self._index_to_skill) if s.startswith("vocab:")]

    @property
    def all_indices(self) -> List[int]:
        return list(range(len(self)))

    @property
    def skill_difficulties(self) -> NDArray[np.float64]:
        return self._skill_difficulties

    @property
    def skills(self) -> tuple:
        return self._index_to_skill
