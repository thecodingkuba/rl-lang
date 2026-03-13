from dataclasses import dataclass, field
import json
from typing import Dict, List


@dataclass
class EnvConfig:
    skills: List[str] = field(default_factory=lambda: [
        "grammar:present",
        "grammar:past",
        "grammar:future",
        "grammar:articles",
        "grammar:prepositions",
        "vocab:1",
        "vocab:2",
        "vocab:3",
        "vocab:4",
        "vocab:5",
    ])

    init_mastery: float = 0.25
    learning_rate: float = 0.6393
    decay_rate: float = 0.0000005
    sigmoid_beta: float = 5.23
    min_mastery: float = 0.0
    max_mastery: float = 1.0
    episode_length: int = 200

    solver_costs: Dict[str, float] = field(default_factory=lambda: {
        "vocab_drill": 0.10,
        "grammar_explanation": 0.47,
        "mixed_quiz": 0.43,
        "spaced_repetition": 0.15,
        "free_form": 0.50,
    })

    cost_penalty_lambda: float = 0.01
    mastery_noise_std: float = 0.05
    discount_gamma: float = 0.99

    skill_difficulties: Dict[str, float] = field(default_factory=lambda: {
        "grammar:present": 0.124,
        "grammar:past": 0.140,
        "grammar:future": 0.500,
        "grammar:articles": 0.088,
        "grammar:prepositions": 0.180,
        "vocab:1": 0.124,
        "vocab:2": 0.093,
        "vocab:3": 0.145,
        "vocab:4": 0.127,
        "vocab:5": 0.131,
    })

    @classmethod
    def from_fitted(cls, json_path: str = "data/fitted_params.json", **overrides) -> "EnvConfig":
        with open(json_path) as f:
            data = json.load(f)
        env_params = data["env_params"]
        skill_diffs = data.get("skill_difficulties", {})
        kwargs = dict(
            learning_rate=env_params["learning_rate"],
            decay_rate=env_params["decay_rate"],
            sigmoid_beta=env_params["beta"],
        )
        if skill_diffs:
            kwargs["skill_difficulties"] = skill_diffs
        kwargs.update(overrides)
        return cls(**kwargs)
