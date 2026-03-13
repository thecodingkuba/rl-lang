from dataclasses import dataclass, field
import json
from typing import Dict, List


@dataclass
class EnvConfig:
    """
    Stores fixed experiment parameters for the tutoring environment.

    Defines the skill set and hyperparameters for learner initialization and dynamics.
    Created once and passed to environment/learner as read-only.
    """

    skills: List[str] = field(default_factory=lambda: [
        # grammar
        "grammar:present",
        "grammar:past",
        "grammar:future",
        "grammar:articles",
        "grammar:prepositions",
        # vocab
        "vocab:1",
        "vocab:2",
        "vocab:3",
        "vocab:4",
        "vocab:5",
    ])

    init_mastery: float = 0.25
    learning_rate: float = 0.6393      # fitted from Duolingo SLAM (alpha_0)
    decay_rate: float = 0.0000005      # fitted from SLAM (delta_per_day=0.0001, converted to per-step)
    sigmoid_beta: float = 5.23         # fitted from SLAM (IRT discrimination)
    min_mastery: float = 0.0
    max_mastery: float = 1.0
    episode_length: int = 200

    # per-solver computational costs
    solver_costs: Dict[str, float] = field(default_factory=lambda: {
        "vocab_drill": 0.1,
        "grammar_explanation": 0.3,
        "mixed_quiz": 0.3,
        "spaced_repetition": 0.2,
        "free_form": 0.5,
    })

    cost_penalty_lambda: float = 0.01
    mastery_noise_std: float = 0.05
    discount_gamma: float = 0.99

    @classmethod
    def from_fitted(cls, json_path: str = "data/fitted_params.json", **overrides) -> "EnvConfig":
        """Load an EnvConfig with parameters calibrated from Duolingo SLAM data."""
        with open(json_path) as f:
            data = json.load(f)
        env_params = data["env_params"]
        return cls(
            learning_rate=env_params["learning_rate"],
            decay_rate=env_params["decay_rate"],
            sigmoid_beta=env_params["beta"],
            **overrides,
        )
