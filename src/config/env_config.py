from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class EnvConfig:
    """
    Stores fixed experiment parameters for the tutoring environment.

    Defines the skill set and hyperparameters for learner initialization and dynamics.
    Created once and passed to environment/learner as read-only.

    Parameter rationale
    -------------------
    init_mastery = 0.25
        Learners start with low but non-zero knowledge.  0.25 puts the sigmoid
        IRT response probability at ~38 % for difficulty=0.5, meaning early
        questions are hard enough to drive learning without being demoralising.

    learning_rate = 0.10
        With the decaying schedule alpha_k = 0.10 / sqrt(1 + n_k), the first
        interaction moves mastery by ~0.05-0.06 (half of alpha at n=1), with
        diminishing updates as evidence accumulates.  This mirrors empirical
        learning curves that are steep early and plateau with practice.

    decay_rate = 0.001
        Per-step multiplicative forgetting for unpractised skills.
        Over a 500-step episode, an unpractised skill retains 0.999^500 ≈ 60 %
        of its mastery — realistic forgetting without wiping out progress.
        0.01 (original) caused an 87 % drop in 200 steps, overwhelming learning.

    mastery_noise_std = 0.05
        Adds diversity between learners at episode start without letting any
        skill stray far from init_mastery (3-sigma ≈ ±0.15).

    episode_length = 500
        500 interactions at ~2-3 skills touched per step means every skill is
        reviewed 100-150 times on average — enough for mastery to converge and
        for RL agents to observe a meaningful learning trajectory per episode.
        200 (original) was too short; agents had little time to distinguish
        strategies before termination.

    cost_penalty_lambda = 0.05
        Scales the solver-cost penalty relative to mastery gains.  A typical
        step moves mean mastery by ~0.003-0.005; with lambda=0.05 the FreeForm
        cost (0.5) subtracts 0.025 — enough to matter but not dominating the
        reward so that the agent still prefers learning over pure cost saving.
        0.1 (original) made the cost penalty too large, discouraging expensive
        but effective solvers.
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
    learning_rate: float = 0.10
    decay_rate: float = 0.001
    min_mastery: float = 0.0
    max_mastery: float = 1.0
    episode_length: int = 500

    # per-solver computational costs
    solver_costs: Dict[str, float] = field(default_factory=lambda: {
        "vocab_drill": 0.1,
        "grammar_explanation": 0.3,
        "mixed_quiz": 0.3,
        "spaced_repetition": 0.2,
        "free_form": 0.5,
    })

    cost_penalty_lambda: float = 0.05
    mastery_noise_std: float = 0.05
    discount_gamma: float = 0.99
