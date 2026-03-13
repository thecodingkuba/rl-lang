from dataclasses import dataclass


@dataclass
class TrainConfig:
    """
    Configuration for the training pipeline.

    Parameter rationale
    -------------------
    eval_freq = 10_000
        Evaluate every 10 000 steps (20 episodes at length=500).  More frequent
        evaluation (original 5 000) adds overhead and noisy checkpoints without
        much benefit at 1 M total timesteps.

    n_eval_episodes = 50
        50 evaluation episodes gives a low-variance estimate of mean reward
        (stderr ≈ std/sqrt(50)).  20 (original) produced high-variance model
        selection, sometimes saving a lucky checkpoint rather than the best one.
    """

    seed: int = 42
    log_dir: str = "logs"
    model_dir: str = "models"
    results_dir: str = "results"
    eval_freq: int = 10_000
    n_eval_episodes: int = 50
    verbose: int = 1
