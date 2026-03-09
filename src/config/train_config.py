from dataclasses import dataclass


@dataclass
class TrainConfig:
    """Configuration for the training pipeline."""

    seed: int = 42
    log_dir: str = "logs"
    model_dir: str = "models"
    results_dir: str = "results"
    eval_freq: int = 5_000
    n_eval_episodes: int = 20
    verbose: int = 1
