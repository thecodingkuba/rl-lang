from dataclasses import dataclass


@dataclass
class TrainConfig:
    seed: int = 42
    log_dir: str = "logs"
    model_dir: str = "models"
    results_dir: str = "results"
    eval_freq: int = 5_000
    n_eval_episodes: int = 20
    verbose: int = 1
    wandb_project: str = "rl-lang-tutoring"
    use_wandb: bool = True
