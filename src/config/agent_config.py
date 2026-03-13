from dataclasses import dataclass


@dataclass
class AgentConfig:
    """Configuration for the routing agent."""

    algorithm: str = "ppo"  # ppo | dqn | linucb | random
    total_timesteps: int = 1_000_000
    learning_rate: float = 3e-4
    gamma: float = 0.99
    n_envs: int = 4
    batch_size: int = 64
    linucb_alpha: float = 1.0
