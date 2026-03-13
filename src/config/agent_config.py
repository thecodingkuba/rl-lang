from dataclasses import dataclass


@dataclass
class AgentConfig:
    """
    Configuration for the routing agent.

    Parameter rationale
    -------------------
    total_timesteps = 1_000_000
        With episode_length=500, this is 2 000 full episodes — enough for PPO
        and DQN to see a wide variety of learner trajectories and converge.
        200 000 (original) was only 400 episodes, far too few for credit
        assignment across a 500-step horizon.

    n_envs = 8
        More parallel rollout workers means more diverse experience per update
        step.  8 is a practical limit before CPU overhead dominates on a laptop;
        scales proportionally on a multi-core server.

    batch_size = 256
        Larger mini-batches reduce gradient variance and make better use of
        vectorised operations.  64 (original) was too small given the long
        episodes and high-dimensional observation (20-d).

    linucb_alpha = 0.5
        Controls exploration width of the UCB bonus.  alpha=1.0 (original) was
        over-exploring in a low-noise environment; 0.5 keeps the bandit
        reasonably explorative without constant arm-switching.
    """

    algorithm: str = "ppo"  # ppo | dqn | linucb | random
    total_timesteps: int = 1_000_000
    learning_rate: float = 3e-4
    gamma: float = 0.99
    n_envs: int = 8
    batch_size: int = 256
    linucb_alpha: float = 0.5
