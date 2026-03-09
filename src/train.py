"""
Training pipeline for all four methods:
  - PPO  (Stable-Baselines3)
  - DQN  (Stable-Baselines3)
  - LinUCB contextual bandit
  - Random baseline

Usage:
    python -m src.train --algo ppo
    python -m src.train --algo dqn
    python -m src.train --algo linucb
    python -m src.train --algo random
    python -m src.train --algo all
"""

import argparse
import os
import pickle
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback

from src.config.env_config import EnvConfig
from src.config.agent_config import AgentConfig
from src.config.train_config import TrainConfig
from src.env.env import LanguageTutoringEnv
from src.baselines.random_policy import RandomPolicy
from src.baselines.linucb import LinUCB


def make_env(config: EnvConfig):
    def _init():
        return LanguageTutoringEnv(config)
    return _init


def train_ppo(env_config: EnvConfig, agent_config: AgentConfig, train_config: TrainConfig):
    print("=== Training PPO ===")
    vec_env = make_vec_env(
        make_env(env_config), n_envs=agent_config.n_envs, seed=train_config.seed
    )
    eval_env = LanguageTutoringEnv(env_config)

    log_path = os.path.join(train_config.log_dir, "ppo")
    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=agent_config.learning_rate,
        gamma=agent_config.gamma,
        batch_size=agent_config.batch_size,
        verbose=train_config.verbose,
        tensorboard_log=log_path,
        seed=train_config.seed,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(train_config.model_dir, "ppo"),
        log_path=log_path,
        eval_freq=train_config.eval_freq,
        n_eval_episodes=train_config.n_eval_episodes,
        deterministic=True,
    )

    model.learn(total_timesteps=agent_config.total_timesteps, callback=eval_callback)
    save_path = os.path.join(train_config.model_dir, "ppo", "final_model")
    model.save(save_path)
    print(f"PPO model saved to {save_path}")
    vec_env.close()
    eval_env.close()


def train_dqn(env_config: EnvConfig, agent_config: AgentConfig, train_config: TrainConfig):
    print("=== Training DQN ===")
    env = LanguageTutoringEnv(env_config)
    eval_env = LanguageTutoringEnv(env_config)

    log_path = os.path.join(train_config.log_dir, "dqn")
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=agent_config.learning_rate,
        gamma=agent_config.gamma,
        batch_size=agent_config.batch_size,
        verbose=train_config.verbose,
        tensorboard_log=log_path,
        seed=train_config.seed,
        exploration_fraction=0.3,
        exploration_final_eps=0.05,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(train_config.model_dir, "dqn"),
        log_path=log_path,
        eval_freq=train_config.eval_freq,
        n_eval_episodes=train_config.n_eval_episodes,
        deterministic=True,
    )

    model.learn(total_timesteps=agent_config.total_timesteps, callback=eval_callback)
    save_path = os.path.join(train_config.model_dir, "dqn", "final_model")
    model.save(save_path)
    print(f"DQN model saved to {save_path}")
    env.close()
    eval_env.close()


def train_baseline(
    algo_name: str,
    env_config: EnvConfig,
    agent_config: AgentConfig,
    train_config: TrainConfig,
):
    """Train LinUCB or Random via a simple rollout loop."""
    print(f"=== Training {algo_name.upper()} ===")
    env = LanguageTutoringEnv(env_config)
    obs_dim = 2 * len(env_config.skills)

    if algo_name == "linucb":
        policy = LinUCB(
            n_actions=5, obs_dim=obs_dim, alpha=agent_config.linucb_alpha
        )
    else:
        policy = RandomPolicy(n_actions=5, seed=train_config.seed)

    total_steps = 0
    episode = 0
    all_rewards = []

    while total_steps < agent_config.total_timesteps:
        obs, _ = env.reset(seed=train_config.seed + episode)
        ep_reward = 0.0
        done = False

        while not done:
            action = policy.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            policy.update(obs, action, reward, next_obs)
            obs = next_obs
            ep_reward += reward
            total_steps += 1
            done = terminated or truncated

        all_rewards.append(ep_reward)
        episode += 1

        if episode % 50 == 0:
            recent = all_rewards[-50:]
            print(
                f"  Episode {episode} | Steps {total_steps} | "
                f"Mean reward (last 50): {np.mean(recent):.4f}"
            )

    save_dir = os.path.join(train_config.model_dir, algo_name)
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(save_dir, "policy.pkl"), "wb") as f:
        pickle.dump(policy, f)
    print(f"{algo_name} policy saved to {save_dir}/policy.pkl")
    env.close()


def main():
    parser = argparse.ArgumentParser(description="Train RL language tutoring agents")
    parser.add_argument(
        "--algo",
        type=str,
        default="all",
        choices=["ppo", "dqn", "linucb", "random", "all"],
        help="Which algorithm to train",
    )
    args = parser.parse_args()

    env_config = EnvConfig()
    agent_config = AgentConfig()
    train_config = TrainConfig()

    Path(train_config.model_dir).mkdir(parents=True, exist_ok=True)
    Path(train_config.log_dir).mkdir(parents=True, exist_ok=True)

    runners = {
        "ppo": lambda: train_ppo(env_config, agent_config, train_config),
        "dqn": lambda: train_dqn(env_config, agent_config, train_config),
        "linucb": lambda: train_baseline("linucb", env_config, agent_config, train_config),
        "random": lambda: train_baseline("random", env_config, agent_config, train_config),
    }

    if args.algo == "all":
        for name, run in runners.items():
            run()
    else:
        runners[args.algo]()


if __name__ == "__main__":
    main()
