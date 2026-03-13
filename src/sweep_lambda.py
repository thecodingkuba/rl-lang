import os
import pickle
from copy import deepcopy
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO, DQN

from src.agent.policy import SB3PolicyWrapper
from src.baselines.linucb import LinUCB
from src.baselines.random_policy import RandomPolicy
from src.config.agent_config import AgentConfig
from src.config.env_config import EnvConfig
from src.config.train_config import TrainConfig
from src.env.env import LanguageTutoringEnv

LAMBDAS = [0.0, 0.005, 0.01, 0.02, 0.05]
N_EVAL = 50
METHODS = ["PPO", "DQN", "LinUCB", "Random"]


def train_and_eval_for_lambda(
    lam: float,
    agent_config: AgentConfig,
    base_train_config: TrainConfig,
) -> Dict[str, dict]:
    env_config = EnvConfig(cost_penalty_lambda=lam)
    obs_dim = 2 * len(env_config.skills)

    tag = f"lam_{lam:.2f}"
    train_config = deepcopy(base_train_config)
    train_config.log_dir = os.path.join("sweep_logs", tag)
    train_config.model_dir = os.path.join("sweep_models", tag)
    Path(train_config.log_dir).mkdir(parents=True, exist_ok=True)
    Path(train_config.model_dir).mkdir(parents=True, exist_ok=True)

    policies = {}

    print(f"  [lambda={lam}] Training PPO...")
    from stable_baselines3.common.env_util import make_vec_env

    def _make():
        return LanguageTutoringEnv(env_config)

    vec_env = make_vec_env(_make, n_envs=agent_config.n_envs, seed=train_config.seed)
    ppo = PPO(
        "MlpPolicy", vec_env,
        learning_rate=agent_config.learning_rate, gamma=agent_config.gamma,
        batch_size=agent_config.batch_size, verbose=0, seed=train_config.seed,
    )
    ppo.learn(total_timesteps=agent_config.total_timesteps)
    policies["PPO"] = SB3PolicyWrapper(ppo)
    vec_env.close()

    print(f"  [lambda={lam}] Training DQN...")
    dqn_env = LanguageTutoringEnv(env_config)
    dqn = DQN(
        "MlpPolicy", dqn_env,
        learning_rate=agent_config.learning_rate, gamma=agent_config.gamma,
        batch_size=agent_config.batch_size, verbose=0, seed=train_config.seed,
        exploration_fraction=0.3, exploration_final_eps=0.05,
    )
    dqn.learn(total_timesteps=agent_config.total_timesteps)
    policies["DQN"] = SB3PolicyWrapper(dqn)
    dqn_env.close()

    print(f"  [lambda={lam}] Training LinUCB...")
    linucb = LinUCB(n_actions=5, obs_dim=obs_dim, alpha=agent_config.linucb_alpha)
    env = LanguageTutoringEnv(env_config)
    total_steps, episode = 0, 0
    while total_steps < agent_config.total_timesteps:
        obs, _ = env.reset(seed=train_config.seed + episode)
        done = False
        while not done:
            a = linucb.select_action(obs)
            nobs, r, term, trunc, _ = env.step(a)
            linucb.update(obs, a, r, nobs)
            obs = nobs
            total_steps += 1
            done = term or trunc
        episode += 1
    policies["LinUCB"] = linucb
    env.close()

    policies["Random"] = RandomPolicy(n_actions=5, seed=train_config.seed)

    results = {}
    eval_env = LanguageTutoringEnv(env_config)
    for name, pol in policies.items():
        ep_rewards, final_profs = [], []
        for ep in range(N_EVAL):
            obs, _ = eval_env.reset(seed=5000 + ep)
            ep_r, done = 0.0, False
            while not done:
                a = pol.select_action(obs)
                obs, r, term, trunc, _ = eval_env.step(a)
                ep_r += r
                done = term or trunc
            ep_rewards.append(ep_r)
            final_profs.append(float(obs[:len(env_config.skills)].mean()))
        results[name] = {
            "mean_reward": np.mean(ep_rewards),
            "std_reward": np.std(ep_rewards),
            "mean_proficiency": np.mean(final_profs),
            "std_proficiency": np.std(final_profs),
        }
    eval_env.close()
    return results


def main():
    agent_config = AgentConfig()
    train_config = TrainConfig(verbose=0)
    Path("sweep_results").mkdir(exist_ok=True)

    all_results: Dict[float, Dict[str, dict]] = {}

    for lam in LAMBDAS:
        print(f"\n{'='*50}")
        print(f"Lambda = {lam}")
        print(f"{'='*50}")
        all_results[lam] = train_and_eval_for_lambda(lam, agent_config, train_config)

        for name, res in all_results[lam].items():
            print(f"  {name:<10} prof={res['mean_proficiency']:.3f}  reward={res['mean_reward']:.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = {"PPO": "#4C72B0", "DQN": "#55A868", "LinUCB": "#C44E52", "Random": "#8172B2"}

    for method in METHODS:
        profs = [all_results[l][method]["mean_proficiency"] for l in LAMBDAS]
        stds = [all_results[l][method]["std_proficiency"] for l in LAMBDAS]
        axes[0].errorbar(LAMBDAS, profs, yerr=stds, marker="o", label=method,
                         color=colors[method], capsize=4)
    axes[0].set_xlabel("Cost Penalty (lambda)")
    axes[0].set_ylabel("Final Proficiency")
    axes[0].set_title("Proficiency vs Cost Penalty")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    for method in METHODS:
        rews = [all_results[l][method]["mean_reward"] for l in LAMBDAS]
        stds = [all_results[l][method]["std_reward"] for l in LAMBDAS]
        axes[1].errorbar(LAMBDAS, rews, yerr=stds, marker="o", label=method,
                         color=colors[method], capsize=4)
    axes[1].set_xlabel("Cost Penalty (lambda)")
    axes[1].set_ylabel("Cumulative Reward")
    axes[1].set_title("Reward vs Cost Penalty")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("sweep_results/lambda_sweep.png", dpi=150)
    plt.close()

    print(f"\n\n{'='*80}")
    print(f"{'Lambda':<10}", end="")
    for method in METHODS:
        print(f"{method + ' Prof':<14}{method + ' Rew':<14}", end="")
    print()
    print("=" * 80)
    for lam in LAMBDAS:
        print(f"{lam:<10.2f}", end="")
        for method in METHODS:
            r = all_results[lam][method]
            print(f"{r['mean_proficiency']:<14.3f}{r['mean_reward']:<14.2f}", end="")
        print()
    print("=" * 80)
    print("\nSweep plot saved to sweep_results/lambda_sweep.png")


if __name__ == "__main__":
    main()
