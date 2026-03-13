import os
import itertools
from pathlib import Path
from copy import deepcopy

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from src.config.env_config import EnvConfig
from src.config.train_config import TrainConfig
from src.env.env import LanguageTutoringEnv
from src.agent.policy import SB3PolicyWrapper

SWEEP_TIMESTEPS = 200_000
N_EVAL = 50
SEED = 42

LEARNING_RATES = [1e-4, 3e-4, 1e-3]
LAMBDAS = [0.0, 0.01, 0.1]


def make_env(config: EnvConfig):
    def _init():
        return LanguageTutoringEnv(config)
    return _init


def evaluate(policy, env_config: EnvConfig, n_episodes: int, base_seed: int = 5000):
    env = LanguageTutoringEnv(env_config)
    rewards, profs, costs = [], [], []
    solver_costs_arr = np.array([s.cost for s in env.solvers])

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=base_seed + ep)
        ep_reward, ep_cost, done = 0.0, 0.0, False
        while not done:
            action = policy.select_action(obs)
            ep_cost += solver_costs_arr[action]
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            done = terminated or truncated
        rewards.append(ep_reward)
        profs.append(float(obs[:len(env_config.skills)].mean()))
        costs.append(ep_cost)
    env.close()
    return np.mean(profs), np.std(profs), np.mean(rewards), np.std(rewards), np.mean(costs), np.std(costs)


def main():
    base_env_config = EnvConfig()
    train_config = TrainConfig()

    sweep_dir = os.path.join(train_config.model_dir, "sweep")
    Path(sweep_dir).mkdir(parents=True, exist_ok=True)

    results = []

    for lr, lam in itertools.product(LEARNING_RATES, LAMBDAS):
        tag = f"lr{lr:.0e}_lam{lam}"
        print(f"\n{'='*60}")
        print(f"  PPO sweep: lr={lr}, lambda={lam}")
        print(f"{'='*60}")

        env_config = deepcopy(base_env_config)
        env_config.cost_penalty_lambda = lam

        vec_env = make_vec_env(make_env(env_config), n_envs=4, seed=SEED)
        model = PPO(
            "MlpPolicy", vec_env,
            learning_rate=lr,
            gamma=0.99,
            batch_size=64,
            verbose=0,
            seed=SEED,
        )
        model.learn(total_timesteps=SWEEP_TIMESTEPS)

        save_path = os.path.join(sweep_dir, tag)
        model.save(save_path)
        vec_env.close()

        policy = SB3PolicyWrapper(PPO.load(save_path))
        prof_m, prof_s, rew_m, rew_s, cost_m, cost_s = evaluate(policy, env_config, N_EVAL)
        results.append((lr, lam, prof_m, prof_s, rew_m, rew_s, cost_m, cost_s))

        print(f"  -> Proficiency: {prof_m:.3f}±{prof_s:.3f}  "
              f"Reward: {rew_m:+.3f}±{rew_s:.3f}  Cost: {cost_m:.1f}±{cost_s:.1f}")

    print(f"\n\n{'='*80}")
    print("PPO Hyperparameter Sweep Results (200K steps, 50 eval episodes)")
    print(f"{'='*80}")
    print(f"{'LR':<10} {'Lambda':<10} {'Proficiency':<18} {'Reward':<18} {'Cost':<15}")
    print("-" * 71)
    for lr, lam, pm, ps, rm, rs, cm, cs in results:
        print(f"{lr:<10.0e} {lam:<10} {pm:.3f} ± {ps:.3f}     {rm:+.3f} ± {rs:.3f}     {cm:.1f} ± {cs:.1f}")

    results_path = os.path.join(train_config.results_dir, "sweep_results.txt")
    Path(train_config.results_dir).mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        f.write("LR,Lambda,Prof_Mean,Prof_Std,Reward_Mean,Reward_Std,Cost_Mean,Cost_Std\n")
        for lr, lam, pm, ps, rm, rs, cm, cs in results:
            f.write(f"{lr},{lam},{pm:.4f},{ps:.4f},{rm:.4f},{rs:.4f},{cm:.2f},{cs:.2f}\n")
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
