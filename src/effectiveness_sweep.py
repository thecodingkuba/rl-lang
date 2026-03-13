import os
import pickle
from copy import deepcopy
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from src.config.env_config import EnvConfig
from src.config.train_config import TrainConfig
from src.env.env import LanguageTutoringEnv
from src.env.core.solver_router import build_solver_list
from src.agent.policy import SB3PolicyWrapper
from src.baselines.random_policy import RandomPolicy
from src.baselines.linucb import LinUCB

SWEEP_TIMESTEPS = 200_000
N_EVAL = 50
SEED = 42

EFFECTIVENESS_SCALES = [0.5, 0.75, 1.0, 1.25, 1.5]

BASE_EFFECTIVENESS = {
    "vocab_drill": 1.0,
    "grammar_explanation": 1.2,
    "mixed_quiz": 1.4,
    "spaced_repetition": 1.5,
    "free_form": 2.0,
}


def patch_solver_effectiveness(env, scale):
    for solver in env.solvers:
        solver.effectiveness = BASE_EFFECTIVENESS[solver.name] * scale


def make_env_factory(config, scale):
    def _init():
        env = LanguageTutoringEnv(config)
        patch_solver_effectiveness(env, scale)
        return env
    return _init


def evaluate(policy, env_config, scale, n_episodes, base_seed=5000):
    env = LanguageTutoringEnv(env_config)
    patch_solver_effectiveness(env, scale)
    rewards, profs, costs = [], [], []
    solver_costs_arr = np.array([s.cost for s in env.solvers])

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=base_seed + ep)
        if hasattr(policy, "reset"):
            policy.reset()
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
    env_config = EnvConfig()
    train_config = TrainConfig()
    obs_dim = 2 * len(env_config.skills)

    sweep_dir = os.path.join(train_config.model_dir, "eff_sweep")
    Path(sweep_dir).mkdir(parents=True, exist_ok=True)

    results = []

    for scale in EFFECTIVENESS_SCALES:
        print(f"\n{'='*60}")
        print(f"  Effectiveness scale = {scale}x")
        print(f"  Solver effectiveness: " + ", ".join(
            f"{k}={v*scale:.2f}" for k, v in BASE_EFFECTIVENESS.items()))
        print(f"{'='*60}")

        print("  Training PPO...")
        vec_env = make_vec_env(make_env_factory(env_config, scale), n_envs=4, seed=SEED)
        model = PPO(
            "MlpPolicy", vec_env,
            learning_rate=3e-4,
            gamma=0.99,
            batch_size=64,
            verbose=0,
            seed=SEED,
        )
        model.learn(total_timesteps=SWEEP_TIMESTEPS)
        tag = f"ppo_eff{scale:.2f}"
        save_path = os.path.join(sweep_dir, tag)
        model.save(save_path)
        vec_env.close()

        ppo_policy = SB3PolicyWrapper(PPO.load(save_path))

        print("  Training LinUCB...")
        linucb = LinUCB(n_actions=5, obs_dim=obs_dim, alpha=1.0)
        linucb_env = LanguageTutoringEnv(env_config)
        patch_solver_effectiveness(linucb_env, scale)
        total_steps, episode = 0, 0
        while total_steps < SWEEP_TIMESTEPS:
            obs, _ = linucb_env.reset(seed=SEED + episode)
            done = False
            while not done:
                action = linucb.select_action(obs)
                next_obs, reward, terminated, truncated, _ = linucb_env.step(action)
                linucb.update(obs, action, reward, next_obs)
                obs = next_obs
                total_steps += 1
                done = terminated or truncated
            episode += 1
        linucb_env.close()

        random_policy = RandomPolicy(n_actions=5, seed=SEED)

        print("  Evaluating...")
        for name, policy in [("PPO", ppo_policy), ("LinUCB", linucb), ("Random", random_policy)]:
            pm, ps, rm, rs, cm, cs = evaluate(policy, env_config, scale, N_EVAL)
            results.append((scale, name, pm, ps, rm, rs, cm, cs))
            print(f"    {name:<8} Prof: {pm:.3f}±{ps:.3f}  Rew: {rm:+.3f}±{rs:.3f}  Cost: {cm:.1f}±{cs:.1f}")

    print(f"\n\n{'='*90}")
    print("Effectiveness Robustness Sweep (200K steps, 50 eval episodes)")
    print(f"{'='*90}")
    print(f"{'Scale':<8} {'Method':<10} {'Proficiency':<18} {'Reward':<18} {'Cost':<15}")
    print("-" * 69)
    for scale, name, pm, ps, rm, rs, cm, cs in results:
        print(f"{scale:<8} {name:<10} {pm:.3f} ± {ps:.3f}     {rm:+.3f} ± {rs:.3f}     {cm:.1f} ± {cs:.1f}")

    results_path = os.path.join(train_config.results_dir, "effectiveness_sweep.txt")
    Path(train_config.results_dir).mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        f.write("Scale,Method,Prof_Mean,Prof_Std,Reward_Mean,Reward_Std,Cost_Mean,Cost_Std\n")
        for scale, name, pm, ps, rm, rs, cm, cs in results:
            f.write(f"{scale},{name},{pm:.4f},{ps:.4f},{rm:.4f},{rs:.4f},{cm:.2f},{cs:.2f}\n")
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
