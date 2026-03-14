import argparse
import os
import pickle
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO, DQN

from src.config.env_config import EnvConfig
from src.config.train_config import TrainConfig
from src.env.env import LanguageTutoringEnv

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

N_EVAL_LEARNERS = 100


class SB3PolicyWrapper:
    def __init__(self, model):
        self.model = model

    def select_action(self, obs):
        action, _ = self.model.predict(obs, deterministic=True)
        return int(action)


def load_policies(model_dir: str, obs_dim: int) -> Dict[str, object]:
    policies = {}

    ppo_path = os.path.join(model_dir, "ppo", "best_model.zip")
    if os.path.exists(ppo_path):
        policies["PPO"] = SB3PolicyWrapper(PPO.load(ppo_path))
    else:
        ppo_final = os.path.join(model_dir, "ppo", "final_model.zip")
        if os.path.exists(ppo_final):
            policies["PPO"] = SB3PolicyWrapper(PPO.load(ppo_final))

    dqn_path = os.path.join(model_dir, "dqn", "best_model.zip")
    if os.path.exists(dqn_path):
        policies["DQN"] = SB3PolicyWrapper(DQN.load(dqn_path))
    else:
        dqn_final = os.path.join(model_dir, "dqn", "final_model.zip")
        if os.path.exists(dqn_final):
            policies["DQN"] = SB3PolicyWrapper(DQN.load(dqn_final))

    for name in ["linucb", "random", "curriculum", "mpc"]:
        pkl_path = os.path.join(model_dir, name, "policy.pkl")
        if os.path.exists(pkl_path):
            with open(pkl_path, "rb") as f:
                label = {
                    "linucb": "LinUCB", "random": "RANDOM",
                    "curriculum": "Curriculum", "mpc": "MPC",
                }[name]
                policies[label] = pickle.load(f)

    return policies


def evaluate_policy(policy, env: LanguageTutoringEnv, n_episodes: int, base_seed: int = 1000):
    episode_rewards: List[float] = []
    final_proficiencies: List[float] = []
    episode_costs: List[float] = []
    solver_counts = np.zeros(env.action_space.n)
    all_mastery_curves: List[List[float]] = []
    solver_costs = np.array([s.cost for s in env.solvers])

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=base_seed + ep)
        if hasattr(policy, "reset"):
            policy.reset()
        ep_reward = 0.0
        ep_cost = 0.0
        done = False
        mastery_curve = [float(obs[: env.K].mean())]

        while not done:
            action = policy.select_action(obs)
            solver_counts[action] += 1
            ep_cost += solver_costs[action]
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            mastery_curve.append(float(obs[: env.K].mean()))
            done = terminated or truncated

        episode_rewards.append(ep_reward)
        episode_costs.append(ep_cost)
        final_proficiencies.append(float(obs[: env.K].mean()))
        all_mastery_curves.append(mastery_curve)

    return {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "mean_proficiency": np.mean(final_proficiencies),
        "std_proficiency": np.std(final_proficiencies),
        "mean_cost": np.mean(episode_costs),
        "std_cost": np.std(episode_costs),
        "solver_distribution": solver_counts / solver_counts.sum(),
        "mastery_curves": all_mastery_curves,
    }


def plot_results(results: Dict[str, dict], results_dir: str):
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    methods = list(results.keys())
    proficiencies = [results[m]["mean_proficiency"] for m in methods]
    prof_stds = [results[m]["std_proficiency"] for m in methods]
    rewards = [results[m]["mean_reward"] for m in methods]
    reward_stds = [results[m]["std_reward"] for m in methods]

    colors = {"PPO": "#4C72B0", "DQN": "#55A868", "LinUCB": "#C44E52", "RANDOM": "#8172B2", "Curriculum": "#CCB974", "MPC": "#DD8452"}
    bar_colors = [colors.get(m, "#333333") for m in methods]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].bar(methods, proficiencies, yerr=prof_stds, capsize=5, color=bar_colors)
    axes[0].set_ylabel("Final Proficiency")
    axes[0].set_title("Final Proficiency by Method")
    axes[0].set_ylim(0, 1)

    axes[1].bar(methods, rewards, yerr=reward_stds, capsize=5, color=bar_colors)
    axes[1].set_ylabel("Cumulative Reward")
    axes[1].set_title("Cumulative Reward by Method")

    costs = [results[m]["mean_cost"] for m in methods]
    cost_stds = [results[m]["std_cost"] for m in methods]
    axes[2].bar(methods, costs, yerr=cost_stds, capsize=5, color=bar_colors)
    axes[2].set_ylabel("Total Solver Cost")
    axes[2].set_title("Computational Cost by Method")

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "comparison_bars.png"), dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 6))
    for m in methods:
        ax.errorbar(
            results[m]["mean_cost"], results[m]["mean_proficiency"],
            xerr=results[m]["std_cost"], yerr=results[m]["std_proficiency"],
            fmt="o", markersize=10, capsize=5,
            color=colors.get(m, "#333333"), label=m,
        )
    ax.set_xlabel("Total Solver Cost (per episode)")
    ax.set_ylabel("Final Proficiency")
    ax.set_title("Proficiency vs Computational Cost")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "proficiency_vs_cost.png"), dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 6))
    for method_name, res in results.items():
        curves = res["mastery_curves"]
        max_len = max(len(c) for c in curves)
        padded = np.array([c + [c[-1]] * (max_len - len(c)) for c in curves])
        mean_curve = padded.mean(axis=0)
        std_curve = padded.std(axis=0)
        steps = np.arange(max_len)
        color = colors.get(method_name, "#333333")
        ax.plot(steps, mean_curve, label=method_name, color=color)
        ax.fill_between(steps, mean_curve - std_curve, mean_curve + std_curve, alpha=0.15, color=color)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Mean Belief Mastery")
    ax.set_title("Mastery Progression Over Episode")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "mastery_curves.png"), dpi=150)
    plt.close()

    solver_names = ["VocabDrill", "Grammar", "MixedQuiz", "SpacedRep", "FreeForm"]
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(methods))
    bottom = np.zeros(len(methods))
    for s_idx, s_name in enumerate(solver_names):
        vals = [results[m]["solver_distribution"][s_idx] for m in methods]
        ax.bar(x, vals, bottom=bottom, label=s_name, width=0.5)
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("Proportion")
    ax.set_title("Solver Selection Distribution")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "solver_distribution.png"), dpi=150)
    plt.close()

    print(f"Plots saved to {results_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained agents")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    env_config = EnvConfig()
    train_config = TrainConfig()
    env = LanguageTutoringEnv(env_config)
    obs_dim = 2 * len(env_config.skills)

    policies = load_policies(train_config.model_dir, obs_dim)

    if not policies:
        print("No trained models found. Run `python -m src.train --algo all` first.")
        return

    use_wandb = not args.no_wandb and WANDB_AVAILABLE and train_config.use_wandb
    wb_run = None
    if use_wandb:
        wb_run = wandb.init(
            project=train_config.wandb_project,
            name="evaluation",
            job_type="eval",
            config={"n_eval_learners": N_EVAL_LEARNERS},
        )

    print(f"Loaded policies: {list(policies.keys())}")
    print(f"Evaluating each on {N_EVAL_LEARNERS} fresh learners...\n")

    results = {}
    for name, policy in policies.items():
        print(f"Evaluating {name}...")
        results[name] = evaluate_policy(policy, env, N_EVAL_LEARNERS)

    print("\n" + "=" * 85)
    print(f"{'Method':<12} {'Final Proficiency':<22} {'Cumulative Reward':<22} {'Total Cost':<20}")
    print("=" * 85)
    for name, res in results.items():
        print(
            f"{name:<12} "
            f"{res['mean_proficiency']:.3f} ± {res['std_proficiency']:.3f}       "
            f"{res['mean_reward']:.2f} ± {res['std_reward']:.2f}          "
            f"{res['mean_cost']:.1f} ± {res['std_cost']:.1f}"
        )
    print("=" * 85)

    plot_results(results, train_config.results_dir)

    if wb_run:
        for name, res in results.items():
            wandb.log({
                f"{name}/proficiency_mean": res["mean_proficiency"],
                f"{name}/proficiency_std": res["std_proficiency"],
                f"{name}/reward_mean": res["mean_reward"],
                f"{name}/reward_std": res["std_reward"],
                f"{name}/cost_mean": res["mean_cost"],
                f"{name}/cost_std": res["std_cost"],
            })
            for i, s_name in enumerate(["vocab_drill", "grammar", "mixed_quiz", "spaced_rep", "free_form"]):
                wandb.log({f"{name}/solver_{s_name}": res["solver_distribution"][i]})

        for fname in ["comparison_bars.png", "mastery_curves.png", "solver_distribution.png", "proficiency_vs_cost.png"]:
            fpath = os.path.join(train_config.results_dir, fname)
            if os.path.exists(fpath):
                wandb.log({fname.replace(".png", ""): wandb.Image(fpath)})

        table = wandb.Table(
            columns=["Method", "Proficiency", "Prof_Std", "Reward", "Reward_Std", "Cost", "Cost_Std"],
            data=[[name, res["mean_proficiency"], res["std_proficiency"],
                    res["mean_reward"], res["std_reward"],
                    res["mean_cost"], res["std_cost"]]
                   for name, res in results.items()],
        )
        wandb.log({"results_table": table})
        wb_run.finish()

    env.close()


if __name__ == "__main__":
    main()
