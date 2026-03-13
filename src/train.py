import argparse
import os
import pickle
from dataclasses import asdict
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback

from src.config.env_config import EnvConfig
from src.config.agent_config import AgentConfig
from src.config.train_config import TrainConfig
from src.env.env import LanguageTutoringEnv
from src.baselines.random_policy import RandomPolicy
from src.baselines.linucb import LinUCB
from src.baselines.curriculum import CurriculumPolicy
from src.baselines.model_based import collect_transitions, train_models, MPCPolicy

try:
    import wandb
    from wandb.integration.sb3 import WandbCallback
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def _wandb_config(env_config, agent_config, train_config, algo):
    cfg = {"algorithm": algo}
    cfg.update({f"env/{k}": v for k, v in asdict(env_config).items()})
    cfg.update({f"agent/{k}": v for k, v in asdict(agent_config).items()})
    cfg.update({f"train/{k}": v for k, v in asdict(train_config).items()})
    return cfg


def _init_wandb(algo, env_config, agent_config, train_config):
    if not train_config.use_wandb or not WANDB_AVAILABLE:
        return None
    run = wandb.init(
        project=train_config.wandb_project,
        name=algo,
        config=_wandb_config(env_config, agent_config, train_config, algo),
        reinit=True,
    )
    return run


def make_env(config: EnvConfig):
    def _init():
        return LanguageTutoringEnv(config)
    return _init


def train_ppo(env_config: EnvConfig, agent_config: AgentConfig, train_config: TrainConfig):
    print("=== Training PPO ===")
    run = _init_wandb("ppo", env_config, agent_config, train_config)

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

    callbacks = [
        EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(train_config.model_dir, "ppo"),
            log_path=log_path,
            eval_freq=train_config.eval_freq,
            n_eval_episodes=train_config.n_eval_episodes,
            deterministic=True,
        ),
    ]
    if run and WANDB_AVAILABLE:
        callbacks.append(WandbCallback(verbose=0))

    model.learn(total_timesteps=agent_config.total_timesteps, callback=callbacks)
    save_path = os.path.join(train_config.model_dir, "ppo", "final_model")
    model.save(save_path)
    print(f"PPO model saved to {save_path}")
    vec_env.close()
    eval_env.close()
    if run:
        run.finish()


def train_dqn(env_config: EnvConfig, agent_config: AgentConfig, train_config: TrainConfig):
    print("=== Training DQN ===")
    run = _init_wandb("dqn", env_config, agent_config, train_config)

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

    callbacks = [
        EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(train_config.model_dir, "dqn"),
            log_path=log_path,
            eval_freq=train_config.eval_freq,
            n_eval_episodes=train_config.n_eval_episodes,
            deterministic=True,
        ),
    ]
    if run and WANDB_AVAILABLE:
        callbacks.append(WandbCallback(verbose=0))

    model.learn(total_timesteps=agent_config.total_timesteps, callback=callbacks)
    save_path = os.path.join(train_config.model_dir, "dqn", "final_model")
    model.save(save_path)
    print(f"DQN model saved to {save_path}")
    env.close()
    eval_env.close()
    if run:
        run.finish()


def train_baseline(
    algo_name: str,
    env_config: EnvConfig,
    agent_config: AgentConfig,
    train_config: TrainConfig,
):
    print(f"=== Training {algo_name.upper()} ===")
    run = _init_wandb(algo_name, env_config, agent_config, train_config)

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
        ep_mastery_start = float(obs[:len(env_config.skills)].mean())
        done = False
        solver_counts = np.zeros(5)

        while not done:
            action = policy.select_action(obs)
            solver_counts[action] += 1
            next_obs, reward, terminated, truncated, info = env.step(action)
            policy.update(obs, action, reward, next_obs)
            obs = next_obs
            ep_reward += reward
            total_steps += 1
            done = terminated or truncated

        all_rewards.append(ep_reward)
        episode += 1

        ep_mastery_end = float(obs[:len(env_config.skills)].mean())

        if run:
            wandb.log({
                "episode": episode,
                "total_steps": total_steps,
                "episode_reward": ep_reward,
                "mastery_start": ep_mastery_start,
                "mastery_end": ep_mastery_end,
                "mastery_gain": ep_mastery_end - ep_mastery_start,
                "solver/vocab_drill": solver_counts[0] / solver_counts.sum(),
                "solver/grammar": solver_counts[1] / solver_counts.sum(),
                "solver/mixed_quiz": solver_counts[2] / solver_counts.sum(),
                "solver/spaced_rep": solver_counts[3] / solver_counts.sum(),
                "solver/free_form": solver_counts[4] / solver_counts.sum(),
            })

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
    if run:
        run.finish()


def train_curriculum(env_config: EnvConfig, train_config: TrainConfig):
    print("=== Saving Curriculum Policy ===")
    policy = CurriculumPolicy(
        skill_names=env_config.skills,
        skill_difficulties=env_config.skill_difficulties,
        episode_length=env_config.episode_length,
    )
    save_dir = os.path.join(train_config.model_dir, "curriculum")
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(save_dir, "policy.pkl"), "wb") as f:
        pickle.dump(policy, f)
    print(f"Curriculum policy saved to {save_dir}/policy.pkl")


def train_mpc(env_config: EnvConfig, agent_config: AgentConfig, train_config: TrainConfig):
    print("=== Training Model-Based MPC ===")
    run = _init_wandb("mpc", env_config, agent_config, train_config)

    n_collect_episodes = 2000
    print(f"  Collecting {n_collect_episodes} episodes of random transitions...")
    data = collect_transitions(env_config, n_episodes=n_collect_episodes, seed=train_config.seed)
    print(f"  Collected {len(data['obs'])} transitions")

    print("  Training transition and reward models...")
    trans_model, rew_model = train_models(data, n_actions=5, epochs=50, verbose=True)

    policy = MPCPolicy(
        trans_model, rew_model,
        n_actions=5, horizon=5, n_trajectories=200,
        gamma=agent_config.gamma,
    )

    save_dir = os.path.join(train_config.model_dir, "mpc")
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(save_dir, "policy.pkl"), "wb") as f:
        pickle.dump(policy, f)
    print(f"  MPC policy saved to {save_dir}/policy.pkl")

    if run:
        wandb.log({"mpc/n_transitions": len(data["obs"])})
        run.finish()


def main():
    parser = argparse.ArgumentParser(description="Train RL language tutoring agents")
    parser.add_argument(
        "--algo",
        type=str,
        default="all",
        choices=["ppo", "dqn", "linucb", "random", "curriculum", "mpc", "all"],
    )
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    env_config = EnvConfig()
    agent_config = AgentConfig()
    train_config = TrainConfig()

    if args.no_wandb:
        train_config.use_wandb = False

    if train_config.use_wandb and not WANDB_AVAILABLE:
        print("WARNING: wandb not installed. Continuing without W&B...\n")

    Path(train_config.model_dir).mkdir(parents=True, exist_ok=True)
    Path(train_config.log_dir).mkdir(parents=True, exist_ok=True)

    runners = {
        "ppo": lambda: train_ppo(env_config, agent_config, train_config),
        "dqn": lambda: train_dqn(env_config, agent_config, train_config),
        "linucb": lambda: train_baseline("linucb", env_config, agent_config, train_config),
        "random": lambda: train_baseline("random", env_config, agent_config, train_config),
        "curriculum": lambda: train_curriculum(env_config, train_config),
        "mpc": lambda: train_mpc(env_config, agent_config, train_config),
    }

    if args.algo == "all":
        for name, run in runners.items():
            run()
    else:
        runners[args.algo]()


if __name__ == "__main__":
    main()
