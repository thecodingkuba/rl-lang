"""
Model-Based RL via learned transition/reward models + Model Predictive Control.

Pipeline:
  1. Collect (obs, action, reward, next_obs) transitions from random rollouts
  2. Train an MLP transition model:  f(obs, action_onehot) -> next_obs
  3. Train an MLP reward model:      g(obs, action_onehot) -> reward
  4. At test time, use MPC: simulate H steps ahead for each action sequence,
     pick the first action of the best trajectory.
"""

import os
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from numpy.typing import NDArray

from src.config.env_config import EnvConfig
from src.env.env import LanguageTutoringEnv


class TransitionModel(nn.Module):
    """Predicts next_obs given (obs, action_onehot)."""

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + n_actions, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, obs_dim),
        )

    def forward(self, obs, action_onehot):
        x = torch.cat([obs, action_onehot], dim=-1)
        return self.net(x)


class RewardModel(nn.Module):
    """Predicts scalar reward given (obs, action_onehot)."""

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + n_actions, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs, action_onehot):
        x = torch.cat([obs, action_onehot], dim=-1)
        return self.net(x).squeeze(-1)


class MPCPolicy:
    """
    Model Predictive Control using learned transition and reward models.

    At each step, simulates all possible action sequences for H steps,
    evaluates cumulative reward, and picks the first action of the best
    trajectory. Uses random shooting for tractability.
    """

    def __init__(
        self,
        transition_model: TransitionModel,
        reward_model: RewardModel,
        n_actions: int = 5,
        horizon: int = 5,
        n_trajectories: int = 200,
        gamma: float = 0.99,
        device: str = "cpu",
    ):
        self.transition_model = transition_model.to(device)
        self.reward_model = reward_model.to(device)
        self.n_actions = n_actions
        self.horizon = horizon
        self.n_trajectories = n_trajectories
        self.gamma = gamma
        self.device = device

        self.transition_model.eval()
        self.reward_model.eval()

    def select_action(self, obs: NDArray[np.float64]) -> int:
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
            obs_batch = obs_t.unsqueeze(0).expand(self.n_trajectories, -1)

            # sample random action sequences: (n_traj, horizon)
            action_seqs = torch.randint(
                0, self.n_actions, (self.n_trajectories, self.horizon),
                device=self.device,
            )

            cumulative_rewards = torch.zeros(self.n_trajectories, device=self.device)
            current_obs = obs_batch.clone()

            for t in range(self.horizon):
                actions = action_seqs[:, t]
                action_onehot = torch.zeros(
                    self.n_trajectories, self.n_actions, device=self.device
                )
                action_onehot.scatter_(1, actions.unsqueeze(1), 1.0)

                reward = self.reward_model(current_obs, action_onehot)
                cumulative_rewards += (self.gamma ** t) * reward

                current_obs = self.transition_model(current_obs, action_onehot)

            # group trajectories by first action, pick the best mean
            first_actions = action_seqs[:, 0]
            best_action = 0
            best_value = -float("inf")
            for a in range(self.n_actions):
                mask = first_actions == a
                if mask.any():
                    mean_val = cumulative_rewards[mask].mean().item()
                    if mean_val > best_value:
                        best_value = mean_val
                        best_action = a

            return best_action

    def update(self, obs, action, reward, next_obs) -> None:
        pass


def collect_transitions(
    env_config: EnvConfig,
    n_episodes: int = 2000,
    seed: int = 0,
) -> dict:
    """Collect transitions from random rollouts."""
    env = LanguageTutoringEnv(env_config)
    all_obs, all_actions, all_rewards, all_next_obs = [], [], [], []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        while not done:
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, _ = env.step(action)
            all_obs.append(obs.copy())
            all_actions.append(action)
            all_rewards.append(reward)
            all_next_obs.append(next_obs.copy())
            obs = next_obs
            done = terminated or truncated

    env.close()
    return {
        "obs": np.array(all_obs),
        "actions": np.array(all_actions),
        "rewards": np.array(all_rewards),
        "next_obs": np.array(all_next_obs),
    }


def train_models(
    data: dict,
    n_actions: int = 5,
    hidden: int = 128,
    epochs: int = 50,
    batch_size: int = 512,
    lr: float = 1e-3,
    val_fraction: float = 0.1,
    verbose: bool = True,
) -> tuple:
    """Train transition and reward models on collected data."""
    obs_dim = data["obs"].shape[1]
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    n = len(data["obs"])
    idx = np.random.permutation(n)
    n_val = int(n * val_fraction)
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    obs_all = torch.tensor(data["obs"], dtype=torch.float32)
    act_onehot = torch.zeros(n, n_actions)
    act_onehot.scatter_(1, torch.tensor(data["actions"]).unsqueeze(1), 1.0)
    rewards_all = torch.tensor(data["rewards"], dtype=torch.float32)
    next_obs_all = torch.tensor(data["next_obs"], dtype=torch.float32)

    train_ds = TensorDataset(
        obs_all[train_idx], act_onehot[train_idx],
        rewards_all[train_idx], next_obs_all[train_idx],
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    # validation data
    val_obs = obs_all[val_idx].to(device)
    val_act = act_onehot[val_idx].to(device)
    val_rew = rewards_all[val_idx].to(device)
    val_next = next_obs_all[val_idx].to(device)

    trans_model = TransitionModel(obs_dim, n_actions, hidden).to(device)
    rew_model = RewardModel(obs_dim, n_actions, hidden).to(device)

    trans_opt = optim.Adam(trans_model.parameters(), lr=lr)
    rew_opt = optim.Adam(rew_model.parameters(), lr=lr)
    mse = nn.MSELoss()

    best_trans_loss = float("inf")
    best_rew_loss = float("inf")
    best_trans_state = None
    best_rew_state = None

    for epoch in range(epochs):
        trans_model.train()
        rew_model.train()
        epoch_trans_loss = 0.0
        epoch_rew_loss = 0.0

        for obs_b, act_b, rew_b, next_b in train_loader:
            obs_b = obs_b.to(device)
            act_b = act_b.to(device)
            rew_b = rew_b.to(device)
            next_b = next_b.to(device)

            # transition model
            trans_opt.zero_grad()
            pred_next = trans_model(obs_b, act_b)
            t_loss = mse(pred_next, next_b)
            t_loss.backward()
            trans_opt.step()
            epoch_trans_loss += t_loss.item() * len(obs_b)

            # reward model
            rew_opt.zero_grad()
            pred_rew = rew_model(obs_b, act_b)
            r_loss = mse(pred_rew, rew_b)
            r_loss.backward()
            rew_opt.step()
            epoch_rew_loss += r_loss.item() * len(obs_b)

        # validation
        trans_model.eval()
        rew_model.eval()
        with torch.no_grad():
            val_trans_loss = mse(trans_model(val_obs, val_act), val_next).item()
            val_rew_loss = mse(rew_model(val_obs, val_act), val_rew).item()

        if val_trans_loss < best_trans_loss:
            best_trans_loss = val_trans_loss
            best_trans_state = {k: v.cpu().clone() for k, v in trans_model.state_dict().items()}
        if val_rew_loss < best_rew_loss:
            best_rew_loss = val_rew_loss
            best_rew_state = {k: v.cpu().clone() for k, v in rew_model.state_dict().items()}

        if verbose and (epoch + 1) % 10 == 0:
            n_train = len(train_idx)
            print(
                f"  Epoch {epoch+1:3d}/{epochs}"
                f"  trans_loss={epoch_trans_loss/n_train:.6f}"
                f"  rew_loss={epoch_rew_loss/n_train:.6f}"
                f"  val_trans={val_trans_loss:.6f}"
                f"  val_rew={val_rew_loss:.6f}"
            )

    if best_trans_state:
        trans_model.load_state_dict(best_trans_state)
    if best_rew_state:
        rew_model.load_state_dict(best_rew_state)

    trans_model.cpu()
    rew_model.cpu()

    if verbose:
        print(f"  Best val transition MSE: {best_trans_loss:.6f}")
        print(f"  Best val reward MSE:     {best_rew_loss:.6f}")

    return trans_model, rew_model
