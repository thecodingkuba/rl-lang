"""
Behavior Cloning: train a supervised MLP to imitate the best RL policy (DQN).

Pipeline:
  1. Load the trained DQN model
  2. Roll it out to collect (observation, action) demonstration pairs
  3. Train an MLP classifier via cross-entropy loss
  4. Save the trained BC policy for evaluation
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
from stable_baselines3 import DQN

from src.agent.policy import SB3PolicyWrapper
from src.config.env_config import EnvConfig
from src.env.env import LanguageTutoringEnv


class BCNet(nn.Module):
    """Two-hidden-layer MLP for action classification."""

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class BCPolicy:
    """Wraps a trained BCNet for the common select_action interface."""

    def __init__(self, model: BCNet, device: str = "cpu"):
        self.model = model
        self.device = device
        self.model.eval()

    def select_action(self, obs: NDArray[np.float64]) -> int:
        with torch.no_grad():
            x = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            logits = self.model(x)
            return int(logits.argmax(dim=1).item())

    def update(self, obs, action, reward, next_obs) -> None:
        pass


def collect_demonstrations(
    env_config: EnvConfig,
    dqn_model_dir: str,
    n_episodes: int = 500,
    seed: int = 0,
) -> tuple:
    """Roll out the trained DQN to collect (obs, action) pairs."""
    dqn_path = os.path.join(dqn_model_dir, "best_model.zip")
    if not os.path.exists(dqn_path):
        dqn_path = os.path.join(dqn_model_dir, "final_model.zip")
    expert = SB3PolicyWrapper(DQN.load(dqn_path))

    env = LanguageTutoringEnv(env_config)
    all_obs, all_actions = [], []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        while not done:
            action = expert.select_action(obs)
            all_obs.append(obs.copy())
            all_actions.append(action)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

    env.close()
    return np.array(all_obs), np.array(all_actions)


def train_bc(
    obs: np.ndarray,
    actions: np.ndarray,
    n_actions: int = 5,
    hidden: int = 64,
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
    val_fraction: float = 0.1,
    verbose: bool = True,
) -> BCNet:
    """Train a BCNet on demonstration data."""
    obs_dim = obs.shape[1]
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    n_val = int(len(obs) * val_fraction)
    idx = np.random.permutation(len(obs))
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    X_train = torch.tensor(obs[train_idx], dtype=torch.float32)
    y_train = torch.tensor(actions[train_idx], dtype=torch.long)
    X_val = torch.tensor(obs[val_idx], dtype=torch.float32)
    y_val = torch.tensor(actions[val_idx], dtype=torch.long)

    train_loader = DataLoader(
        TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True
    )

    model = BCNet(obs_dim, n_actions, hidden).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_state = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(xb)

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val.to(device))
            val_preds = val_logits.argmax(dim=1).cpu()
            val_acc = (val_preds == y_val).float().mean().item()
            train_loss = total_loss / len(X_train)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if verbose and (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs}  loss={train_loss:.4f}  val_acc={val_acc:.3f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    model.cpu()

    if verbose:
        print(f"  Best validation accuracy: {best_val_acc:.3f}")

    return model
