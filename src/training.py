"""MLP training on a target policy grid (ported from paper #1's zoo)."""

import numpy as np
import torch
import torch.nn as nn

from mdp2d import RustMDP2D
from train_agents import MLPAgent
from train_agents2d import all_state_features


def train_on_target(target_flat: np.ndarray, mdp: RustMDP2D,
                    n_epochs: int = 3000, lr: float = 1e-3,
                    seed: int = 0) -> MLPAgent:
    torch.manual_seed(seed)
    feats = all_state_features(mdp)
    target = torch.tensor(np.clip(target_flat, 1e-6, 1 - 1e-6),
                          dtype=torch.float32)
    agent = MLPAgent(n_features=feats.shape[1])
    opt = torch.optim.Adam(agent.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    for _ in range(n_epochs):
        opt.zero_grad()
        loss = loss_fn(agent(feats), target)
        loss.backward()
        opt.step()
    return agent
