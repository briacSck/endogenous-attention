"""Neural agents for the 2-D environment, as a one-parameter family.

Agent_lambda is trained on the target policy
    target_lam(x, c) = lam * CCP*(x, c) + (1 - lam) * CCPbar*(x)
where CCP* is the optimal policy and CCPbar* its c-average (the cost-regime-
blind policy). lam = 1 is the full clone; lam = 0 is an agent that never uses
the cost regime; intermediate lam under-uses it.

All agents share the same architecture and the same inputs (including c), so
any difference in whether c is *causally used* lives in the learned weights —
exactly what the internal audit must detect. The family exists to test the
lit review's Q2: does an internal-audit statistic predict counterfactual
error monotonically across agents?
"""

import numpy as np
import torch
import torch.nn as nn

from mdp2d import RustMDP2D, Solution2D
from train_agents import MLPAgent


def state_features_2d(x: np.ndarray, c: np.ndarray, mdp: RustMDP2D) -> np.ndarray:
    xn = x / (mdp.n_mileage - 1)
    cn = c / (mdp.n_cost - 1)
    return np.stack([xn, cn, xn**2, xn * cn,
                     np.cos(np.pi * xn), np.cos(2 * np.pi * xn)], axis=-1)


def all_state_features(mdp: RustMDP2D) -> torch.Tensor:
    x, c = mdp.state_grid()
    return torch.tensor(state_features_2d(x, c, mdp), dtype=torch.float32)


def target_policy(solution: Solution2D, lam: float) -> np.ndarray:
    """Blend of optimal and cost-regime-blind policies, flattened."""
    grid = solution.ccp_grid()                       # (n_mileage, n_cost)
    blind = grid.mean(axis=1, keepdims=True)         # stationary c is uniform
    return (lam * grid + (1 - lam) * blind).ravel()


def train_agent_2d(solution: Solution2D, lam: float, n_epochs: int = 3000,
                   lr: float = 1e-3, seed: int = 0) -> MLPAgent:
    torch.manual_seed(seed)
    mdp = solution.mdp
    feats = all_state_features(mdp)
    target = torch.tensor(np.clip(target_policy(solution, lam), 1e-6, 1 - 1e-6),
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


def agent_policy(agent: MLPAgent, mdp: RustMDP2D) -> np.ndarray:
    """Flattened CCPs implied by the agent, over the full state grid."""
    with torch.no_grad():
        return torch.sigmoid(agent(all_state_features(mdp))).numpy()


if __name__ == "__main__":
    sol = RustMDP2D().solve()
    for lam in (1.0, 0.0):
        agent = train_agent_2d(sol, lam)
        pol = agent_policy(agent, sol.mdp).reshape(sol.mdp.n_mileage,
                                                   sol.mdp.n_cost)
        spread = np.abs(pol[:, -1] - pol[:, 0]).max()
        print(f"lambda={lam}: max policy spread across cost regimes = "
              f"{spread:.4f} (clone high, blind ~0)")
