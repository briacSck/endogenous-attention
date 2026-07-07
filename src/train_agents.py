"""Neural agents for the ground-truth MDP (week 3).

Two agents, deliberately different *inside* while similar in behavior:

Agent A ("clone"):    small MLP trained by behavioral cloning on the optimal
                      logit policy — should internalize something like the
                      value-difference structure.
Agent B ("heuristic"): an agent trained or constructed to use a mechanistically
                      different rule (e.g., a noisy mileage threshold, or an
                      RL agent stopped early) that matches Agent A's choice
                      frequencies on the visited-state distribution but not
                      off-distribution.

The paper's punchline lives in the contrast: structural estimation (see
estimate.py) fits both; linear probes (see probes.py) decode V(x) from both
and so do NOT tell them apart — only causal patching audits do; and
counterfactuals fail for exactly the agent whose internals lack the
value-function representation.

Requires torch (see requirements.txt). State is fed as a normalized scalar
plus simple basis features so linear probes have something meaningful to
read from hidden layers.
"""

import numpy as np
import torch
import torch.nn as nn

from mdp import RustMDP, Solution


def state_features(x: np.ndarray, n_states: int) -> np.ndarray:
    """Feature map for mileage bins: normalized level, square, and cosine basis."""
    z = x / (n_states - 1)
    return np.stack([z, z**2, np.cos(np.pi * z), np.cos(2 * np.pi * z)], axis=-1)


class MLPAgent(nn.Module):
    """Small MLP mapping state features -> logit of P(replace | x)."""

    def __init__(self, n_features: int = 4, hidden: int = 32, n_layers: int = 2):
        super().__init__()
        layers: list[nn.Module] = []
        d = n_features
        for _ in range(n_layers):
            layers += [nn.Linear(d, hidden), nn.ReLU()]
            d = hidden
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)

    def hidden_activations(self, features: torch.Tensor) -> list[torch.Tensor]:
        """Post-ReLU activations per layer, for probing (see probes.py)."""
        acts, h = [], features
        for layer in self.net:
            h = layer(h)
            if isinstance(layer, nn.ReLU):
                acts.append(h.detach())
        return acts


def train_clone(solution: Solution, n_epochs: int = 2000,
                lr: float = 1e-3, seed: int = 0) -> MLPAgent:
    """Behavioral cloning: fit the MLP to the optimal CCPs on all states.

    Trains on the exact CCPs (cross-entropy against P(replace|x)) rather than
    sampled actions — cleaner ground truth, same estimand.
    """
    torch.manual_seed(seed)
    k = solution.mdp.n_states
    feats = torch.tensor(state_features(np.arange(k), k), dtype=torch.float32)
    target = torch.tensor(solution.ccp_replace, dtype=torch.float32)

    agent = MLPAgent()
    opt = torch.optim.Adam(agent.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    for _ in range(n_epochs):
        opt.zero_grad()
        loss = loss_fn(agent(feats), target)
        loss.backward()
        opt.step()
    return agent


def train_heuristic(solution: Solution, n_buckets: int = 5,
                    n_epochs: int = 2000, lr: float = 1e-3,
                    seed: int = 1) -> MLPAgent:
    """Mechanistically different agent: same MLP architecture as the clone,
    but fitted to a coarse bucket rule instead of the optimal policy.

    The target policy is the optimal CCP averaged within n_buckets mileage
    buckets (weights = ergodic visit frequencies), i.e. an agent that only
    tracks "roughly how worn is the engine" rather than a value function.
    By construction it matches optimal behavior closely where the state
    distribution has mass, and diverges in fine structure and in the tails.
    """
    torch.manual_seed(seed)
    k = solution.mdp.n_states

    panel = solution.simulate(n_buses=2000, n_periods=200,
                              rng=np.random.default_rng(seed))
    _, counts = panel.empirical_ccp(k)
    weights = counts + 1e-9

    edges = np.linspace(0, k, n_buckets + 1).astype(int)
    target_np = np.empty(k)
    for b in range(n_buckets):
        lo, hi = edges[b], edges[b + 1]
        target_np[lo:hi] = np.average(solution.ccp_replace[lo:hi],
                                      weights=weights[lo:hi])

    feats = torch.tensor(state_features(np.arange(k), k), dtype=torch.float32)
    target = torch.tensor(np.clip(target_np, 1e-6, 1 - 1e-6),
                          dtype=torch.float32)
    agent = MLPAgent()
    opt = torch.optim.Adam(agent.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    for _ in range(n_epochs):
        opt.zero_grad()
        loss = loss_fn(agent(feats), target)
        loss.backward()
        opt.step()
    return agent


def simulate_agent(agent: MLPAgent, mdp: RustMDP, n_buses: int, n_periods: int,
                   seed: int = 0):
    """Simulate a Panel under the agent's policy (mirror of Solution.simulate)."""
    rng = np.random.default_rng(seed)
    k = mdp.n_states
    feats = torch.tensor(state_features(np.arange(k), k), dtype=torch.float32)
    with torch.no_grad():
        ccp = torch.sigmoid(agent(feats)).numpy()

    from mdp import Panel
    deltas = rng.choice(len(mdp.trans_probs), p=mdp.trans_probs,
                        size=(n_buses, n_periods))
    states = np.zeros((n_buses, n_periods), dtype=np.int64)
    choices = np.zeros((n_buses, n_periods), dtype=np.int64)
    x = np.zeros(n_buses, dtype=np.int64)
    for t in range(n_periods):
        states[:, t] = x
        replace = rng.random(n_buses) < ccp[x]
        choices[:, t] = replace
        x = np.minimum(np.where(replace, 0, x) + deltas[:, t], k - 1)
    return Panel(states=states, choices=choices)


if __name__ == "__main__":
    mdp = RustMDP()
    sol = mdp.solve()
    agent = train_clone(sol)
    feats = torch.tensor(state_features(np.arange(mdp.n_states), mdp.n_states),
                         dtype=torch.float32)
    with torch.no_grad():
        fitted = torch.sigmoid(agent(feats)).numpy()
    err = np.max(np.abs(fitted - sol.ccp_replace))
    print(f"clone max CCP error vs optimal policy: {err:.5f}")
