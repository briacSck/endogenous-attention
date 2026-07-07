"""Ground-truth environment: Rust (1987) bus-engine replacement MDP.

This module is the "laboratory referee" of the project: it defines the true
data-generating process, solves it exactly by value iteration, and simulates
choice panels. Everything downstream (neural agents, probes, structural
estimation) is validated against the objects computed here.

Model
-----
State  x in {0, ..., n_states-1}: discretized mileage since last replacement.
Choice d in {0 (keep), 1 (replace)} each period.

Flow utility (before the Type-I extreme-value shock):
    u(x, keep)    = -theta1 * x
    u(x, replace) = -rc          (mileage resets to 0 before transiting)

Mileage transition under keep: x' = min(x + delta, n_states-1),
delta ~ Categorical(trans_probs). Under replace the same increment applies
from x = 0.

With additive T1EV shocks the choice probabilities are binary logit in the
choice-specific values v(x, d), and the integrated (ex-ante) value function
satisfies  V(x) = logsumexp_d v(x, d)  (Euler constant dropped: it shifts V
by a constant and leaves choice probabilities unchanged).
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.special import logsumexp


@dataclass
class RustMDP:
    n_states: int = 90
    theta1: float = 0.05   # maintenance cost slope per mileage bin
    rc: float = 10.0       # replacement cost
    beta: float = 0.95     # discount factor (held fixed in estimation, as in Rust 1987)
    trans_probs: tuple = (0.35, 0.60, 0.05)  # P(delta = 0, 1, 2)

    def flow_utility(self) -> np.ndarray:
        """(n_states, 2) array of flow utilities u(x, d)."""
        x = np.arange(self.n_states)
        u = np.empty((self.n_states, 2))
        u[:, 0] = -self.theta1 * x
        u[:, 1] = -self.rc
        return u

    def transition_matrix(self) -> np.ndarray:
        """(n_states, n_states) mileage transition matrix under keep."""
        k = self.n_states
        p = np.zeros((k, k))
        for x in range(k):
            for delta, prob in enumerate(self.trans_probs):
                p[x, min(x + delta, k - 1)] += prob
        return p

    def solve(self, tol: float = 1e-10, max_iter: int = 10_000) -> "Solution":
        """Solve for the integrated value function by value iteration."""
        u = self.flow_utility()
        p_keep = self.transition_matrix()
        p_replace = p_keep[0]  # replacing resets mileage to 0 before transiting

        v_bar = np.zeros(self.n_states)
        for it in range(max_iter):
            cv = np.empty_like(u)  # choice-specific values v(x, d)
            cv[:, 0] = u[:, 0] + self.beta * p_keep @ v_bar
            cv[:, 1] = u[:, 1] + self.beta * p_replace @ v_bar
            v_new = logsumexp(cv, axis=1)
            if np.max(np.abs(v_new - v_bar)) < tol:
                v_bar = v_new
                break
            v_bar = v_new
        else:
            raise RuntimeError("value iteration did not converge")

        ccp_replace = np.exp(cv[:, 1] - logsumexp(cv, axis=1))
        return Solution(mdp=self, v_bar=v_bar, choice_values=cv,
                        ccp_replace=ccp_replace, n_iterations=it + 1)


@dataclass
class Solution:
    mdp: RustMDP
    v_bar: np.ndarray          # integrated value function V(x)
    choice_values: np.ndarray  # v(x, d), shape (n_states, 2)
    ccp_replace: np.ndarray    # P(replace | x)
    n_iterations: int = 0

    def simulate(self, n_buses: int, n_periods: int,
                 rng: np.random.Generator | None = None) -> "Panel":
        """Simulate a balanced choice panel under the optimal (logit) policy."""
        rng = rng or np.random.default_rng(0)
        k = self.mdp.n_states
        deltas = rng.choice(len(self.mdp.trans_probs), p=self.mdp.trans_probs,
                            size=(n_buses, n_periods))
        states = np.zeros((n_buses, n_periods), dtype=np.int64)
        choices = np.zeros((n_buses, n_periods), dtype=np.int64)
        x = np.zeros(n_buses, dtype=np.int64)
        for t in range(n_periods):
            states[:, t] = x
            replace = rng.random(n_buses) < self.ccp_replace[x]
            choices[:, t] = replace
            base = np.where(replace, 0, x)
            x = np.minimum(base + deltas[:, t], k - 1)
        return Panel(states=states, choices=choices)


@dataclass
class Panel:
    states: np.ndarray   # (n_buses, n_periods) int mileage bins
    choices: np.ndarray  # (n_buses, n_periods) int {0, 1}

    def empirical_ccp(self, n_states: int) -> tuple[np.ndarray, np.ndarray]:
        """Frequency estimate of P(replace | x) and per-state visit counts."""
        counts = np.bincount(self.states.ravel(), minlength=n_states)
        replaces = np.bincount(self.states.ravel(),
                               weights=self.choices.ravel(), minlength=n_states)
        with np.errstate(invalid="ignore"):
            ccp = np.where(counts > 0, replaces / np.maximum(counts, 1), np.nan)
        return ccp, counts


if __name__ == "__main__":
    mdp = RustMDP()
    sol = mdp.solve()
    print(f"value iteration converged in {sol.n_iterations} iterations")
    print(f"P(replace) at x = 0, 30, 60, 89: "
          f"{sol.ccp_replace[[0, 30, 60, 89]].round(4)}")
    panel = sol.simulate(n_buses=200, n_periods=500)
    print(f"simulated replacement rate: {panel.choices.mean():.4f}")
