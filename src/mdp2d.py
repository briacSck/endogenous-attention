"""2-D ground-truth environment: bus replacement with an observable cost shifter.

Extension of mdp.py answering the lit review's Q3 ("does the diagnostic
survive on a multi-dimensional state?") and enabling Q1 (causal separation).

State  s = (x, c): mileage bin x in {0..n_mileage-1} and an observable cost
regime c in {0..n_cost-1} (think fuel price, labor cost, or a compliance
regime) following a persistent Markov chain independent of choices.

Flow utility:
    u((x, c), keep)    = -theta1 * x * (1 + theta2 * c / (n_cost - 1))
    u((x, c), replace) = -rc

theta2 governs how much the cost regime scales maintenance costs. An agent
that ignores c (the 2-D heuristic) can stay behaviorally close on the ergodic
distribution but has no internal representation of the regime to use — which
is exactly what causal patching should detect and what a regime-shift
counterfactual should expose.

States are flattened as s = x * n_cost + c so Panel utilities from mdp.py
can be reused.
"""

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp

from mdp import Panel


@dataclass
class RustMDP2D:
    n_mileage: int = 60
    n_cost: int = 5
    theta1: float = 0.05
    theta2: float = 0.8
    rc: float = 10.0
    beta: float = 0.95
    mileage_probs: tuple = (0.35, 0.60, 0.05)
    cost_persist: float = 0.9  # P(c' = c); rest split between c +/- 1

    @property
    def n_states(self) -> int:
        return self.n_mileage * self.n_cost

    def state_grid(self) -> tuple[np.ndarray, np.ndarray]:
        """(x, c) arrays of length n_states, in flattened order."""
        s = np.arange(self.n_states)
        return s // self.n_cost, s % self.n_cost

    def flow_utility(self) -> np.ndarray:
        x, c = self.state_grid()
        u = np.empty((self.n_states, 2))
        u[:, 0] = -self.theta1 * x * (1 + self.theta2 * c / (self.n_cost - 1))
        u[:, 1] = -self.rc
        return u

    def cost_transition(self) -> np.ndarray:
        kc = self.n_cost
        pc = np.zeros((kc, kc))
        spill = 1 - self.cost_persist
        for c in range(kc):
            pc[c, c] += self.cost_persist
            lo, hi = max(c - 1, 0), min(c + 1, kc - 1)
            pc[c, lo] += spill / 2
            pc[c, hi] += spill / 2
        return pc

    def transition_matrix(self) -> np.ndarray:
        """Flattened transition under keep: mileage and cost move independently."""
        kx, kc = self.n_mileage, self.n_cost
        px = np.zeros((kx, kx))
        for x in range(kx):
            for delta, prob in enumerate(self.mileage_probs):
                px[x, min(x + delta, kx - 1)] += prob
        return np.kron(px, self.cost_transition())

    def solve(self, tol: float = 1e-10, max_iter: int = 20_000,
              v_init: np.ndarray | None = None) -> "Solution2D":
        u = self.flow_utility()
        p_keep = self.transition_matrix()
        # Replacing resets mileage to 0 (then transits); cost regime unaffected
        # by the choice, so the replace-transition rows are those of x = 0
        # paired with the current c.
        _, c = self.state_grid()
        p_replace = p_keep[c]  # row for flattened state (0, c) is index c

        # v_init warm-starts value iteration; used by NFXP where successive
        # candidate parameters are close and convergence takes a few passes.
        v_bar = np.zeros(self.n_states) if v_init is None else v_init.copy()
        for it in range(max_iter):
            cv = np.empty_like(u)
            cv[:, 0] = u[:, 0] + self.beta * p_keep @ v_bar
            cv[:, 1] = u[:, 1] + self.beta * p_replace @ v_bar
            v_new = logsumexp(cv, axis=1)
            if np.max(np.abs(v_new - v_bar)) < tol:
                v_bar = v_new
                break
            v_bar = v_new
        else:
            raise RuntimeError("value iteration did not converge")

        ccp = np.exp(cv[:, 1] - logsumexp(cv, axis=1))
        return Solution2D(mdp=self, v_bar=v_bar, choice_values=cv,
                          ccp_replace=ccp, n_iterations=it + 1)


@dataclass
class Solution2D:
    mdp: RustMDP2D
    v_bar: np.ndarray
    choice_values: np.ndarray
    ccp_replace: np.ndarray   # flattened (x * n_cost + c)
    n_iterations: int = 0

    def ccp_grid(self) -> np.ndarray:
        """CCP reshaped to (n_mileage, n_cost)."""
        return self.ccp_replace.reshape(self.mdp.n_mileage, self.mdp.n_cost)

    def simulate(self, n_buses: int, n_periods: int,
                 rng: np.random.Generator | None = None,
                 policy: np.ndarray | None = None) -> Panel:
        """Simulate a panel; `policy` (flattened CCPs) defaults to optimal."""
        rng = rng or np.random.default_rng(0)
        ccp = self.ccp_replace if policy is None else policy
        mdp = self.mdp
        kx, kc = mdp.n_mileage, mdp.n_cost
        pc = mdp.cost_transition()
        pc_cum = pc.cumsum(axis=1)

        states = np.zeros((n_buses, n_periods), dtype=np.int64)
        choices = np.zeros((n_buses, n_periods), dtype=np.int64)
        x = np.zeros(n_buses, dtype=np.int64)
        c = rng.integers(0, kc, size=n_buses)
        for t in range(n_periods):
            s = x * kc + c
            states[:, t] = s
            replace = rng.random(n_buses) < ccp[s]
            choices[:, t] = replace
            base = np.where(replace, 0, x)
            deltas = rng.choice(len(mdp.mileage_probs), p=mdp.mileage_probs,
                                size=n_buses)
            x = np.minimum(base + deltas, kx - 1)
            c = (rng.random(n_buses)[:, None] > pc_cum[c]).sum(axis=1)
        return Panel(states=states, choices=choices)


if __name__ == "__main__":
    mdp = RustMDP2D()
    sol = mdp.solve()
    grid = sol.ccp_grid()
    print(f"value iteration converged in {sol.n_iterations} iterations")
    print(f"P(replace | x=40, c): {grid[40].round(3)}  (should rise with c)")
    panel = sol.simulate(200, 300)
    print(f"simulated replacement rate: {panel.choices.mean():.4f}")
