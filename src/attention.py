"""Endogenous sparse-max attention (Gabaix 2014 style) on the 2-D bus MDP.

The agent perceives the cost regime as c' = m*c + (1-m)*cbar and acts
optimally on the perceived state. Attention m is CHOSEN:

    m*(kappa, env) = argmax_{m in [0,1]}  B(m; env) - kappa * m^2

where B(m; env) is the expected discounted payoff of the m-perception policy
in environment env, computed exactly by policy evaluation on the solved MDP.
Expected flow utilities only (no taste-shock entropy term) — a simplification
stated in the design doc, to be stress-tested in an appendix.

Stakes enter through the environment: `scale_stakes(mdp, s)` scales flow
utilities by s while holding the T1EV shock scale at 1, so higher stakes
raise the value of attending.
"""

from dataclasses import replace

import numpy as np

from mdp2d import RustMDP2D


def scale_stakes(mdp: RustMDP2D, s: float) -> "ScaledMDP":
    """Environment with all flow utilities scaled by s (shock scale fixed)."""
    return ScaledMDP(mdp, s)


class ScaledMDP(RustMDP2D):
    """RustMDP2D with flow utilities scaled by a stakes shifter s."""

    def __init__(self, base: RustMDP2D, s: float):
        super().__init__(n_mileage=base.n_mileage, n_cost=base.n_cost,
                         theta1=base.theta1, theta2=base.theta2, rc=base.rc,
                         beta=base.beta, mileage_probs=base.mileage_probs,
                         cost_persist=base.cost_persist)
        self.stakes = s

    def flow_utility(self) -> np.ndarray:
        return self.stakes * super().flow_utility()


def perception_policy(solution, m: float) -> np.ndarray:
    """Flattened CCPs of the sparse-max agent with attention m.

    Acts optimally on the perceived regime c' = m*c + (1-m)*cbar
    (linear interpolation between the optimal policy's columns) —
    the `sparse_attention` construction of paper #1's zoo_v2.
    """
    grid = solution.ccp_grid()
    kx, kc = grid.shape
    cbar = (kc - 1) / 2
    out = np.empty_like(grid)
    for c in range(kc):
        cp = m * c + (1 - m) * cbar
        lo = int(np.floor(cp))
        hi = min(lo + 1, kc - 1)
        w = cp - lo
        out[:, c] = (1 - w) * grid[:, lo] + w * grid[:, hi]
    return out.ravel()


def policy_value(mdp: RustMDP2D, policy_flat: np.ndarray,
                 tol: float = 1e-9, max_iter: int = 20_000,
                 v_init: np.ndarray | None = None) -> np.ndarray:
    """Exact policy evaluation: V^pi(s) under stochastic policy pi.

    V^pi = sum_d pi(d|s) [u(s,d) + beta * P_d V^pi], expected flow utilities
    only (no shock term). `v_init` warm-starts the iteration — essential
    inside the joint estimator, where successive candidate parameters are
    close and cold starts would dominate the likelihood cost.
    """
    u = mdp.flow_utility()
    p_keep = mdp.transition_matrix()
    _, c = mdp.state_grid()
    p_replace = p_keep[c]
    p1 = policy_flat
    p0 = 1 - p1

    v = np.zeros(mdp.n_states) if v_init is None else v_init.copy()
    for _ in range(max_iter):
        v_new = (p0 * (u[:, 0] + mdp.beta * (p_keep @ v))
                 + p1 * (u[:, 1] + mdp.beta * (p_replace @ v)))
        if np.max(np.abs(v_new - v)) < tol:
            return v_new
        v = v_new
    raise RuntimeError("policy evaluation did not converge")


def attention_benefit(mdp: RustMDP2D, m: float,
                      solution=None) -> float:
    """B(m): expected discounted payoff of the m-perception policy,
    averaged over a uniform initial state distribution."""
    sol = solution if solution is not None else mdp.solve()
    pol = perception_policy(sol, m)
    return float(policy_value(mdp, pol).mean())


def m_star(kappa: float, mdp: RustMDP2D, grid_size: int = 21,
           solution=None) -> float:
    """Chosen attention: argmax_m B(m) - kappa * m^2 over an m grid."""
    sol = solution if solution is not None else mdp.solve()
    ms = np.linspace(0.0, 1.0, grid_size)
    objective = [attention_benefit(mdp, m, solution=sol) - kappa * m**2
                 for m in ms]
    return float(ms[int(np.argmax(objective))])


if __name__ == "__main__":
    base = RustMDP2D()
    for s in (0.5, 1.0, 2.0):
        env = scale_stakes(base, s)
        sol = env.solve()
        for kappa in (0.0, 0.5, 5.0):
            print(f"stakes={s}, kappa={kappa}: m* = "
                  f"{m_star(kappa, env, solution=sol):.2f}")
