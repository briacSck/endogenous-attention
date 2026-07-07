"""Coherent joint deep estimator: (theta1, theta2, RC, kappa) by pooled
full-solution MLE with the attention fixed point inside the likelihood.

Pre-registered: design/prereg_joint_estimator.md (committed before this
implementation). Model-implied CCPs for market s at psi:

    env_s(psi) = ScaledMDP(theta1, theta2, RC; s)
    m_s(psi)   = argmax_m B(m; env_s) - kappa * m^2   (quadratic-smoothed)
    CCP_s(psi) = perception_policy(solve(env_s), m_s)

Numerical mitigations (all pre-committed):
- m* continuous via local quadratic fit around the grid argmax (trap 1);
- warm-started value iteration AND per-(market, m-point) warm-started
  policy evaluations cached across likelihood calls (trap 2);
- kappa profile likelihood reported for corner/set-identification (trap 3);
- utilities are s * u(theta) with s observable: 4 common parameters, no
  per-market scales (trap 4).
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from attention import perception_policy, policy_value, scale_stakes
from mdp import Panel
from mdp2d import RustMDP2D


M_GRID = np.linspace(0.0, 1.0, 21)


@dataclass
class JointResult:
    theta1: float
    theta2: float
    rc: float
    kappa: float
    log_likelihood: float
    converged: bool
    n_evals: int = 0


class _Cache:
    """Warm starts across likelihood evaluations."""

    def __init__(self):
        self.v_solve = {}   # s -> V for env_s value iteration
        self.v_pol = {}     # (s, m_index) -> V^pi for policy evaluation


def _m_star_smooth(benefits: np.ndarray, kappa: float) -> float:
    """Continuous maximizer of B(m) - kappa m^2 via local quadratic fit."""
    obj = benefits - kappa * M_GRID**2
    i = int(np.argmax(obj))
    if i == 0 or i == len(M_GRID) - 1:
        return float(M_GRID[i])
    x = M_GRID[i - 1:i + 2]
    y = obj[i - 1:i + 2]
    denom = (x[0] - x[1]) * (x[0] - x[2]) * (x[1] - x[2])
    a = (x[2] * (y[1] - y[0]) + x[1] * (y[0] - y[2])
         + x[0] * (y[2] - y[1])) / denom
    b = (x[2]**2 * (y[0] - y[1]) + x[1]**2 * (y[2] - y[0])
         + x[0]**2 * (y[1] - y[2])) / denom
    if a >= 0:  # not locally concave; keep grid argmax
        return float(M_GRID[i])
    return float(np.clip(-b / (2 * a), M_GRID[i - 1], M_GRID[i + 1]))


def market_ccp(psi: np.ndarray, s: float, base: RustMDP2D,
               cache: _Cache) -> tuple[np.ndarray, float]:
    """Model-implied flattened CCPs and chosen attention for market s."""
    theta1, theta2, rc, kappa = psi
    env = scale_stakes(
        RustMDP2D(n_mileage=base.n_mileage, n_cost=base.n_cost,
                  theta1=theta1, theta2=theta2, rc=rc, beta=base.beta,
                  mileage_probs=base.mileage_probs,
                  cost_persist=base.cost_persist), s)
    sol = env.solve(tol=1e-8, v_init=cache.v_solve.get(s))
    cache.v_solve[s] = sol.v_bar

    benefits = np.empty(len(M_GRID))
    for i, m in enumerate(M_GRID):
        pol = perception_policy(sol, m)
        v = policy_value(env, pol, tol=1e-7,
                         v_init=cache.v_pol.get((s, i)))
        cache.v_pol[(s, i)] = v
        benefits[i] = v.mean()

    m_s = _m_star_smooth(benefits, kappa)
    return perception_policy(sol, m_s), m_s


def estimate_joint(panels: dict[float, Panel], base: RustMDP2D,
                   x0=(0.04, 1.0, 8.0, 0.3)) -> JointResult:
    """Pooled MLE over markets {s: Panel}."""
    cache = _Cache()
    n_evals = [0]

    def nll(psi: np.ndarray) -> float:
        theta1, theta2, rc, kappa = psi
        if theta1 <= 0 or rc <= 0 or kappa < 0 or theta2 < -0.99:
            return 1e12
        n_evals[0] += 1
        total = 0.0
        for s, panel in panels.items():
            ccp, _ = market_ccp(psi, s, base, cache)
            p = np.clip(ccp[panel.states], 1e-12, 1 - 1e-12)
            total -= float(np.sum(np.where(panel.choices == 1,
                                           np.log(p), np.log1p(-p))))
        return total

    res = minimize(nll, x0=np.asarray(x0, dtype=float),
                   method="Nelder-Mead",
                   options={"xatol": 1e-4, "fatol": 1e-3, "maxiter": 2000})
    return JointResult(theta1=res.x[0], theta2=res.x[1], rc=res.x[2],
                       kappa=res.x[3], log_likelihood=-res.fun,
                       converged=res.success, n_evals=n_evals[0])


def kappa_profile(panels: dict[float, Panel], base: RustMDP2D,
                  psi_hat: np.ndarray,
                  kappa_grid: np.ndarray) -> np.ndarray:
    """Profile log-likelihood in kappa at the joint optimum's other
    coordinates (cheap diagnostic for corner set-identification)."""
    cache = _Cache()
    lls = []
    for kappa in kappa_grid:
        psi = np.array([psi_hat[0], psi_hat[1], psi_hat[2], kappa])
        total = 0.0
        for s, panel in panels.items():
            ccp, _ = market_ccp(psi, s, base, cache)
            p = np.clip(ccp[panel.states], 1e-12, 1 - 1e-12)
            total += float(np.sum(np.where(panel.choices == 1,
                                           np.log(p), np.log1p(-p))))
        lls.append(total)
    return np.array(lls)
