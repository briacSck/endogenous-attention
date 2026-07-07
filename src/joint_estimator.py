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

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from attention import perception_policy, policy_value, scale_stakes
from mdp import Panel
from mdp2d import RustMDP2D


M_GRID = np.linspace(0.0, 1.0, 11)


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
    """Warm starts + precomputed transition structure (theta-independent).

    T3 note: on this machine LAPACK dense solves are ~100x slower than
    normal (165 ms per 300x300 solve — likely emulated BLAS), so exact
    linear-solve policy evaluation LOSES to warm-started fixed-point
    iteration. We therefore iterate, with matrices precomputed once and
    values warm-started per (market, m-grid-point) across likelihood calls.
    """

    def __init__(self):
        self.v_solve = {}     # s -> V for env_s value iteration warm start
        self.v_pol = {}       # (s, m_index) -> V^pi warm start
        self.p_keep = None    # transitions do not depend on theta or s
        self.p_replace = None

    def transitions(self, mdp: RustMDP2D):
        if self.p_keep is None:
            self.p_keep = mdp.transition_matrix()
            _, c = mdp.state_grid()
            self.p_replace = self.p_keep[c]
        return self.p_keep, self.p_replace


def _policy_value_iter(u: np.ndarray, p_keep, p_replace, beta: float,
                       pol: np.ndarray, v_init=None, tol: float = 1e-3,
                       max_iter: int = 5000) -> np.ndarray:
    # tol 1e-3: B(m) only needs to rank adjacent m-grid points, whose
    # benefit gaps are ~3e-2 — precision beyond 1e-3 buys nothing (T3).
    """Warm-started fixed-point policy evaluation with precomputed
    transitions (no per-call matrix rebuilds)."""
    u_pi = (1 - pol) * u[:, 0] + pol * u[:, 1]
    v = np.zeros(len(u_pi)) if v_init is None else v_init
    for _ in range(max_iter):
        v_new = u_pi + beta * ((1 - pol) * (p_keep @ v)
                               + pol * (p_replace @ v))
        if np.max(np.abs(v_new - v)) < tol:
            return v_new
        v = v_new
    raise RuntimeError("policy evaluation did not converge")


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

    p_keep, p_replace = cache.transitions(env)
    u = env.flow_utility()
    benefits = np.empty(len(M_GRID))
    for i, m in enumerate(M_GRID):
        pol = perception_policy(sol, m)
        v = _policy_value_iter(u, p_keep, p_replace, env.beta, pol,
                               v_init=cache.v_pol.get((s, i)))
        cache.v_pol[(s, i)] = v
        benefits[i] = v.mean()

    m_s = _m_star_smooth(benefits, kappa)
    return perception_policy(sol, m_s), m_s


class _TimeBudget(Exception):
    """Raised inside the optimizer callback to stop before a kill window."""


def estimate_joint(panels: dict[float, Panel], base: RustMDP2D,
                   x0=(0.04, 1.0, 8.0, 0.3), checkpoint: str | None = None,
                   time_budget: float | None = None,
                   elapsed0: float = 0.0) -> JointResult:
    """Pooled MLE over markets {s: Panel}.

    Resumable (T3, slow-BLAS machine): pass ``checkpoint`` to persist the
    best point every optimizer iteration and ``time_budget`` (seconds) to
    stop gracefully before a background-kill window. A follow-up call that
    reads x0 from the same checkpoint restarts the simplex warm near the
    optimum, so a killed run loses at most one iteration of progress.
    """
    cache = _Cache()
    n_evals = [0]
    best = {"x": np.asarray(x0, dtype=float), "f": np.inf}
    t0 = time.time()
    ck = Path(checkpoint) if checkpoint else None

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
        if total < best["f"]:
            best["f"], best["x"] = total, np.array(psi, dtype=float)
        return total

    def _save(converged):
        ck.write_text(json.dumps({"x": np.asarray(best["x"]).tolist(),
                                  "f": best["f"], "n_evals": n_evals[0],
                                  "elapsed": elapsed0 + (time.time() - t0),
                                  "converged": bool(converged)}))

    def cb(xk):
        if ck is not None:
            _save(False)
        if time_budget is not None and time.time() - t0 > time_budget:
            raise _TimeBudget

    # fatol 0.5 on a pooled log-likelihood of magnitude ~3e4 is ample;
    # tighter budgets triple the wall clock for no parameter movement (T3).
    converged = False
    try:
        res = minimize(nll, x0=np.asarray(x0, dtype=float),
                       method="Nelder-Mead", callback=cb,
                       options={"xatol": 1e-3, "fatol": 0.5, "maxiter": 600})
        best["f"], best["x"] = res.fun, res.x
        converged = bool(res.success)
    except _TimeBudget:
        pass  # stopped on wall-clock budget; best-so-far persisted below
    if ck is not None:
        _save(converged)
    x = np.asarray(best["x"], dtype=float)
    return JointResult(theta1=x[0], theta2=x[1], rc=x[2], kappa=x[3],
                       log_likelihood=-best["f"], converged=converged,
                       n_evals=n_evals[0])


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
