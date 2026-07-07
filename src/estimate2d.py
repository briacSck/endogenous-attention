"""NFXP maximum likelihood for the 2-D environment: (theta1, theta2, rc)."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from mdp import Panel
from mdp2d import RustMDP2D


@dataclass
class EstimationResult2D:
    theta1: float
    theta2: float
    rc: float
    log_likelihood: float
    converged: bool


def estimate_nfxp_2d(panel: Panel, template: RustMDP2D,
                     x0: tuple[float, float, float] = (0.02, 0.3, 5.0)
                     ) -> EstimationResult2D:
    warm = {"v": None}  # warm-start VI across likelihood evaluations

    def nll(params: np.ndarray) -> float:
        theta1, theta2, rc = params
        if theta1 <= 0 or rc <= 0 or theta2 < -0.99:
            return 1e12
        mdp = RustMDP2D(n_mileage=template.n_mileage, n_cost=template.n_cost,
                        theta1=theta1, theta2=theta2, rc=rc,
                        beta=template.beta,
                        mileage_probs=template.mileage_probs,
                        cost_persist=template.cost_persist)
        sol = mdp.solve(tol=1e-8, v_init=warm["v"])
        warm["v"] = sol.v_bar
        p = np.clip(sol.ccp_replace[panel.states], 1e-12, 1 - 1e-12)
        ll = np.where(panel.choices == 1, np.log(p), np.log1p(-p))
        return -ll.sum()

    res = minimize(nll, x0=np.asarray(x0), method="Nelder-Mead",
                   options={"xatol": 1e-5, "fatol": 1e-5, "maxiter": 3000})
    return EstimationResult2D(theta1=res.x[0], theta2=res.x[1], rc=res.x[2],
                              log_likelihood=-res.fun, converged=res.success)


if __name__ == "__main__":
    truth = RustMDP2D()
    panel = truth.solve().simulate(n_buses=500, n_periods=200)
    print("estimating on optimal-policy data "
          "(truth: theta1=0.05, theta2=0.8, rc=10.0)...")
    r = estimate_nfxp_2d(panel, truth)
    print(f"theta1={r.theta1:.4f}, theta2={r.theta2:.4f}, rc={r.rc:.3f}, "
          f"converged={r.converged}")
