"""Paper #2 environment calibration.

Chosen by the pre-registered instrument-design step (design doc §7 risk 1),
executed 2026-07-07 BEFORE any failure metric was computed: theta2 = 1.5 and
cost_persist = 0.95 give an attention-benefit range B(1)-B(0) ~ 0.30 and
subsidy-induced attention repricing |m1* - m0*| up to ~0.175 across the
kappa grid, with the corner structure (repricing ~ 0 at kappa -> 0 and
kappa -> inf) required by H1. The 50% RC subsidy (10 -> 5) is inherited
from paper #1.
"""

from mdp2d import RustMDP2D

THETA2 = 1.5
COST_PERSIST = 0.95
RC_PRE = 10.0
SUBSIDY = 0.5

KAPPA_GRID = (0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.60, 1.00)


def env_pre(**overrides) -> RustMDP2D:
    kw = dict(theta2=THETA2, cost_persist=COST_PERSIST, rc=RC_PRE)
    kw.update(overrides)
    return RustMDP2D(**kw)


def env_post(**overrides) -> RustMDP2D:
    return env_pre(rc=RC_PRE * SUBSIDY, **overrides)
