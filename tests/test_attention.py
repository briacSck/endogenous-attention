"""Comparative-statics tests for the endogenous-attention module.

These encode the design doc's predictions (design/research_design.md §2.3):
m* increasing in stakes, decreasing in kappa; corner solutions at the
extremes; and B(m) increasing in m (attention is weakly valuable).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from attention import (attention_benefit, m_star, perception_policy,
                       policy_value, scale_stakes)
from mdp2d import RustMDP2D


@pytest.fixture(scope="module")
def base():
    return RustMDP2D()


@pytest.fixture(scope="module")
def solution(base):
    return base.solve()


def test_policy_value_converges_and_is_negative(base, solution):
    v = policy_value(base, solution.ccp_replace)
    assert np.all(np.isfinite(v))
    assert v.mean() < 0  # all flow utilities are costs


def test_full_attention_policy_is_optimal_policy(solution):
    pol = perception_policy(solution, 1.0)
    assert np.allclose(pol, solution.ccp_replace)


def test_benefit_weakly_increasing_in_attention(base, solution):
    bs = [attention_benefit(base, m, solution=solution)
          for m in (0.0, 0.5, 1.0)]
    assert bs[0] <= bs[1] + 1e-9
    assert bs[1] <= bs[2] + 1e-9


def test_m_star_corner_solutions(base, solution):
    assert m_star(0.0, base, solution=solution) == pytest.approx(1.0)
    assert m_star(1e6, base, solution=solution) == pytest.approx(0.0)


def test_m_star_decreasing_in_kappa(base, solution):
    kappas = (0.0, 0.05, 0.5, 5.0)
    ms = [m_star(k, base, solution=solution) for k in kappas]
    assert all(a >= b - 1e-9 for a, b in zip(ms, ms[1:]))


def test_m_star_increasing_in_stakes():
    base = RustMDP2D()
    kappa = 0.2  # interior region
    ms = []
    for s in (0.5, 1.0, 2.0):
        env = scale_stakes(base, s)
        ms.append(m_star(kappa, env))
    assert all(a <= b + 1e-9 for a, b in zip(ms, ms[1:]))
    assert ms[-1] > ms[0]  # strictly more attention at higher stakes
