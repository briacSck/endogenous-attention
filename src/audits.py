"""Conformity audits, self-contained port from paper #1.

Per-channel dispersion of locally patched causal sensitivities along the
readout direction, with the insensitivity gate (a uniformly insensitive
channel is CONFORMING — some parameter value fits it exactly). See paper #1
`results/c_conformity.md` for validation (Spearman 0.87 on its 13-agent zoo).
"""

import numpy as np
import torch

from mdp2d import RustMDP2D
from probes import _forward_from, readout_direction
from train_agents2d import all_state_features


def _local_sensitivities(agent, mdp: RustMDP2D, pairs, layer: int = 0):
    d = readout_direction(agent, layer)
    u = torch.tensor(d / np.linalg.norm(d), dtype=torch.float32)
    feats = all_state_features(mdp)
    sens = []
    with torch.no_grad():
        acts = agent.hidden_activations(feats)[layer]
        for s_lo, s_hi in pairs:
            a_lo, a_hi = acts[s_lo], acts[s_hi]
            patched = a_lo + torch.dot(a_hi - a_lo, u) * u
            p_lo = torch.sigmoid(_forward_from(agent, layer, a_lo)).item()
            p_pa = torch.sigmoid(_forward_from(agent, layer, patched)).item()
            sens.append(abs(p_pa - p_lo))
    return np.array(sens)


def _gated_cv(sens: np.ndarray, floor: float = 1e-3,
              gate: float = 0.01) -> float:
    mean = float(sens.mean())
    return float(sens.std() / (mean + floor)) if mean >= gate else 0.0


def x_dispersion(agent, mdp: RustMDP2D, layer: int = 0,
                 step: int = 2) -> float:
    c_mid = mdp.n_cost // 2
    pairs = [(x * mdp.n_cost + c_mid, (x + step) * mdp.n_cost + c_mid)
             for x in range(4, 46 - step)]
    return _gated_cv(_local_sensitivities(agent, mdp, pairs, layer))


def c_dispersion(agent, mdp: RustMDP2D, layer: int = 0) -> float:
    pairs = [(x * mdp.n_cost + c, x * mdp.n_cost + c + 1)
             for x in range(5, 45, 5) for c in range(mdp.n_cost - 1)]
    return _gated_cv(_local_sensitivities(agent, mdp, pairs, layer))
