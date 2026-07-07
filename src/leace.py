"""LEACE-style linear concept erasure and the amnesic audit.

Replaces the ad hoc ridge-direction patching audit, per the feynman audit's
top recommendation. LEACE (Belrose et al. 2023, arXiv:2306.03819) is the
minimal-change affine map guaranteeing that NO linear probe can recover the
concept from the edited representations. For a scalar concept z the erased
subspace is one-dimensional in whitened coordinates: u ~ W sigma_xz with
W = Sigma_xx^{-1/2}.

The audit is amnesic-style (Elazar et al. 2021, TACL): erase all linear
information about the concept from a hidden layer, resume the forward pass,
and measure how much the output moves. Mean |dP| over in-distribution states
= the causal use of the concept. Decodability is irrelevant by construction —
we remove everything linearly available and see whether behavior cares.
"""

import numpy as np
import torch


def leace_eraser(activations: np.ndarray, concept: np.ndarray,
                 ridge: float = 1e-6):
    """Return a function mapping activations -> concept-erased activations.

    activations: (n, d) matrix; concept: (n,) scalar concept values.
    """
    x_mean = activations.mean(axis=0)
    xc = activations - x_mean
    z = concept - concept.mean()

    sigma = xc.T @ xc / len(xc) + ridge * np.eye(xc.shape[1])
    evals, evecs = np.linalg.eigh(sigma)
    evals = np.maximum(evals, ridge)
    w = evecs @ np.diag(evals ** -0.5) @ evecs.T        # whitening
    w_inv = evecs @ np.diag(evals ** 0.5) @ evecs.T     # unwhitening

    sigma_xz = xc.T @ z / len(xc)
    u = w @ sigma_xz
    norm = np.linalg.norm(u)
    if norm < 1e-12:  # concept not linearly present at all
        return lambda a: a
    u = u / norm

    # x' = x - W^+ u u^T W (x - mu): project out the concept direction in
    # whitened space, minimal squared change in the original space.
    eraser_mat = w_inv @ np.outer(u, u) @ w

    def erase(a: np.ndarray) -> np.ndarray:
        return a - (a - x_mean) @ eraser_mat.T

    return erase


def amnesic_audit(agent, layer: int, activations: np.ndarray,
                  concept: np.ndarray, state_mask: np.ndarray) -> float:
    """Mean |dP(replace)| from LEACE-erasing the concept at `layer`.

    activations: (n_states, d) hidden activations at `layer` for ALL states;
    concept: (n_states,) concept values; state_mask: boolean, which states
    count toward the audit (in-distribution only).
    """
    from probes import _forward_from

    erase = leace_eraser(activations, concept)
    erased = erase(activations)

    a_clean = torch.tensor(activations[state_mask], dtype=torch.float32)
    a_erased = torch.tensor(erased[state_mask], dtype=torch.float32)
    with torch.no_grad():
        p_clean = torch.sigmoid(_forward_from(agent, layer, a_clean))
        p_erased = torch.sigmoid(_forward_from(agent, layer, a_erased))
    return float((p_clean - p_erased).abs().mean())


def linear_r2_after_erasure(activations: np.ndarray,
                            concept: np.ndarray) -> float:
    """Sanity check: held-out R^2 of a probe AFTER erasure (should be ~0)."""
    from probes import ridge_probe

    erased = leace_eraser(activations, concept)(activations)
    return ridge_probe(erased, concept.astype(float))["r2_test"]


if __name__ == "__main__":
    from mdp2d import RustMDP2D
    from train_agents2d import all_state_features, train_agent_2d

    mdp = RustMDP2D()
    sol = mdp.solve()
    _, c = mdp.state_grid()
    feats = all_state_features(mdp)
    mask = np.ones(mdp.n_states, dtype=bool)

    for lam, name in ((1.0, "clone"), (0.0, "c-blind")):
        agent = train_agent_2d(sol, lam)
        with torch.no_grad():
            acts = agent.hidden_activations(feats)[0].numpy()
        audit = amnesic_audit(agent, 0, acts, c.astype(float), mask)
        r2_post = linear_r2_after_erasure(acts, c.astype(float))
        print(f"{name}: amnesic audit = {audit:.4f}, "
              f"probe R2 after erasure = {r2_post:.4f} (should be ~0)")
