"""Interpretability tools for the neural agents (week 4).

Core question: do the agent's hidden layers *linearly encode* the objects the
structural model assumes it optimizes — the integrated value function V(x),
the choice-value difference v(x,1) - v(x,0), or just the raw state?

Method 1 — linear probes (Alain & Bengio 2016): ridge-regress a target
quantity on hidden activations across states; report held-out R^2. High R^2
for V(x) in the clone but not the heuristic agent is the paper's first
exhibit.

Method 2 — activation patching (week 4, TODO): replace an internal
activation computed at state x with the one from state x' and check whether
the output moves as the value function predicts (causal use, not just
decodability).
"""

import numpy as np


def ridge_probe(activations: np.ndarray, target: np.ndarray,
                l2: float = 1e-3, train_frac: float = 0.7,
                seed: int = 0) -> dict:
    """Fit target ~ activations by ridge regression; return held-out R^2.

    activations: (n_states, n_units) hidden activations, one row per state.
    target:      (n_states,) quantity to decode, e.g. Solution.v_bar.
    """
    rng = np.random.default_rng(seed)
    n = len(target)
    idx = rng.permutation(n)
    n_train = int(train_frac * n)
    train, test = idx[:n_train], idx[n_train:]

    x_mean = activations[train].mean(axis=0)
    y_mean = target[train].mean()
    xc, yc = activations[train] - x_mean, target[train] - y_mean

    gram = xc.T @ xc + l2 * np.eye(xc.shape[1])
    weights = np.linalg.solve(gram, xc.T @ yc)

    pred = (activations[test] - x_mean) @ weights + y_mean
    ss_res = np.sum((target[test] - pred) ** 2)
    ss_tot = np.sum((target[test] - target[test].mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return {"r2_test": float(r2), "weights": weights,
            "train_idx": train, "test_idx": test}


def probe_agent_for_value(agent, solution) -> list[dict]:
    """Probe every hidden layer of an MLPAgent for the integrated value function."""
    import torch
    from train_agents import state_features

    k = solution.mdp.n_states
    feats = torch.tensor(state_features(np.arange(k), k), dtype=torch.float32)
    with torch.no_grad():
        layer_acts = agent.hidden_activations(feats)

    results = []
    for layer_i, acts in enumerate(layer_acts):
        out = ridge_probe(acts.numpy(), solution.v_bar)
        results.append({"layer": layer_i, "r2_test": out["r2_test"]})
    return results


def _forward_from(agent, layer: int, activation):
    """Resume the MLP forward pass from the post-ReLU activation of `layer`.

    MLPAgent.net is [Linear, ReLU, Linear, ReLU, ..., Linear]; the post-ReLU
    activation of hidden layer k sits after net index 2k + 1.
    """
    import torch

    h = activation
    for module in agent.net[2 * layer + 2:]:
        h = module(h)
    return h.squeeze(-1)


def directional_patch(agent, layer: int, direction: np.ndarray,
                      source_state: int, target_state: int,
                      n_states: int) -> dict:
    """Patch only the `direction` component of layer-`layer` activations.

    Runs the agent on target_state but overwrites the activation component
    along `direction` with its value at source_state, leaving the orthogonal
    complement untouched. Returns output logits for the clean target, the
    patched run, and the clean source.

    If the probed direction is causally used, the patched logit moves toward
    the source logit; if the probe was merely correlational, it barely moves.
    """
    import torch
    from train_agents import state_features

    u = direction / np.linalg.norm(direction)
    feats = torch.tensor(
        state_features(np.array([source_state, target_state]), n_states),
        dtype=torch.float32)
    with torch.no_grad():
        acts = agent.hidden_activations(feats)[layer]
        a_source, a_target = acts[0], acts[1]
        u_t = torch.tensor(u, dtype=torch.float32)
        shift = torch.dot(a_source - a_target, u_t) * u_t
        a_patched = a_target + shift
        logit_target = _forward_from(agent, layer, a_target).item()
        logit_patched = _forward_from(agent, layer, a_patched).item()
        logit_source = _forward_from(agent, layer, a_source).item()
    return {"logit_target": logit_target, "logit_patched": logit_patched,
            "logit_source": logit_source}


def readout_direction(agent, layer: int) -> np.ndarray:
    """First-order direction the network actually reads from `layer`.

    For the last hidden layer this is exactly the output weights w_out; for
    earlier layers we propagate through the downstream linear maps
    (ignoring ReLU gating, so it is a first-order approximation):
    d = W_{L}^T ... W_{layer+2}^T w_out.
    """
    import torch

    linears = [m for m in agent.net if isinstance(m, torch.nn.Linear)]
    d = linears[-1].weight.detach().numpy().ravel()
    for lin in reversed(linears[layer + 1:-1]):
        d = lin.weight.detach().numpy().T @ d
    return d


def patching_recovery(agent, solution, layer: int,
                      min_state: int = 10, max_state: int = 75,
                      min_gap: int = 20, seed: int = 0) -> dict:
    """Mean causal recovery of the V(x) probe direction vs a random control.

    Recovery for a (source, target) pair = fraction of the clean
    logit gap (source minus target) reproduced by patching only the probed
    direction. ~1 means the direction carries the decision-relevant signal;
    ~0 means the probe was decodable but causally idle.
    """
    import torch
    from train_agents import state_features

    k = solution.mdp.n_states
    feats = torch.tensor(state_features(np.arange(k), k), dtype=torch.float32)
    with torch.no_grad():
        acts = agent.hidden_activations(feats)[layer].numpy()

    probe_dir = ridge_probe(acts, solution.v_bar)["weights"]
    rng = np.random.default_rng(seed)
    random_dir = rng.standard_normal(probe_dir.shape)
    readout_dir = readout_direction(agent, layer)

    recoveries = {"probe": [], "random": [], "readout": []}
    states = np.arange(min_state, max_state)
    for target in states[::5]:
        for source in states[::5]:
            if abs(int(source) - int(target)) < min_gap:
                continue
            for name, d in (("probe", probe_dir), ("random", random_dir),
                            ("readout", readout_dir)):
                out = directional_patch(agent, layer, d, int(source),
                                        int(target), k)
                gap = out["logit_source"] - out["logit_target"]
                if abs(gap) < 0.5:  # skip pairs with no behavioral contrast
                    continue
                recoveries[name].append(
                    (out["logit_patched"] - out["logit_target"]) / gap)
    return {name: float(np.mean(vals)) if vals else np.nan
            for name, vals in recoveries.items()}


if __name__ == "__main__":
    from mdp import RustMDP
    from train_agents import train_clone

    sol = RustMDP().solve()
    agent = train_clone(sol)
    for row in probe_agent_for_value(agent, sol):
        print(f"layer {row['layer']}: held-out R^2 for V(x) = {row['r2_test']:.4f}")
