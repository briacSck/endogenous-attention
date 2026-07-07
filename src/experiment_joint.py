"""T1-T3 + HJ1 for the joint deep estimator (prereg_joint_estimator.md).

T1 recovery: joint estimator on panels simulated from the true DGP
   (kappa = 0.2, theta2 = 1.5) across five markets — must recover all four
   parameters, in particular theta2 (two-step plug-in got 1.256).
T2 corner: kappa = 0 DGP — estimator at the corner, flat kappa profile
   below the pinning threshold reported, not hidden.
T3 cost: wall-clock per evaluation and total, printed.
HJ1: deep counterfactuals from the JOINT estimates vs E3's as-if — must win
   in the majority of markets to reverse the plug-in failure.

Outputs: results/joint_raw.md.
"""

import json
import os
import time
import zlib
from pathlib import Path

import numpy as np

from attention import m_star, perception_policy, scale_stakes
from config import env_post, env_pre
from joint_estimator import estimate_joint, kappa_profile, market_ccp
from mdp2d import RustMDP2D
from train_agents2d import agent_policy
from training import train_on_target

RESULTS = Path(__file__).resolve().parents[1] / "results"
STAKES = (0.5, 0.75, 1.0, 1.5, 2.0)
KAPPA_TRUE = 0.20

# E3's as-if grid RMSE per market (results/e3_raw.md), the benchmark to beat
ASIF_RMSE = {0.5: 0.02483, 0.75: 0.01188, 1.0: 0.00720,
             1.5: 0.01583, 2.0: 0.02231}


def simulate_true_panels(kappa: float) -> dict:
    panels = {}
    for s in STAKES:
        env = scale_stakes(env_pre(), s)
        sol = env.solve()
        m0 = m_star(kappa, env, grid_size=41, solution=sol)
        pol = perception_policy(sol, m0)
        panels[s] = sol.simulate(
            n_buses=500, n_periods=200,
            rng=np.random.default_rng(zlib.crc32(f"joint{s}".encode())),
            policy=pol)
    return panels


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    base = env_pre()

    from joint_estimator import JointResult

    # Wall-clock budget per session: self-stop and persist best-so-far well
    # before a background-kill window, so a relaunch resumes near-optimal.
    BUDGET = float(os.environ.get("JOINT_BUDGET", "480"))

    def resume(ckpt: Path, panels_, x0):
        """Run/continue the joint estimator until it converges, one budgeted
        session per call. Returns the JointResult and cumulative wall clock."""
        elapsed0 = 0.0
        if ckpt.exists():
            d = json.loads(ckpt.read_text())
            if d.get("converged"):
                x = d["x"]
                print(f"  {ckpt.name}: converged checkpoint loaded.")
                return (JointResult(x[0], x[1], x[2], x[3], -d["f"], True,
                                    d.get("n_evals", 0)), d.get("elapsed", 0.0))
            x0 = tuple(d["x"])
            elapsed0 = d.get("elapsed", 0.0)
            print(f"  {ckpt.name}: resuming from x0={x0}, "
                  f"elapsed {elapsed0:.0f}s, {d.get('n_evals', 0)} evals so far.")
        res_ = estimate_joint(panels_, base, x0=x0, checkpoint=str(ckpt),
                              time_budget=BUDGET, elapsed0=elapsed0)
        d = json.loads(ckpt.read_text())
        return res_, d.get("elapsed", 0.0)

    # ---- T1: recovery (resumable to survive 10-min run windows) ----------
    ckpt = RESULTS / "joint_t1.json"
    panels = simulate_true_panels(KAPPA_TRUE)
    print("T1: joint estimator on true-DGP panels...")
    res, t1_time = resume(ckpt, panels, (0.04, 1.0, 8.0, 0.3))
    if not res.converged:
        print(f"T1: budget hit, checkpoint saved (elapsed {t1_time:.0f}s). "
              "Relaunch to resume.")
        return
    print(f"T1 estimates: theta1={res.theta1:.4f} (0.05), "
          f"theta2={res.theta2:.3f} (1.5), rc={res.rc:.3f} (10), "
          f"kappa={res.kappa:.3f} (0.2); converged={res.converged}")
    print(f"T3: {res.n_evals} likelihood evals in {t1_time:.0f}s "
          f"({t1_time / max(res.n_evals, 1):.2f}s/eval)")

    kgrid = np.linspace(0.02, 0.6, 15)
    prof = kappa_profile(panels, base,
                         np.array([res.theta1, res.theta2, res.rc,
                                   res.kappa]), kgrid)
    k_argmax = float(kgrid[int(np.argmax(prof))])

    # ---- T2: corner case (resumable) ---------------------------------------
    ckpt2 = RESULTS / "joint_t2.json"
    panels0 = simulate_true_panels(0.0)
    print("T2: kappa = 0 DGP...")
    res0, _ = resume(ckpt2, panels0, (0.04, 1.0, 8.0, 0.05))
    if not res0.converged:
        print("T2: budget hit, checkpoint saved. Relaunch to resume.")
        return
    prof0 = kappa_profile(panels0, base,
                          np.array([res0.theta1, res0.theta2, res0.rc,
                                    res0.kappa]), kgrid)
    flat_band = float(prof0.max() - prof0.min())
    print(f"T2 estimates: kappa_hat={res0.kappa:.3f} (truth 0: corner), "
          f"theta2={res0.theta2:.3f}; kappa-profile range {flat_band:.1f} "
          f"log-lik units (flat = set-identified)")

    # ---- HJ1: coherent deep counterfactuals vs E3's as-if -----------------
    print("HJ1: deep counterfactuals from joint estimates...")
    psi_hat = np.array([res.theta1, res.theta2, res.rc, res.kappa])
    rows = []
    for s in STAKES:
        post_true = scale_stakes(env_post(), s)
        sol_post = post_true.solve()
        m1_true = m_star(KAPPA_TRUE, post_true, grid_size=41,
                         solution=sol_post)
        agent_post = train_on_target(perception_policy(sol_post, m1_true),
                                     post_true)
        pol_adapted = agent_policy(agent_post, post_true)

        # Coherent deep prediction: same psi_hat, subsidized RC, re-priced m
        env_hat_post = scale_stakes(
            RustMDP2D(theta1=res.theta1, theta2=res.theta2,
                      rc=res.rc * 0.5, cost_persist=base.cost_persist), s)
        sol_hat = env_hat_post.solve()
        m1_hat = m_star(res.kappa, env_hat_post, grid_size=41,
                        solution=sol_hat)
        pol_deep = perception_policy(sol_hat, m1_hat)

        rmse_deep = float(np.sqrt(np.mean((pol_deep - pol_adapted) ** 2)))
        rows.append((s, ASIF_RMSE[s], rmse_deep, m1_true, m1_hat))
        print(f"  s={s}: as-if {ASIF_RMSE[s]:.4f} vs deep {rmse_deep:.4f} "
              f"(m1 true {m1_true:.2f}, hat {m1_hat:.2f})")

    wins = sum(1 for _, a, d, *_ in rows if d < a)
    table = "\n".join(f"| {s} | {a:.5f} | {d:.5f} | {mt:.3f} | {mh:.3f} |"
                      for s, a, d, mt, mh in rows)
    summary = f"""# Joint estimator (raw) — T1-T3 + HJ1

## T1 recovery (truth: theta1=0.05, theta2=1.5, RC=10, kappa=0.2)
theta1={res.theta1:.4f}, theta2={res.theta2:.3f}, rc={res.rc:.3f},
kappa={res.kappa:.3f}; converged={res.converged};
kappa profile argmax on grid: {k_argmax:.3f}.

## T2 corner (truth kappa = 0)
kappa_hat={res0.kappa:.3f}, theta2={res0.theta2:.3f};
kappa-profile range {flat_band:.1f} log-lik units over [0.02, 0.6].

## T3 cost
{res.n_evals} likelihood evaluations, {t1_time:.0f}s total
({t1_time / max(res.n_evals, 1):.2f} s/eval).

## HJ1 — coherent deep vs as-if (E3 benchmark)
Deep wins in {wins}/5 markets.

| s | as-if RMSE (E3) | deep RMSE (joint) | m1 true | m1 hat |
|---|---|---|---|---|
{table}
"""
    (RESULTS / "joint_raw.md").write_text(summary, encoding="utf-8")
    print(f"summary -> {RESULTS / 'joint_raw.md'}; HJ1: deep wins {wins}/5")


if __name__ == "__main__":
    main()
