"""E1 + E2: as-if counterfactual failure across the kappa grid, and the
blindness of pre-policy conformity audits.

Pre-registered (design/research_design.md, committed before this run):
- H1: as-if failure ~ 0 at kappa corners, maximal at intermediate kappa;
  monotone in the attention repricing |m1* - m0*|.
- H2: pre-policy conformity audits score all agents as conforming and are
  uncorrelated with E1 failure (the failure lives one level up).

Per kappa: the agent attends at m0* = m*(kappa, pre env), the econometrician
estimates NFXP on its pre-policy panel, the subsidy halves RC, the agent
re-prices attention to m1* = m*(kappa, post env) and re-learns; the as-if
prediction solves the model at (theta1_hat, theta2_hat, RC_hat/2).

Resumable CSV: results/e1_rows.csv. Outputs: results/e1_e2_raw.md + figures.
"""

import csv
import zlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from attention import m_star, perception_policy
from audits import c_dispersion, x_dispersion
from config import KAPPA_GRID, env_post, env_pre
from estimate2d import estimate_nfxp_2d
from mdp2d import RustMDP2D
from train_agents2d import agent_policy
from training import train_on_target

RESULTS = Path(__file__).resolve().parents[1] / "results"
CSV_PATH = RESULTS / "e1_rows.csv"
FIELDS = ["kappa", "m0", "m1", "dm", "cv_x", "cv_c", "theta1", "theta2",
          "rc", "rmse_grid", "rmse_w", "rate_pred", "rate_actual", "rate_err"]


def load_done():
    if not CSV_PATH.exists():
        return {}
    with CSV_PATH.open(newline="", encoding="utf-8") as f:
        return {row["kappa"]: row for row in csv.DictReader(f)}


def append_row(row):
    new = not CSV_PATH.exists()
    with CSV_PATH.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def replacement_rate(sol, policy_flat) -> float:
    panel = sol.simulate(n_buses=1000, n_periods=300,
                         rng=np.random.default_rng(77), policy=policy_flat)
    return float(panel.choices.mean())


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    pre, post = env_pre(), env_post()
    sol_pre, sol_post = pre.solve(), post.solve()

    ref = sol_pre.simulate(n_buses=2000, n_periods=200,
                           rng=np.random.default_rng(9))
    _, counts = ref.empirical_ccp(pre.n_states)
    ergodic = counts / counts.sum()

    done = load_done()
    for kappa in KAPPA_GRID:
        key = f"{kappa}"
        if key in done:
            print(f"kappa={kappa}: already done, skipping")
            continue
        m0 = m_star(kappa, pre, grid_size=41, solution=sol_pre)
        m1 = m_star(kappa, post, grid_size=41, solution=sol_post)
        print(f"kappa={kappa}: m0={m0:.3f}, m1={m1:.3f} — training pre agent...")

        agent_pre = train_on_target(perception_policy(sol_pre, m0), pre)
        cv_x = x_dispersion(agent_pre, pre)
        cv_c = c_dispersion(agent_pre, pre)

        pol_pre = agent_policy(agent_pre, pre)
        panel = sol_pre.simulate(
            n_buses=500, n_periods=200,
            rng=np.random.default_rng(zlib.crc32(key.encode())),
            policy=pol_pre)
        print("  estimating NFXP...")
        est = estimate_nfxp_2d(panel, pre)

        # Adapted truth: attention repriced to m1, policy re-derived, re-learned
        agent_post = train_on_target(perception_policy(sol_post, m1), post)
        pol_adapted = agent_policy(agent_post, post)

        mdp_hat = RustMDP2D(n_mileage=pre.n_mileage, n_cost=pre.n_cost,
                            theta1=est.theta1, theta2=est.theta2,
                            rc=est.rc * 0.5, beta=pre.beta,
                            mileage_probs=pre.mileage_probs,
                            cost_persist=pre.cost_persist)
        pol_pred = mdp_hat.solve().ccp_replace

        rmse_grid = float(np.sqrt(np.mean((pol_pred - pol_adapted) ** 2)))
        rmse_w = float(np.sqrt(np.sum(ergodic * (pol_pred - pol_adapted) ** 2)
                               / ergodic.sum()))
        rate_pred = replacement_rate(sol_post, pol_pred)
        rate_actual = replacement_rate(sol_post, pol_adapted)

        append_row(dict(kappa=key, m0=f"{m0:.3f}", m1=f"{m1:.3f}",
                        dm=f"{abs(m1 - m0):.3f}", cv_x=f"{cv_x:.3f}",
                        cv_c=f"{cv_c:.3f}", theta1=f"{est.theta1:.5f}",
                        theta2=f"{est.theta2:.4f}", rc=f"{est.rc:.3f}",
                        rmse_grid=f"{rmse_grid:.5f}", rmse_w=f"{rmse_w:.5f}",
                        rate_pred=f"{rate_pred:.4f}",
                        rate_actual=f"{rate_actual:.4f}",
                        rate_err=f"{abs(rate_pred - rate_actual):.4f}"))
        print(f"  dm={abs(m1-m0):.3f}, theta2_hat={est.theta2:.3f}, "
              f"RMSE={rmse_grid:.4f}, rate_err={abs(rate_pred-rate_actual):.4f}, "
              f"cv=({cv_x:.2f},{cv_c:.2f})")

    build_outputs()


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def build_outputs() -> None:
    rows = list(load_done().values())
    if len(rows) < 4:
        print("not enough rows for outputs")
        return
    kappa = np.array([float(r["kappa"]) for r in rows])
    dm = np.array([float(r["dm"]) for r in rows])
    rmse = np.array([float(r["rmse_grid"]) for r in rows])
    conf = np.maximum(np.array([float(r["cv_x"]) for r in rows]),
                      np.array([float(r["cv_c"]) for r in rows]))

    rho_dm = spearman(dm, rmse)
    rho_conf = spearman(conf, rmse)
    print(f"n={len(rows)}: Spearman(|dm|, RMSE)={rho_dm:.2f}; "
          f"Spearman(conformity, RMSE)={rho_conf:.2f}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    order = np.argsort(kappa)
    axes[0].plot(kappa[order], rmse[order], "o-")
    axes[0].set_xlabel("$\\kappa$ (attention cost)")
    axes[0].set_ylabel("as-if counterfactual RMSE")
    axes[0].set_title("E1: failure across the attention-cost grid")
    axes[1].scatter(dm, rmse, s=70, zorder=3)
    axes[1].set_xlabel("attention repricing $|m_1^* - m_0^*|$")
    axes[1].set_ylabel("as-if counterfactual RMSE")
    axes[1].set_title(f"E1: failure vs repricing (Spearman {rho_dm:.2f})")
    axes[2].scatter(conf, rmse, s=70, zorder=3, color="tab:red")
    axes[2].set_xlabel("pre-policy conformity score (max cv)")
    axes[2].set_ylabel("as-if counterfactual RMSE")
    axes[2].set_title(f"E2: audit blindness (Spearman {rho_conf:.2f})")
    fig.tight_layout()
    fig.savefig(RESULTS / "e1_e2.png", dpi=150)

    table = "\n".join(
        f"| {r['kappa']} | {r['m0']} | {r['m1']} | {r['dm']} | {r['cv_x']} | "
        f"{r['cv_c']} | {r['theta2']} | {r['rc']} | {r['rmse_grid']} | "
        f"{r['rate_err']} |" for r in sorted(rows, key=lambda r: float(r['kappa'])))
    summary = f"""# E1 + E2 (raw)

Environment: theta2={1.5}, cost_persist={0.95}, RC 10 -> 5.
Spearman(|dm|, as-if RMSE) = {rho_dm:.2f} (H1).
Spearman(pre-policy conformity, as-if RMSE) = {rho_conf:.2f} (H2 expects ~0).

| kappa | m0 | m1 | \\|dm\\| | cv_x | cv_c | theta2_hat | RC_hat | as-if RMSE | rate err |
|---|---|---|---|---|---|---|---|---|---|
{table}
"""
    (RESULTS / "e1_e2_raw.md").write_text(summary, encoding="utf-8")
    print(f"outputs -> {RESULTS / 'e1_e2_raw.md'}, {RESULTS / 'e1_e2.png'}")


if __name__ == "__main__":
    main()
