"""E3: estimating one level deeper restores counterfactual validity.

Pre-registered H3 (design/research_design.md): deep counterfactuals (based
on the recovered attention cost kappa) beat as-if counterfactuals by an
order of magnitude, uniformly over markets.

Design: a cross-section of markets with stakes shifters s in S. In market s
the agent attends at m0*(kappa_true, env_s). The econometrician estimates
per-market as-if parameters (theta2_hat(s) absorbs the attention level).
Deep step: recover (kappa, theta2_deep) by minimum distance on the moment
condition theta2_hat(s) ~ m*(kappa, env_s) * theta2_deep, exploiting the
known attention technology. Under the subsidy, the deep model re-prices
attention (m1 = m*(kappa_hat, post env_s)) while the as-if model holds
theta2_hat(s) fixed.

Modeling choices documented for the note:
- The analyst's non-attention parameters for the deep prediction are the
  market's own as-if (theta1_hat, RC_hat) — their contamination by
  inattention is second-order (paper #1 zoo: sparse-max RC_hat 10.05-10.17).
- Stakes s are observable (market size); estimation uses the unscaled
  template so theta1_hat, RC_hat absorb s while theta2_hat does not.

Resumable CSV: results/e3_rows.csv. Outputs: results/e3_raw.md.
"""

import csv
import zlib
from pathlib import Path

import numpy as np

from attention import m_star, perception_policy, scale_stakes
from config import env_post, env_pre
from estimate2d import estimate_nfxp_2d
from mdp2d import RustMDP2D
from train_agents2d import agent_policy
from training import train_on_target

RESULTS = Path(__file__).resolve().parents[1] / "results"
CSV_PATH = RESULTS / "e3_rows.csv"
FIELDS = ["s", "m0", "m1_true", "theta1", "theta2", "rc",
          "rmse_asif", "rmse_deep", "rate_err_asif", "rate_err_deep"]

KAPPA_TRUE = 0.20
STAKES = (0.5, 0.75, 1.0, 1.5, 2.0)


def load_done():
    if not CSV_PATH.exists():
        return {}
    with CSV_PATH.open(newline="", encoding="utf-8") as f:
        return {row["s"]: row for row in csv.DictReader(f)}


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


def stage1_markets() -> None:
    """Per-market: agent at m0*(kappa_true), NFXP, subsidy CF (as-if part +
    ingredients for the deep part)."""
    done = load_done()
    for s in STAKES:
        key = f"{s}"
        if key in done:
            print(f"s={s}: already done, skipping")
            continue
        pre_s = scale_stakes(env_pre(), s)
        post_s = scale_stakes(env_post(), s)
        sol_pre, sol_post = pre_s.solve(), post_s.solve()

        m0 = m_star(KAPPA_TRUE, pre_s, grid_size=41, solution=sol_pre)
        m1_true = m_star(KAPPA_TRUE, post_s, grid_size=41, solution=sol_post)
        print(f"s={s}: m0={m0:.3f}, m1={m1_true:.3f} — training + NFXP...")

        agent_pre = train_on_target(perception_policy(sol_pre, m0), pre_s)
        pol_pre = agent_policy(agent_pre, pre_s)
        panel = sol_pre.simulate(
            n_buses=500, n_periods=200,
            rng=np.random.default_rng(zlib.crc32(key.encode())),
            policy=pol_pre)
        est = estimate_nfxp_2d(panel, env_pre())  # unscaled template

        # Adapted truth under the subsidy
        agent_post = train_on_target(perception_policy(sol_post, m1_true),
                                     post_s)
        pol_adapted = agent_policy(agent_post, post_s)

        # As-if prediction: model at (theta1_hat, theta2_hat, RC_hat/2)
        mdp_asif = RustMDP2D(theta1=est.theta1, theta2=est.theta2,
                             rc=est.rc * 0.5, cost_persist=0.95)
        pol_asif = mdp_asif.solve().ccp_replace

        rmse_asif = float(np.sqrt(np.mean((pol_asif - pol_adapted) ** 2)))
        rate_actual = replacement_rate(sol_post, pol_adapted)
        rate_asif = replacement_rate(sol_post, pol_asif)

        append_row(dict(
            s=key, m0=f"{m0:.3f}", m1_true=f"{m1_true:.3f}",
            theta1=f"{est.theta1:.5f}", theta2=f"{est.theta2:.4f}",
            rc=f"{est.rc:.3f}", rmse_asif=f"{rmse_asif:.5f}",
            rmse_deep="", rate_err_asif=f"{abs(rate_asif - rate_actual):.4f}",
            rate_err_deep=""))
        print(f"  theta2_hat={est.theta2:.3f}, as-if RMSE={rmse_asif:.4f}")


def stage2_deep() -> None:
    """Recover (kappa, theta2_deep) from the theta2_hat(s) cross-section,
    then compute deep counterfactuals and rewrite the CSV with them."""
    rows = sorted(load_done().values(), key=lambda r: float(r["s"]))
    stakes = np.array([float(r["s"]) for r in rows])
    t2_hat = np.array([float(r["theta2"]) for r in rows])

    # Attention technology known to the analyst: m*(kappa, env_s) computable.
    kappa_grid = np.linspace(0.02, 0.6, 30)
    pre_sols = {s: scale_stakes(env_pre(), s).solve() for s in stakes}
    best = None
    for kappa in kappa_grid:
        m_of_s = np.array([m_star(kappa, scale_stakes(env_pre(), s),
                                  grid_size=41, solution=pre_sols[s])
                           for s in stakes])
        if np.all(m_of_s < 1e-9):
            continue
        theta2_deep = float(m_of_s @ t2_hat / max(m_of_s @ m_of_s, 1e-12))
        dist = float(np.sum((t2_hat - m_of_s * theta2_deep) ** 2))
        if best is None or dist < best[0]:
            best = (dist, kappa, theta2_deep)
    dist, kappa_hat, theta2_deep = best
    print(f"deep recovery: kappa_hat={kappa_hat:.3f} (true {KAPPA_TRUE}), "
          f"theta2_deep={theta2_deep:.3f} (true 1.5), min dist={dist:.5f}")

    out_rows = []
    for r in rows:
        s = float(r["s"])
        post_s = scale_stakes(env_post(), s)
        sol_post = post_s.solve()
        m1_hat = m_star(kappa_hat, post_s, grid_size=41, solution=sol_post)

        # Deep prediction: re-priced attention applied to the analyst's model
        mdp_deep = RustMDP2D(theta1=float(r["theta1"]), theta2=theta2_deep,
                             rc=float(r["rc"]) * 0.5, cost_persist=0.95)
        sol_deep = mdp_deep.solve()
        pol_deep = perception_policy(sol_deep, m1_hat)

        m1_true = float(r["m1_true"])
        agent_post = train_on_target(
            perception_policy(sol_post, m1_true), post_s)
        pol_adapted = agent_policy(agent_post, post_s)

        rmse_deep = float(np.sqrt(np.mean((pol_deep - pol_adapted) ** 2)))
        rate_actual = replacement_rate(sol_post, pol_adapted)
        rate_deep = replacement_rate(sol_post, pol_deep)
        r["rmse_deep"] = f"{rmse_deep:.5f}"
        r["rate_err_deep"] = f"{abs(rate_deep - rate_actual):.4f}"
        out_rows.append(r)
        print(f"  s={s}: as-if RMSE={r['rmse_asif']}, deep RMSE={rmse_deep:.5f}")

    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(out_rows)

    table = "\n".join(
        f"| {r['s']} | {r['m0']} | {r['m1_true']} | {r['theta2']} | "
        f"{r['rmse_asif']} | {r['rmse_deep']} | {r['rate_err_asif']} | "
        f"{r['rate_err_deep']} |" for r in out_rows)
    ratio = np.mean([float(r["rmse_asif"]) / max(float(r["rmse_deep"]), 1e-6)
                     for r in out_rows])
    summary = f"""# E3 (raw) — deep vs as-if counterfactuals

kappa_true = {KAPPA_TRUE}; recovered kappa_hat = {kappa_hat:.3f},
theta2_deep = {theta2_deep:.3f} (true 1.5). Mean RMSE ratio (as-if / deep)
= {ratio:.1f}x.

| s | m0 | m1_true | theta2_hat(s) | as-if RMSE | deep RMSE | as-if rate err | deep rate err |
|---|---|---|---|---|---|---|---|
{table}
"""
    (RESULTS / "e3_raw.md").write_text(summary, encoding="utf-8")
    print(f"summary -> {RESULTS / 'e3_raw.md'}")


if __name__ == "__main__":
    RESULTS.mkdir(exist_ok=True)
    stage1_markets()
    stage2_deep()
