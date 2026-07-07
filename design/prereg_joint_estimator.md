# Pre-registration — the coherent joint deep estimator (completing E3)

**Date: 2026-07-07, committed BEFORE implementation and any recovery run.**

## Question
E3 showed the deep primitive is identified (kappa_hat = 0.200 exact by
minimum distance) but plug-in deep predictions LOSE to as-if (pseudo-true-
value compensation). The paper's constructive claim therefore requires the
coherent estimator: joint full-solution MLE of (theta1, theta2, RC, kappa)
with the attention technology inside the likelihood.

## Specification
Data: pooled panels from M = 5 markets with observable stakes s (utilities
scaled by s, shock scale 1 — as in E3). Model-implied CCPs for market s at
candidate psi = (theta1, theta2, RC, kappa):

    env_s(psi)   = ScaledMDP(theta1, theta2, RC; s)
    m_s(psi)     = argmax_m B(m; env_s(psi)) - kappa * m^2
    CCP_s(psi)   = perception_policy( solve(env_s(psi)), m_s(psi) )

Pooled log-likelihood over all markets' (state, choice) observations.
Optimizer: Nelder-Mead over psi (4 parameters), bounds enforced by penalty.

## Known numerical traps and their pre-committed mitigations
1. **Step-function m\*** (grid argmax makes the likelihood piecewise-
   constant in kappa): fit a local quadratic to B(m) around the grid argmax
   and use its continuous maximizer (clipped to [0,1]).
2. **Cost:** each likelihood eval needs, per market, one model solve + a
   B(m) curve (many policy evaluations). Mitigations: warm-started value
   iteration AND warm-started policy evaluations cached per (s, m) across
   likelihood calls; coarse m grid (21 points) + quadratic refinement.
3. **Corner degeneracy:** if m_s is pinned at 1 (or 0) in ALL markets at
   the optimum, kappa is set-identified only (any kappa below/above the
   pinning threshold fits). Report the profile likelihood in kappa, not
   just the point estimate.
4. **Scale absorption:** stakes s are data (observable market size);
   utilities are s * u(theta) by construction, so theta is common across
   markets — no per-market free scale parameters. This is what makes 4
   parameters (not 12) sufficient and the estimator coherent.

## Pre-registered test battery (run BEFORE any headline claim)
- **T1 (recovery):** simulate panels from the TRUE DGP (exact perception
  policies at kappa = 0.2, theta2 = 1.5) across the 5 markets; the joint
  estimator must recover all four parameters — in particular theta2 near
  1.5 (the number two-step plug-in missed: 1.256) and kappa near 0.2.
- **T2 (corner case):** DGP with kappa = 0 (full attention everywhere);
  estimator must go to the corner and the kappa profile must be flat below
  the pinning threshold (set-identification detected, not hidden).
- **T3 (cost):** wall-clock per likelihood eval and total; if > ~30 min
  total, report and reduce (fewer m points / smaller panels) BEFORE the
  headline run, never after seeing results.

## Pre-registered headline hypothesis
**HJ1:** deep counterfactuals computed from the JOINT estimates
(re-price attention at m*(kappa_hat, post env), predict with the coherent
psi_hat) beat the as-if counterfactuals of E3 in the majority of markets,
reversing the plug-in failure. Metrics identical to E3 (policy RMSE vs the
adapted agent; replacement-rate error). If HJ1 fails too, the paper reports
that coherent depth does not pay at this sample size and scopes the
hierarchy claim accordingly.

## Deliverables
`src/joint_estimator.py` (estimator + caching), `src/experiment_joint.py`
(T1-T3 + HJ1), `results/joint_raw.md`, annotated `results/joint.md`.
