# Joint deep estimator — annotated (run of 2026-07-07)

Raw: `joint_raw.md`. Pre-registration: `design/prereg_joint_estimator.md`
(committed before implementation and any recovery run). This is E3's
designated completion, framed as a section of the paper, not a companion.

## Verdict in one line
**HJ1 CONFIRMED (5/5), but not for the reason the pre-registration
expected.** Coherent joint estimation reverses E3's plug-in failure — yet
it does so *without* sharpening the deep parameters. It sharpens the object
that actually maps into behavior: the attention allocation.

## HJ1 — headline: coherent depth wins 5/5 (plug-in won 1/5)

| s | as-if RMSE (E3) | deep RMSE (joint) | ratio |
|---|---|---|---|
| 0.5 | 0.02483 | 0.00282 | 8.8× |
| 0.75 | 0.01188 | 0.00327 | 3.6× |
| 1.0 | 0.00720 | 0.00542 | 1.3× |
| 1.5 | 0.01583 | 0.00318 | 5.0× |
| 2.0 | 0.02231 | 0.00320 | 7.0× |

The coherent deep counterfactual (same ψ̂, subsidized RC, attention
re-priced at the new fixed point) beats the as-if benchmark in every
market, by 1.3–8.8×. E3's plug-in deep prediction was REJECTED (worse than
as-if in 4/5). The reversal is complete. The constructive claim of the
paper — that respecting the attention fixed point inside estimation lets
you predict one Lucas level up — holds.

## T1 recovery — the finding hiding under the headline

Truth θ1=0.05, θ2=1.5, RC=10, κ=0.2. Joint estimates:

| param | truth | joint | two-step (E3) | recovered? |
|---|---|---|---|---|
| θ1 | 0.05 | 0.0536 | — | ✓ |
| RC | 10 | 9.987 | — | ✓ clean |
| θ2 | 1.5 | **1.264** | 1.256 | ✗ same bias |
| κ | 0.2 | **0.138** | 0.200 (exact) | ✗ worse |

This is the honest core of the section. The pre-registration expected joint
estimation to pull θ2 to ~1.5 and κ to ~0.2. **It did neither.** θ2 sits at
1.264 — essentially where two-step plug-in left it (1.256) — and κ actually
degraded from the two-step's exact 0.200 to 0.138. Converged (150 evals,
2.18 s/eval, well inside the T3 budget). So this is not non-convergence; the
likelihood genuinely prefers (θ2≈1.26, κ≈0.14) about as much as the truth.

## T2 corner — why the θ2 bias is not a bug

Truth κ=0 (full attention everywhere). Joint estimates: κ̂=0.006 (clean
corner recovery), **θ2=1.496 (perfect)**. κ-profile range 3804 log-lik
units over [0.02, 0.6] — steeply identified at the corner, not flat.

The contrast with T1 is the whole point: **when there is no attention cost,
θ2 is recovered exactly.** The θ2 bias in T1 appears *only* under interior
attention (κ=0.2). So it is not an estimator defect — it is a structural
θ2–κ ridge that opens up precisely when attention is interior.

## The mechanism (the program's recurring lesson, third instance)

Under interior attention, θ2 (deep cost curvature) and κ (attention cost)
are jointly weakly identified from choice data: many (θ2, κ) pairs imply
nearly the same conditional choice probabilities, because what the CCPs
respond to is the *attention allocation* m*(θ2, κ), not the two parameters
separately. The joint MLE lands somewhere on that ridge — (1.264, 0.138)
rather than (1.5, 0.2) — but the ridge is a level set of m*(·). Look at the
HJ1 table: m1_hat tracks m1_true closely (0.33 vs 0.28, 0.50 vs 0.43, 0.50
vs 0.50, 0.62 vs 0.55, 0.75 vs 0.68) even though θ2 and κ are individually
off. The estimator recovers the attention *technology's output*, not its
two inputs.

And that output is exactly what the deep counterfactual consumes: to
predict post-subsidy behavior you re-solve the MDP and re-price attention;
the re-pricing needs m*(·) to be right, not the θ2/κ split. So coherence,
not point-accuracy, is what pays. This is the same lesson as #1 (probes vs
conformity see different channels) and #2 in this paper (audits blind to the
allocation shift): **the instrument identifies the channel it couples to.**
The coherent likelihood couples to the attention allocation → CCP channel;
it is blind to the θ2-vs-κ decomposition, and it does not need it.

## What this means for the paper's claim (scoping)

- The **hierarchy-of-invariance** claim survives and is now *constructive*:
  coherent estimation predicts one level up (post-subsidy) where plug-in
  fails. This is the positive headline.
- But it must be stated at the right level of ambition. Coherent depth does
  NOT deliver the deep structural parameters (θ2, κ) under interior
  attention — those are weakly jointly identified. It delivers the deep
  *predictive object* (the re-priced allocation). "Depth wholesale, not
  retail" (E3's phrase) is now sharpened: you get the whole coherent
  counterfactual, but not the retail parameter decomposition.
- Honest limitation to state: with a different κ-identifying moment (e.g.
  exogenous variation in attention cost, or an attention-cost instrument),
  the ridge would break and θ2 would be point-identified. That is the
  designated next experiment, in-paper: an attention-cost shifter that
  identifies the θ2–κ split, testing whether it also improves (already
  excellent) counterfactuals or merely the point estimates.

## Reproduce
`JOINT_BUDGET=480 .venv/Scripts/python src/experiment_joint.py`
(resumable: writes `results/joint_t{1,2}.json` each optimizer iteration and
self-stops before a background-kill window; relaunch to continue). Cold
full battery ≈ 5.5 min wall clock here.
