# Research design — Endogenous Attention and the Hierarchy of Invariance
### Paper #2 of the "Microeconomics of Artificial Agents" program

**Status:** design finalized 2026-07-07, before any experiment. Written to be
executable by a cold-start session: everything needed is in this file, the
copied `src/` modules, and paper #1's repo (`../structural_ground_truth/`,
GitHub `briacSck/structural-interp`).

## 1. Question and thesis

Paper #1 established that structural estimation is robust to bounded
rationality *inside* the model class: fixed-attention agents (Gabaix
sparse-max, Matějka–McKay RI) are absorbed into as-if parameters, and
counterfactuals — including a subsidy under which agents re-learn — stay
accurate. That result conditioned on one assumption: the cognitive primitive
(attention) is held fixed under the intervention.

This paper breaks that assumption, on purpose. In rational-inattention
theory, attention responds to stakes. A policy that changes stakes therefore
moves the primitive itself, and the as-if parameters estimated pre-policy are
no longer invariant. This is the Lucas critique reconstituted one level up —
for cognitive parameters instead of behavioral ones — and its resolution is
the same as Lucas's, iterated:

> **Thesis (hierarchy of invariance).** Behavioral parameters < as-if
> preferences < cognitive primitives. Each class of policy interventions
> requires estimating one level deeper than the level the policy moves.
> Interpretability audits certify conformity at a level; they cannot see
> one level up.

## 2. The model

### 2.1 Environment (reuse `src/mdp2d.py` as-is)
The 2-D bus environment from paper #1: mileage x (60 bins), cost regime c
(5-state persistent chain), truth (θ₁, θ₂, RC) = (0.05, 0.8, 10), β = 0.95.

### 2.2 Markets with stakes variation
A cross-section of M markets indexed by a **stakes shifter** s ∈ S (grid,
e.g. S = {0.5, 0.75, 1.0, 1.5, 2.0}): in market s, all flow utilities are
scaled by s (equivalently: bigger buses, costlier downtime). Scaling all
payoffs leaves the *optimal policy* unchanged (logit CCPs are invariant to
common scaling ONLY IF the shock scale is also scaled — decision: hold the
T1EV shock scale at 1, so higher s makes payoff differences larger relative
to noise; both the optimal policy and the value of attention then vary with
s. This is the intended design: s moves the benefit of attention).

### 2.3 Endogenous attention (new module `src/attention.py`)
Sparse-max agents perceive c′ = m·c + (1−m)·c̄ and act optimally given the
perceived state (the `sparse_attention` construction of paper #1's zoo_v2).
Attention is now chosen:

    m*(κ, env) = argmax_{m ∈ [0,1]}  B(m; env) − κ·m²

where B(m; env) is the agent's expected discounted payoff under the
m-perception policy in environment env, computed EXACTLY by policy
evaluation (iterative V^π on the solved MDP; expected flow utilities, no
shock-entropy term — a simplification to state in the paper and stress-test
in an appendix). Quadratic cost gives interior solutions; grid search over
m ∈ {0, 0.05, ..., 1} suffices (no closed form needed since the environment
is exactly solvable).

Predicted comparative statics (unit tests): m* increasing in s, decreasing
in κ; m*(κ=0) = 1; m*(κ large) = 0.

### 2.4 Agents
Grid of agents: κ ∈ K (e.g. 5 values spanning "always attend" to "never
attend") × markets s ∈ S. Each agent = MLP trained (paper #1 pipeline,
`train_on_target`) on the policy induced by m*(κ, s). Same architecture,
same seeds convention (torch seed 0, crc32(name) panel seeds).

## 3. The intervention
The paper #1 subsidy: RC → RC/2 (50% proportional), which changes value
differences and hence the stakes of attending. Post-policy attention:
m₁* = m*(κ, env_post) ≠ m₀*. Adaptation = re-derive the sparse-max policy at
m₁* and retrain (stable κ — the DEEP primitive is the attention COST, not
the attention LEVEL; this is the paper's whole point).

## 4. Exhibits and pre-registered hypotheses

**E1 — As-if failure is hump-shaped in κ.** The econometrician estimates
(θ̂₁, θ̂₂, R̂C) per market pre-policy (NFXP, `src/estimate2d.py`) and
predicts post-subsidy behavior at (θ̂₁, θ̂₂, R̂C/2). Failure metric: policy
RMSE + replacement-rate error vs the adapted agent (paper #1 metrics).
  - **H1:** failure ≈ 0 at κ → 0 (m pinned at 1 pre and post) and κ → ∞
    (m pinned at 0), maximal at intermediate κ where |m₁* − m₀*| is largest.
    Failure is monotone in the attention repricing |m₁* − m₀*|.

**E2 — Pre-policy audits cannot see it.** Run paper #1's conformity audit
(x- and c-dispersion with insensitivity gate; port `c_conformity.py`) on all
agents pre-policy.
  - **H2:** audit scores are LOW (conforming) for all fixed-m sparse-max
    agents regardless of κ, and uncorrelated with E1 failure. The audit
    certifies mechanism-class conformity, which genuinely holds pre-policy;
    the failure comes from one level up. (This is a *feature* of the story,
    not a bug: it delimits what audits certify.)

**E3 — Estimating one level deeper restores validity.** Use the
cross-section: pre-policy, θ̂₂(s) ≈ m*(κ, s)·θ₂ varies across markets. With
the attention model known up to (κ, θ₂), recover both by minimum distance on
{θ̂₂(s)}_{s ∈ S}. Then predict post-policy attention m₁*(κ̂, env_post) and
behavior.
  - **H3:** deep counterfactuals (κ̂-based) beat as-if counterfactuals
    (θ̂₂-based) by an order of magnitude on the E1 metrics, uniformly over
    the κ grid.

**E4 (stretch) — the audit that WOULD work.** A "stress-test audit":
simulate a stakes change on the agent's INPUTS (scale payoff-relevant
features) and check whether internals reallocate. Only meaningful if agents
are trained with an attention-choice mechanism inside the network (hard);
default is to discuss this as future work rather than implement.

## 5. Implementation plan (for the executing session)

1. `src/attention.py` (scaffolded + unit-tested tonight): policy evaluation
   `policy_value(mdp, policy_flat)`, benefit curve `B(m)`, `m_star(kappa,
   mdp, stakes)`.
2. `src/markets.py`: stakes-scaled environment wrapper (utility scaling on
   top of RustMDP2D — scale `flow_utility()` by s).
3. `src/experiment_e1.py`: κ × s agent grid; NFXP per market; subsidy CF
   with m₁* adaptation; failure vs |m₁* − m₀*| figure. Resumable CSV
   (paper #1 `zoo_v2.py` pattern).
4. `src/experiment_e2.py`: conformity audits pre-policy (port from
   `../structural_ground_truth/src/c_conformity.py` + `audit_battery.py`).
5. `src/experiment_e3.py`: minimum-distance recovery of (κ, θ₂) from
   θ̂₂(s); deep vs as-if CF comparison.
6. Note drafting: reuse paper #1's `note.tex` skeleton; the discussion
   section of paper #1 already contains this paper's framing paragraph.

Conventions (inherited from paper #1, non-negotiable): crc32 seeding;
scripts write `*_raw.md`, hand annotations in separate files; pre-register
H1–H3 by committing this file BEFORE running experiments (done);
tolerances/tests before science.

## 6. Positioning and must-cites
Everything in `../structural_ground_truth/paper/positioning.md` plus:
Gabaix (2014) §on endogenous sparsity; Maćkowiak–Matějka–Wiederholt (2023,
JEL) for attention-responds-to-stakes; Lucas (1976) obviously; paper #1
(cite as companion). Check during lit pass: whether anyone has framed
"attention non-invariance" as a Lucas problem for structural work —
candidates to search: "rational inattention Lucas critique",
"policy-invariant attention", work by Caplin & Dean on identifying attention
costs (Caplin, Dean & Leahy — "Rationally Inattentive Behavior:
Characterizing and Generalizing Shannon Entropy", JPE 2022 — likely relevant
for E3's identification argument; verify before citing).

## 7. Risks
- E1's hump could be flat if the subsidy barely moves B(m): tune SUBSIDY
  size / θ₂ / regime persistence until |m₁* − m₀*| spans a meaningful range
  (this is instrument design, do it BEFORE looking at failure metrics).
- E3's minimum distance may be weakly identified with |S| = 5 markets:
  increase the s grid if needed (cheap).
- Conceptual referee risk: "of course non-invariant primitives break
  counterfactuals." Answer: the contribution is the hierarchy + the audit
  boundary + the constructive E3, in a lab where each level is verifiable.
