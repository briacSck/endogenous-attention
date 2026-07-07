# Endogenous Attention and the Hierarchy of Invariance
### Paper #2 of the "Microeconomics of Artificial Agents" program

**Question.** Paper #1 ([briacSck/structural-interp](https://github.com/briacSck/structural-interp))
showed structural estimation absorbs *fixed* inattention into as-if
parameters, and counterfactuals survive. What happens when attention is
**endogenous** — chosen against stakes, as rational-inattention theory says
it must be — and a policy changes the stakes?

**Thesis (hierarchy of invariance).** Behavioral parameters < as-if
preferences < cognitive primitives. Each class of policy interventions
requires estimating one level deeper than the level the policy moves. The
Lucas critique, one level up — with Lucas's own resolution, iterated:
identify the attention *cost* from cross-sectional stakes variation, and
deep counterfactuals work again. Interpretability audits certify conformity
at a level and are provably blind one level up — this paper measures that
boundary in a lab where every level is observable.

**Status.** Design finalized and pre-registered:
[`design/research_design.md`](design/research_design.md) (model, exhibits
E1–E4, hypotheses H1–H3, implementation plan, risks). Scaffold implemented
and tested: `src/attention.py` (exact policy evaluation + m*(κ, stakes) by
grid search on the solved MDP) with comparative-statics unit tests. Core lab
modules copied from paper #1 (`mdp2d`, `estimate2d`, `train_agents2d`,
`probes`, `leace`).

## Reproduce the scaffold

```powershell
py -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python -m pytest tests -v   # comparative statics: m* in stakes/kappa
.venv\Scripts\python src\attention.py     # m* table across stakes x kappa
```

Conventions inherited from paper #1: crc32 seeding, `*_raw.md` script
outputs never overwrite hand annotations, hypotheses committed before runs.
