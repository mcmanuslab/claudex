# RESEARCH_LOG.md

Chronological. Includes the things that did not work, because those were the
most informative part of the day.

---

## 2026-09-25 — environment

4 CPU cores, 15 GB RAM, **no GPU**. PyTorch 2.14 CPU from PyPI
(`download.pytorch.org` is blocked by the egress proxy; plain PyPI works).
Measured 530 GFLOP/s on a 1024³ GEMM, ~26k tok/s fwd+bwd at
`d=128, L=4, ff=512`.

**Consequence, recorded up front:** the brief asks for 10–30M starting
parameters and <100M final. At `d=512, L=8` this box does ~500 tok/s, so a
single run of the requested size would take on the order of a week, and the
study needs ~18 of them. The requested scale is **not reachable here** by
roughly two orders of magnitude.

Decision: keep the protocol exactly as specified and run it at the scale the
hardware allows, with `configs/exp001_paper_scale.json` holding the requested
size so the identical code can be launched on a GPU unchanged. The brief's own
instruction — "begin even smaller during debugging", "do not scale until there
is evidence worth scaling" — points the same way. Every scale claim in
`RESULTS_001.md` is labelled.

---

## Prior art (before any code)

See `PRIOR_ART.md`. The decision that came out of it: **do not grow
`d_model`.** MSG's whole contribution is solving the LayerNorm dilemma that
`d_model` growth creates. Restricting to depth growth and MLP-branch growth
makes both operators exactly identity-preserving with no LN surgery at all.
That turns Gate 1 from "a result" into "an assertion", which is where it
belongs.

Recorded prediction before running anything: Q1 and Q2 likely pass, Q3 is a
coin flip (random structured pruning is a known-strong baseline), Q4 most
likely returns NO-GO.

---

## Gate 1 — passed immediately

`logit_delta == 0.0` exactly, for depth growth and MLP-branch growth, on both a
random and a trained model. Not `1e-7`. The test asserts exact equality.

The non-obvious part was **not** the operator, it was the optimizer. A newborn
AdamW parameter has zero moments, and AdamW's first step on such a parameter is
~`lr` in magnitude no matter how small the gradient is — so a perfectly
identity-preserving block destroys the parent function on its first update.
`GrowthAwareOptimizer` gives each newborn cohort its own group ramping from
exactly zero LR, and carries the existing moments across by tensor identity.
This is Staged Training's "preserve the training dynamics, not just the loss"
point, and it is the thing a reimplementation would most likely miss.

---

## Pilot 1 — thrown away. Uniform sampling starved the axioms.

Symptom: `SHELL` validation loss stuck at 4.8–7.4 while everything else learned.

Cause: the binary relations have ~2256 facts each, the unary ones have 48. Under
uniform sampling the model sees a `SHELL` fact 0.5% of the time and never learns
the latent period at all. But period is exactly what the D3 bucket requires the
model to have grounded.

This would have produced a clean, completely bogus finding: "no model can
extrapolate the mass rule", caused entirely by the sampler.

Fix: temperature-`α=0.5` relation balancing (`P(rel) ∝ n^0.5`), the standard
multi-task choice, applied identically to every group. SHELL goes from 0.5% to
3.4% of the mixture.

---

## Pilot 1 — second flaw. D3 was contaminated.

D3 was defined as "HEAVIER or REACTSOL facts touching a period-5 entity". But
`REACTSOL(e1,e2) = SOLUBILITY[(g1+g2) mod 6]` **depends on group alone** —
period is irrelevant to it. So half of the flagship extrapolation bucket
required no extrapolation whatsoever, and was measuring interpolation while
wearing a D3 label.

Fix: D3 is now HEAVIER-touching-period-5 **only**. REACTSOL moved to D2, where
it is the pure two-hop composition test it always was. Each bucket now isolates
exactly one mechanism.

---

## Pilot 2 — thrown away. The axioms arrived too late.

Even with balanced sampling, `SHELL` stayed unlearned. The remaining cause: the
~35 pool SHELL facts were spread across 8 revelation tranches, so a period-5
entity's grounding fact could first appear at age 6 — after most of the
developmental trajectory we wanted to measure.

That makes "the model failed to extrapolate at age 2" indistinguishable from
"the axiom had not been stated yet at age 2". It would have been a test of the
revelation schedule, not of the model.

Fix: all **atomic** facts (VAL, SHELL, METAL, SOLUBLE) in the pool are revealed
at age 0; progressive revelation applies to the relational facts. This mirrors
the atomic-vs-inferred split from *Grokked Transformers are Implicit Reasoners*,
and is now pinned by `test_atomic_facts_are_all_available_at_age_zero` and
`test_distant_bucket_is_grounded_but_withheld`.

---

## Pilot 3 — the leakage alarm was calibrated against the wrong null.

D4 accuracy reached 0.592 with n=233. Against a 0.5 null that is z = 2.8 and
the leakage alarm fires — on a model doing nothing wrong.

The salted hash does not land exactly 50/50 on a finite holdout; its majority
rate is ~0.536 on these worlds. A model that simply learns the marginal
("OMEN usually answers 1") legitimately achieves that rate. Testing against 0.5
therefore flags honest models.

Fix: the null is the bucket's own **majority-class rate**, not 0.5. Verified in
both directions: an honest majority-guesser now scores z = +0.00, an oracle
scores z = +14.19.

A false leakage alarm would have been worse than no alarm at all — it would
have invalidated a valid experiment.

---

## Pilot 3 — the finding that forced a scale change

At `d=128, L=4` (0.55M params), by ~3500 steps:

| bucket | accuracy | chance |
|---|---|---|
| D0 interpolation | 0.89 | 0.35 |
| D1 near | 0.94 | 0.31 |
| D2 compositional | 0.98 | 0.50 |
| **D3 distant** | **0.49** | **0.50** |
| D4 unsupported | 0.56 | 0.54 |

Two things, both important:

1. **D1 and D2 saturate.** FIXED SMALL already solves them. If the starting
   model solves the task, growth has nothing to repair and the experiment has
   no power — any group difference would be a ceiling artefact. The precondition
   for this study is that FIXED SMALL plateaus at a *non-trivial* error, and at
   this scale it does not.
2. **D3 sits at chance and does not move.** The model does not learn
   `mass = f(period, group)` as a composable rule; it appears to learn a
   per-entity mass scalar, which is simply untrained for entities that appear in
   no HEAVIER fact. This is a real negative finding about transformers of this
   size and it will be reported as one — but it also means D3 contributes
   variance without signal to the primary endpoint.

Response: shrink the starting model so the task is hard relative to capacity,
keeping the endpoint definition untouched. Calibration runs on `world_seed=99`,
which is never used in the frozen experiment.
