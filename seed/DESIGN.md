# DESIGN.md — how the pieces work and why each one is shaped that way

This is the engineering rationale. The frozen experimental protocol is in
`EXPERIMENT_001.md`; the results are in `RESULTS_001.md`.

---

## 1. Growth: why `d_model` never changes

The single most common way a growth experiment quietly fails is that the growth
operator is not actually function preserving, so "growth hurt performance"
really means "we perturbed the model and called it growth".

MSG (ICLR 2024) showed that the hard case is **LayerNorm**: widen `d_model` and
LN suddenly normalises over dimensions that did not exist a moment ago, so the
function changes even when every new weight is zero. Their fix is to fold a
mask into the LN statistics.

We take the other road and **restrict the operator set so the problem cannot
arise**:

| operator | what it adds | why it is exactly the identity at birth |
|---|---|---|
| `GROW_DEPTH` | a whole pre-norm block | the block computes `x + attn(ln(x)) + mlp(ln(x))`; with `W_proj = 0` and every branch's `W_down = 0`, it adds exactly `0` |
| `ADD_MLP_CAPACITY` | a parallel MLP branch inside a block | same argument; `d_ff` is never normalised over, so LN never sees it |

Neither touches `d_model`, so `logit_delta` is `0.0` exactly — not `1e-7`,
not "close enough". `tests/test_gates.py` asserts `== 0.0`, including on a
model that has actually been trained.

This is a **scope restriction, not a contribution**. We cannot grow `d_model`.
If a later experiment needs to, it should adopt MSG's masked LN rather than
reinvent it.

### The part everyone forgets: optimizer state

Staged Training's real lesson is that loss preservation is worthless on its own.
A freshly created parameter enters AdamW with zero first and second moments,
and AdamW's first update on such a parameter has magnitude ≈ `lr` *regardless
of how small the gradient is*. Drop a perfectly identity-preserving block into a
converged model, take one step, and you have destroyed the function you were
careful to preserve.

`GrowthAwareOptimizer` therefore:

- carries `exp_avg` / `exp_avg_sq` / `step` across every structural change by
  tensor identity (stable, because growth adds and removes modules but never
  re-allocates surviving tensors);
- puts each newborn cohort in its own parameter group whose LR ramps from
  **exactly zero** over `new_param_warmup` steps.

Two tests pin this: `test_new_params_get_lr_warmup_not_a_full_adam_step` and
`test_optimizer_moments_survive_growth`.

### Pruning removes things

`remove_unit` deletes the module. Gating a unit to zero and calling it "pruned"
would leave the model still paying its FLOPs, which would corrupt every
compute-matched comparison in the study.
`test_pruning_actually_removes_parameters_and_flops` pins it.

---

## 2. Compute accounting: the budget is FLOPs, not steps

`flops_per_token()` is computed from the **live** structure every step, so:

- a model that grew takes **fewer steps** for the same budget;
- a model running K competing candidates pays for **K candidates** while they
  are alive.

This is the fairness decision the whole study rests on. If the budget were
steps, "developmental models are better" would reduce to "we gave them more
compute", and nothing else in this repository would matter.
`test_flops_accounting_charges_for_live_candidates` pins it.

---

## 3. The controller

Heuristic, small, and completely logged. It watches a moving window of
validation loss and fires `GROW` when relative improvement over the window
stays below `plateau_rel` for `patience` consecutive observations, subject to a
warmup, a cooldown and a growth budget.

- **What** to grow: depth for the first `depth_first` events, then MLP width.
  Depth first is not arbitrary — G_stack found depthwise stacking the strongest
  operator, and the PNAS induction-head work argues compositional OOD
  generalisation comes from composing attention layers, which is precisely what
  D2 asks for.
- **Where**: the block with the highest gradient-norm-per-parameter. A block
  whose parameters are being pushed hard relative to their number is the one
  under optimisation strain.
- **Why**: every decision records the loss window, the relative improvement, the
  consecutive-stall count, per-category validation losses, and the strain
  ranking.

Per-category loss is recorded as **diagnostic only** and is deliberately not
used to make the decision: the small relations have only a handful of
validation facts each, so "worst category" is far too noisy to steer on.

`RandomController` is the matched control and the reason Question 2 is
answerable: same number of events, same kinds, same order, same cooldown
constraint, **random timing**. If D beats A but not R, then what helps is
getting bigger, not growing under pressure.

---

## 4. Competition and pruning

At a competitive growth event we instantiate `K` candidates that differ only in
initialisation scale (`0.02 × (1 + 0.25·i)`). Because each is exactly the
identity, instantiating all K is *still* exactly the identity — they start on
genuinely equal terms. They then train for `dev_window` steps and are scored by:

- **marginal contribution** — the increase in validation loss when that unit
  alone is masked off;
- **redundancy** — cosine similarity between candidates' contributions to the
  residual stream.

`keep = 1` by default. That is a deliberate choice: it makes group C (simple
growth, one unit per event) and group D (overgrow K, keep 1) end at **identical
final parameter counts**, so C vs D isolates *which* unit survived and what the
competition cost, with no parameter-count confound.

The decomposition this buys:

- **C vs D** — does overgrowing and selecting beat committing to one candidate?
  (differs by both the selection and the competitive dynamics)
- **D vs D_RANDPRUNE** — does the *selection rule* do anything, or would keeping
  a random candidate work as well? (differs by selection only)

`D_RANDPRUNE` exists because "When BERT Plays the Lottery" found random
structured pruning competitive with importance-based pruning. Without it, any
claim that competition works is unsupported.

This is where we differ from Firefly, which selects candidates by a first-order
Taylor surrogate *at the instant of growth*. We let candidates actually
differentiate under training first, and we pay the FLOPs for doing so.

---

## 5. The hidden world

Facts are `[REL] [ARG1] [ARG2] [=] [ANSWER]`, loss on the answer position only,
so accuracy and confidence are unambiguous and the model earns no free loss
from predicting the format.

Everything derives from latent per-entity coordinates (group, period) that the
model never observes. Three defences are built in:

1. **Token ids are decorrelated from the latent grid by rejection sampling**
   (`|pearson| < 0.05`). One random permutation leaves r ≈ 1/√N ≈ 0.15, which
   is a real side channel: a model could "extrapolate" by reading the token id.
2. **D4 (`OMEN`) is a salted hash.** Ground truth exists but is unpredictable in
   principle. It is the leakage alarm: significantly above chance means
   information is reaching the model that should not be, and every other number
   in the study is void. `test_leakage_alarm_fires_on_an_oracle` checks the
   alarm actually rings.
3. **The atomic facts are all revealed at age 0.** Two pilot runs were thrown
   away to learn this (see `RESEARCH_LOG.md`): drip-feeding the axioms means a
   failure to extrapolate is really "the axiom had not been stated yet", which
   tests the revelation schedule rather than the model.

### The five buckets, each a single mechanism

| bucket | contents | what it actually requires |
|---|---|---|
| D0 interpolation | held-out facts, well-observed relations and entities | generalise within the observed regime |
| D1 near | BOND/REACT on distant-period entities | apply a group rule to less-observed entities |
| D2 compositional | REACTSOL only | chain `REACT` then `SOLUBLE`, never composed in training |
| D3 distant | HEAVIER touching a period-5 entity | learn `mass = f(period, group)` on periods 0–4 and **extrapolate it to a period never seen in any HEAVIER fact**, with that period stated by a SHELL fact |
| D4 unsupported | OMEN | nothing. It is unpredictable. |

The first pilot exposed that D3 originally also contained REACTSOL facts. That
was wrong: `REACTSOL` depends on group alone, so its period-5 instances need no
extrapolation at all and were inflating the flagship bucket. They are now D2.

---

## 6. The ledger

Hash-chained append-only JSONL. Records contain **no ground truth** —
`Ledger.append` raises if you try. Truth is joined afterwards, in
`evaluate.py`, straight from the generator. `verify()` fails if any record was
edited after the fact. "We promise we didn't peek" is not a protocol.

---

## 7. Metrics

The primary endpoint is the equally-weighted mean of D1/D2/D3 holdout accuracy
at matched FLOPs. Equal weights matter: D3 has ~1.6x the facts of D2 and ~1.6x
D1, so an unweighted pooled mean would quietly become "how well do you do on
D3".

We deliberately do **not** build a `correctness × novelty × calibration`
scalar. Composite endpoints have free parameters and free parameters in an
endpoint are how experiments get massaged. The components are reported
separately and the trade-off is shown as a Pareto plot.

`info_gain_bits` is the direct answer to "a model can game accuracy with safe
predictions": each correct prediction is worth `log2(1/P(answer | relation))`
bits against the empirical prior computed on **revealed facts only**, so a
majority-class guesser scores near zero.
`test_majority_guesser_scores_near_zero_information_gain` pins it.

With 3–5 seeds a t-test is a fiction. We report a paired bootstrap CI, Cohen's
d, and the raw seed-level win count side by side, and we do not pretend that
n=3 is anything other than n=3.
