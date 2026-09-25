# openevo — open-ended neuroevolution of tiny transformers

A research system for one question:

> When a population of very small transformers evolves across many procedurally generated
> worlds, and computation costs something, does evolution produce **better capacity to
> adapt** — and do architectures grow on their own?

And a second, harder one: can evolution select for **evolvability** itself — mutation
strategies, learning parameters and growth tendencies that make descendants better at
adapting to genuinely novel causal structure?

Complexity here is an **observable, never an objective**. Nothing in the system rewards
parameter count. Architectures may grow, shrink, diversify into size niches, or stay put.

---

## Start here

| Document | What it is |
|---|---|
| **[DESIGN.md](DESIGN.md)** | Critique of the original proposal and the design actually built. Read §3 first. |
| [RESEARCH.md](RESEARCH.md) | Literature audit, prior-art verdict, and what the audit changed |
| [EXPERIMENTS.md](EXPERIMENTS.md) | Stages, conditions, ablations, sweeps, metrics |
| [PREREGISTRATION.md](PREREGISTRATION.md) | Hypotheses and analysis, frozen before the held-out set is opened |

---

## Three findings from building it

**1. The naive design would have reported a result that wasn't there.** Under *no selection
at all*, with growth and shrinkage equally likely, parameter count drifts upward at
+0.0067 nats/generation — a **28-fold increase over 500 generations**, entirely passive.
The cause is Gould's left wall inside the operator set: with single-block founders,
`remove_block` can never fire while `add_block` always can. Seeding founders one rung clear
of the floor reduces the drift to statistical noise. Every complexity claim is reported
against a neutral arm for this reason. ([DESIGN.md §3](DESIGN.md))

**2. "Adaptation speed improved" is not a finding.** Four mechanisms produce that curve and
only one is the hypothesis. Frozen organisms are therefore re-scored under five transplant
conditions — the sharpest being `no_feedback`, which zeroes the previous-action and
previous-reward tokens. An organism blinded to the consequences of its own actions but
still improving across a context was never adapting; it had a better reactive prior.
([DESIGN.md §1](DESIGN.md))

**3. A 983-parameter transformer already expresses a near-reference policy** (0.919 on the
random-to-reference scale). Representational capacity is not the binding constraint
anywhere near 5K — in-context *inference* is. So if architectures grow in this system, it
cannot be in order to represent a better policy, which makes the hypothesis sharper and
more falsifiable. ([EXPERIMENTS.md](EXPERIMENTS.md))

---

## Design choices that differ from the brief

| Brief | Here | Why |
|---|---|---|
| fitness = performance − λ·FLOPs | compute as a finite per-island **resource** that limits birth rate | λ *is* the answer: small λ grows architectures, large λ shrinks them. A budget has no exchange rate, and sweeping it yields a phase diagram instead of a binary |
| adaptation speed as the metric | adaptation speed **decomposed** across five transplant conditions | separates adaptive machinery from priors, capacity and hyperparameter tuning |
| held out from *selection* | held out from selection **and** split into C-dev / C-test | a researcher iterating while watching alien scores leaks through their own choices |
| lifetime learning = SGD with heritable LR | **two channels**: in-context (primary, no hyperparameter) and in-weights (secondary) | the primary metric must not be confounded by an evolved learning rate |
| environment co-evolution deferred to phase 5 | world grammar **compositionally unbounded from phase 1** | with a bounded environment and a cost on compute, optimal size is bounded and growth *must* stop — deferring would pre-commit to a null result |
| ancestral scales 5K / 20K / 80K | **1K / 5K / 20K** | measured: ~1K suffices for a near-reference policy |
| ~256 organisms | 192, but **≥10 seeds per condition** | throughput is not the constraint; seed variance is |

---

## Quick start

```bash
cd openevo && pip install -e ".[dev]"
python -m pytest -q

python scripts/min_viable_size.py                                  # stage 3
python scripts/run_evolution.py --config configs/pilot.json        # the pilot
python scripts/run_evolution.py --config configs/neutral.json      # the null (required)
python scripts/report.py results/pilot/run.db \
       --neutral results/neutral/run.db --figures

python scripts/bench_backend.py --backends numpy mlx               # on Apple Silicon
```

Class C-test stays sealed: `run_evolution.py` will not touch it without `--open-test-set`.

---

## How it works

An **organism** is an architecture genome, a weight set, and strategy genes
(exploration temperature, learning rate, structural-mutation rate, weight-mutation
magnitude, and `p_growth_bias` — the heritable fraction of structural mutations that grow
rather than shrink). It lives on an island, is evaluated on that island's own overlapping
sample of worlds, and competes for a finite generation compute budget.

A **world** composes a latent dynamics with chains of observation and reward modifiers.
Class A uses familiar structure, Class B withheld *combinations* of familiar mechanisms,
Class C mechanisms never seen under selection. Every world shares one fixed interface, and
each is normalised by its own measured random and reference-policy scores so that scores
are comparable across classes.

Four properties are enforced by tests rather than by intention:

- **Growth is exactly function-preserving** at zero noise, for all five growth operators —
  verified against the forward pass, not asserted.
- **Grow and shrink are exact inverses.** Dimensions live on a geometric ladder and
  mutations move ±1 rung, so `shrink(grow(x)) == x` and the neutral walk has no built-in
  direction.
- **Accounted compute is not wall-clock.** Fitness is charged from an analytic FLOP model,
  so batching, padding and dispatch overhead cannot leak into any result.
- **The probe cannot reach the population.** It runs on deep copies and writes only to the
  database; `population.py` is asserted not to mention the probe module or any held-out
  class name.

Gradients for the slow channel are hand-derived and checked against finite differences to
1.7e-7.

---

## Status and honest limits

Stages 0–6 and 9 are implemented and exercised; stages 7–8 are implemented but not yet run
at statistical power; stage 10 (recombination) is implemented but unevaluated; stage 11
(environment co-evolution) is not implemented.

**No number in this repository was measured on Apple Silicon.** Development ran on a
4-core Linux container with a NumPy reference implementation. The case for MLX rests on an
arithmetic-intensity argument — a 5K organism's context is ~2,176 kernel dispatches at
~428 FLOP each, i.e. dispatch-bound by four orders of magnitude, so graph fusion matters
far more than kernel quality — and `scripts/bench_backend.py` exists to confirm or refute
it on the target machine. Because fitness is charged analytically, the backend choice
affects only how long a run takes, never what it concludes.
