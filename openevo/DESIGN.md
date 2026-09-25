# DESIGN.md — critique of the proposed experiment, and the design actually built

The brief asked to be challenged rather than agreed with. This document does that first,
then specifies what was built instead. Every number quoted here was measured in this
repository; the commands that produce them are given so they can be re-run.

**Summary of the verdict.** The central question is well posed and the environment
philosophy (many causal structures, per-organism ecological samples, held-out mechanisms)
is right and is kept. Seven things are changed, one of which is not a refinement but a
correction: **as originally specified, the experiment would have reported spontaneous
complexity growth that was entirely an artefact of the mutation operators.** That is shown
by measurement in §3, not argued.

---

## 1. The load-bearing problem: "adaptation speed improved" is not a finding

The brief nominates adaptation efficiency on alien worlds as the primary metric and
"adaptation_speed(generation t) improves" as the central test. But at least four different
mechanisms produce a rising curve, and only one of them is the hypothesis:

1. genuine in-context adaptive machinery — the hypothesis;
2. a better **reactive prior** that happens to suit held-out worlds — transfer, not
   adaptation, and the default expectation from any multi-task training;
3. **capacity** — larger descendants learn faster per interaction for trivial reasons;
4. **hyperparameters** — exploration temperature and learning rate tuned to the fixed
   evaluation budget, i.e. fitting the protocol rather than adapting.

The proposed design cannot separate these, so it cannot distinguish its own hypothesis
from three mundane alternatives. This is the reason for the largest addition to the
system.

**What was built** (`src/openevo/metrics/probe.py`). Frozen organisms are re-scored under
five transplant conditions on the *same* world instances with common random numbers:

| Condition | What it removes | What a surviving effect means |
|---|---|---|
| `full` | nothing | the headline number |
| `no_feedback` | previous action and reward tokens zeroed | improvement that survives is mechanism 2, a reactive prior — **not** in-context learning |
| `reinit_weights` | learned weights; architecture and genes kept | improvement that survives is carried by the *substrate*: evolvability in the strict sense (cf. Weight Agnostic Neural Networks) |
| `capacity_matched` | the evolved shape; ancestral shape re-scaled to the same parameter count, random weights | isolates mechanism 3 |
| `fixed_hparams` | evolved exploration temperature, reset to ancestral | isolates mechanism 4 |

`no_feedback` is the sharpest of these. In-context adaptation is *defined* by using the
consequences of one's own actions; an organism blinded to reward and action history but
still improving across the context was never adapting. The test that it actually bites is
`tests/test_isolation.py::test_no_feedback_condition_actually_blinds_the_organism`.

---

## 2. The other leak: the experimenter

The brief carefully forbids alien-world performance from entering reproduction, mutation
selection, hyperparameter tuning, environment generation and early stopping. It omits the
largest channel: **a researcher who iterates on the system while watching alien-world
numbers leaks through their own choices just as surely as the fitness function would.**

Built instead: Class C is split into **C-dev**, inspectable freely during engineering, and
**C-test**, which the runner refuses to touch unless `--open-test-set` is passed. Family
partitions are a deterministic function of a seed and are hashed (`SuiteSplit.seal()`) into
each run's manifest, so a run can prove which partition it used. `PREREGISTRATION.md`
records predictions and the analysis before C-test is opened.

Class isolation is enforced by tests rather than by intent: families are disjoint across
classes, alien mechanisms cannot appear in A or B, Class B is a pure *recombination* of
Class A's vocabulary, and `population.py` is asserted not to reference the probe machinery
or any held-out class name at all.

---

## 3. The complexity claim would have been an artefact — measured, not argued

The brief asks whether architectures grow "spontaneously" when compute has a cost. The
implicit null is that nothing happens without selection. That null is false, for two
reasons the brief does not mention.

**Reason one: operator asymmetry.** If growth multiplies a dimension by 1.25 and shrinkage
divides by it, integer rounding makes the steps unequal in log space — 22 → 28 is +0.241
nats while 22 → 18 is only −0.201. A neutral population with `p_growth_bias = 0.5` then
drifts upwards for purely arithmetic reasons. **Fix:** every dimension lives on a fixed
geometric ladder of integers and mutations move ±1 rung, so the rung walk is provably
unbiased and `shrink(grow(x)) == x` exactly. Verified by
`tests/test_models.py::test_grow_then_shrink_is_identity_on_the_ladder`.

**Reason two: the left wall.** Even with a symmetric ladder, a population sitting against
its lower bound can only move one way. This is precisely Gould's left wall and McShea's
passive-versus-driven distinction, and it is not removable by better engineering.

Measured, with **no selection at all** (60 neutral generations × 80 independent lineages,
`p_growth_bias = 0.5`):

| Founder | Δ rung (95% CI) | Δ ln(params)/generation | implied over 500 generations |
|---|---|---|---|
| 5131 params, `n_blocks=1` | −0.20 ± 1.30 | **+0.00666 ± 0.00447** | **28×** |
| 5360 params, `n_blocks=2` | −0.38 ± 1.22 | +0.00235 ± 0.00470 | 3.2× (CI spans 1) |
| 20550 params, `n_blocks=2` | −1.15 ± 1.21 | −0.00222 ± 0.00432 | 0.3× (CI spans 1) |
| 81356 params, `n_blocks=2` | −0.11 ± 1.32 | +0.00358 ± 0.00472 | 6.0× (CI spans 1) |

So a naive run would have shown a 28-fold "spontaneous" increase in parameter count with
no selection whatsoever. The cause is specific and instructive: with single-block founders
the depth dimension sits on the floor of its ladder, so `remove_block` can never fire while
`add_block` always can. Founders are therefore seeded one rung clear of the floor
(`scale_to_params(..., n_blocks=2)`), which reduces the passive drift to statistical
noise.

An alternative fix — sample the dimension first and abort when the move is illegal — was
implemented, measured, and **rejected**: it is markedly *worse* (+2.75 ± 1.12 rungs),
because blocked shrinks vanish while the matching growth still succeeds. The rejected
variant and its measurement are recorded in the code comment at
`evolution/organism.py:reproduce`, since the reasoning is not obvious and the intuitive
choice is the wrong one.

**What this implies for the protocol.** Four things become mandatory rather than optional:

1. **A neutral arm in every experiment** (`PhaseConfig(neutral=True)`: identical operators
   and demography, fitness randomised). Complexity claims are reported as selected-minus-neutral,
   never as raw growth.
2. **Track the whole size distribution**, especially the *minimum*. A driven trend moves
   the minimum; passive diffusion off a wall does not (McShea's test).
3. **Displaced founders** (`displaced_founder_params`): seed lineages well above the bulk.
   Under a driven trend they keep growing; under passive diffusion they regress toward the
   bulk. This is the decisive subclade test and it is cheap.
4. **Report effective parameters** alongside raw ones. The genetic-programming bloat
   literature is unambiguous that variable-size representations grow for reasons unrelated
   to fitness; raw parameter count cannot distinguish complexity from introns.

---

## 3b. Aggregate metrics cannot tell you whether the system is working

This was not anticipated, and it is the most transferable lesson from the build.

The first complete pilot produced: mean fitness rising 0.019 → 0.206; Class C-dev
adaptation AUC rising from ~0 to +0.155; 246 distinct architectures over 3,449 births; the
neutral arm flat at 0.007 throughout; exploration temperature collapsing 1.00 → 0.18 while
the neutral arm drifted to 1.22. A clean, internally consistent, publishable-looking set of
curves with a working control.

The transformer was doing nothing at all.

Weight mutation perturbed each tensor by `sigma * rms(tensor)`. That is scale-invariant,
which is desirable, but it makes **zero an absorbing state** — and both `init_params` and
every function-preserving growth operator deliberately start each block's output path
(`Wo`, `W2`) at exactly zero, because that is what makes a new block an exact identity.
Those tensors therefore had an effective mutation size of about 5e-6 and never left zero.
Attention and the FFN stayed pinned at the identity for the entire run, and evolution was
optimising the embeddings and the output head and nothing else.

Measured on the champion: ablating **every** block output path changed the action
distribution by a total variation of **0.00019**. Individual unit ablations came in at
~1e-5, four orders of magnitude below any threshold one might pick, so 0 of 99 units
counted as effective.

Nothing in fitness, the architecture statistics, the species counts, the gene trajectories
or the adaptation probes revealed this. The only instrument that did was the
effective-parameter ablation — which existed only because `PREREGISTRATION.md` H3 demanded
it as a *bloat* check, for a completely different reason.

Two consequences beyond the fix itself:

* That run's `no_feedback` result — improvement attributable to a reactive prior rather
  than in-context learning — is exactly what a policy with no working attention produces.
  It must not be read as a finding about evolution. The decomposition was reporting
  correctly; the substrate was broken.
* An ablation-based health check belongs in the standard metric set for any system with
  function-preserving growth, not just as a bloat control. Zero-initialised output paths
  are the *standard* construction for identity-preserving growth (Net2Net, network
  morphism, and this repository), and any mutation operator whose step size vanishes with
  the tensor will silently freeze them.

Fixed at `evolution/organism.py:_perturb_weights` by flooring the step at the scale the
tensor would have had at initialisation. Regression tests cover the operator and an
end-to-end lineage. The broken run is kept at `results/pilot_inert_stack/`.

---

## 4. `λ·FLOPs` is a free parameter that decides the answer

The brief proposes fitness ≈ `performance − λ·inference − α·learning − β·memory`, while
also warning against scalarising everything. The warning is right and the formula is the
problem: λ *is* the result. Choose it small and architectures grow; choose it large and
they shrink. The experiment would report the experimenter's choice back to them, and there
is no principled way to pick λ because performance and FLOPs have no common unit.

**Built instead: compute as a finite ecological resource.** Each island receives a
per-generation FLOP budget. Offspring are produced until it is exhausted, and each is
charged the accounted cost of its own evaluation. A larger organism pays no fitness tax; it
simply means fewer offspring fit into the generation. Growth is favoured exactly when
return *per unit compute* justifies it, and no exchange rate is ever chosen.

This also upgrades the question. Instead of "do architectures grow (under my λ)?", the
experiment sweeps the budget and asks **at what compute abundance does growth become
viable?** A phase boundary is a far stronger result than a binary answer, and it is
falsifiable in a way the binary version is not. `scalarised` and `pareto` selection remain
implemented, specifically so that λ-dependence can be *demonstrated* as a finding.

---

## 5. Complexity growth needs an expanding niche, so difficulty is unbounded from day one

The brief defers environment co-evolution to phase 5. But under a fixed environment
distribution with a cost on compute, the optimal architecture size is *bounded*, and growth
must stop. Unbounded complexity growth requires an expanding niche — which is why POET and
ATEP couple agent complexity to an environment generator, and why ATEP's stated motivation
is the "complexity ceiling" of fixed-topology agents.

Deferring co-evolution entirely would therefore pre-commit the experiment to a null result.
Built instead: the world grammar is **compositionally unbounded from phase 1** — worlds are
compositions of a latent dynamics with chains of observation and reward modifiers, with no
ceiling on depth — and sampled from a fixed heavy-tailed depth distribution. Headroom
always exists without needing POET's machinery. Phase 5 co-evolution then becomes a genuine
enhancement rather than a prerequisite, and when it arrives it should use
novelty-and-learnability (Hughes et al. 2024) or regret (ACCEL), not "defeats the agents".

---

## 6. Where adaptation lives: two channels, measured separately

The brief treats lifetime learning as gradient descent with a heritable learning rate. That
makes the primary metric confounded by construction — "evolved evolvability" would largely
mean "evolved learning rate", which is mechanism 4 above.

Built instead, two explicitly separated channels:

* **Fast / in-context.** The context window is scored in four blocks; improvement from the
  first to the last is adaptation with *no weight change at all*. This is the **primary**
  metric precisely because there is no lifetime hyperparameter to confound it.
* **Slow / in-weights.** Bounded REINFORCE on the organism's own trajectory
  (`evaluate.lifetime_learn`), with a heritable learning rate — reported separately.

And a control knob that makes the relationship between them experimental rather than
observational: `WorldSpec.n_variants`, the number of distinct instance-parameter draws a
world has. At `n_variants=1` a world's mapping is stable across evolutionary time and can be
assimilated into weights (the Baldwin regime); at `n_variants=0` every context draws a fresh
mapping, nothing instance-specific is heritable, and the only route to reward is in-context
inference. Sweeping it tests the prediction from *An evolutionary perspective on modes of
learning in Transformers* — that environmental change rate determines whether adaptation
lives in weights or in context — inside an evolutionary system rather than a training run.

**This turned out to matter immediately.** The first capacity probe (§8) was
uninterpretable until the two regimes were separated.

---

## 7. Statistical power

Evolutionary runs have enormous seed-to-seed variance; with three seeds nothing here is
claimable. The design targets **≥10 seeds per condition**, which is affordable only because
the worlds are cheap and the primary channel needs no gradients. The test is pre-specified
in `PREREGISTRATION.md`: a permutation test on the slope of adaptation-AUC against
generation with seed as the unit of resampling, effect sizes with bootstrap CIs, and no
optional stopping.

---

## 8. Recomputed settings, from measurement

### 8.1 Minimum viable size — the answer is much smaller than 5K, and the question was wrong

The brief proposes ancestral scales of ~5K / ~20K / ~80K and asks for the minimum viable
size to be established empirically. Doing so (`scripts/min_viable_size.py`) produced a
result that changes the scale ladder.

First attempt, behaviour-cloning against the reference policy with a fresh instance every
context, scored on held-out worlds:

| params | CE loss | acted score |
|---|---|---|
| 909 | 1.080 | 0.097 |
| 5131 | 0.532 | 0.055 |
| 20168 | 0.127 | 0.050 |

Cross-entropy falls fourfold while acted score *falls*. It would be easy to read that as
"these architectures cannot do the task". It is not: the probe was conflating
representational capacity with in-context inference. With a *stable* world
(`n_variants=1`), scored on worlds it was trained on:

| params | CE loss | acted score |
|---|---|---|
| 983 | 0.209 | **0.919** |
| 2238 | 0.110 | **0.972** |

(Re-measured after per-block normalisation was introduced in §6: 983 → 0.868,
5360 → 0.967. Slightly lower, same conclusion.)

**A 983-parameter transformer expresses a near-reference policy.** Representational
capacity is not the binding constraint anywhere near 5K; the constraint is in-context
inference of the instance. Consequences:

* the ancestral ladder starts **lower** — roughly 1K / 5K / 20K — so that growth has room
  to be observed rather than being where the ladder begins;
* the hypothesis sharpens usefully. If architectures grow, it cannot be to represent a
  better policy, since ~1K suffices. It would have to be to support *inference*. That is a
  far more interesting claim, and it is directly testable with the `no_feedback` and
  `n_variants` machinery.

### 8.2 Throughput and the backend decision

Measured here (4-core container, NumPy float32, whole population batched per architecture
bucket):

| params | env-steps/s | contexts/s |
|---|---|---|
| 5131 | 67,216 | 1,050 |
| 20168 | 33,695 | 527 |
| 80855 | 16,425 | 257 |

The decisive number is arithmetic intensity per dispatch. A 5K organism's context is
2,176 kernel dispatches and 0.93 MFLOP — **428 FLOP per dispatch**. At a ~10 µs dispatch
overhead that is 0.04 GFLOP/s, four orders of magnitude below what the hardware can do. The
workload is *entirely* dispatch-bound, and batching is not an optimisation but the whole
game:

| contexts per batched call | effective GFLOP/s |
|---|---|
| 1 | 0.04 |
| 256 | 11.0 |
| 2,048 | 87.8 |
| 16,384 | 702 |

This is why organisms are bucketed by architecture signature and each bucket runs as one
batched graph, and it is the strongest argument for MLX over PyTorch-MPS on the target
machine: MPS dispatches eagerly, one command buffer per op, while MLX builds a graph and
fuses at `mx.eval()`. Graph fusion is worth more here than kernel quality.

**This remains a hypothesis.** This container is Linux on 4 CPU cores; no Apple Silicon was
available, so no MLX number in this repository is measured. `scripts/bench_backend.py` is
written to settle it on the target machine, and the backend interface is deliberately thin
so that swapping in MLX touches only `models/transformer.py`. Fitness is charged from the
analytic FLOP model, never from wall-clock, so the backend choice cannot affect any
scientific result — only how long it takes.

### 8.3 Population and generation budget

At pilot settings — 8 islands × (24 adults + 24 offspring) × 7 credited worlds × 8
instances = **21,504 contexts/generation**:

| machine | s/generation | generations/hour | births/hour |
|---|---|---|---|
| this 4-core container | 20.5 | 176 | 67,000 |
| M3 Ultra @ 30× (conservative) | 0.68 | 5,275 | 2.0 M |
| M3 Ultra @ 100× (optimistic) | 0.20 | 17,582 | 6.8 M |

Even the conservative figure makes the brief's "hundreds or thousands of evolutionary
turnovers" comfortable: 1,000 generations is roughly 11 minutes, so **10 seeds × 8
conditions ≈ 15 hours**. The binding constraint is not throughput but statistical power,
which is why the budget is spent on seeds and conditions rather than on population size.

Revised defaults, with reasons:

| Setting | Brief | Here | Why |
|---|---|---|---|
| ancestral scales | 5K / 20K / 80K | 1K / 5K / 20K | 983 params already expresses a near-reference policy |
| founder depth | (unspecified) | `n_blocks = 2` | at `n_blocks=1` neutral drift is 28×/500 generations |
| population | ~256 | 8 islands × 24 = 192 | throughput is not the constraint; seeds are. Spend the budget on ≥10 seeds per condition |
| lifetime | 512–4096 interactions | 64-step context, fast channel | the primary metric needs no gradients; the slow channel adds interactions only where it is being studied |
| compute cost | `λ·FLOPs` | ecological FLOP budget | λ determines the sign of the answer |
| structural mutation rate | 10–20 % | 15 % initial, then heritable | as proposed; the ladder makes it interpretable |
| memory ceiling | ~400 GB | 2 GB per evaluation group, chunked | groups are split *before* running, by analytic estimate |

---

## 9. What would count as evidence

**For evolved adaptability** (rather than benchmark overfitting) — all four required:

1. Class C adaptation AUC rises with generation, with a permutation-test slope over seeds,
   while **Class A performance has plateaued**. Divergence between the two is the signature;
   a rise in both is consistent with ordinary transfer.
2. The rise **does not survive `no_feedback`**. If a blinded organism improves just as much,
   the system evolved a better reactive prior and the adaptability claim fails.
3. The rise **exceeds `capacity_matched`**. Otherwise it is capacity, available to any
   equally large random network.
4. The rise **survives `fixed_hparams`**. Otherwise it is exploration-schedule tuning.

A further, stronger result would be a rise that survives `reinit_weights` — adaptability
carried by the architecture itself, independent of learned content.

**For spontaneous complexity growth** — all four required:

1. Parameter growth exceeds the **neutral arm** at matched generation count, over seeds.
2. The **minimum** of the size distribution moves, not only the mean and maximum
   (McShea's driven-trend test).
3. **Displaced founders do not regress** toward the bulk (the subclade test).
4. **Effective** parameters grow, not just raw ones — otherwise it is bloat.

If growth appears but fails (2)–(4), the correct report is *passive diffusion off a lower
bound*, which is a real and publishable finding about this system and is exactly what the
neutral arm exists to detect.

**Negative results explicitly preserved.** That complexity never increases; that it rises
while generalisation does not; that mutation rates collapse (which Clune et al. 2008
predicts on rugged landscapes); that Lamarckian inheritance harms long-run evolvability;
that many small specialists beat few large generalists. None of these are failure modes of
the experiment. Only an uninterpretable result is.

---

## 10. System layout

```
src/openevo/
  models/        genome.py     exact parameter + FLOP accounting, scale ladder
                 transformer.py batched tiny transformer; KV-cached acting,
                                teacher-forced learning, hand-derived backward
                                (gradient-checked to 1.7e-7)
                 morphisms.py   ladder-based growth/shrink; growth exactly
                                function-preserving at noise=0
  environments/  worlds.py      compositional POMDP grammar, paired evaluation
                                with common random numbers, n_variants stability knob
                 suites.py      sealed A / B / C-dev / C-test split, reference
                                normalisation, discriminativeness filter
  evolution/     organism.py    genotype: architecture + weights + strategy genes
                 evaluate.py    batched rollout, feedback ablation, slow channel
                 population.py  islands, ecological compute budget, selection, archive
  metrics/       probe.py       the five-condition evolvability decomposition
  storage/       run.py         manifest + SQLite metrics
                 scheduler.py   analytic peak-memory estimates, pre-flight chunking
```

Two structural invariants hold the design together, and both are tested:

* **Accounted compute is not wall-clock.** Fitness is charged from
  `ArchGenome.flops_forward`, so padding, bucketing and dispatch overhead cannot leak into
  any result.
* **The probe cannot reach the population.** It runs on deep copies, returns plain numbers,
  and writes only to the database. `test_isolation.py` asserts that `population.py` does
  not so much as mention the probe module or any held-out class name.
