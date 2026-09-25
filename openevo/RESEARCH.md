# RESEARCH.md — literature audit

Audit performed 2026-09-25. The question it has to answer first is blunt: **has someone
already run this experiment?** The short answer is no, but roughly 80% of the components
exist in mature form and several of them contradict assumptions in the original brief.
Where that happens it is called out, because a design that ignores a known negative
result is not a new experiment, it is a repeat of an old one.

---

## 1. Has this exact experiment been done?

The proposed stack is: minimal transformers + architecture growth/shrinkage + inherited
weights + heritable mutation parameters + speciated populations + many procedural worlds
+ compute cost + long-term selection + held-out alien worlds measuring adaptation speed
+ tracking whether complexity grows without being rewarded.

**No published system combines all of these.** The five nearest neighbours, and exactly
what each one is missing:

| Work | Covers | Missing relative to this study |
|---|---|---|
| **ATEP / Augmentative Topology EPOET** ([Nasir et al. 2022](https://arxiv.org/abs/2210.11442)) | Topology growth *inside* an open-ended environment generator; explicitly motivated by the "complexity ceiling" of fixed-topology POET agents; species-based variant | NEAT networks not transformers; no compute cost; no heritable mutation parameters; no passive-vs-driven analysis of the complexity trend; adaptation measured as transfer, not as adaptation *rate* |
| **AdA** ([Bauer et al. 2023](https://arxiv.org/abs/2301.07608)) | Human-timescale in-context adaptation on genuinely held-out tasks; scaling laws in model size, memory length and task-distribution richness | A single agent trained by RL, not a population; fixed architecture; no evolution; enormous compute |
| **GPICL** ([Kirsch & Schmidhuber 2022](https://arxiv.org/abs/2212.04458)) | Tiny transformers meta-learning *general-purpose* in-context learning algorithms; phase transitions from memorisation → system identification → general learning | Gradient meta-training, not evolution; no architecture change; no complexity question |
| **POET / Enhanced POET** ([Wang et al. 2019](https://arxiv.org/abs/1901.01753), [2020](https://proceedings.mlr.press/v119/wang20l/wang20l.pdf)) | Agent–environment co-evolution, transfer between niches, the ANNECS open-endedness metric | Fixed agent architecture; no compute cost; no evolvability genes |
| **Polyworld passive/driven analysis** ([Yaeger & Griffith 2011](https://arxiv.org/abs/1112.4906)) | Precisely the statistical machinery for asking whether a complexity trend is *driven* or *passive diffusion off a lower bound* | Neural-complexity measure, not parameter count; no transformers; no held-out adaptation metric |

**Conclusion:** the combination is novel, and the contribution is mostly the *measurement
design* rather than the mechanism. Individual mechanisms should be adopted from the work
above rather than reinvented; what does not exist anywhere is a clean decomposition of
"adaptation improved" into its possible causes.

---

## 2. Complexity growth — the literature that the brief omits, and that decides the result

This is the most consequential gap in the original design. The brief asks whether
complexity will "spontaneously" increase and treats the answer as self-evidently
meaningful. Three separate literatures say it is not.

**Passive diffusion versus driven trends.** [McShea (1994)](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1558-5646.1994.tb02211.x)
distinguishes *driven* trends (a homogeneous force pushing every lineage) from *passive*
ones (an unbiased walk against a reflecting lower boundary). Gould's "left wall" is the
same argument: a drunkard's walk beside a wall produces a rising mean with no directional
force at all. [Yaeger & Griffith (2011)](https://arxiv.org/abs/1112.4906) apply the test
inside an artificial-life system, and [Butterworth et al. (2025)](https://besjournals.onlinelibrary.wiley.com/doi/full/10.1111/2041-210X.70104)
give current methods for distinguishing the two in the presence of boundaries. The
discriminating observables are (a) the *minimum* of the distribution, which a driven
trend moves and a passive one does not, and (b) **subclades founded away from the wall**,
which keep rising under a driven trend and regress under a passive one.

This directly produced two design requirements: track the whole size distribution rather
than mean and maximum, and seed *displaced founders* above the bulk. It also produced a
measurement (§6 below) showing the original operator design would have manufactured a
28× passive increase.

**Bloat.** Variable-length evolutionary systems grow for reasons unrelated to fitness —
removal bias, crossover bias, hitchhiking, and the simple fact that far more long
representations than short ones encode any given behaviour
([Luke & Panait 2006](https://www.researchgate.net/publication/6883218_A_Comparison_of_Bloat_Control_Methods_for_Genetic_Programming);
[Silva et al. on Operator Equalisation](https://www.researchgate.net/publication/257564492_Operator_equalisation_for_bloat_free_genetic_programming_and_a_survey_of_bloat_control_methods)).
Growth in neutral or near-neutral regions of the genome is the *expected* outcome, not a
surprising one. Hence the requirement to report **effective** parameters (those whose
ablation changes behaviour) alongside raw parameter count.

**Digital evolution of complex features.** [Lenski, Ofria, Pennock & Adami (2003)](https://www.nature.com/articles/nature01568)
remains the reference result for complexity arising by selection in a digital system, and
its key finding is a constraint on this design: complex functions evolved *only* when the
simpler intermediates were themselves selectively favoured. An environment distribution
with no graded stepping stones will not produce complexity however long it runs.

---

## 3. Evolvability — what "selecting for evolvability" actually buys

[Wagner & Altenberg (1996)](https://onlinelibrary.wiley.com/doi/10.1111/j.1558-5646.1996.tb02339.x)
is the canonical framing. Direct optimisation for it exists: evolvability search
([Mengistu, Lehman & Clune 2016](https://arxiv.org/abs/1606.05365)) and
[Evolvability ES (Gajewski et al. 2019)](https://arxiv.org/abs/1907.06077) evaluate a
parent by the behavioural *variance* of its offspring. Both are directly relevant as
baselines: they tell us what evolvability looks like when it is the explicit objective,
which is the comparison an "emergent evolvability" claim needs.

Two results reshape the secondary question:

* [Canaan & Ofria, PNAS 2024](https://www.pnas.org/doi/10.1073/pnas.2413930121) find that
  environmental change produces **two distinct routes** to evolvability: raising the
  mutation rate, and biasing the *distribution of mutational effects* towards beneficial
  phenotypes. Critically, these have different signatures — evolved mutational
  neighbourhoods speed adaptation to *previously encountered* environments, while raw
  mutation rate helps with *genuinely novel* ones. This study therefore must measure both,
  and must not read a flat mutation rate as "no evolvability evolved".
* [Clune et al. (2008)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2527516/) show
  natural selection systematically evolves mutation rates *far below* the long-term
  optimum on rugged landscapes. A predicted, literature-grounded outcome is therefore that
  evolved mutation rates **collapse**, and that this harms long-term adaptability. That is
  a negative result the design should be able to detect rather than be surprised by.

Self-adaptive mutation rates are long-established in evolution strategies, so the
mechanism itself is not novel; what is measured about it can be.

---

## 4. Architecture change with inherited function

[Net2Net (Chen, Goodfellow & Shlens 2015)](https://arxiv.org/abs/1511.05641) introduced
function-preserving widening and deepening; [Network Morphism (Wei et al. 2016)](https://arxiv.org/abs/1603.01670)
generalised it beyond identity-initialised and idempotent-activation cases; a
[2024 treatment](https://arxiv.org/abs/2410.11038) gives a more complete theory of
function-preserving transforms. The known failure mode is that perfectly
function-preserving growth can create capacity with *zero gradient*, which then never
trains — addressed here by zeroing only the output path of new capacity (so the gradient
with respect to it is non-zero) plus optional symmetry-breaking noise.

For transformers specifically, one detail is not in the literature and had to be solved
here: **RoPE frequencies are a function of `head_dim`**, so changing head width silently
re-assigns the frequency of every existing dimension pair. Ordering the frequency
exponents by the van der Corput sequence makes each smaller head's frequency set an exact
*prefix* of every larger one, which restores exact function preservation under head-width
growth. See `models/transformer.py:rope_tables`.

Related: [CoDeepNEAT](https://arxiv.org/abs/1703.00548), [Regularized/Aging Evolution](https://arxiv.org/abs/1802.01548)
(whose age-based removal is adopted implicitly via finite lifetimes),
[AutoML-Zero](https://arxiv.org/abs/2003.03384), [Evolving RL Algorithms](https://arxiv.org/abs/2101.03958),
[The Evolved Transformer](https://arxiv.org/abs/1901.11117), and
[Weight Agnostic Neural Networks](https://arxiv.org/abs/1906.04358) — the last as the
clean demonstration that architecture alone can carry task performance, which is exactly
what the `reinit_weights` probe condition tests for here.

---

## 5. Learning inside a lifetime, and where adaptation lives

The Baldwin effect ([Downing 2010](https://folk.idi.ntnu.no/keithd/publications/downloads/gecco-baldwin2010.pdf);
[Fernando et al. 2018, "Meta-Learning by the Baldwin Effect"](https://arxiv.org/abs/1806.07917);
[Weber et al. 2023](https://arxiv.org/abs/2306.11761) on when learning and evolution
combine usefully) is the classical frame. The transformer-specific version is newer and
more directly usable: [*An evolutionary perspective on modes of learning in Transformers*](https://arxiv.org/abs/2505.09855)
casts in-weight learning as the slow, heritable channel and in-context learning as the
fast, within-lifetime one, and shows the balance between them is set by **how fast the
environment changes**. [Hebbian and gradient-based plasticity in transformers](https://arxiv.org/abs/2510.21908)
gives a fast-weights middle path.

This is the single most useful import into the design. It converts a vague "study
evolution plus learning" into a controlled experiment: make environmental stability an
explicit parameter (`WorldSpec.n_variants`), and measure whether adaptation migrates
between the weight channel and the context channel as it varies.

---

## 6. Open-endedness: definitions and metrics

[Hughes et al., ICML 2024](https://arxiv.org/abs/2406.04268) give the current formal
framing via **novelty and learnability**, which is directly reusable as the environment
reproduction criterion for phase 5 (and is a better criterion than "defeats the agents",
which the brief already correctly rejects). For measurement,
[the MODES toolbox (Dolson, Vostinar, Wiser & Ofria, *Artificial Life* 2019)](https://direct.mit.edu/artl/article/25/1/50/2915/The-MODES-Toolbox-Measurements-of-Open-Ended)
provides change, novelty, complexity and ecology potential — each **defined relative to a
neutral model**, which is precisely the discipline this study needs. Enhanced POET's
**ANNECS** (accumulated number of novel environments created and solved) is the
complementary environment-side metric.

Unsupervised environment design is the mature version of phase 5:
[PAIRED](http://aima.eecs.berkeley.edu/~russell/papers/neurips20-paired.pdf) and
[ACCEL](https://arxiv.org/abs/2203.01302) use regret-based curricula and are cheaper and
better-behaved than POET's full machinery.

Recent open-ended systems worth watching but *not* adoptable here, because they are
LLM-mediated rather than neural-substrate evolution:
[Darwin Gödel Machine](https://arxiv.org/abs/2505.22954),
[ShinkaEvolve](https://arxiv.org/abs/2509.19349), and
[AI-GAs](https://arxiv.org/abs/1905.10985) as the programme statement.

---

## 7. Backend

[MLX vs PyTorch-MPS benchmarking](https://github.com/TristanBilot/mlx-benchmark) and a
[comparative study of local inference stacks on Apple Silicon](https://arxiv.org/abs/2511.05502)
both report MLX ahead for transformer workloads, for a reason that matters here: PyTorch
MPS dispatches eagerly, one Metal command buffer per op, while MLX builds a graph and
fuses at `mx.eval()`. At this model scale the workload is *entirely* dispatch-bound — see
`DESIGN.md §6` for the measured arithmetic-intensity argument — so graph fusion is worth
far more than kernel quality. That is a hypothesis to be confirmed by
`scripts/bench_backend.py` on the target machine, not an assumption.

---

## 8. What the audit changed

1. Added the passive-vs-driven battery (neutral arm, displaced founders, whole-distribution
   tracking, effective-parameter reporting). Without it the headline complexity claim is
   uninterpretable — and the measurement in `DESIGN.md §3` shows the naive operator set
   would have produced a 28× passive increase under no selection at all.
2. Replaced `λ·FLOPs` with compute as a finite ecological resource, because λ determines
   the sign of the result.
3. Made environmental stability an explicit parameter, importing the in-weight/in-context
   framing, and made the *fast* channel the primary adaptation metric because it carries
   no learning-rate confound.
4. Added the evolvability decomposition (`metrics/probe.py`): five transplant conditions
   that separate genuine adaptive machinery from a better prior, from capacity, and from
   hyperparameter tuning.
5. Adopted MODES-style neutral-model-relative metrics and ANNECS rather than inventing
   open-endedness measures.
6. Kept the mutation-rate question but split it in two, per Canaan & Ofria, and recorded
   Clune et al.'s prediction that evolved rates will sit below the long-term optimum.
