# DESIGN.md — NEMO: Neuroevolution of Modular Organisms

**Status:** recommended design, revised against the original proposal.
Evidence: `RESEARCH.md` (literature), `scripts/compute_budget.py` (arithmetic),
`scripts/bench_backend.py` (hardware measurement plan).

---

## 0. Verdict on the original proposal

The proposal is well-conceived and most of it survives. Its instincts about
scientific discipline (operational definitions, ablation assays, phylogeny,
negative results being valuable, benchmark-before-committing) are exactly right
and are adopted wholesale.

**Six changes are recommended, in descending order of importance.**

| # | Change | Why | Cost |
|---|---|---|---|
| **1** | **Reframe the question from "does modularity emerge?" to "which ecological conditions are sufficient?"**, and run it as a 3×2×2 factorial with a fitness-shuffled drift control | The strong form is already known false: under fixed goals, evolution reliably produces *non-modular* networks (Kashtan & Alon 2005; Clune et al. 2013). As written, the experiment's most likely outcome is an uninterpretable null | Cheaper: each cell is smaller, and all 144 runs execute in one batch |
| **2** | **Make the environment compositional, with a shared subgoal basis, so Modularly Varying Goals is constructible** | MVG is the strongest known driver. *Randomly* varying goals — which is what "overlapping but nonidentical world samples" gives — is specifically known **not** to work | Neutral |
| **3** | **Write the whole population as one vectorised tensor program; never loop over organisms** | Measured in the model: naive 3.7 h/generation vs vectorised 0.41 s/generation — **32,768×**. This decides whether the experiment exists | Constrains module shape uniformity (§7) |
| **4** | **Drop lifetime learning from phase 1** | Gradient descent on two duplicate copies makes them differ. That is not evolutionary divergence, and it cannot be disentangled post hoc if learning is on from generation 0 | Saves 3×; but the real reason is causal identifiability |
| **5** | **Move copy-on-write out of the execution path into storage** | At 2,145 params/module the modelled live-path speedup from deduplication is **1.00×** (dispatch-bound regime), and COW indirection breaks the dense layout that delivers change #3. In storage it is genuinely valuable | Saves implementation effort |
| **6** | **Add an encapsulation operator** | Duplication + rewiring produces a *flat* graph. Nothing in the proposal can create a level above "module", so "hierarchical complexity" would return a structural null, not a biological one. Prior art: modular CGP, Modular NEAT, TPG | One operator |

**Rejected from the proposal, with reasons:** a 350 GB memory budget (modelled
pilot footprint is 0.26 GB — a ceiling 1,000× above the workload hides bugs
rather than preventing them; use 32 GB soft / 64 GB hard); per-module vocabulary
embeddings (they make modules non-interchangeable, destroying the meaning of
duplication and rearrangement); and genome *order* as the carrier of functional
meaning (replaced by explicit wiring, so that wiring, regulation and weight
mutations are separable — which is what makes the factorial possible at the
operator level).

**Adopted unchanged:** the module/organism/population/ecology layering;
~2.1K parameters per module and ~9K per ancestor (the proposal's arithmetic is
correct — see §6); island structure with local competition; asexual first;
metabolism on *executed* compute rather than genome length; ablation assays as
observational rather than selective; strict ancestral/recombination/alien
environment separation; module- and organism-level phylogeny as a primary
scientific output; the demand for operational definitions of every biological
term.

---

## 1. The reframed central question

> **Original:** Can Darwinian evolution on variable-length genomes of tiny
> cooperating neural modules spontaneously produce redundancy, duplication,
> divergence, specialisation, regulation, modularity and evolvability *without
> explicitly rewarding any of them*?

The answer to that, for the fixed-goal case, is already in the literature: **no.**
Kashtan & Alon evolved networks under a fixed goal and got "highly non-modular"
networks that "lack understandable motifs"; Clune, Mouret & Lipson report that
without a connection cost "modules never form." Running the proposal as written
risks spending weeks confirming a 2005 result.

> **Recommended:** **What is the minimal ecological condition sufficient for
> Darwinian evolution of neural-module assemblies to produce
> duplication → divergence → specialisation, and does the organisation that
> results confer measurable evolvability on unseen causal structure?**

This keeps everything the proposal cares about — nothing about modularity,
redundancy or specialisation is ever rewarded *within* a condition — while
making the experiment falsifiable, because different conditions carry different
predicted signs. It is also strictly more informative: a null in the fixed-goal
cell plus a positive in the MVG cell is a result; a global null is not.

### The novel part

The literature establishes the three drivers for *abstract networks* (logic
circuits, Boolean GRNs, small MLPs). Nobody has tested whether they transfer
when the unit of duplication is a **parameterised neural module with its own
regulatory state inside a variable-length genome**. Three specific things could
break: (a) modules are large enough that duplication is never near-neutral;
(b) message-passing between modules provides enough flexibility that entangled
solutions stay reachable, defeating the connection-cost mechanism; (c) module
internal capacity substitutes for organisation, so assemblies never need to
specialise. Any of those would be a real negative result worth publishing.

---

## 2. Operational definitions

Every biological term used in this project has exactly one computational
meaning. Implemented in `src/nemo/metrics/`. No claim may use a term outside
this table.

| Term | Operational definition | Null model |
|---|---|---|
| **Gene** | One entry in the genome: a tuple (module weight-slot ref, input-source refs, execution round, regulatory parameters). The unit of duplication and deletion. | — |
| **Module** | The parameter tensor a gene refers to. Modules are shape-identical and weight-distinct. | — |
| **Organism** | The executable assembly produced by developing one genome. Unit of selection. | — |
| **Active module** | Gate output > τ on ≥5% of lifetime timesteps. Modules below τ are present in the genome but pay no execution cost. | — |
| **Redundancy** | Pair (i,j) such that single ablation of either costs <5% of fitness while joint ablation costs >20%. Reported as a *count of pairs*, normalised by module count. | Ablation of randomly chosen size-matched pairs |
| **Specialisation** | Normalised entropy of module i's causal-contribution vector across the K subgoal channels: `S_i = 1 − H(c_i)/log K`, where `c_{ik}` is the fitness drop on subgoal k under ablation of i, normalised to a simplex. | Permutation over subgoal labels, per organism, 1,000 draws |
| **Division of labour** | Mean pairwise Jensen–Shannon divergence between modules' contribution vectors `c_i`. | Same permutation null |
| **Structural modularity** | `Q_str`: Newman modularity of the organism's realised message-passing graph, best over greedy + spectral partitions. | Degree-preserving rewired graph, 1,000 draws (controls the sparsity confound) |
| **Functional modularity** | `Q_cor`: Newman modularity of the thresholded module-activity correlation matrix over a lifetime. Reported **alongside** `Q_str`, never instead of it — structural modularity does not imply functional specialisation (Nature Comms 2024). | Phase-shuffled activity traces |
| **Regulatory organisation** | Mutual information `I(context ; gate_i)` between environment subgoal-phase and module i's gate output, in bits, minus the shuffled-context baseline. | Circular shift of the context stream |
| **Robustness** | Area under the fitness-vs-ablation-fraction curve, ablating uniformly at random, 32 draws per fraction ∈ {0, .125, .25, .5}. | Parameter-matched monolithic control |
| **Duplication event** | A birth in which genome length increased by a copy of an existing gene. Recorded with parent gene id. |
| **Divergence** | Post-duplication L2 distance between the two copies' weights, normalised by the mean weight norm, as a function of generations since duplication. | Sister lineages that did *not* duplicate, matched by generation |
| **Duplication→specialisation event** | A duplication whose two descendant copies, ≥50 generations later, both survive, and whose contribution vectors have JS divergence exceeding the 95th percentile of the permutation null. **This is the primary outcome measure.** | Permutation null, above |
| **Evolvability** | Normalised area under the adaptation curve on a held-out alien environment, at a fixed offspring budget, with pre-adaptation fitness as a covariate. | Fitness-shuffled drift control |
| **Metabolic cost** | Sum of `flops_forward(in_degree)` over modules whose gate fired, over the lifetime. Exact, from `ModuleSpec.flops_forward`. | — |

Two terms the proposal uses that are **deliberately not defined and not used**:
"cancer" and "cooperation." Neither has an unambiguous operational meaning in
this system in phase 1 (there is no module-level reproduction, so there is
nothing for a module to defect *from*). They may be defined in phase 3 if
module-level reproduction is introduced.

---

## 3. Experimental design

### 3.1 Factors (phase 1)

| Factor | Levels | Rationale |
|---|---|---|
| **G** — goal structure | **MVG** (goal switches every `g_switch` generations among combinations of a fixed subgoal basis) · **RVG** (subgoals resampled independently each switch — no shared basis) · **FIX** (one goal throughout) | The strongest known driver, and the one the proposal gets wrong. RVG is the control that isolates *modular* variation from *mere* variation |
| **M** — metabolic cost | on (`λ>0` on executed FLOPs) · off (`λ=0`) | Connection-cost analogue. Must be a factor, or a positive result means only "we paid for parsimony" |
| **D** — duplication operator | enabled · disabled | Tests whether duplication is *necessary*, not just present |

3 × 2 × 2 = **12 cells**, **12 replicate runs per cell** = **144 runs**, all
executed as independent lanes of one batched tensor program. Modelled cost:
**~11 min for 1,000 generations, ~3.9 h for 5,000.**

### 3.2 Controls (run as additional lanes in the same batch)

| Control | Purpose |
|---|---|
| **Fitness-shuffled drift** | Fitness assigned at random each generation. Establishes the null for genome-length growth, duplication rate, and every modularity metric. **Absent from the original proposal; it is the single most important addition.** Without it, genome bloat (well documented in GP, and known to drive Tierra's complexity growth) is indistinguishable from adaptive complexification |
| **Monolithic, matched** | One module with `d_model` solved so parameters match the modular ancestor. Implemented in `ModuleSpec.match_monolithic`. **Parameters and FLOPs both scale as `d_model²`, so a single `d_model` matches both budgets to within 2.5%** — one control answers "did modularity win, or did more parameters win?" without needing a separate compute-matched arm. Runs as its own invocation, because the batched engine takes one module shape per run (§7.2) |
| **Modular-fixed-topology** | Modules present, wiring and regulation frozen |
| **Regulation-frozen** | Wiring mutable, gate parameters frozen. Isolates factor 7 of the proposal's operator list |
| **Global selection** | Replaces island structure. Tests the proposal's (correct) intuition that global top-k is harmful |

### 3.3 Pre-registered predictions and decision rule

Written before any run. Primary outcome: **duplication→specialisation event
rate** (definition in §2), over the final 1,000 generations.

| Cell | Prediction |
|---|---|
| MVG + M-on + D-on | Rate significantly above drift null; `Q_str` and `Q_cor` both above rewired null; alien adaptation improves with evolutionary time |
| FIX + M-off | Rate indistinguishable from drift null on every metric |
| RVG + M-on | Intermediate; **if RVG matches MVG, Kashtan & Alon does not transfer to this substrate — a genuinely novel negative result** |
| Any cell, D-off | Specialisation may still occur via weight mutation alone; if it matches D-on, duplication is not necessary and a central premise of the proposal is wrong |

**Decision rule.** Cliff's δ ≥ 0.47 (large) between cell and its drift control,
Mann–Whitney *p* < 0.01 after Holm–Bonferroni across the 12 cells, n = 12
replicates. Power analysis and the final n are fixed after the pilot's observed
variance, not after seeing effects. **Negative results are reported with the
same prominence as positive ones**, per the proposal's instruction.

---

## 4. Representation

### 4.1 Genome

A genome is a variable-length list of **genes** plus organism-level scalars:

```
Genome = {
  genes : [ Gene ],                    # length 1..G_max, variable
  meta  : MutationRates,               # heritable, 8 log-scale scalars
  iface : (sensor_id, actuator_id),    # interface weight slots
}

Gene = {
  module   : slot index into the organism's module weight tensor,
  inputs   : [ source refs ],          # up to K; SENSOR | gene id | SELF
  round    : execution round 0..R-1,   # WHEN, not what
  gate_b   : float,                    # regulatory bias
  gate_s   : float,                    # regulatory slope
  innov    : innovation id,            # NEAT-style, for homology + phylogeny
  parent   : innovation id of the gene this was duplicated from (or none),
}
```

**Why this and not the proposal's `[A,B,C,D]` ordered list.** The proposal makes
genome *order* functionally significant. That conflates three things — which
module, where it sits in the graph, and when it fires — into one mutable
quantity, so a "rearrangement" mutation silently changes wiring and regulation
together. Splitting them into `inputs` (wiring), `round` (regulation) and
`module` (computation) makes the proposal's own mutation classes 1, 6 and 7
*causally separable*, which is a precondition for the operator-level factorial
and for the claim "regulatory mutation, specifically, did this."

The `innov`/`parent` fields give NEAT-style homology, which the proposal
correctly identifies as the right basis for recombination later, and which is
what makes the module phylogeny reconstructible.

### 4.2 Module

Not a language-model transformer block. A **message block**: it attends over a
masked set of incoming latent messages from its own recurrent state, emits one
latent message and one regulatory gate logit.

```
q   = W_q h                              # query from own state
K,V = W_k M_in , W_v M_in                # over ≤K incoming messages (masked)
c   = W_o softmax(qKᵀ/√d) V
h₁  = RMSNorm(h + c)
h₂  = RMSNorm(h₁ + W₂ σ(W₁ h₁ + b₁) + b₂)
out = h₂ ,  gate = σ(w_g·h₁ + b_g)
```

Exact counts (`src/nemo/modules/spec.py`, unit-tested):

| d_model | d_ff | params | KB (bf16) | FLOP/step |
|---|---|---|---|---|
| 8 | 16 | 561 | 1.10 | 1,988 |
| 12 | 24 | 1,225 | 2.39 | 4,316 |
| **16** | **32** | **2,145** | **4.19** | **7,540** |
| 20 | 40 | 3,321 | 6.49 | 11,660 |
| 24 | 48 | 4,753 | 9.28 | 16,676 |
| 32 | 64 | 8,385 | 16.38 | 29,396 |

This covers the proposal's requested 500 / 1K / 2K / 3K / 5K / 10K sweep with a
single knob (`d_model`, `d_ff = 2·d_model`). **The proposal's ~2,500/module
estimate is confirmed correct.**

**Why modules have no vocabulary.** The proposal specifies vocab ≈32 and context
16–64 per module. Giving each module its own embedding table would cost
+512 params and, far worse, would make modules **non-interchangeable** — a
duplicated or rearranged module would be reading a different alphabet than its
neighbours. Duplication, deletion and rearrangement are only meaningful if any
module can be wired to any other. So symbol↔latent conversion happens exactly
once, at the organism's sensor and actuator. This is not a simplification for
speed; it is what makes the operators coherent.

**Why attention at all, given the module could be an MLP.** Because module
in-degree *changes* when wiring mutates. Attention over the incoming message set
is the natural size-agnostic aggregator; mean-pooling would discard source
identity and a fixed concatenation would break under rewiring. One head, no
causal masking over time — the attention is over *sources*, not over a token
sequence.

### 4.3 Organism

Ancestor: 4 genes → 4 × 2,145 + sensor 256 + actuator 136 = **8,972 neural
parameters** + 44 regulatory scalars. The proposal's ~10K estimate is confirmed.

Starting-complexity controls at 1, 2, 4, 8 genes, as the proposal requests
(run as extra lanes; cost is negligible).

### 4.4 Development

Genome → phenotype is deterministic and cheap: resolve `inputs` refs, drop
dangling edges, topologically place genes into `round` buckets, allocate module
weight slots. No growth process, no NDP. *Justification:* a developmental
encoding would add a second source of structure that is not under the
experiment's control, confounding the measurement of *evolved* organisation.
Developmental encodings are a phase-4 question, deliberately not phase 1.

---

## 5. Environment: compositional worlds with a shared subgoal basis

This is where change #2 lives. The environment must make MVG *constructible*,
which requires an explicit, shared, recombinable subgoal basis.

### 5.1 Subgoal basis

A fixed library of **causal primitives**, each a small tensor-expressible
finite-state mechanism over a 16-symbol alphabet:

| Primitive | Causal demand |
|---|---|
| `RECALL(k)` | Emit the symbol seen k steps ago (working memory) |
| `COUNT(m)` | Act on every m-th occurrence of a cue (periodic state) |
| `DELAY(d)` | Reward arrives d steps after the action (credit assignment) |
| `XOR(a,b)` | Response depends on conjunction of two channels (composition) |
| `SWITCH(c)` | A context symbol flips the mapping (regulation) |
| `DECOY(p)` | A locally rewarding action is globally costly (deception) |
| `NOISE(σ)` | Observation corrupted with probability σ (stochasticity) |
| `GATE(h)` | Hidden state determines which channel is live (partial observability) |
| `IRREV(a)` | One action permanently closes a branch (irreversibility) |
| `DRIFT(r)` | The mapping changes slowly mid-lifetime (nonstationarity) |

A **world** is a subset of primitives bound to observation channels, plus a
reward rule combining their per-primitive scores. Every world is a tensor
program with per-lane parameters — **no Python in the rollout loop**, which is
what makes the whole population steppable in one graph (§7).

### 5.2 Goal-structure conditions

- **MVG**: goal = a subset of size 3 from a fixed basis of 6 primitives, switched
  every `g_switch` generations among a fixed set of combinations. Subgoals recur
  in different combinations — exactly Kashtan & Alon's construction.
- **RVG**: goal = 3 primitives drawn from the full library of 10, resampled
  independently at each switch. Same switching *rate* and same task difficulty
  distribution, no shared basis. **This is the control that separates "modular
  variation" from "variation".**
- **FIX**: one combination for the whole run.

Difficulty is matched across G levels by calibrating each primitive's solo
difficulty for a random policy before the run, and reporting the calibration.

### 5.3 Alien-world adaptation, operationally

The headline evolvability measurement (`src/nemo/ecology/alien.py`,
`scripts/alien_test.py`):

1. Take a fossil; **deep-copy it**.
2. Evaluate it unchanged on a world built only from held-out primitives. That
   is the pre-adaptation level.
3. Evolve the copy there for a **fixed offspring budget** — the same number of
   births, not the same wall clock, so a larger organism does not get more
   adaptation for being slower.
4. Report normalised AUC with pre-adaptation fitness subtracted, which is what
   controls regression to the mean (a population starting lower has more room
   to improve).

`adapt()` asserts its goal is a subset of `ALIEN_POOL` and refuses otherwise;
`tests/test_alien_adaptation.py` verifies the live population is byte-identical
after an adaptation run. The isolation is structural, not procedural.

### 5.4 Environment classes (strict separation, as the proposal requires)

- **A — Ancestral:** primitives and combinations used for reproductive fitness.
- **B — Recombination:** familiar primitives, unseen combinations.
- **C — Alien:** primitives (`COUNT`, `IRREV`, `DRIFT`, and two held out entirely)
  **never** used for reproductive fitness in any condition.

Alien evaluation runs on *copies* of fossilised organisms in a separate code
path that cannot write to the selection tensor. Enforced by a unit test
(`tests/test_alien_isolation.py`) that asserts the alien evaluator's output
never reaches any field read by selection.

---

## 6. Ecology, selection and reproduction

- **Population.** 256 organisms per run, 8 islands × 32. The proposal's numbers
  are kept — **but for a reason the proposal does not give.** The model shows
  organisms are a *parallel* axis that rides free inside existing dispatches;
  the binding constraint is the FLOP roof at ~1.3×10⁵ lanes, shared with
  episodes and replicates. Given that budget, spending it on **replicate runs**
  (statistical power for a rare-event rate) beats spending it on a bigger
  population. 144 runs × 256 organisms × 8 episodes = 2.9×10⁵ lanes.
- **Episodes.** 8–16 per organism. This is the arithmetic-intensity lever: at
  E=1 the workload sits 4.6× below the M3 Ultra ridge point; at E=16 it is
  comfortably compute-bound. Also reduces fitness-estimation variance, which
  matters more than population size for detecting a rare event.
- **Lifetime.** 256 steps for the pilot, 512 for the main run. **This is the one
  genuinely expensive axis** (serial), so it is chosen by measurement: the
  minimum lifetime at which the drift control and the MVG cell are
  distinguishable.
- **Selection.** Local, within-island, tournament of 4. No global top-k, no
  global champion migration — adopted from the proposal. Migration 1 organism
  per island per 20 generations.
- **Reproduction.** Asexual. 2 offspring per selected event; island carrying
  capacity fixed at 32; offspring displace the tournament loser. Population size
  is constant, so no exponential growth.
- **Fitness.** Reproductive success is a **rank within island** on the Pareto
  front of (mean episode reward, −metabolic cost), with ties broken by reward.
  Rank-based, not scalar `reward − λ·compute`: a scalar λ silently sets the
  exchange rate between performance and parsimony, and the proposal is right
  that this is the wrong knob to fix arbitrarily. With `M=off`, cost is dropped
  from the front and it reduces to reward rank.
- **Dormancy discount.** A gene whose gate does not fire pays no execution cost,
  only a small presence cost. This is what lets redundancy survive if it earns
  its keep, exactly as the proposal argues it should.

---

## 7. Execution architecture

The single most consequential engineering decision, and the justification for
several representation constraints above.

### 7.1 Population-vectorised execution

Every organism in every replicate run is a **lane** in one set of dense tensors.
Per timestep:

```
for r in range(R):                      # R=4 execution rounds, unrolled
    msgs   = gather(message_buffer, input_index[:, r])     # 1 kernel
    q,k,v  = batched_matmul(W_qkv[:, r], ...)              # fused
    ctx    = attention(q, k, v, mask[:, r])
    out    = ffn(norm(ctx))
    gate   = sigmoid(...)
    message_buffer = scatter(out * gate * alive_mask[:, r])
```

All tensors carry a leading lane dimension `(n_runs × n_organisms × n_episodes)`.
**Nothing in the rollout loop is a Python conditional over organisms.**

Modelled cost: 32 dispatches/timestep → 0.41 s/generation, versus 1.05M
dispatches/timestep → 3.7 h/generation for the naive form. **32,768×.**

### 7.2 What this constrains

- **One module shape per run.** Modules differ in *weights*, not dimensions.
  Architectural diversity comes from wiring, regulation and weights. This is a
  real restriction relative to the proposal's "modify one module's internal
  architecture" (mutation class 2) — that operator is **dropped** for phase 1,
  and the module-size sweep runs *between* runs instead of within. Justified:
  the central question is about organisation, not per-module architecture
  search, and CoDeepNEAT already covers the latter.
- **Fixed capacity with masks.** Genome length varies up to `G_max`; tensors are
  allocated at `G_max` and masked. `mx.compile` is shape-dependent, so varying
  shapes would trigger recompilation every generation.
- **Environments must be tensor programs** (§5.1). A Python environment would
  force a host sync every timestep, destroying MLX's lazy graph batching.

### 7.3 Backend

MLX is the default, on the reasoning that unified memory suits a workload where
control (genome bookkeeping) and compute interleave, and that `mx.compile` +
lazy evaluation are the right tools for many small fused ops. **But this is a
hypothesis, not a conclusion.** `scripts/bench_backend.py` measures MLX-GPU,
MLX-CPU, and a NumPy reference on the same kernels, and there is a decision rule
per measurement. The reference implementation is written against a thin backend
shim (`src/nemo/backend.py`) so the choice is reversible.

**The one number to measure first** is per-dispatch overhead. The model is
4× sensitive to it across the plausible 5–100 µs range, and it decides whether
the pilot is dispatch-bound or FLOP-bound. Everything else can be tuned later.

### 7.4 Resource ceilings

Modelled footprints: pilot **0.26 GB**; pilot + lifetime learning **1.7 GB**;
64 runs × 512 organisms × 32 genes **8.2 GB**. Recommended ceiling: **32 GB
soft** (warn, shed episode batch) / **64 GB hard** (abort). The proposal's
350 GB is ~1,000× the workload and would let a genome-bloat bug run for hours
before tripping.

---

## 8. Mutation

### 8.1 Operators (phase 1)

| # | Operator | Acts on | Notes |
|---|---|---|---|
| 1 | Weight perturbation | module tensor | Gaussian, per-gene, scale heritable |
| 2 | Regulatory mutation | `gate_b`, `gate_s`, `round` | **Changes only WHEN a module fires** |
| 3 | Wiring mutation | `inputs` | Rewire one edge |
| 4 | Gene duplication | genome | Exact copy; new `innov`, `parent` recorded |
| 5 | Gene deletion | genome | |
| 6 | Subsystem duplication | genome | Copies a connected set with internal wiring preserved |
| 7 | **Encapsulation** | genome | Freezes a connected subgraph as a reusable composite gene. **The designated hierarchy operator** — added because nothing else in the design can produce a level above "module" |
| — | Fusion / fission / recombination / HGT | — | Deferred to phases 2–3, per the proposal |

Operators 1, 2 and 3 are causally separable by construction (§4.1) — this is the
payoff of changing the genome representation.

### 8.2 Distribution

The proposal's bootstrap (70/15/5/5/3/2) is adopted **as a bootstrap only**, with
one disagreement: **5% duplication is too high given no bloat penalty.** At 5%
per birth with near-neutral duplication, the model of a neutral birth-death
walk gives a genome-length doubling time of a few hundred generations — fast
enough that bloat will dominate the measurement window. The recommendation is:

- Start at **2% duplication / 2% deletion** (deletion equal to duplication, so
  the neutral expectation on genome length is *flat*, not upward).
- Run the sensitivity sweep the proposal asks for: duplication ∈ {1, 2, 5, 10}%
  as extra lanes.
- Report the fitness-shuffled drift control's genome-length curve alongside every
  experimental curve. **If the experimental curve does not separate from drift,
  observed "complexification" is bloat.**

Meta-rates become heritable (log-normal mutation within bounds `[10⁻⁴, 0.5]`)
in the **evolvable-mutation-rate** condition, which the proposal correctly
identifies as its own experimental factor rather than a default.

---

## 9. Phylogeny and storage

- **Organism phylogeny:** parent id per birth.
- **Module phylogeny:** `innov` ids with `parent` links, giving the exact tree
  the proposal wants to inspect — C → C1/C2 → C2 diverges → C2a/C2b.
- **Storage:** SQLite for events (births, deaths, mutations, duplications,
  migrations) + Parquet for per-generation metric time series. No per-organism
  JSON files.
- **Content-addressed module tensors** (BLAKE3 of the bf16 bytes) in the
  *archive*, so 1,000 organisms sharing an ancestral module store it once. This
  is where the proposal's copy-on-write idea belongs — the storage layer, where
  it genuinely saves disk and makes the module phylogeny exact, rather than the
  execution path, where the model shows it saves nothing (1.00× at pilot module
  size, ~1.2× only above 33K params/module).
- **Retention:** full tensors for living organisms, periodic fossils (every 100
  generations), lineage MRCAs, and species representatives. Compact records
  (id, parents, genome, mutation history, fitness/environment history, resource
  use, behavioural summary) for everything else, forever. Hard disk budget with
  an enforced ceiling.

---

## 10. Phasing

| Phase | Contents | Modelled cost | Gate to proceed |
|---|---|---|---|
| **0 Benchmark** | `bench_backend.py` on the M3 Ultra; fix backend, E, lifetime | minutes | Dispatch overhead measured; vectorised path ≥1,000× naive |
| **1 Smoke** | 144 runs × 100 generations | ~1.2 min | Selection works (fitness > drift control); mutation rates sane; no NaNs; alien isolation test passes |
| **2 Pilot** | 144 runs × 1,000 generations + all controls | ~11 min | Variance estimate for power analysis; genome-length curve separates from drift, or duplication rate is retuned |
| **3 Main** | 144 runs × 5,000 generations, lifetime 512, E=16 | ~3.9 h | Pre-registered decision rule evaluated |
| **4 Learning** | + lifetime learning factor: none / reset / Lamarckian / partial | ~11.7 h | Only if phase 3 has a clear answer without it |
| **5 Sex & HGT** | + recombination, + horizontal transfer, as separate runs | ~23 h | Never mid-run — HGT is itself a modularity driver |

Total to a decisive answer on the central question: **under 5 hours of compute**,
not the multi-day figure the proposal anticipates. That difference is entirely
the vectorisation.

---

## 11. What would falsify this

- **Central hypothesis falsified** if the MVG + metabolism + duplication cell
  shows a duplication→specialisation rate statistically indistinguishable from
  the fitness-shuffled drift control at n=12, with the pre-registered effect
  size. That would mean the three established drivers do **not** transfer to
  neural-module substrates, which is a publishable negative result and the
  single most likely way this project produces a real contribution.
- **Design falsified** (rather than hypothesis) if genome length in the drift
  control tracks the experimental cells — meaning the measurement is dominated
  by bloat and the metabolism/deletion balance needs retuning before any claim.
- **Metrics falsified** if `Q_str` rises above the rewired null while `Q_cor` and
  ablation-based specialisation do not. Per Nature Comms 2024 that pattern means
  structural modularity without functional specialisation, and the structural
  number alone must not be reported as "modularity emerged."
