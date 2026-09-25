# RESEARCH.md — Literature and Prior-Implementation Audit

**Scope.** Prior work bearing on the proposed experiment: Darwinian evolution of
variable-length genomes of tiny cooperating neural modules, asking whether
redundancy, duplication, divergence, specialisation, regulation, modularity and
evolvability appear without being rewarded.

**Bottom line, stated up front.**

1. **The exact system proposed does not exist in the literature.** No prior work
   combines tiny attention-based modules + variable-length assembly genome +
   module duplication/deletion + regulatory mutation + whole-organism ecological
   fitness + module-level phylogeny + held-out "alien" adaptation tests. The
   novelty claim is supportable, but it is a novelty of *combination*, not of
   any single mechanism — every individual mechanism has prior art (§2).
2. **The strong form of the central hypothesis is already known to be false**,
   and this is the single most important finding of this audit. Under a *fixed*
   fitness function, evolutionary algorithms reliably produce **non-modular,
   entangled** networks. Modularity is not a free lunch that falls out of
   selection-plus-variation. Three drivers are independently established as
   sufficient (§3). The proposal's design contains two of them by accident and
   does not control any of them. Run as written, the most likely outcome is a
   null result that reproduces Kashtan & Alon (2005) without citing it.
3. **The right experiment is therefore not "does modularity emerge?" but "which
   ecological conditions are sufficient, and does the resulting organisation buy
   evolvability?"** That version is falsifiable, decisive, cheaper, and novel.
   See DESIGN.md.
4. **Structural modularity does not imply functional specialisation** (§6).
   Any claim here needs both metric families plus null models, or it is not a
   claim.

---

## 1. What this audit could and could not access

Searches were run in September 2026 across the evolutionary-computation,
artificial-life, and machine-learning literatures. `arxiv.org` is blocked by
this container's network egress proxy, so arXiv preprints were read through
search snippets, mirrors (Semantic Scholar, PMC, MIT Press, PLOS, PNAS, ACM,
Springer) and publisher pages rather than fetched directly. Where a claim rests
only on a search snippet it is marked *(snippet)*. Nothing below is cited from
memory alone without a retrievable link.

No prior neuroevolution codebase exists in this account (checked: 30
repositories under `mcmanuslab`). **The "procedural ecological environment
system from the previous experiment" referenced in the proposal is not
available and is therefore built from scratch here.**

---

## 2. The closest prior work, and exactly how this differs

### 2.1 Cooperative coevolution of neural components

| System | What it does | What it lacks relative to this proposal |
|---|---|---|
| **SANE** (Moriarty & Miikkulainen 1996) | Coevolves a population of *individual neurons* that are assembled at random into networks; network fitness is credited back to participating neurons. Explicitly reports that symbiotic evolution "promotes both cooperation and specialization." | Neurons, not modules; fixed-size assemblies; no genome that persists across generations; no duplication operator; no regulation; no lifetime learning. |
| **ESP / Enforced SubPopulations** (Gomez & Miikkulainen 1997, 1999) | Each hidden neuron evolves in its *own* subpopulation; recombination confined within subpopulation. | Specialisation is **imposed by construction** (segregated subpopulations), which is precisely what this experiment wants to *measure*, not assume. Fixed topology. |
| **COVNET** (García-Pedrajas et al. 2003) | Coevolves subnetworks ("nodules") and combinations of them. | Fixed nodule count; no variable-length genome; no duplication/divergence tracking. |
| **Modular NEAT** (Reisinger, Stanley & Miikkulainen 2004) | Coevolves *modules* and *blueprints* that specify which module slots into which position; the same module may repeat in a blueprint. | This is the closest classical ancestor of the proposed representation. But: modules and blueprints live in separate populations (so there is no single organism genome), reuse is by reference only, and there is no duplication→divergence event to track, no regulation, no metabolism, no phylogeny of modules. |
| **CoDeepNEAT** (Miikkulainen et al. 2017) | Generalises Modular NEAT to deep networks; blueprints + module populations; discovers repeated motifs like those in GoogLeNet/ResNet. | Supervised NAS with gradient training; fitness is validation accuracy; no ecology, no metabolism, no organism-level lifecycle, no open-endedness. |

**Difference:** in all of the above, the *module population is a separate
entity* from the network. In this proposal the modules are **inside one
organism's heritable genome**, so duplication, deletion and divergence are
events in a *lineage*, which is what makes duplication→divergence→specialisation
observable at all.

- Moriarty & Miikkulainen, *Efficient Reinforcement Learning through Symbiotic Evolution*, Machine Learning 22 (1996). <https://link.springer.com/article/10.1023/A:1018004120707>
- Gomez & Miikkulainen, *Solving Non-Markovian Control Tasks with Neuro-Evolution*, IJCAI 1999. <https://mlanthology.org/ijcai/1999/gomez1999ijcai-solving/>
- Reisinger, Stanley & Miikkulainen, *Evolving Reusable Neural Modules*, GECCO 2004. <https://link.springer.com/chapter/10.1007/978-3-540-24855-2_7>
- Miikkulainen et al., *Evolving Deep Neural Networks* (CoDeepNEAT), 2017. <https://arxiv.org/pdf/1703.00548>
- Stanley & Miikkulainen, *Evolving Neural Networks through Augmenting Topologies* (NEAT), Evolutionary Computation 10(2), 2002. <https://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf>

### 2.2 Emergent modularity in genetic programming

- **Tangled Program Graphs** (Kelly & Heywood, IJCAI 2018; TELO 2021) — the
  strongest existing example of *emergent* modularity in evolutionary
  computation: programs self-organise into teams, teams into graphs of teams,
  with no explicit reward for doing so. Matches DQN on Atari at orders of
  magnitude lower complexity.
  <https://www.ijcai.org/proceedings/2018/740> · <http://www.cs.mun.ca/~banzhaf/papers/telo2021.pdf>
  **Difference:** programs, not differentiable modules; no lifetime learning; no
  metabolism; hierarchy arises from a specific team-referencing operator rather
  than from a genome with duplication.
- **Modular CGP** (Walker & Miller, IEEE TEC 2008) — module acquisition,
  evolution and reuse in Cartesian GP; modules are *captured* and *destroyed* by
  dedicated operators. Establishes that explicit encapsulation operators speed
  evolution on larger problems. <http://gpbib.cs.ucl.ac.uk/gp-html/Walker_2008_TEC.html>
  **Relevance:** this is the prior art for the *encapsulation* operator that
  DESIGN.md adds — the proposal as written has no operator capable of producing
  a hierarchy level above "module" (see §7, confounder C9).
- **Automatically Defined Functions** (Koza) — the original reuse abstraction in GP.
- Survey: *A survey of modularity in genetic programming*, IEEE 2016. <https://ieeexplore.ieee.org/document/7748328/>

### 2.3 Digital evolution / artificial life

- **Tierra** (Ray, 1991) — self-replicating assembly programs; produced
  parasites, hyper-parasites, immunity, sociality and cheaters *spontaneously*
  from one ancestor, and compressed an 80-instruction ancestor to 45
  instructions within hours. The canonical demonstration that open-ended
  ecological dynamics arise without being specified.
  <https://csmgeo.csm.jmu.edu/geollab/complexevolutionarysystems/Documents/Tierra2010.pdf>
  Caveat literature: *Tierra's missing neutrality: case solved* <https://arxiv.org/pdf/nlin/0404012>
  and *The influence of parsimony and randomness on complexity growth in Tierra*
  <https://arxiv.org/pdf/nlin/0604026> — both show Tierra's complexity dynamics
  are highly sensitive to the parsimony/cost parameterisation. **Directly
  relevant warning for the metabolic-cost design decision here.**
- **Avida** (Ofria, Lenski, Adami) — the reference platform. Key results:
  - *The evolutionary origin of complex features*, Nature 423 (2003): complex
    logic functions evolve by building on simpler ones that were themselves
    selectively favoured. <https://www.nature.com/articles/nature01568>
  - *Evolution of new tissues through gene duplication and divergence in
    multithreaded digital organisms*, 2003 — **the single closest prior result
    to this proposal's central hypothesis**, in a non-neural substrate.
    <https://ieeexplore.ieee.org/document/1237676>
  - *Using digital organisms to study the evolutionary consequences of whole
    genome duplication and polyploidy*, PLOS ONE 2019. <https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0220257>
  - *Different evolutionary paths to complexity for small and large populations*,
    PLOS Comp Biol 2016 — population size changes *which* path to complexity is
    taken. **Relevant to the population-size decision.** <https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1005066>
  - *Horizontal Gene Transfer Leads to Increased Task Acquisition and Genomic
    Modularity in Digital Organisms*, ALIFE 2019 — HGT is an established
    *driver* of modularity, not a neutral add-on. The proposal's plan to add HGT
    "later" would confound the modularity measurement if added mid-run.
    <https://direct.mit.edu/isal/proceedings/isal2019/31/243/99227>
- **Transitions in individuality**: Goldsby et al. on division of labour and the
  origin of soma in digital multicellularity; Moreno & Ofria, *Toward Open-Ended
  Fraternal Transitions in Individuality*, Artificial Life 25(2), 2019.
  <https://direct.mit.edu/artl/article/25/2/117/2929/> ·
  <https://arxiv.org/pdf/2104.10081>
- *The Surprising Creativity of Digital Evolution* (Lehman, Clune, Misevic et al.)
  — catalogue of ways digital evolution satisfies the letter of a fitness
  function while violating its intent. **Required reading before interpreting
  any positive result here.** <https://arxiv.org/pdf/1803.03453>

**Difference:** Avida/Tierra organisms are assembly programs; their "genes" are
instruction sequences, not parameterised differentiable modules, and there is no
lifetime learning. This proposal is, in effect, *Avida with neural modules as
the unit of duplication* — which is a genuine gap, and the most defensible
framing of its novelty.

### 2.4 Modern neuroevolution and modular deep learning

- **HyperNEAT** (Stanley, D'Ambrosio, Gauci 2009) — indirect/generative encoding.
- **Weight Agnostic Neural Networks** (Gaier & Ha, NeurIPS 2019) — architecture
  alone can encode behaviour; relevant to the no-lifetime-learning control.
- **PathNet** (Fernando et al. 2017) — evolves *pathways* through a fixed matrix
  of modules; modules are frozen after a task is learned. Closest prior art for
  "evolve routing over modules." **Difference:** fixed module inventory, no
  duplication, no variable-length genome, no organism lifecycle.
- **AutoML-Zero** (Real, Liang, So & Le, ICML 2020) — evolves whole ML algorithms
  from primitive ops; the reference point for "start minimal, let evolution
  build." <https://arxiv.org/abs/2003.03384> · code:
  <https://github.com/google-research/google-research/blob/master/automl_zero/README.md>
- **ModuleFormer**, **EMO: Pretraining MoE for Emergent Modularity**, *Unlocking
  Emergent Modularity in Large Language Models* — MoE-family work showing
  modular structure emerging under gradient training. <https://arxiv.org/html/2310.10908v2>
  **Difference:** gradient descent, fixed expert count, no evolution, no
  duplication events, no lineage.
- **Neural Developmental Programs** (Najarro, Sudhakaran & Risi, ALIFE 2023) and
  *Evolving Self-Assembling Neural Networks* (Plantec et al., ALIFE 2024) —
  developmental encodings that *grow* networks by local rules.
  <https://arxiv.org/abs/2307.08197> · <https://arxiv.org/html/2406.09787>
  **Difference:** development from a single rule-set, not a variable-length
  gene list; growth is not a heritable duplication event.
- *The Generalist Brain Module: Module Repetition in Neural Networks in Light of
  the Minicolumn Hypothesis* (2025) <https://arxiv.org/pdf/2507.12473> — direct
  support for the "many copies of one small module" architecture chosen here.
- Survey: *A Review of Modularization Techniques in Artificial Neural Networks*
  <https://arxiv.org/pdf/1904.12770>; *A Systematic Literature Review of the
  Successors of NeuroEvolution of Augmenting Topologies*, Evolutionary
  Computation 29(1) <https://direct.mit.edu/evco/article-pdf/29/1/1/1888486/evco_a_00282.pdf>

### 2.5 Open-endedness and quality-diversity

- **Novelty search** (Lehman & Stanley) and **MAP-Elites** (Mouret & Clune 2015)
  <https://members.loria.fr/jbmouret/qd.html>
- **POET / Enhanced POET** (Wang, Lehman, Clune, Stanley 2019, 2020) —
  coevolution of environments and agents; combines novelty search, MAP-Elites and
  minimal criterion coevolution. <https://arxiv.org/pdf/1901.01753> ·
  <https://proceedings.mlr.press/v119/wang20l/wang20l.pdf>
  **Relevance:** the proposal defers environment coevolution to "later." Good
  call — POET showed environment coevolution *dominates* the dynamics, which
  would swamp the module-level signal.
- **ASAL** (Kumar, Lu, Clune, Cully, Ha et al., *Artificial Life* 31(3), 2025) —
  foundation models as the search signal for ALife substrates.
  <https://sakana.ai/asal/> · <https://github.com/SakanaAI/asal>
  **Difference:** searches over *simulation rules*, not over evolving organisms
  with genomes.

### 2.6 Hardware-accelerated evolution (the engineering precedent)

- **evosax** (Lange 2022) <https://github.com/RobertTLange/evosax>,
  **EvoJAX** (Tang, Tian & Ha 2022) <https://github.com/google/evojax>,
  **TensorNEAT** (2025) <https://arxiv.org/pdf/2504.08339>
  — all establish the same lesson: **the entire population must be one
  vectorised tensor program.** TensorNEAT specifically solves the hard case,
  *heterogeneous topologies under `vmap`*, which is exactly this experiment's
  execution problem. Its approach (pad to fixed capacity, mask, gather/scatter)
  is adopted here.
- MLX: <https://ml-explore.github.io/mlx/> ; *Writing Fast MLX* (Awni Hannun)
  <https://gist.github.com/awni/4beb1f7dfefc6f9426f3a7deee74af50> ; lazy
  evaluation and `mx.compile` semantics
  <https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html>.
  Published MLX guidance: graphs of "a few tens to many thousands of operations
  per evaluation" are fine, `mx.compile` is **shape-dependent** (recompiles when
  input shapes change). *(snippet)* — this directly forces the fixed-capacity
  padded-tensor design in DESIGN.md.
- Comparative Apple-Silicon measurements: *Production-Grade Local LLM Inference
  on Apple Silicon* <https://arxiv.org/pdf/2511.05502>; *Profiling Apple Silicon
  Performance for ML Training* <https://arxiv.org/pdf/2501.14925>;
  mlx-benchmark <https://github.com/TristanBilot/mlx-benchmark>.
  **Caveat: every published MLX benchmark found is LLM inference.** None
  measures the regime that matters here (tens of thousands of 16×16 GEMMs with
  gather/scatter). No published number can be reused; it must be measured
  (`scripts/bench_backend.py`).

---

## 3. The decisive finding: modularity does **not** evolve for free

This is the part of the audit that should change the experiment.

**Three mechanisms are independently established as sufficient to evolve
modularity, and the absence of all three is established as sufficient to
*prevent* it:**

1. **Modularly Varying Goals (MVG).** Kashtan & Alon, *Spontaneous evolution of
   modularity and network motifs*, PNAS 102(39), 2005. Evolution under a goal
   that **switches between several goals, each a different combination of the
   same subgoals**, spontaneously produces modular structure and recurring
   motifs. Evolution under a *fixed* goal produces highly non-modular networks
   that "lack understandable motifs."
   <https://www.pnas.org/doi/10.1073/pnas.0503610102>
   **Critically: randomly varying goals are not MVG and do not work.** The
   subgoal basis must be shared and recombined. The proposal's "overlapping but
   nonidentical world samples" is randomly varying goals, not MVG.
2. **Connection cost.** Clune, Mouret & Lipson, *The evolutionary origins of
   modularity*, Proc. R. Soc. B 280, 2013: adding a cost for connections makes
   modules "immediately appear"; without a cost "modules never form."
   <https://royalsocietypublishing.org/rspb/article/280/1755/20122863/74559/> ·
   <https://arxiv.org/pdf/1207.2743v1>
   Extended to HyperNEAT: Huizinga, Clune & Mouret, GECCO 2014.
   <http://www.cmap.polytechnique.fr/~nikolaus.hansen/proceedings/2014/GECCO/proceedings/p697.pdf>
   Recent replication in free-form MLPs: <https://link.springer.com/article/10.1007/s00521-023-09117-4>
3. **Selection for specialisation.** Espinosa-Soto & Wagner, *Specialization Can
   Drive the Evolution of Modularity*, PLOS Comp Biol 6(3), 2010. Modularity
   increases when selection favours *specialised activity patterns*.
   <https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1000719>

Supporting/qualifying results: sparseness as a partial driver
<https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1006172>;
mutation-rule effects on modularity <https://arxiv.org/pdf/1302.4267>;
noise-motivated regularisation producing modular connectivity
<https://arxiv.org/pdf/2512.13707>; HGT as a driver (Avida, §2.3).

### Consequence for this proposal

The proposal states: *"without explicitly rewarding any of those properties."*
Taken literally — fixed goal, no metabolic cost, no specialisation pressure —
**the expected result is no modularity**, and the experiment would spend weeks
re-deriving a 2005 PNAS result.

But the proposal *does* contain two of the three drivers, unlabelled:

- **"Computational metabolism"** (fitness discounted by executed module FLOPs)
  is a connection-cost analogue. It is driver #2, smuggled in.
- **"Procedural ecological diversity"** across islands is goal variation — but
  *randomly* varying, which Kashtan & Alon showed is **not** sufficient.

So the proposal is in the worst position: it has one driver present but
uncontrolled (so a positive result is uninterpretable — "modularity emerged"
would just mean "we paid for parsimony"), and the strongest driver present in a
form known not to work.

**The fix is small and makes the experiment better on every axis:** promote goal
structure and metabolic cost from implementation details to *experimental
factors*, and give the environment an explicit shared subgoal basis so MVG is
constructible. This converts an experiment likely to produce an uninterpretable
result into a 3×2×2 factorial with a pre-registered, known-sign prediction.
See DESIGN.md §3.

---

## 4. Lifetime learning, Lamarck, and the Baldwin effect

- Baldwin: learning reshapes the fitness landscape without genetic transmission
  of what was learned; strictly Darwinian.
- Lamarckian neuroevolution (final learned weights inherited) typically converges
  in fewer generations but is prone to premature convergence; Baldwinian can
  reach better optima but needs more generations. *(snippet; see
  <https://www.researchgate.net/publication/220489390_Lamarckian_Evolution_and_the_Baldwin_Effect_in_Evolutionary_Neural_Networks>)*
- *Meta-Learning by the Baldwin Effect* (Fernando et al. 2018) <https://arxiv.org/pdf/1806.07917>
- *Embodied Intelligence via Learning and Evolution* (Gupta et al., Nature Comms
  2021) — morphology/control co-evolution with lifetime learning; establishes
  that learning **speeds** morphological evolution (the "Baldwin effect in
  morphology"). <https://arxiv.org/pdf/2102.02202>

**Consequence for this proposal.** Lifetime learning is a *confounder for the
central measurement*, not merely an extra condition. If two duplicate modules
diverge, gradient descent on independent data alone will make them differ —
that is not evolutionary divergence, and no amount of post-hoc analysis can
separate the two if learning is on from generation 0. **Phase 1 must run without
lifetime learning.** (The compute argument for this is weak — see §5 — the
argument is causal identifiability.)

---

## 5. Hardware arithmetic: what the numbers actually say

Full model and all assumptions: `scripts/compute_budget.py` (runnable;
every hardware constant is named and overridable). Exact parameter counts:
`src/nemo/modules/spec.py`, unit-tested against a materialised reference.

Headline results for the recommended module (d_model 16, d_ff 32, K=4):

| Quantity | Value |
|---|---|
| Parameters / module | **2,145** (attn 1,024 · ffn 1,072 · norm 32 · gate 17) |
| Ancestral organism, 4 genes | **8,972** neural params + 44 regulatory scalars |
| Forward FLOPs / module-step | 7,540 |
| Arithmetic intensity, 1 episode | **1.76 FLOP/byte** |
| M3 Ultra ridge point (at 25% small-GEMM efficiency) | **8.1 FLOP/byte** |

Three findings that change the design:

1. **Naive per-organism execution is not merely slow, it is impossible.**
   A Python loop over organisms × modules costs ~1.05M kernel dispatches per
   timestep → **3.7 hours per generation**. The population-vectorised form costs
   32 dispatches per timestep → **0.41 s per generation**. Ratio **32,768×**.
   This is not an optimisation; it decides whether the experiment exists.
2. **Replicates and factorial cells are nearly free; lifetime length is not.**
   Parallel axes (organisms, episodes, replicate runs, conditions) ride inside
   the same dispatches until the FLOP roof at ~1.3×10⁵ lanes. Serial axes
   (lifetime steps × generations) set the wall clock. The whole 144-run
   factorial for 1,000 generations costs **~11 minutes**; the 5,000-generation
   main experiment **~3.9 h**; phase 2 with lifetime learning **~11.7 h**.
   *The proposal's "multi-day evolution" framing is an artefact of assuming
   per-organism execution.*
3. **At 1 episode/organism the workload is 4.6× below the ridge point** — pure
   memory-bandwidth waste. Running E parallel episodes per organism multiplies
   arithmetic intensity by E. E=16 puts it at 28 FLOP/byte, comfortably compute
   bound. **Episode batching, not population size, is the efficiency lever.**

And two findings that contradict parts of the proposal:

4. **Copy-on-write buys nothing in the live execution path at this scale.** The
   model shows deduplication gives 1.00× speedup for modules ≤8K params (the
   regime is dispatch-bound, not bandwidth-bound) and only ~1.2× at 33K+ params.
   Meanwhile COW indirection *breaks* the dense-tensor layout that delivers the
   32,768×. **Keep content-addressing in the storage/archive layer (where it is
   genuinely valuable for genealogy and disk), not in the execution path.**
   Duplication semantics — both copies start identical — cost 4.3 KB to
   implement by plain copy.
5. **The ~350 GB working-memory budget is ~1,000× too large.** The pilot's full
   resident footprint (weights + activations, 144 runs × 256 organisms) is
   **0.26 GB**; with lifetime learning **1.7 GB**; a 64-run × 512-organism ×
   32-gene configuration is **8.2 GB**. Recommendation: 32 GB soft / 64 GB hard
   ceiling. A budget of 350 GB does not protect the machine, it hides bugs —
   a runaway genome-length bloat would consume 300 GB before tripping a limit
   that should have fired at 32.

**No MLX measurement was possible in this container** (Linux, 4 cores, no Apple
Silicon). `scripts/bench_backend.py` is written to run on the M3 Ultra and
measures the five constants the model depends on, with an explicit decision rule
attached to each. The model's most sensitive assumption is per-dispatch overhead
(25 µs central): at 5 µs the pilot is FLOP-bound and 25% faster; at 100 µs it is
4× slower. **That one number should be measured before anything else is tuned.**

---

## 6. Measuring modularity, specialisation and evolvability

**Structural modularity is not sufficient evidence.** *Dynamics of specialization
in neural modules under resource constraints*, Nature Communications 15 (2024)
— the most directly relevant recent methods paper — finds explicitly that
**"structural modularity does not in general guarantee functional specialization
across multiple measures."**
<https://www.nature.com/articles/s41467-024-55188-9> · <https://arxiv.org/html/2106.02626>
It also supplies the three-metric protocol adopted here: **module probing**
(can a module's state be decoded for a given function?), **module ablation**
(performance loss under masking), and **hidden-state correlation**. It further
establishes that *resource constraints* modulate specialisation — another
independent confirmation of §3 driver #2.

Additional metric sources:
- Structural Q vs functional Q_cor (weight-matrix vs activity-correlation
  modularity), same paper.
- *Are Neural Nets Modular? Inspecting Functional Modularity Through
  Differentiable Weight Masks* <https://arxiv.org/pdf/2010.02066> — cautionary:
  naive modularity probes find "modules" in networks that have none.
- *Emergence of functionally differentiated structures via mutual information
  minimization in RNNs* (2025) <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12618794/>
- *Towards Understanding the Link Between Modularity and Performance in Neural
  Networks for RL* <https://arxiv.org/pdf/2205.06451> — modularity and
  performance are **not** monotonically related; do not assume a modular winner.
- Evolvability: Wagner & Altenberg (1996); Altenberg, *Complex Adaptations and
  the Evolution of Evolvability* <https://dynamics.org/~altenber/PAPERS/CAEE/>;
  *Evolvability Is Inevitable: Increasing Evolvability Without the Pressure to
  Adapt* (Lehman & Stanley 2013) <https://arxiv.org/pdf/1302.1143>;
  *Evolvability signatures of generative encodings* (Tarapore & Mouret 2014)
  <https://arxiv.org/pdf/1410.4985>.
  **Note Lehman & Stanley's result carefully: evolvability can increase for
  reasons having nothing to do with selection for it.** A rising-evolvability
  curve is therefore *not* on its own evidence that modular organisation caused
  it — the drift control in DESIGN.md §6 exists for this reason.
- Evolution of mutation rates: <https://www.sciencedirect.com/science/article/abs/pii/S0303264702001375>

---

## 7. Confounders this audit identifies

| # | Confounder | Why it matters | Mitigation (DESIGN.md) |
|---|---|---|---|
| C1 | Fixed goals cannot produce modularity | Known null result (Kashtan & Alon) | Goal structure is factor **G** ∈ {MVG, RVG, FIX} |
| C2 | Metabolic cost *is* a modularity driver | A positive result would be uninterpretable | Metabolism is factor **M** ∈ {on, off} |
| C3 | Sparsity inflates Q | Fewer active modules mechanically raises modularity scores | Degree-preserving rewired null + active-module-matched controls |
| C4 | Lifetime learning manufactures divergence | Gradient noise ≠ evolutionary divergence | Phase 1 has no lifetime learning; it enters as factor **L** in phase 2 |
| C5 | **Genome bloat masquerading as complexity** | Near-neutral duplication random-walks genome length upward (GP bloat; constructive neutral evolution). Tierra's complexity growth is known to be parsimony-parameter-sensitive | **Fitness-shuffled drift control** (absent from the proposal) establishes the null growth curve; report active vs inactive module counts separately |
| C6 | Alien-adaptation speed confounded by starting fitness | Regression to the mean | Pre-adaptation fitness as covariate; matched compute budget; normalised AUC |
| C7 | Modular vs monolithic confounded by scale | "More parameters won" is not a finding | Parameter-, compute- and interaction-matched monolithic control |
| C8 | Duplicate-pair "specialisation" by chance | Two random modules differ | Permutation null over module identity; pre-registered effect size |
| C9 | **No operator can produce hierarchy** | Duplication + rewiring yields a flat graph; "hierarchical complexity" would return a structural null, not a biological one | Add an **encapsulation** operator (prior art: modular CGP, Modular NEAT, TPG) |
| C10 | HGT/sex added mid-run | HGT is itself a modularity driver (Avida 2019) | Separate phase, separate runs, never mid-run |

---

## 8. Novelty statement (defensible form)

> Prior work has evolved cooperating neural components (SANE, ESP, Modular NEAT,
> CoDeepNEAT), evolved emergent modular program graphs (TPG, modular CGP),
> demonstrated gene duplication and divergence in digital organisms (Avida), and
> established three distinct drivers of the evolution of modularity (Kashtan &
> Alon 2005; Espinosa-Soto & Wagner 2010; Clune, Mouret & Lipson 2013). We are
> not aware of prior work in which **the unit of heritable duplication is a
> parameterised neural module inside a single organism's variable-length genome**,
> with regulatory mutations separable from wiring and weight mutations, under an
> explicit compute metabolism, with module-level phylogeny tracked across
> lineages, and with the *sufficiency of each proposed driver* tested factorially
> against a fitness-shuffled drift control.

What is **not** novel and must not be claimed as such: that modularity can
evolve; that duplication and divergence occur in digital evolution; that
modularity aids evolvability; that cooperating neural modules can be evolved.
All four are established.

---

## 9. Sources

All links appear inline above. Principal sources:

- [Kashtan & Alon, *Spontaneous evolution of modularity and network motifs*, PNAS 2005](https://www.pnas.org/doi/10.1073/pnas.0503610102)
- [Clune, Mouret & Lipson, *The evolutionary origins of modularity*, Proc. R. Soc. B 2013](https://royalsocietypublishing.org/rspb/article/280/1755/20122863/74559/)
- [Espinosa-Soto & Wagner, *Specialization Can Drive the Evolution of Modularity*, PLOS CB 2010](https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1000719)
- [*Dynamics of specialization in neural modules under resource constraints*, Nature Communications 2024](https://www.nature.com/articles/s41467-024-55188-9)
- [Moriarty & Miikkulainen, *Efficient RL through Symbiotic Evolution*, ML 1996](https://link.springer.com/article/10.1023/A:1018004120707)
- [Reisinger, Stanley & Miikkulainen, *Evolving Reusable Neural Modules*, GECCO 2004](https://link.springer.com/chapter/10.1007/978-3-540-24855-2_7)
- [Miikkulainen et al., *Evolving Deep Neural Networks* (CoDeepNEAT)](https://arxiv.org/pdf/1703.00548)
- [Stanley & Miikkulainen, *NEAT*, Evol. Comp. 2002](https://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf)
- [Kelly & Heywood, *Emergent Tangled Program Graphs in Multi-Task Learning*, IJCAI 2018](https://www.ijcai.org/proceedings/2018/740)
- [Walker & Miller, *Automatic Acquisition, Evolution and Reuse of Modules in CGP*, IEEE TEC 2008](http://gpbib.cs.ucl.ac.uk/gp-html/Walker_2008_TEC.html)
- [Lenski, Ofria, Pennock & Adami, *The evolutionary origin of complex features*, Nature 2003](https://www.nature.com/articles/nature01568)
- [*Evolution of new tissues through gene duplication and divergence in multithreaded digital organisms*, 2003](https://ieeexplore.ieee.org/document/1237676)
- [*HGT Leads to Increased Task Acquisition and Genomic Modularity in Digital Organisms*, ALIFE 2019](https://direct.mit.edu/isal/proceedings/isal2019/31/243/99227)
- [Moreno & Ofria, *Toward Open-Ended Fraternal Transitions in Individuality*, Artificial Life 2019](https://direct.mit.edu/artl/article/25/2/117/2929/)
- [Wang, Lehman, Clune & Stanley, *POET*, 2019](https://arxiv.org/pdf/1901.01753)
- [Wang et al., *Enhanced POET*, ICML 2020](https://proceedings.mlr.press/v119/wang20l/wang20l.pdf)
- [Real, Liang, So & Le, *AutoML-Zero*, ICML 2020](https://arxiv.org/abs/2003.03384)
- [Najarro, Sudhakaran & Risi, *Neural Developmental Programs*, ALIFE 2023](https://arxiv.org/abs/2307.08197)
- [Kumar et al., *Automating the Search for Artificial Life with Foundation Models*, Artificial Life 2025](https://sakana.ai/asal/)
- [Lehman, Clune, Misevic et al., *The Surprising Creativity of Digital Evolution*](https://arxiv.org/pdf/1803.03453)
- [Lehman & Stanley, *Evolvability Is Inevitable*, 2013](https://arxiv.org/pdf/1302.1143)
- [Lange, *evosax: JAX-Based Evolution Strategies*](https://github.com/RobertTLange/evosax)
- [Tang, Tian & Ha, *EvoJAX*](https://github.com/google/evojax)
- [*TensorNEAT: A GPU-accelerated Library for NeuroEvolution*, 2025](https://arxiv.org/pdf/2504.08339)
- [Ray, *Tierra*](https://csmgeo.csm.jmu.edu/geollab/complexevolutionarysystems/Documents/Tierra2010.pdf)
- [Hannun, *Writing Fast MLX*](https://gist.github.com/awni/4beb1f7dfefc6f9426f3a7deee74af50)
- [MLX lazy evaluation docs](https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html)
