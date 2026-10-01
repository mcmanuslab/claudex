# Modular recursive self-improvement with tiny transformers

**Goal.** Show that (1) a population of *tiny* transformer modules can improve itself
recursively, improving both the solutions and the process that produces them;
(2) the modules can bind into **polymers / communities** that solve problems no single
module can; and (3) this scales: capability grows by recruiting and composing modules,
not by retraining one big model. The long-run motivation is to grow from ~4k-parameter
modules toward very large modular systems.

Everything runs on a 4-core CPU (≈3 hours for the full suite). About **0.55 M monomer
variants** and **1.9 M polymer variants** were generated and evaluated. Every monomer
variant's lineage (id, parent, generation, fitness, improver genes, selected?) is
logged in `results/lineage_*.npz`.

![generations](figures/generations.png)

---

## The system

| Piece | What it is |
|---|---|
| **Monomer** (`modrsi/monomer.py`) | 1-layer transformer, 2 heads, MLP, **3,872 parameters**. Reads a context of relation tables and a query symbol, and outputs a distribution over symbols. |
| **Shared interface** (`modrsi/world.py`) | A fixed "genetic code" embedding that every module reads and writes. Because input and output are the same kind of object (a symbol distribution), modules plug into each other with no glue. |
| **Skills (niches)** | `A`, `B` (look x up in table A/B), `Ainv`, `Binv` (reverse lookup), `succ` (x+1). Exp 5 adds tables C, D and `pred`, `neg`, `triple`. One layer can do about **one relational hop**, so a k-step problem is out of reach for any single monomer. |
| **Genome** | Weights + **improver genes** (mutation size σ, drift `m` and spread `a` along the lineage path, spread `b` in the elite subspace, lifetime learning rate `lr`) + an inherited **lineage path** and optimizer state. |
| **Polymer** (`modrsi/polymer.py`) | Ordered chain of ≤5 monomers; each one's output is the next one's query. |
| **Community** | Modules + a tiny evolved router that maps instruction tokens to modules, so it can recruit *n* modules for an *n*-step program. |
| **Macro** (`modrsi/hierarchy.py`) | A polymer that worked, frozen into a new reusable unit. Macros can contain macros. |

### One generation (per niche, `modrsi/evolve.py`)
1. **Vary.** Children copy a parent and mutate:
   `δ/σ = z + (m + a·g)·κ·p̂ + b·Uᵀh`. Here `p̂` is the remembered lineage path, `κ` is how
   consistent that path has been, and `U` is the top directions of recently accepted steps
   across lineages. Variants are concentrated *around the path that worked*.
2. **Develop.** 16 steps of lifetime learning (Adam). Learned weights and optimizer
   state are inherited (Lamarckian).
3. **Select.** (μ+λ) truncation on held-out fitness, plus credit earned inside working polymers.
4. **Remember.** Every variant goes into the lineage log; the paths of selected children
   are extended.

### Where the "recursive" part lives
The improver genes ride along with the weights they produced, so selection acts on
**how the system improves** as well as on what it knows. The design for this went through
three versions, and the failures are part of the result (see *What went wrong*):

* **Island niches (PBT).** Each niche is 4 islands, and each island has one improver.
  Weights are selected every generation within an island. Improvers are compared every
  5 generations, between islands; the losing island copies the winner and perturbs its
  improver.
* **Stress response.** When the progress log shows a plateau, improver genes hypermutate.
  After 15 stagnant generations below mastery, the worst island restarts its weights with
  fresh ones **but keeps the best improver** (the know-how is kept, the stuck solution is not).
* **Path credit (Exp 5).** After each new skill is mastered, the champion's ancestry is
  walked back to its founder through the lineage log. The improver genes along that
  winning branch (discounted 0.9 per step from the leaf) seed the next acquisition.
  The champion's total edit ("module X → skill Y") joins an **edit memory** that later
  mutations explore.

---

## Results

### Exp 1: mutate → polymerize → dissociate cycles (3 seeds × steered/blind, 100 generations)
`python run.py main --mode steered|blind --seed S`

| | steered (islands + path memory) | blind (fixed σ, fixed lr, isotropic) |
|---|---|---|
| Generations to master a lookup skill (9 niche-runs) | **43.0** (15–66) | 54.9 (43–75) |
| Fraction of new variants already successful (acc ≥ 0.9), mean over all generations | **0.57** | 0.47 |
| All 4 skills mastered | 3/3 seeds | 3/3 seeds |

* Steered leaves chance about 25 generations before blind and keeps its lead to 100%
  (figure above, panel b). There is **no final plateau**.
* The evolved learning rate climbs from 0.02 to about 0.06 (panel d): the improver
  learns to learn faster.
* **Polymers solve what monomers can't.** On the four hidden multi-step problems
  (`A→B`, `B→succ→Ainv`, `A→succ→A→B`, `Ainv→B→succ→A→Ainv`), evolved polymers reach
  **1.00** accuracy in every seed. The best single monomer scores **0.14**, which is
  chance (0.125). Polymers are often functionally equivalent rather than literal; for
  example `A→B` was found as `A,B,A,Ainv`, where the last two cancel.

![polymers](figures/2_polymers.png)

### Exp 2: does the improver improve? (4 seeds, 5 skills learned in sequence)
`python run.py curriculum --improver inherit|fresh|blind --seed S`

| Generations to learn A, Ainv, B, succ, Binv | total (mean of 4 seeds) |
|---|---|
| steered, default improver for each skill | **146.5** |
| steered, *inheriting* the previous skill's evolved improver | 172.5 |
| blind | 225.8 |

**Honest finding:** the steered machinery cuts learning time by about 35% compared with
blind. But simply **handing the evolved improver to the next skill did not help, and
slightly hurt.** An improver tuned to the end state of one skill is not the right starting
point for the next. This is what motivated path credit and duplication in Exp 5.

![curriculum](figures/3_recursive_improver.png)

### Exp 3: scaling by recruitment (community router)
`python run.py scale`

The router (instruction → module) is evolved on programs of length ≤3, then tested
**without any retraining** on programs of length n = 1…12:

| n | 1 | 4 | 8 | 12 |
|---|---|---|---|---|
| community, 4 skills | 1.00 | 1.00 | 1.00 | 1.00 |
| community, 5 skills (Binv added as a new species) | 1.00 | 0.99 | 0.99 | 0.99 |
| best single monomer | 0.34 | 0.13 | 0.13 | 0.15 |

At n = 12 with 5 skills there are 5¹² ≈ **244 million distinct programs**. They are solved
by 12 recruited copies drawn from 5 modules of 3,872 parameters each. Adding a new
species (`Binv`) extended the community's reach without touching the other modules.

![scaling](figures/4_scaling.png)

### Exp 4: compounding growth (the exponential)
`python run.py grow [--flat] --seed S`

Each cycle, the environment poses 4 new problems, each a concatenation of 2–3 problems
from the previous level. The system only sees (context, query) → answer examples.
Polymers that solve a problem are **frozen into macros** and become building blocks.
Every problem gets the same search budget (6,000 polymer evaluations).

| cycle | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| longest solved, hierarchical (geo-mean of 3 seeds) | 3 | 8 | 18 | 46 | 113 | 325 | 780 | 2,084 | 5,476 | **12,717** |
| longest solved, flat control (5 monomers, ≤5 per polymer) | 3 | 6 | 8 | 9 | 9 | 9 | 9 | 9 | 9 | 9 |

* **Capability grows ×2.54 per cycle** (1.35 doublings per cycle): a straight line on a
  log scale. The flat control stalls at about 9 steps.
* Search cost does **not** grow with problem length: about 1,100 evaluations per problem
  from 3-step to 20,000-step problems. 113 of 120 problems were solved.
* The macro shortcut was checked against real execution: 85 macros were re-run
  monomer by monomer, up to **1,771 transformer calls in a row**.
* **Caveat:** accuracy does erode with depth. The minimum re-run accuracy was 0.89, and
  validation accuracy at 10k+ steps is 0.95–0.98, because rare per-step errors compound.
  Error correction will matter at larger scales.

![growth](figures/6_exponential_growth.png)

### Exp 5: does it get smarter as it grows?
EXP5_PLACEHOLDER

---

## What went wrong along the way (kept on purpose)
1. **Pure mutation could not learn lookup.** Evolution strategies on 3.9k parameters sat
   at chance for 300+ generations. Attention-based lookup has a plateau: query/key
   matching and the value path are both needed before either gets any signal. This is
   the same phenomenon as induction-head formation. Gradient descent escapes it after
   about 250 steps, which is why monomers now develop with lifetime learning (Lamarckian
   evolution).
2. **Per-generation self-adaptation was myopic.** On a plateau, a child that barely
   learns always looks slightly better than one that learns. Selection drove the
   learning rate toward zero and lineages stalled.
3. **Greedy improvers fell into partial-solution basins.** A high learning rate escaped
   chance quickly, then dragged whole populations into 60–80% solutions with collapsed
   diversity. Islands with improvers compared every 5 generations, plus improver-preserving
   restarts, fixed both 2 and 3.
4. **Inheriting the improver alone did not speed up new skills** (Exp 2).

## What this does and does not show
* **Shows:** tiny modules that evolve, that improve their own improvement process
  (learning-rate genes, mutation steering, island selection), and that compose into
  polymers and communities solving problems far beyond any single module. Through
  hierarchical reuse, problem size grows **exponentially at roughly constant cost per
  cycle**.
* **Does not show:** exponential growth of general intelligence. The exponential in
  Exp 4 is in *problem size reachable*, and the environment poses geometrically growing,
  compositional problems. What is tested is whether a modular system keeps up at
  constant cost by reusing what it built. A monolithic search does not keep up (flat
  control). The domain is small and synthetic (8 symbols, permutation tables).

## Next steps toward large n
* Learned error correction / redundancy between modules (majority-vote polymers) to stop
  error compounding at 10⁴+ steps.
* A learned polymerization policy (a small transformer that proposes assemblies), so
  search improves with experience.
* Replace the hand-built curriculum with an open-ended environment where the system picks
  its own next problems.
* Richer modules (2–4 layers, larger vocabularies) and more varied skills, to test
  whether duplication + edit memory keep reducing the cost of new skills.

## Reproduce
```
pip install -r requirements.txt
sh -c 'cat jobs.txt | xargs -P 4 -I{} sh -c "python3 run.py {}"'   # Exp 1 + 2
python run.py scale                                                # Exp 3
python run.py grow --seed 0 ; python run.py grow --flat --seed 0   # Exp 4
python run.py acquire --mode smart|dup|scratch --seed 0            # Exp 5
python run.py report ; python plot_generations.py                  # figures/, results/summary.json
```
