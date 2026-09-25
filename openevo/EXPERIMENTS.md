# EXPERIMENTS.md — protocol, conditions, metrics

Staged so that each stage produces a decision, and no stage begins before the previous one
has answered its question. Stages 0–4 and 9 are implemented and have been run; the rest are
scaffolded with the interfaces they need.

---

## Stage status

| Stage | Question | Status |
|---|---|---|
| 0 | Literature audit; has this been done? | done — `RESEARCH.md` |
| 1 | Which backend on the target machine? | harness written (`scripts/bench_backend.py`); **must be run on Apple Silicon** |
| 2 | Are the worlds learnable and discriminative? | done — reference filter + `tests/test_environments.py` |
| 3 | What is the minimum viable transformer? | done — **~1K parameters**, see below |
| 4 | Does fixed-architecture asexual evolution work? | done — `configs/pilot.json` |
| 5 | Are architecture mutations valid and function-preserving? | done — `tests/test_models.py`, all five growth ops exact |
| 6 | Do islands, migration and the archive behave? | done — `population.py` |
| 7 | Darwinian vs Lamarckian vs partial inheritance | implemented as `phase.weight_inheritance`; ablation not yet run at power |
| 8 | Does heritable mutation strategy change outcomes? | implemented as `phase.heritable_mutation` |
| 9 | Does adaptation speed on alien worlds improve? | probe implemented and running; needs the full seed budget |
| 10 | Recombination | implemented (`organism.recombine`), module-level, signature-gated; not evaluated |
| 11 | Environment co-evolution | not implemented; see `DESIGN.md §5` for why it is not a prerequisite |

---

## Stage 3 result: minimum viable size

`python scripts/min_viable_size.py --steps 250 --worlds 6 --targets 900 2000 5000 20000 --variants 1 0`

Behaviour cloning against the reference policy, then scored by *acting*.

**Stable worlds (`n_variants=1`), scored on training worlds — representational capacity:**

| params | CE loss | acted score (random = 0, reference = 1) |
|---|---|---|
| 983 | 0.209 | 0.919 |
| 2238 | 0.110 | 0.972 |
| 5360 | 0.042 | 0.978 |
| 20550 | 0.031 | 0.976 |

Re-measured after per-block normalisation was introduced (`results/min_viable_size_recheck.json`):
983 → 0.868, 5360 → 0.967. Slightly lower, same conclusion.

**Variable worlds (`n_variants=0`), held-out worlds — in-context inference:**

| params | CE loss | acted score |
|---|---|---|
| 909 | 1.069 | 0.145 |
| 2150 | 0.793 | 0.216 |

**Decision.** ~1K parameters already expresses a near-reference policy, so the ancestral
ladder starts at **1K / 5K / 20K** rather than 5K / 20K / 80K. The binding constraint is
in-context inference, not capacity — which sharpens the hypothesis: if architectures grow,
it cannot be in order to represent a better policy.

---

## The neutral arm is not optional

Every complexity claim is reported as **selected minus neutral** at matched generation
count. The neutral arm (`configs/neutral.json`, `phase.neutral = true`) uses identical
operators, demography, world sampling and compute budget, with fitness replaced by a
uniform random draw.

Measured null, 60 generations × 80 lineages, no selection (`tests/test_neutrality.py`):

| founder | Δ ln(params)/generation | implied over 500 generations |
|---|---|---|
| 5131 params, `n_blocks=1` | +0.00666 ± 0.00447 | **28×** |
| 5360 params, `n_blocks=2` | +0.00235 ± 0.00470 | 3.2× (CI spans 1) |

Founders are seeded at `n_blocks=2` for this reason. See `DESIGN.md §3`.

---

## Conditions

Each is one config; each runs at **≥10 seeds**. The factor under test is the only thing
that changes.

### Primary

| # | Condition | Config | Tests |
|---|---|---|---|
| P1 | full system | `pilot.json` | the hypothesis |
| P2 | **neutral** | `neutral.json` | the passive-diffusion null for every complexity claim |
| P3 | displaced founders | `--set displaced_founder_params='[40000]'` | McShea subclade test: driven vs passive |

### Ablations

| # | Condition | Config change | Isolates |
|---|---|---|---|
| A1 | fixed architecture | `phase.structural_mutation=false` | whether architecture change contributes at all |
| A2 | single environment family | `island_family_fraction≈0.02` | whether ecological diversity is doing the work |
| A3 | global top-k | `evo.n_islands=1` | whether island structure matters |
| A4 | fixed mutation strategy | `phase.heritable_mutation=false` | whether evolvability genes matter |
| A5 | Darwinian inheritance | `phase.weight_inheritance="darwinian"` | whether inherited weights help or hurt long-run evolvability |
| A6 | partial inheritance | `phase.weight_inheritance="partial"` | inheritance without perturbation |
| A7 | growth only | `p_growth_bias` fixed at 0.98 | whether shrinkage is load-bearing |
| A8 | scalarised fitness | `evo.selection="scalarised"`, λ ∈ {0, 0.01, 0.05, 0.2} | **demonstrates that λ decides the answer** |
| A9 | Pareto selection | `evo.selection="pareto"` | performance/compute front instead of a budget |
| A10 | archive re-entry | `evo.archive_reentry=true` | whether protecting niches changes the size distribution |
| A11 | random search | `evo.tournament_k=1` | the floor any evolutionary claim must clear |

### Sweeps

| # | Sweep | Range | Produces |
|---|---|---|---|
| S1 | compute abundance | `flop_budget_per_island` × {⅛, ¼, ½, 1, 2, 4, 8} | **the phase diagram**: at what compute abundance does growth become viable? |
| S2 | environmental stability | `n_variants` ∈ {1, 2, 4, 16, 0} | where adaptation lives: weights vs context |
| S3 | ancestral scale | 1K / 5K / 20K | whether the outcome depends on where it starts |

S1 replaces the brief's binary "does complexity grow?" with a quantitative boundary, and is
the single most informative experiment in the set.

---

## Metrics

**Per generation** (`generation` table): population size; parameter count median, mean,
min, max, log-mean, log-SD; mean and max score; mean in-context gain; species count;
archive coverage and QD score; cumulative births; cumulative accounted FLOPs; and the
median of every strategy gene.

Tracking `params_min` is not decoration — it is McShea's driven-trend discriminator.

**Per organism** (`organism` table): id, parents, lineage root, birth generation, island,
architecture signature and every dimension, mutation operators applied at birth, cumulative
structural events, fitness components, behavioural descriptor, accounted FLOPs, age,
death reason, and all strategy genes. Retained for every organism that ever lived,
including the extinct — weight tensors are freed on death, metadata never is.

**Per probe** (`probe` table): (generation × class × condition) → score, gain, final block,
mean parameter count, cohort size. This is the evolvability decomposition.

**Derived, in the report:** adaptation AUC by class and generation; the `full` −
`no_feedback` difference (in-context adaptation attributable to feedback); the `full` −
`capacity_matched` difference (adaptation beyond capacity); selected − neutral parameter
drift; lineage tree with architectural innovations marked; growth/shrink event rates;
survival by mutation operator; per-FLOP performance.

---

## Pre-specified analysis

Fixed before C-test is opened; see `PREREGISTRATION.md`.

* **Unit of analysis is the seed**, not the organism. Organisms within a run are not
  independent.
* **Primary test:** permutation test on the slope of Class C adaptation AUC against
  generation, resampling seeds, two-sided, α = 0.05.
* **Complexity test:** difference in Δ ln(params) between the selected and neutral arms at
  matched generation count, bootstrap CI over seeds.
* **No optional stopping.** Generation count is fixed in the config before the run.
* **Multiplicity:** Holm correction across the four required adaptability criteria.
* Effect sizes with bootstrap CIs are reported for everything; p-values alone are not
  reported for any claim.

---

## Reproducing

```bash
cd openevo && pip install -e ".[dev]"
python -m pytest -q                                    # 71 tests
python scripts/min_viable_size.py                      # stage 3
python scripts/run_evolution.py --config configs/pilot.json
python scripts/run_evolution.py --config configs/neutral.json
python scripts/report.py results/pilot/run.db --neutral results/neutral/run.db
python scripts/bench_backend.py                        # on Apple Silicon
```

Every run writes `run.manifest.json` with the config, its hash, the git commit, the seed,
the platform, and the sealed hash of the world-family split.
