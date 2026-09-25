# NEMO — Neuroevolution of Modular Organisms

An experiment on whether Darwinian selection over variable-length genomes of
tiny cooperating neural modules produces duplication, functional divergence,
specialisation, regulation and evolvability — and, more precisely, **under
which ecological conditions it does**.

*(This directory is self-contained and unrelated to the `claudex` plugin at the
repository root.)*

> **Read this first.** The strong version of the question — "does modularity
> emerge with nothing rewarding it?" — already has an answer in the literature,
> and the answer is **no**. Under a fixed fitness function, evolutionary
> algorithms reliably produce entangled, non-modular networks
> ([Kashtan & Alon 2005](https://www.pnas.org/doi/10.1073/pnas.0503610102);
> [Clune, Mouret & Lipson 2013](https://royalsocietypublishing.org/rspb/article/280/1755/20122863/74559/)).
> So this project does not ask whether organisation appears from nothing. It
> asks **which ecological condition is sufficient**, tests the candidates
> factorially, and measures every claim against a fitness-shuffled drift
> control. See [RESEARCH.md §3](RESEARCH.md).

---

## Documents

| File | What is in it |
|---|---|
| [RESEARCH.md](RESEARCH.md) | Literature audit, closest prior work, the finding that reshapes the design, hardware arithmetic, confounders, defensible novelty statement |
| [DESIGN.md](DESIGN.md) | Critique of the original proposal, the recommended design, operational definitions of every biological term, execution architecture |
| [EXPERIMENTS.md](EXPERIMENTS.md) | Pre-registration, decision rule, phases, run log, deviation log |

## Code

```
src/nemo/
  modules/spec.py        exact parameter and FLOP accounting (unit-tested)
  genome/population.py   dense population state -- every organism is a lane
  organisms/execute.py   population-vectorised rollout
  mutation/operators.py  the seven operators + innovation registry
  environments/          causal primitives and the MVG / RVG / FIX goal structures
  selection/islands.py   island ecology, performance/cost Pareto ranking
  metrics/               operational definitions, null models, periodic assays
  ecology/evolve.py      the generation loop
  storage/db.py          SQLite events + content-addressed module archive
scripts/
  compute_budget.py      roofline + dispatch model for the M3 Ultra
  bench_backend.py       measures the five constants that model assumes
  module_sweep.py        smallest viable module, empirically
  run_experiment.py      runs an experiment
  analyse.py             evaluates the pre-registered decision rule
  report.py              self-contained HTML research dashboard
tests/                   47 tests
```

## Quick start

```bash
cd nemo
pip install numpy pytest
python3 -m pytest tests/ -q

python3 scripts/compute_budget.py                 # what the hardware allows
python3 scripts/run_experiment.py --preset smoke  # ~10 s
python3 scripts/analyse.py results/smoke
python3 scripts/report.py results/smoke           # open results/smoke/report.html
```

On the target machine:

```bash
NEMO_BACKEND=mlx python3 scripts/bench_backend.py --device gpu   # phase 0
NEMO_BACKEND=mlx python3 scripts/run_experiment.py --preset main --subset full
```

---

## The three things worth knowing

### 1. Modularity has known causes, so the experiment tests causes

Three mechanisms are independently established as sufficient: **modularly
varying goals**, **connection cost**, and **selection for specialisation**. The
design promotes the first two to experimental factors instead of leaving them
implicit:

| Factor | Levels |
|---|---|
| **G** goal structure | MVG (subgoals recur in new combinations) · RVG (subgoals never recur identically) · FIX |
| **M** metabolic cost | on · off |
| **D** duplication operator | on · off |

12 cells × 12 replicates + 12 seed-matched drift controls. MVG and RVG are
matched on difficulty, goal diversity and switching rate; they differ *only* in
whether a subgoal is a stable, reusable target.

### 2. The population is one tensor program, not a loop over organisms

Modelled on the M3 Ultra: a Python loop over organisms costs **3.7 hours per
generation**. The vectorised form costs **0.41 seconds**. That is **32,768×**,
and it decides whether the experiment exists at all.

It also inverts the intuition about cost. Organisms, episodes, replicate runs
and factorial cells are *parallel* axes that ride inside the same kernel
dispatches; only lifetime × generations is serial. **Statistical power is
almost free; long lifetimes are not.** The full 144-run factorial at 5,000
generations is modelled at ~3.9 h, not the multiple days the original proposal
anticipated.

### 3. Nothing is a result without its null

| Claim | Null it is measured against |
|---|---|
| Genome length grew | fitness-shuffled drift control (otherwise it is bloat) |
| Graph is modular | degree-preserving rewiring (otherwise it is sparsity) |
| Modules specialised | drift control (two random modules always differ) |
| Duplicates diverged | drift control at matched age since duplication |
| Evolvability rose | drift control, with pre-adaptation fitness as covariate |

Structural modularity is never reported without functional modularity beside
it: [Nature Communications 15 (2024)](https://www.nature.com/articles/s41467-024-55188-9)
shows structural modularity does not imply functional specialisation.

The need for that discipline is not hypothetical here. In a pilot assay, a
drift-control duplicate pair with **byte-identical weights** showed the
**maximum possible** contribution divergence — pure estimation noise. Reported
without its null it would have read as a textbook specialisation event.

---

## Key numbers

| Quantity | Value | Source |
|---|---|---|
| Parameters per module | **2,145** (d=16, d_ff=32, K=4) | `ModuleSpec`, unit-tested |
| 4-gene ancestor | **8,972** neural + 44 regulatory | `OrganismSpec` |
| Forward FLOPs / module-step | 7,540 | `ModuleSpec.flops_forward` |
| Pilot resident memory (modelled) | **0.26 GB** | `compute_budget.py` §9 |
| Recommended memory ceiling | 32 GB soft / 64 GB hard | DESIGN.md §7.4 |

The original proposal's estimates of ~2,500 parameters per module and ~10K per
ancestor are **confirmed correct**. Its 350 GB working-memory budget is ~1,000×
the workload, and its copy-on-write scheme buys 1.00× in the execution path at
this module size — so content-addressing lives in the storage layer instead,
where it genuinely pays.

---

## Pilot outcome

The pipeline has been run end to end. It reported **a clean null on every
outcome** — no condition's duplication→specialisation rate exceeds its drift
control, and no condition's graphs are more modular than degree-matched random
ones. That is the machinery working: it declined to find a pattern.

The informative part is *why*. The largest effect in the run is genome
shrinkage (MVG 4.0 → 1.4 genes, Cliff's δ = −1.00 against drift), because **a
single 2,145-parameter module holds median performance on a 3-subgoal goal at
lifetime 32**. When one module solves the task there is no selective reason for
modularity, and selection is right to discard the rest. That is an environment
problem, not a selection problem, and [EXPERIMENTS.md §6](EXPERIMENTS.md) sets
out the fix and the diagnostic to run before spending compute on the main
experiment: *a 1-gene organism should not be able to clear the minimal
criterion.*

This does **not** show that modularity fails to evolve. The pilot is n=4 over
180 generations — roughly 1/300th of the designed experiment — in an
environment that does not require an organism.

## Status

Implemented and tested: representation, mutation operators, vectorised
execution, compositional environments, island selection, metrics with nulls,
periodic assays, storage, analysis and reporting. Phase-1 smoke and pilot runs
execute on the NumPy reference backend.

**Not yet done: any measurement on Apple Silicon.** Every hardware figure above
comes from the analytic model in `scripts/compute_budget.py`, not from a
stopwatch. `scripts/bench_backend.py` measures what the model assumes, and the
model is most sensitive to per-dispatch overhead — that is the first number to
take on the real machine.
