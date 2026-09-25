# EXPERIMENTS.md — Protocol, Pre-Registration and Log

This file is the experimental record. Section 1 is written **before** any run
and must not be edited afterwards except to record deviations in §5.

---

## 1. Pre-registration (phase 1)

### 1.1 Question

What is the minimal ecological condition sufficient for Darwinian evolution of
neural-module assemblies to produce duplication → divergence → specialisation,
and does the organisation that results confer measurable evolvability on unseen
causal structure?

### 1.2 Design

3 (goal structure: MVG / RVG / FIX) × 2 (metabolism: on / off) × 2 (duplication:
on / off) = 12 cells, 12 replicates each = 144 runs, plus 12 fitness-shuffled
drift controls whose seeds are matched one-to-one with the MVG / metabolism-on /
duplication-on replicates.

All runs execute as lanes of one batched program. No condition gets a different
code path.

### 1.3 Primary outcome

**Duplication→specialisation event rate**, measured over the final 25% of
generations.

A duplicate pair counts as an event when the JS divergence between the two
copies' subgoal-contribution vectors, measured at a matched age since
duplication, exceeds the 95th percentile of the same quantity in the
seed-matched drift control.

Implementation: `nemo.metrics.definitions.duplication_specialised_vs_drift`.
Calibration (drift-vs-drift gives a 5% false-positive rate) is asserted in
`tests/test_selection_and_metrics.py::test_primary_outcome_has_calibrated_false_positive_rate`.

### 1.4 Secondary outcomes

| Outcome | Metric | Null |
|---|---|---|
| Structural modularity | `Q_str` of the realised message graph | degree-preserving rewiring, 1,000 draws |
| Functional modularity | `Q_cor` of the activity-correlation graph | phase-shuffled traces |
| Specialisation | `S_i = 1 − H(c_i)/log K` | drift control (NOT a permutation — see §1.7) |
| Division of labour | mean pairwise JS of contribution vectors | independent label permutation |
| Redundancy | pairs where neither ablation alone costs >5% but the joint costs >20% | size-matched random pairs |
| Regulatory organisation | `I(subgoal phase ; gate_i)` in bits | circular context shift |
| Robustness | AUC of fitness vs ablation fraction | parameter-matched monolithic control |
| Evolvability | normalised AUC of the alien adaptation curve, pre-adaptation fitness as covariate | drift control |

### 1.5 Decision rule

Cliff's δ ≥ 0.47 (large) **and** Mann–Whitney *p* < 0.01 after Holm–Bonferroni
across all cell×metric contrasts, at n = 12 replicates. Both conditions must
hold. Implemented in `scripts/analyse.py`; the verdict column prints
`SUPPORTED`, `large, n.s.`, `sig., small` or `null`.

### 1.6 Predictions, with signs fixed in advance

| Cell | Prediction |
|---|---|
| MVG + M-on + D-on | Event rate above drift; `Q_str` **and** `Q_cor` above their nulls; alien adaptation improves with evolutionary time |
| FIX + M-off | Indistinguishable from drift on every metric |
| RVG + M-on | Intermediate. **If RVG matches MVG, Kashtan & Alon (2005) does not transfer to this substrate** — a novel negative result, and the most interesting thing this experiment could find |
| Any cell, D-off | If specialisation matches D-on, duplication is not necessary and a central premise of the original proposal is wrong |

### 1.7 Statistical decisions taken in advance

- **The permutation null is not used for the primary outcome.** Permuting a
  module's subgoal labels preserves its entropy exactly, so for concentrated
  contributions the null piles up at maximum divergence and the test has no
  power. The drift control is the null. This was established by testing before
  any experimental run (see §5, deviation D2).
- **Structural modularity is never reported alone.** `Q_str` above its rewired
  null with `Q_cor` and ablation-based specialisation at null is reported as
  *structural modularity without functional specialisation*, per Nature
  Communications 15 (2024).
- **Genome length is never reported as "complexity"** without the drift
  control's genome-length curve beside it.
- Negative results are reported with the same prominence as positive ones.

---

## 2. Phases

| Phase | Contents | Modelled cost | Gate |
|---|---|---|---|
| 0 Benchmark | `bench_backend.py` on the M3 Ultra | minutes | dispatch overhead measured; vectorised ≥1,000× naive |
| 1 Smoke | 144 runs × 100 gen | ~1.2 min | selection beats drift; no NaNs; alien isolation passes |
| 2 Pilot | 144 runs × 1,000 gen + controls | ~11 min | variance estimate for power; genome-length curve separates from drift |
| 3 Main | 144 runs × 5,000 gen, lifetime 512, E=16 | ~3.9 h | decision rule evaluated |
| 4 Learning | + lifetime learning: none / reset / Lamarckian / partial | ~11.7 h | only after phase 3 answers without it |
| 5 Sex & HGT | + recombination, + horizontal transfer | ~23 h | separate runs, never mid-run |

Costs from `scripts/compute_budget.py` §11 (M3 Ultra, modelled).

---

## 3. How to run

```bash
python3 scripts/compute_budget.py                    # arithmetic and projections
NEMO_BACKEND=mlx python3 scripts/bench_backend.py    # phase 0, on the M3 Ultra
python3 scripts/run_experiment.py --preset smoke
python3 scripts/run_experiment.py --preset pilot  --subset full
python3 scripts/run_experiment.py --preset main   --subset full --out results/main
python3 scripts/analyse.py results/main
python3 scripts/report.py  results/main            # HTML research dashboard
python3 scripts/module_sweep.py                    # smallest viable module, empirically
python3 scripts/alien_test.py results/main        # evolvability on held-out causal structure
python3 scripts/run_experiment.py --preset main --control monolithic \
        --out results/main_mono                   # the matched monolithic control
```

`--subset core` runs the four lane-groups carrying the central contrast
(MVG / RVG / FIX / DRIFT); `--subset full` runs the complete 12-cell factorial.

---

## 4. Run log

| Date | Preset | Backend | Runs × organisms | Gens | Wall clock | Result |
|---|---|---|---|---|---|---|
| 2026-09-25 | smoke | numpy (4-core container) | 4 × 16 | 60 | 8 s | Selection works; drift stays flat. Exposed the DECOY calibration flaw (§5 D3) |
| 2026-09-25 | pilot | numpy (4-core container) | 16 × 64 | 62 (stopped) | 5 min | **Selection collapsed genomes to ~1 gene.** Diagnosed two design flaws (D8, D9). Evidence in `results/pilot_no_min_criterion/` |
| 2026-09-25 | pilot | numpy (4-core container) | 16 × 64 | 180 | 942 s | **Clean null on every outcome.** Pipeline validated end to end; hypothesis untested at this scale. See §6 |

**Not yet run: anything on the M3 Ultra.** Every hardware number in
RESEARCH.md §5 and DESIGN.md §7 is from the analytic model, not measurement.
Phase 0 has to happen on the target machine before the main experiment is
sized.

---

## 6. Pilot outcome (2026-09-25)

**Read this as a pipeline validation, not as a test of the hypothesis.** The
pilot is n=4 replicates × 180 generations × lifetime 32 on a 4-core CPU. The
designed main experiment is n=12 × 5,000 generations × lifetime 512 — roughly
300× more evolutionary time per replicate. Nothing here can confirm or refute
the central hypothesis, and it is not offered as doing so.

### What the machinery did

It ran, it recorded 2,827 duplication events and 970 tracked pair
observations, it computed every metric against its null, and **it reported a
null honestly rather than finding a pattern.** That was the thing to check.

### Primary outcome: duplication → specialisation rate

| condition | pairs | events | rate | Cliff's δ vs drift |
|---|---|---|---|---|
| DRIFT | 347 | 14 | 0.040 | — (calibration; nominal 0.05) |
| FIX | 261 | 17 | 0.065 | +0.012 |
| MVG | 177 | 8 | 0.045 | −0.038 |
| RVG | 169 | 2 | 0.012 | −0.049 |

**No condition exceeds its drift control.** Every effect size is
indistinguishable from zero. The drift control's own rate lands at 0.040
against a nominal 0.05, which is the calibration working.

### Structural modularity vs the degree-preserving rewired null

| condition | Q_str | rewired null | excess |
|---|---|---|---|
| DRIFT | 0.246 | 0.246 | 0.000 |
| FIX | 0.026 | 0.033 | −0.007 |
| RVG | 0.168 | 0.201 | −0.032 |
| MVG | — | — | fewer than 3 live genes; Q undefined |

**No condition is more modular than a degree-matched random graph.** Note that
raw `Q_str` is *highest* in the drift control (0.246) purely because drift has
the longest genomes and more nodes give more chances for modular structure by
accident. Reported without the rewired null, that single number would have
read as "the unselected control evolved the most modularity", which is the
sparsity/size confound the null exists to catch.

### The finding that matters: the task is too easy

Genome length, mean over replicates:

| generation | MVG | RVG | FIX | DRIFT |
|---|---|---|---|---|
| 0 | 3.99 | 4.01 | 4.02 | 3.99 |
| 60 | 3.41 | 3.11 | 3.73 | 4.68 |
| 120 | 2.32 | 3.61 | 3.48 | 4.68 |
| 179 | **1.44** | 3.90 | 3.33 | 4.23 |

The largest effect anywhere in the run is **genome shrinkage** (MVG vs drift,
Cliff's δ = −1.00 on genome length, active modules and metabolic cost alike),
not modularity. D8 and D9 delayed the collapse from generation ~40 to ~120;
they did not prevent it, because they addressed the wrong cause. The minimal
criterion stops a *degenerate* organism from riding the front, but a single
2,145-parameter module apparently holds **median** performance on a 3-subgoal
goal at lifetime 32 — so it clears the criterion honestly, and selection is
right to discard the other three modules.

**There is no selective reason for modularity when one module solves the
task.** That is a property of the environment, not of the selection scheme,
and it is the thing to fix before the main experiment:

1. Raise `goal_arity` from 3 toward 5, so more must be handled at once.
2. Run the module-size sweep (`scripts/module_sweep.py`) and take the
   *smallest viable* `d_model`, not the default 16 — a module large enough to
   solve everything alone is a module that makes the experiment vacuous.
3. Lengthen the lifetime so harder primitives (DELAY, SWITCH) are learnable at
   all, rather than being noise a single module averages over.

The diagnostic to run before committing compute: **a 1-gene organism should
not be able to clear the minimal criterion.** If it can, the environment is
not asking for an organism.

### What is not concluded

That modularity does not evolve. The pilot has neither the generations, the
replicates, nor an environment that requires more than one module — three
independent reasons it could not have detected the effect it is looking for.

---

## 5. Deviations from pre-registration

Recorded as they occur. Deviations found *before* any experimental run are
still recorded, because "we changed the metric after seeing the pilot" and "we
changed the metric after unit-testing it" are different things and the reader
cannot tell them apart otherwise.

**D1 — mutation-class 2 (per-module architecture mutation) dropped from phase 1.**
Population-vectorised execution requires one module *shape* per run. Module size
varies between runs instead (`ModuleSpec` sweep). Reason: DESIGN.md §7.2. The
central question is about organisation, not per-module architecture search,
which CoDeepNEAT already covers.

**D2 — primary outcome null changed from permutation to drift control.**
Found by unit test, before any experimental run. A within-row permutation of a
module's subgoal contributions preserves its entropy exactly, so for two
perfectly concentrated vectors over K channels the null attains maximum
divergence with probability (K−1)/K and the empirical p-value is ~0.83 no matter
how cleanly the modules have specialised. The permutation function is retained
and documented as a limitation, not deleted.

**D3 — DECOY and IRREV given positive reward components.**
Found by the smoke run, before any experimental run. As penalty-only channels
their calibrated range was ~0.06 wide, so normalising by it amplified noise ~16×
and the ceiling was reachable by a degenerate constant-action policy. The
fitness-shuffled drift control consequently appeared to "improve" from 0.01 to
0.42. `world.calibrate` now refuses any scored channel narrower than
`MIN_RANGE = 0.25`, and `tests/test_alien_isolation.py` asserts it.

**D4 — NOISE reclassified from subgoal to modifier.**
Consequence of D3: NOISE has no reward channel of its own, so it cannot be a
member of the subgoal basis. The shared basis is now
(RECALL, XOR, SWITCH, DECOY, GATE, DELAY), giving C(6,3) = 20 MVG goals, and the
alien pool is (COUNT, IRREV, DRIFT).

**D9 — reward aggregation changed from arithmetic mean to conjunctive.**
Found by the pilot. Under an arithmetic mean over active subgoals, an organism
scores well by handling whichever subgoal is easiest and ignoring the rest — so
there is no pressure for one part of the organism to do one job and another
part a different job, which is the pressure this experiment exists to study the
consequences of. The pilot showed it plainly: the MVG cell reached the
**highest reward with the smallest genome** (1.14 genes, 0.51 of them active)
while the drift control sat at 4.28. Reproductive fitness is now
`0.25·arithmetic + 0.75·geometric` over the active subgoals; the geometric term
is ~0 unless every subgoal is above its random baseline, and the arithmetic
minority keeps a gradient for populations that start below baseline everywhere.
`"mean"` is retained as an explicit control condition, not as a default.

**D8 — minimal performance criterion added to Pareto selection.**
Found by the pilot, and the more serious of the two. On a two-objective front
the cheapest organism is **always** non-dominated — nothing can beat it on cost
— so a bare Pareto rank on (reward, −compute) hands rank 0 to the most
degenerate genome in the population regardless of how badly it performs. With
metabolism on, genomes collapsed from 4 genes to ~1 within 40 generations while
the seed-matched drift control held at 3.7. A one-gene organism has no
organisation to measure, so the experiment's central question becomes
unaskable. Selection now restricts the Pareto front to organisms clearing a
performance floor (median by default); below the floor, an organism ranks after
every eligible one however cheap it is. Precedent: minimal criterion
coevolution (Brant & Stanley 2017), as used in POET. This is also what the
original proposal asked for — redundancy should survive when its benefits
justify its cost, rather than being competed away by whatever is smallest.
Evidence preserved in `results/pilot_no_min_criterion/`.

**D7 — duplicate pairs tracked by innovation id rather than by birth lane.**
A correctness fix, not a design change, found while validating the assay.
Offspring overwrite the lanes of the organisms they displace, so looking for a
pair in its birth lane both missed most survivors and could attribute a pair to
whatever unrelated organism now occupied that lane. Observed pairs rose from 4
per 25 generations to 31 per 60 on identical settings. The resulting
observations also make the case for the drift null concrete: median weight
distance across observed pairs is 0.0000 while median contribution divergence
is 0.13 — most pairs are byte-identical and still *appear* to have diverged.

**D6 — pilot duplication rate raised from 2% to 6% (deletion matched).**
Pilot only; the main experiment keeps the pre-registered 2%. At 2% per birth,
a 250-generation run produced too few duplicate pairs surviving to a measurable
age (~20 generations) to estimate the variance the power analysis needs. Both
rates are raised together so the neutral expectation on genome length stays
flat, which is the property that makes the drift control interpretable. The
pilot's job per phase 2 is explicitly "variance estimate ... or duplication rate
is retuned", so this is the pilot working, not a departure from it. Exposed via
`--p-dup`; recorded in each run's `meta.json`.

**D5 — activation changed from GELU to ReLU.**
Performance, measured: the tanh-based GELU was 11% of rollout time on the
reference backend and `where(x>0, x, x*slope)` cost 7.8 ms/round against 1.0 ms
for `maximum(x, 0)`. No behavioural justification is claimed; if phase 4's
lifetime learning shows dead-unit problems at `d_ff=32`, this reverts.
