# EXPERIMENT_001.md — frozen protocol

**Status: FROZEN 2026-09-25, before any group-comparison run was executed.**

Everything below was written before a single number from the comparison groups
existed. Calibration used `world_seed=99` only, which never appears in the
experiment. The go/no-go thresholds in §7 are the ones we will be held to.

---

## 1. The question

Does a transformer whose architecture develops in response to its own failures
become better at predicting structure that was withheld from it — at matched
training compute — than a transformer whose architecture was fixed in advance?

This is a **go/no-go test**, not a demonstration. A clean negative answer is a
successful outcome.

---

## 2. Scale, and an honest statement of the gap

The brief asks for 10–30M starting parameters and <100M final. **This was not
reachable.** The container is 4 CPU cores with no GPU; at `d=512, L=8` it
sustains roughly 500 tok/s, so one run of the requested size would take about a
week and the study needs 18 of them.

| | requested | run here | ratio |
|---|---|---|---|
| start params | 10–30M | **0.077M** | ~170x smaller |
| final params | <100M | **~0.18M** | |
| hardware | unspecified | 4 CPU cores | |

`configs/exp001_paper_scale.json` holds the requested configuration so the
identical protocol can be launched on a GPU without a code change. Every claim
in `RESULTS_001.md` is explicitly scoped to the small scale, and **no result
here should be assumed to transfer to 10M+ parameters.** That is the single
largest limitation of this study and it is a hardware limitation, not a design
choice.

The scale that *was* used is not arbitrary. It is the operating point where
FIXED SMALL plateaus below saturation on every bucket. At `d=128, L=4` the
small model already solved D1 and D2 (0.94 / 0.98), which would have left
growth nothing to repair and the experiment no power.

## 2b. The regime, which matters as much as the scale

The world was enlarged to 80 entities / 26,316 facts after a dry run showed the
model was **memorisation-limited** (13.4 parameters per training fact) rather
than **capacity-limited**. The hypothesis is about capacity failure. If the
model can simply memorise, "persistent failure" means "not enough data", and
adding capacity is answering a question nobody asked. At 80 entities the ratio
is 5.0 parameters per training fact.

---

## 3. The world

Deterministic functions of latent per-entity coordinates (group ∈ 0..9,
period ∈ 0..7) that the model never observes. 9 relation types, 26,316 facts.
Surface form `[REL] [ARG1] [ARG2] [=] [ANSWER]`, loss on the answer position
only.

Splits: 30% never-revealed holdout, 8% validation, the rest a pool revealed
over 8 developmental ages. Atomic facts (VAL / SHELL / METAL / SOLUBLE — the
world's axioms) are all revealed at age 0; progressive revelation applies to
the relational facts.

### Extrapolation-distance buckets

| bucket | n (holdout) | what it requires | majority baseline |
|---|---|---|---|
| D0 interpolation | 4388 | generalise within the observed regime | 0.340 |
| D1 near | 915 | apply a group rule to less-observed entities | 0.287 |
| D2 compositional | 1908 | chain REACT then SOLUBLE, never composed in training | 0.501 |
| D3 distant | 1490 | learn `mass = f(period, group)` on periods 0–6 and extrapolate to period 7, which appears in **no** HEAVIER fact | 0.497 |
| D4 unsupported | 233 | nothing — salted hash, unpredictable in principle | 0.545 |

---

## 4. Groups

| group | description |
|---|---|
| **A** FIXED SMALL | initial architecture throughout |
| **B** FIXED LARGE | trained from scratch at **exactly** D's final architecture for the same seed (two-phase: D runs first, B copies its `final_arch`) |
| **C** GROWTH ONLY | developmental controller, one unit per event, no pruning |
| **D** DEVELOPMENTAL | controller + overgrow K=4 + differentiate 800 steps + competitive prune to 1 |
| **D_RANDPRUNE** | identical to D, survivor chosen at random |
| **R** RANDOM DEV | same event count, kinds and order as D; **random timing** |

3 seeds each = 18 runs.

`keep = 1` makes C and D end at identical parameter counts, so:

- **C vs D** — does overgrowing and selecting beat committing to one candidate?
- **D vs D_RANDPRUNE** — does the *selection rule* do anything, or would a
  random survivor work as well?
- **D vs R** — does growing *under pressure* beat growing *on a schedule*?
- **D vs B** — does the developmental path beat starting at the destination?

---

## 5. Fairness

The budget is **8.0e12 training FLOPs**, identical for every group, computed
from the live structure each step. A grown model therefore takes fewer steps;
a model running 4 candidates pays for 4 candidates. Tokens, steps, wall-clock,
peak RSS and inference FLOPs/token are all recorded, and equal-token results
are reported alongside equal-FLOP.

---

## 6. Endpoints

**Primary (pre-registered, single):** equally-weighted mean holdout accuracy
over D1, D2, D3 at the full FLOPs budget.

Equal weights are deliberate — D0 has 4388 facts and D1 has 915, so a pooled
mean would silently become "how well do you do on D0".

**Secondary:** per-bucket accuracy; validated information gain in bits against
`P(answer | relation)` computed on revealed facts only; Brier; ECE;
precision@50% coverage; accuracy vs developmental age; extrapolation per PFLOP.

**Gating check (not an endpoint):** D4 accuracy against its own majority-class
rate. z > 2.5 voids the run.

We deliberately do **not** construct a `correctness × novelty × calibration`
composite. Composite endpoints have free parameters, and free parameters in an
endpoint are how experiments get massaged. The trade-off is shown as a Pareto
plot.

Statistics: paired bootstrap across seeds, Cohen's d, and the raw win count,
reported together. With n=3 we will not claim significance.

---

## 7. GO / NO-GO criteria — frozen

The brief proposed criteria and asked us to improve them first. Three changes
were made, each closing a hole:

1. **Added an absolute floor.** "Better than FIXED SMALL" is satisfiable by two
   models that are both at chance. Every GO now requires beating the
   majority-class baseline on the relevant buckets.
2. **Added the RANDOM DEV control to the GO criteria, not just the NO-GO.** The
   original phrasing let a model pass by beating fixed baselines while being
   indistinguishable from random growth — which is the null hypothesis this
   whole study exists to test.
3. **Made "no catastrophic forgetting" measurable.** It is now: accuracy on
   previously-mastered categories must not drop by more than 2 points from the
   pre-growth value within 500 steps of any growth event.

### GO — all four must hold

1. D beats A on the primary at matched FLOPs, with a paired bootstrap CI
   excluding 0 and D winning in ≥ 3 of 3 seeds.
2. D beats the majority-class baseline on D1, D2 and D3 individually.
3. D beats **R** on the primary (mean difference > 0, winning in ≥ 3 of 3 seeds).
   *Without this, "growth helps" reduces to "more parameters help".*
4. Either D ≥ B at matched FLOPs, or D's extrapolation-per-PFLOP exceeds B's.

Plus: no catastrophic forgetting as defined above, and no leakage flag.

### STRONG GO — additionally

- the D − A advantage is larger on D3 than on D1 (gains grow with distance);
- surviving branches show measurably concentrated ablation effects on specific
  relation categories;
- D ≥ C (pruning preserves or improves);
- D > D_RANDPRUNE (the selection rule, not just the extra training, matters).

### NO-GO / REDESIGN — any one

- D ≈ A after FLOPs matching;
- D ≈ R (the controller is indistinguishable from a random schedule);
- the advantage vanishes across seeds, or sign-flips;
- B ≥ D at matched FLOPs *and* matched parameters (it is just parameters);
- pruning consistently damages performance (D < C reliably);
- **D4 leakage flag on any run** — voids everything;
- D3 at chance for every group (the flagship bucket has no signal to compare).

---

## 8. The five strongest confounders, and what was done about each

Written as a hostile reviewer, before running.

**1. Compute.** *"The developmental model just got more FLOPs."*
→ Budget is FLOPs, computed from the live structure. Growing means fewer steps.
Overgrowth is charged for all K candidates.
`test_flops_accounting_charges_for_live_candidates`.

**2. It is just more parameters.** *"Any bigger model would do this."*
→ B is built to D's **exact** final architecture and trained from scratch. R
matches D's parameter trajectory with random timing. If D ≈ R, the answer is
"parameters", and we say so.

**3. Growth is an optimizer perturbation, not a capacity change.** *"You are
measuring a learning-rate restart."*
→ New parameters get a warmup from exactly zero LR and existing Adam moments
are carried across, so growth is not a disguised LR event. R receives the
identical perturbation at random times, isolating timing from perturbation.

**4. Leakage / memorisation.** *"The holdout is inferable by a shortcut."*
→ Token ids are rejection-sampled to decorrelate from the latent grid
(|r| < 0.05); one random permutation leaves r ≈ 0.15, a real side channel.
D4 is a salted hash and acts as a live alarm, calibrated against its own
majority rate (calibrating against 0.5 produced a false alarm at z = 2.8 on an
honest model — see `RESEARCH_LOG.md`). Holdout membership is asserted disjoint
from every revealed set at every age.

**5. Researcher degrees of freedom.** *"You tuned this until it worked."*
→ All calibration ran on `world_seed=99`, never used in the experiment. The
primary endpoint is single and pre-registered. This document is frozen before
the first comparison run. The prediction ledger is hash-chained and contains no
ground truth, so predictions provably precede revelation. Four pilots were
discarded and all four are written up in `RESEARCH_LOG.md`, including the ones
that exposed our own mistakes.

**6 (the one the brief did not list).** *"Competitive pruning looks good
because pruning regularises at small scale, not because the competition
selected well."*
→ `D_RANDPRUNE` prunes the same count at random. "When BERT Plays the Lottery"
found random structured pruning competitive with importance-based pruning, so
without this control any claim about competition is unsupported.

---

## 9. Gates

| gate | question | status |
|---|---|---|
| 0 | can we train the tiny transformer at all? | pass — far above majority baseline |
| 1 | can we grow while preserving function? | **pass — `logit_delta == 0.0` exactly**, on trained models, both operators |
| 2 | does growth match or beat a fixed control at matched FLOPs? | measured |
| 3 | does the controller find real plateaus? | measured (vs R) |
| 4 | does growth + competition + pruning beat growth alone? | measured (C vs D, D vs D_RANDPRUNE) |
| 5 | does either strategy improve prediction of withheld structure? | measured |

---

## 10. Reproduction

```
python3 tests/test_gates.py
python3 run_experiment.py --config configs/exp001_cpu.json --seeds 0,1,2
python3 evaluate.py --runs 'results/exp001/*/'
python3 visualize.py
```
