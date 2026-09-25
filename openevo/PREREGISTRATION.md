# PREREGISTRATION.md

Recorded **before** the Class C-test suite is opened. The point is narrow and practical:
once alien-world numbers have been seen, every subsequent choice — how long to run, which
metric to feature, which seed looked broken — is a channel through which those numbers can
influence the result. Freezing the analysis closes that channel.

Fill in the run identifiers below and commit this file **before** passing `--open-test-set`.

```
Date sealed:            ____________________
Git commit:             ____________________
Suite seal hash:        e50ad237af389d10   (SuiteSplit(seed=20260101).seal())
Configs:                configs/pilot.json, configs/neutral.json
Seeds:                  0..9  (10 per condition)
Generations per run:    ____  (fixed here; no extension after inspecting results)
```

---

## H1 — Adaptation on novel causal structure improves over evolutionary time

**Primary outcome.** Class C-test adaptation AUC (mean normalised return across the
context, reference-normalised per world) for frozen cohorts, as a function of generation.

**Test.** Permutation test on the per-seed OLS slope of AUC against generation, resampling
seeds, two-sided, α = 0.05. Effect size: mean slope with a bootstrap CI over seeds.

**Confirmed only if all four hold** (Holm-corrected across the four):

1. slope > 0 on `full`;
2. slope on `full` exceeds slope on `no_feedback` — otherwise a reactive prior, not adaptation;
3. slope on `full` exceeds slope on `capacity_matched` — otherwise capacity;
4. slope on `fixed_hparams` remains > 0 — otherwise exploration-schedule tuning.

**Prediction.** (1) and (4) hold; (2) is the coin-flip and the most informative single
number in the study; (3) holds weakly because parameter growth is expected to be modest.

**What would falsify it.** Any of the four failing. In particular, if `no_feedback` tracks
`full`, the honest conclusion is that evolution produced better priors and **not**
adaptability, and that is the reported result.

---

## H2 — Class A performance plateaus while Class C adaptation continues to improve

The divergence, not either curve alone, is the signature of selection acting on the
machinery of adaptation rather than on memorised task structure.

**Test.** Difference in per-seed slopes over the final third of the run, bootstrap CI.

**Prediction.** Weak support at pilot scale. Class A is unlikely to fully plateau within
the pilot's generation budget.

---

## H3 — Architectures grow spontaneously under an ecological compute budget

**Test.** Δ ln(params) per generation, selected arm minus neutral arm, matched generation
count, bootstrap CI over seeds.

**Confirmed only if all four hold:**

1. selected − neutral > 0 with a CI excluding 0;
2. the **minimum** of the size distribution moves, not only mean and max;
3. displaced founders do **not** regress toward the bulk;
4. effective parameters grow, not only raw parameters.

**Prediction.** (1) is marginal at pilot scale; (2) fails; (3) fails. The predicted honest
outcome is therefore **passive diffusion off the lower bound, not a driven trend**. The
measured neutral drift (`DESIGN.md §3`) is the reason this prediction is made in advance
rather than discovered afterwards.

---

## H4 — The heritable growth-bias gene rises above 0.5

`p_growth_bias` is directly heritable and mutates on the logit scale, so under drift its
median stays at 0.5 (asserted in `tests/test_neutrality.py`). A sustained rise is the
sharpest available signature of selection *for* growth, because it does not require
inferring intent from a noisy size trajectory.

**Test.** Median `p_growth_bias` in the final generation, selected minus neutral, bootstrap
CI over seeds.

**Prediction.** No significant departure from 0.5 at pilot scale.

---

## H5 — Adaptation migrates from weights to context as environments become less stable

Across sweep S2 (`n_variants` ∈ {1, 2, 4, 16, 0}), the share of adaptation attributable to
the feedback channel — (`full` − `no_feedback`) / `full` — increases as `n_variants` grows.

**Prediction.** Supported. This is the most likely positive result in the study and the one
with the clearest mechanism.

---

## H6 — Evolved mutation rates fall below the long-run optimum

Following Clune et al. (2008). Compare evolved `p_structural` against a sweep of fixed
values, scored on end-of-run Class C adaptation.

**Prediction.** Supported: evolved rates sit below the value that maximises long-run
adaptability. This is a *negative* result for naive "evolvability evolves" optimism and is
reported as such.

---

## Stopping and reporting rules

* Generation count and seed count are fixed above and are not extended after inspecting
  results.
* C-test is opened **once**, after all runs complete. C-dev may be inspected freely during
  engineering; no C-dev number is reported as a headline result.
* All pre-registered hypotheses are reported whether or not they are supported, with effect
  sizes and CIs, including the ones predicted to fail.
* Any deviation from this document is recorded in `results/DEVIATIONS.md` with its reason
  and its date.
