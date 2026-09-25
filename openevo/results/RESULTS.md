# Pilot results

One seed, 40 generations, 6 islands × 16 organisms, ~3,700 births, 289 GFLOP accounted,
20 minutes wall-clock on 4 CPU cores. Run against a matched **neutral arm** (identical
operators, demography, world sampling and compute budget; fitness replaced by a uniform
random draw).

**Nothing here is a scientific claim.** `PREREGISTRATION.md` requires ≥10 seeds and the
unit of analysis is the seed, not the organism. The pilot's job was to validate the
machinery, and it did that mainly by finding three things wrong with it. What follows is
reported in the pre-registered format because that is the discipline, not because one seed
settles anything.

Reproduce: `python scripts/run_evolution.py --config configs/pilot.json` and
`--config configs/neutral.json`, then
`python scripts/report.py results/pilot/run.db --neutral results/neutral/run.db --figures`.

---

## Headline: under an ecological compute budget, this system evolves *compression*

| | generation 1 | generation 40 |
|---|---|---|
| median parameters | 4,910 | **1,055** |
| minimum parameters | 863 | 735 |
| effective parameters | 2,210 | 847 |
| **effective fraction** | 45.0 % | **86.2 %** |
| mean score | 0.021 | 0.192 |
| stack activity | 0.018 | 0.110 |
| `p_growth_bias` gene | 0.500 | 0.435 |
| `p_structural` gene | 0.150 | 0.062 |
| temperature gene | 1.000 | 0.286 |

Architectures shrank by a factor of 4.7 while performance rose nine-fold and the fraction
of capacity doing measurable work nearly doubled. Organisms became **smaller and denser**,
and used their transformer more (stack activity is the total-variation change from
ablating every block output path at once — it rose 6×).

Against the null: the selected arm drifts at **−0.0484** nats/generation in mean
ln(parameters), the neutral arm at **−0.0112**. Selected minus neutral is **−0.0372
nats/generation**. Selection is not merely failing to favour growth; it is actively
driving size down faster than drift does.

---

## Pre-registered criteria

### H3 — architectures grow spontaneously: **not met, in the opposite direction**

| # | criterion | value | verdict |
|---|---|---|---|
| 1 | selected − neutral drift > 0 | −0.0372 nats/gen | not met |
| 2 | distribution minimum moves up | −4.93/gen | not met |
| 3 | displaced founders do not regress | see caveat below | inconclusive |
| 4 | effective parameters grow | −18.4/gen | not met |

### H4 — the growth-bias gene rises above 0.5: **not met**

0.500 → 0.435. This gene mutates on the logit scale and provably stays at 0.5 under drift
(`tests/test_neutrality.py::test_growth_bias_gene_drifts_symmetrically_about_half`), so a
departure in either direction is selection rather than noise. Evolution reduced its own
tendency to grow. Consistently, the two most-used operators were `contract_ffn` (146
births) and `narrow` (134), and they carried the highest mean scores.

### H1 — adaptation on novel causal structure improves: **slopes met, level ambiguous**

Slope of adaptation AUC against generation on Class C-dev:

| # | criterion | value | verdict |
|---|---|---|---|
| 1 | `full` slope > 0 | +0.00146 | met |
| 2 | `full` > `no_feedback` (not a reactive prior) | +0.00146 vs +0.00085 | met |
| 3 | `full` > `capacity_matched` (not capacity) | +0.00146 vs −0.00005 | met |
| 4 | `fixed_hparams` > 0 (not hyperparameter tuning) | +0.00028 | met |

All four slope criteria are met. But at the **final generation** the levels are a wash:
`full` +0.1412 against `no_feedback` +0.1434, a difference of −0.0023. On Class B
(recombinations of familiar mechanisms) the same comparison is clearly positive
(+0.1521 vs +0.1321, difference +0.0200).

The honest reading is that the *rate* of improvement on alien worlds is partly
feedback-attributable, while the *endpoint* is not distinguishable from a reactive policy.
Taken at face value that would say in-context machinery generalises across recombinations
of familiar mechanisms but not to genuinely novel causal structure — which is a plausible
and interesting claim, and exactly the kind of thing that needs ten seeds and more
generations before anyone believes it. The slope/level disagreement is the first thing the
full experiment should resolve.

### H6 — evolved mutation rates fall below the long-run optimum: **directionally supported**

`p_structural` collapsed 0.150 → 0.062 and exploration temperature 1.000 → 0.286, against
a neutral arm where temperature drifted *upward* to 1.22. This is the outcome
[Clune et al. (2008)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2527516/) predict —
selection systematically evolves mutation rates below the value that maximises long-term
adaptation. Testing it properly means comparing evolved rates against a sweep of fixed
ones (condition A4 plus a rate sweep), which the pilot does not do.

---

## Why compression, and what it means for the design

This is the predicted consequence of `DESIGN.md §5`, arriving earlier than expected. Under
a **fixed** environment difficulty distribution with compute as a finite resource, the
optimal architecture size is bounded, and once the population finds it there is no reason
to grow. A smaller organism is evaluated on more worlds for the same budget and leaves
more offspring. The compute budget worked exactly as intended — it just produced the
boring equilibrium rather than an arms race.

That is an argument for the design decision to make the world grammar compositionally
unbounded from phase 1, and a stronger argument for bringing environment co-evolution
forward. The right next experiment is **sweep S1** (compute abundance × 7) to find where,
if anywhere, the growth threshold sits, rather than more seeds at one budget.

---

## Caveats that matter

* **One seed.** Slopes of ~1e-3 per generation over 8 probe points, on 6-world suites.
  Several of these numbers will not survive replication.
* **The subclade test is confounded as implemented.** Lineages founded at 40K stayed at
  ~39K while small-founded lineages sat at ~3K, which naively reads as "displaced founders
  do not regress" — evidence for a driven trend. But islands hold separate compute budgets
  and migrate only every 10 generations, so large-founded lineages were largely not
  competing with small ones. The test needs displaced founders seeded *within* islands, or
  much higher migration. Reported as inconclusive rather than as support.
* **Class C-test has never been opened.** Everything above is C-dev, which has been looked
  at during engineering and is therefore contaminated by exactly the researcher-degrees-of-freedom
  channel `DESIGN.md §2` describes. No C-dev number is a headline result.
* **40 generations is short** for a claim about plateaus. Class A performance had not
  plateaued, so H2 could not be evaluated at all.

---

## Provenance

| | |
|---|---|
| config hash | `48dd5dd1f5331aa4` |
| world-family seal | `e50ad237af389d10913d8e0b9f027382baa0c01988225971191b8bd138a36b92` |
| family split | A 231 / B 77 / C-dev 1430 / C-test 1430 |
| git commit | `8e8064590cfd` |
| C-test opened | no |

An earlier run of this same config is kept at `results/pilot_inert_stack/`. It produced a
similarly coherent-looking set of curves while the transformer did nothing at all. It is
worth reading before trusting anything here.
