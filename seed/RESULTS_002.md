# RESULTS_002.md — why growth doesn't compound

Three follow-ups to the Experiment 001 NO-GO, run to answer one question: **is
recursive growth — seed to tree, each cycle building on the last — viable?**

Short answer: **not by creating capacity. Possibly by expressing it.**

29 runs. Arms A and B are decisive. Arm C is suggestive and not established.

---

## Arm A — the decisive one: integration decays with birth time

Experiment 001 could not separate *"the unit had more time"* from *"the unit was
born earlier"*, because with a fixed budget `steps_alive = total − birth_step`
makes them the same variable.

Here the birth step is forced and the run then continues for a **fixed 2,400
further steps**, so every unit gets identical time to integrate regardless of
when it was born. Any remaining trend is earliness alone.

| birth step | unit's final ablation effect | model's primary | D3 |
|---|---|---|---|
| 100 | **4.819** | 0.788 | 0.399 |
| 400 | 1.215 | 0.804 | 0.429 |
| 800 | 0.817 | 0.796 | 0.398 |
| 1,600 | 0.176 | 0.805 | 0.422 |
| 2,400 | 0.253 | 0.793 | 0.384 |
| 3,200 | **0.061** | 0.806 | 0.423 |

`pearson(birth step, ablation effect) = −0.559` (n=18). A unit born at step 100
ends up **79× more load-bearing** than one born at step 3,200, on identical
training time.

Log-linear fit: **half-life ≈ 584 training steps** (R² = 0.52). Every ~584 steps
of delay halves how much the new capacity ends up doing.

**It is not a measurement artifact.** Over the same span the *original* units'
ablation effect goes the other way — 75.5 → 91.7 (1.21×) — while the new unit's
share of total functional weight collapses from 6.0% to 0.066%. The decay is
specific to newly added capacity.

Every growth event was exactly function-preserving (`max dlogit = 0.0`).

---

## Arm B — recursion converges, it does not amplify

Double budget, up to 10 cycles, so later generations have room.

| group | primary | D3 | growth events | final params | steps |
|---|---|---|---|---|---|
| A FIXED SMALL | **0.8120** | 0.439 | 0 | 81k | 17,707 |
| D DEVELOPMENTAL | 0.8086 | 0.429 | 5.3 | **270k** | 7,763 |

At 3.3× the parameters and five growth cycles, D still does not beat the model
that never grew.

Per-generation contribution:

| generation | mean ablation effect |
|---|---|
| 1st | **0.597** |
| 2nd | 0.080 |
| 3rd | 0.065 |
| 4th | 0.041 |
| 5th | 0.032 |

The first generation carries **73%** of everything five generations of growth
contributed. This is the arithmetic of Arm A playing out: with a half-life of
584 steps and a 900-step cooldown, each generation should be `2^(−900/584)` ≈
0.34× the last, and the infinite series should converge to ≈1.5× generation one.
Observed: **1.37×**.

**That is the opposite of the seed-to-tree claim.** A tree's later growth
dominates its earlier growth — the trunk is a small fraction of mature biomass.
Here each generation contributes a third of the one before, and the total
converges. More cycles cannot fix this, because the limit is a constant.

---

## Arm C — staged *expression* is different from staged *creation*

A seed's genome is fixed. It does not acquire new DNA; it expresses a blueprint
it already holds. The analogue: allocate the **whole** architecture at step 0 —
never create anything mid-run — and switch units on over time with a gate that
ramps 0→1. Same architecture as D, same timing as D, same FLOPs. The only
difference is whether the capacity was **allocated at step 0 or created mid-run**.

n=5 seeds, identical world, identical 2.0e12 FLOPs:

| group | primary | sd | D3 | params | units created mid-run |
|---|---|---|---|---|---|
| A FIXED SMALL | 0.7960 | 0.053 | 0.393 | 81k | 0 |
| C GROWTH ONLY | 0.7934 | 0.050 | 0.387 | 181k | 3.8 |
| D DEVELOPMENTAL | 0.7944 | 0.051 | 0.390 | 164k | 3.2 |
| **S STAGED EXPRESSION** | **0.8159** | 0.062 | **0.455** | 164k | **0** |
| B FIXED LARGE | **0.8335** | **0.014** | **0.513** | 164k | 0 |

| comparison | primary | 95% CI | wins | D3 |
|---|---|---|---|---|
| S − D | +0.0216 | [−0.0113, +0.0510] | 4/5 | +0.065 |
| S − C | +0.0225 | [−0.0076, +0.0509] | 4/5 | +0.069 |
| S − A | +0.0200 | [−0.0164, +0.0538] | 4/5 | +0.063 |
| S − B | −0.0176 | [−0.0707, +0.0452] | 2/5 | −0.058 |

S lands between the grown models and FIXED LARGE, and it beats every
created-mid-run variant on 4 of 5 seeds in three independent comparisons.

**But no CI excludes zero.** Seed 1 dissents in every comparison. At n=5 with
sd = 0.06 this is *suggestive, not established*, and I am not going to call it a
result. What it is: the first developmental variant in either experiment that
looks like it might be doing something, and the only one worth more seeds.

Note also B's variance: sd 0.014 against everyone else's ~0.05, and D3 in a tight
0.47–0.58 band while A swings 0.21–0.58. Having capacity from the start doesn't
just do better on average — it does so **reliably**.

---

## What this says about recursive growth

1. **Capacity has an integration deadline.** Its eventual functional weight decays
   with a half-life of ~584 steps from the moment the representation starts
   forming. This is measured with the time confound removed.
2. **That makes recursion self-defeating by construction.** Generation N+1 is
   necessarily born later than generation N, so it integrates worse. The series
   converges instead of compounding — confirmed directly at five generations.
3. **The problem is creation, not staging.** Capacity that existed from step 0
   and was merely switched on later looks better than capacity created mid-run,
   on 4/5 seeds. If that survives more seeds, the useful version of the seed
   metaphor is *expression of a pre-allocated blueprint*, not *growth of new
   structure* — and that is a materially different research programme.

## Honest limits

- Arm A's fit is R² = 0.52 on n=18; the 584-step half-life is an order-of-
  magnitude estimate, not a precise constant.
- Arm C is n=5 with every CI spanning zero.
- Same scale caveat as Experiment 001: 0.08M–0.27M parameters on 4 CPU cores.
  The 584-step half-life is a number for *this* model on *this* task; whether the
  deadline scales with model size, data, or LR schedule is unknown and is the
  obvious next question.
- Arm C confounds four things at once (init drawn at step 0 vs later, zero-init
  vs normal init, gate ramp vs LR warmup, optimizer continuity). Which one does
  the work is untested.

## Reproducing

```bash
python3 exp002.py --root results/exp002 --seeds 0,1,2
python3 analyze_exp002.py --root results/exp002
```
