# Pre-registration: a domain where modules must compose

Written and committed before any main run (only smoke tests and timing runs before it).

## Question
When success requires composing learned skills in ways never seen in training, do
separate modules chained together beat monolithic models with the same parameters and
the same training budget? And does an evolutionary outer loop add anything over plain
gradient training?

## Domain
Program execution over relational contexts (`modrsi/world.py`, 4 tables, 12 operations:
A, B, C, D, their inverses, succ, pred, neg, triple; 8 symbols).

* **Input:** a context (four random permutation tables, 32 tagged pair tokens), a query
  symbol x, and an instruction sequence (op₁ … opₙ).
* **Output:** opₙ(…op₁(x)), the program applied to x.
* **Training:** programs of length 1–3 only, with random ops. Only the final answer is
  given; there are no intermediate labels.
* **Test:** fresh contexts at lengths 1, 2, 3, 4, 6, 8, 12 and 16 (1,024 examples each,
  fixed seed). Lengths ≥ 4 are never seen in training, so success there requires
  composition. Chance is 1/8.

## Systems (each about 46–49k parameters, same training steps and batch size)
1. **modular**: 12 blocks of 3,872 params, one per instruction. Step j runs the block
   named by opⱼ. Blocks pass a symbol distribution to each other: soft during training,
   argmax ("hard") at test.
2. **looped**: one shared block (8 heads × 16, MLP 470) plus an instruction embedding,
   applied once per step with the same interface. This is the strongest iterative
   monolithic competitor.
3. **transformer**: a standard 4-layer pre-LN transformer (d=32, 4 heads, FF 128) over
   [context tokens, instruction tokens with sinusoidal positions, query token], read out
   at the query.
4. **evolved**: the modular architecture trained by a population (6 individuals,
   PBT-style). Every 250 steps the bottom half is replaced by mutated copies of the top
   half (weights + noise, learning rate and mutation size inherited and mutated), selected
   on the validation set. It gets the same *total* gradient steps as systems 1–3
   including their learning-rate grid (3 × S), split across the population.

* **Learning rate:** for each of systems 1–3, the learning rate in {1e-3, 3e-3, 1e-2}
  with the best *mean* validation accuracy across the 5 seeds is chosen. Validation uses
  in-distribution programs (lengths 1–3) only; test lengths are never used for selection.
  The 5 seeds at that learning rate are reported. Training: 6,000 steps, batch 128.
* **Seeds:** 5.

## Hypotheses and decision rules (no in-between band)
* **H1, composition beats monoliths.** At length 12, hard interface, mean of 5 seeds:
  * **Pass** if `modular` ≥ `looped` + 0.15 and ≥ `transformer` + 0.15, and each paired
    difference exceeds 2 paired SDs.
  * **Kill** otherwise. If `modular` < `looped` + 0.05, then separate modules give no
    composition advantage over shared iteration.
* **H2, long composition works.** `modular` accuracy at length 16 ≥ 0.80. Otherwise fail.
* **H3, RSI loop.** **Pass** if `evolved` beats `modular` at length 12 by more than 2
  paired SDs. Otherwise fail.

All lengths are reported, with both soft and hard interfaces for the chained systems.
