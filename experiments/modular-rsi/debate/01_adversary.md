# Adversarial review: Level 1b (stacked frozen community) and the program as a whole

## 1. Will 1b close the gap to the monolith? Probably not, and the current data already points that way

- **The gap is large, and it gets wider as the community grows.** Data from `lang/results/grow_rsi_s*.json` and `monolith_K*.json`:
  - The RSI community at 10.9 KB reaches 2.87 bpc. A monolith reaches the same bpc at about **1.3 KB** (K4: 2.87 bpc with 8k steps). That makes the community roughly **8x less byte-efficient**.
  - At 1.36 KB (K=4), the community scores 3.19 bpc. That barely beats the 756-entry **bigram table** (3.24, `probe_size.json`).
  - Over K=8 to K=32, the community gains about 0.115 bpc per doubling of size. The monolith gains about 0.2 per doubling. Gains per added module have fallen to **0.003–0.005 bpc** by the last cycles.
  - Stacking changes the composition rule. It does not change the diminishing per-module return of frozen greedy growth.
- **Frozen greedy layers are a known weak regime.** Cascade-correlation (1990), greedy layer-wise training and progressive nets all show the same thing: features frozen while they were optimized for a *different* objective are poor substrates for later layers. Here every old module was trained to predict the next char directly through its tied embedding/readout. Its hidden state is an output, not a feature basis. Progressive nets also need lateral adapters whose cost grows with depth.
- **Width bottleneck.** If the shared residual stream keeps d=4 (`grow_spell.py` default `d=4, mlp=8`), 32 stacked modules make a 4-wide, 32-deep net. The 10.8 KB monolith has d≈28. A 4-dim stream cannot carry the information spelling needs, no matter how deep the stack is. If d grows instead, 1b turns into progressive nets with growing lateral cost.
- **The monolith comparison is miscalibrated in both directions.** The monolith is *one layer* with a 16-char context. A stacked community gets depth the baseline lacks. So 1b could "beat" this monolith while still losing badly to a jointly trained 2–4-layer model of the same bytes. The monolith's learning rate is also untuned (K32: 8k steps scores worse than 12.8k), so the baseline is weak.
- **Compute.** Each candidate step runs all K frozen modules in sequence, so total cost is O(K²) unless hidden states are cached on a fixed corpus. Level 1 already used 32×12×400 = 153.6k candidate-steps against the monolith's 12.8k.

## 2. Is this the right level? Is it RSI at all?

- **Selection is doing nothing measurable.** In the final cycles, the 12 candidates' bpc spread is about **0.003** (for example `[2.911, 2.909, 2.911, 2.908, 2.91]`). That is within noise of a single fixed 8,192-char validation batch. The evolved σ collapsed to **0.03** in seed 0, which is effectively "copy the parent and run Adam". The language track is **forward-stagewise boosting of logits with warm starts**, so it is gradient training in disguise. Lineage steering, polymerization and dissociation are absent from `lang/` entirely.
- **The "RSI beats fresh" result is confounded.** The rsi mode also evolves the learning rate while random mode fixes it at 0.03. The duplicate-init advantage can be explained by warm start alone, with no evolution involved. No ablation separates these effects (M=1, σ=0, no credit).
- **Real-word rate is a weak metric.**
  - It comes from one 1,500-char sample at temperature 0.8, against a dictionary that includes the corpus's own 3,162 words.
  - It counts two-letter words, which inflates it ("he", "at", "be").
  - It is non-monotonic in size (probe: 236 B scores 0.23, 172 B scores 0.33).
  - Use held-out bpc, plus word-level metrics on many samples with confidence intervals.
- **The corpus is too small for the stated goal.** Alice has 137k chars and 3,162 word types, with a 13.7k-char validation set. "Words to sentences" will run into memorization and overfitting long before any module count matters. GitHub raw is reachable, so a larger corpus (for example tiny-shakespeare at 1.1 MB, or larger GitHub-hosted text) is cheap.

## 3. Where the program may be fooling itself

- **Exp 4's exponential is close to vacuous.** With V=8, every unit or macro is a map on 8 symbols per context (`hierarchy.py` composes tables). A 12,717-step program reduces to one of at most 8⁸ maps. "Problem length" grows geometrically because the curriculum concatenates problems; the *function class* does not grow. The macro accuracy figures are the table-composition shortcut, checked by execution only up to 1,771 steps.
- **"Recursive" is not shown.**
  - Exp 2: inheriting the improver *hurt* (172.5 vs 146.5).
  - Exp 5: the gain arrives once and does not compound (about 32 generations per skill, then about 30).
  - Mutation alone cannot learn lookup ("What went wrong" #1); gradient descent does the learning.
  - What remains is PBT-style hyperparameter tuning plus transfer by duplication. Both are useful, but neither is recursive self-improvement.
- **"Smarter as it grows" is contradicted on the only real task.** Each added language module is *less* useful than the last, and the whole community is far less efficient than one model.

## 4. A hostile expert's single biggest weakness

The learning is done by Adam, and the exponentials are produced by the curriculum. On the one non-synthetic task, the modular system loses about 8x in bytes to a plain one-layer model. Nothing shows the improvement process improving itself in a compounding way. "Modular RSI toward 1T parameters" is extrapolated from evidence that, on real data, points the other way.

## Most informative next experiment (cheap, falsifying)

Run one comparison at a fixed byte budget and at matched FLOPs, on a corpus of at least 1 MB, with held-out bpc and 3+ seeds and confidence intervals:
- (a) stacked frozen growth;
- (b) the *same* stacked architecture trained jointly;
- (c) stacked growth with brief unfreezing;
- (d) a tuned 2–4-layer monolith.

Also ablate the RSI machinery (M=1, σ=0, fixed lr, no credit). **Pre-register** what counts as success, for example within 0.1 bpc of (b) with RSI components contributing more than seed noise. If (a) is about equal to (b), modular growth has real value. If the ablations are about equal to full RSI, drop the RSI framing for this track.

## Ranked objections and what would refute each

1. **Frozen greedy growth is byte-inefficient (8x gap, widening).** *Refuted by:* stacked frozen growth within about 0.1 bpc of a jointly trained, same-architecture model at 11 KB and beyond, with the per-doubling slope matching the monolith's.
2. **The "RSI" components contribute nothing beyond warm-start plus Adam.** *Refuted by:* full RSI beating the M=1 / σ=0 / fixed-lr / no-credit ablations by much more than seed variance (≥3 seeds), with candidate spread clearly above validation noise.
3. **Exp 4's exponential is curriculum-made in an 8-symbol function space.** *Refuted by:* growth in a domain where composed problems need representations not already present, such as larger V, non-closed operations, or new contexts, with search cost staying flat.
4. **No compounding: the improver does not improve its own improvement.** *Refuted by:* cost per new skill falling steadily with library size over a long, diverse skill stream (for example 50+ skills), beyond the one-time drop that duplication gives.
5. **The metric and corpus are too weak to support the language claims.** *Refuted by:* the same conclusions holding on a corpus of at least 1 MB with held-out bpc and multi-sample word metrics with confidence intervals, and a baseline monolith that is properly tuned (lr, depth).
