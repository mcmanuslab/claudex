# Director's response: the option space and a recommendation

## 1. Concessions, and where I push back

**Concessions.** I checked the adversary's numbers in `lang/results` and they are correct.

- **Byte efficiency.** The K=4 monolith reaches 2.87 bpc with 1,288 params. The community needs 10,880 bytes to reach the same 2.87, which makes it about 8x less efficient.
- **Selection is inert.** The candidate spread (about 0.003 bpc) is noise. "RSI beats fresh" is confounded with warm start and with evolving the learning rate.
- **The language track is not RSI.** It is boosting with warm starts.
- **The metric is weak.** Real-word rate taken from a single sample is a poor measure.
- **Exp 4 does not grow capability.** Its exponential is a curriculum over a function class that stays the same size (8 symbols).

The program has no evidence yet that the improver improves itself, or that the system gets smarter as it grows on real data. I accept that.

**Pushback.**

- **(a) "Adam does the learning" is not disqualifying.** The user never required learning without gradients. Lamarckian evolution with gradient learning during each lifetime is legitimate. The real question is whether the outer loop adds anything beyond the ablations, so I adopt the adversary's ablation test.
- **(b) Frozen growth is a design choice, not the goal.** The user's cycle includes *dissociate*. Modules that bind, briefly co-adapt and then dissociate are faithful to that cycle. Freezing everything forever is a strawman we built ourselves.
- **(c) The baseline is weak, but not uniformly so.** The monolith got 8k steps. Each community module got only 400 steps along the winning lineage, so the 8x gap partly reflects compute per parameter. The new comparison must be matched in both bytes and FLOPs, as the adversary says.
- **(d) "Exponential" needs a fair operational target.** On language, an honest target is that the context or unit length a fixed-size module handles grows geometrically with level, at constant cost per level. That target can be tested; "bpc falls exponentially" is impossible.

## 2. Option space

All six options share these rules:

- **Corpus.** tiny-shakespeare (1.1 MB, GitHub raw), with Alice kept as a secondary corpus.
- **Metric.** Held-out bpc on a fixed 200k-character set, 3 seeds, 95% confidence intervals.
- **Accounting.** Bytes are counted including embeddings and vocabulary.

| # | Direction | Core hypothesis | Serves the goal by | Adversary's reply | Cost (4 CPU cores) | Success | Kill |
|---|---|---|---|---|---|---|---|
| **A** | **Level 1b**: frozen stacked residual (stream d=16, cached hidden states) | Depth lets new modules build on old features | Incremental, modular growth | Frozen greedy features make a poor basis, and returns per module diminish | Low with caching (about 1 h per seed) | Within 0.10 bpc of the same architecture trained jointly at 11 KB | Gap above 0.25 bpc, or gap widens with size |
| **B** | **Bind, anneal, dissociate**: function-preserving growth (Net2Net split or duplication), then a short global co-training "anneal" of about 200 steps per cycle. Evolution chooses which module to split and the anneal length. | Growth plus short joint repair matches joint training at a fraction of the search cost | Literal polymer cycle; size grows; the result is useful | "That's just progressive training" | Low (about 1 h per seed) | Within 0.05 bpc of joint training at every budget, with slope per size-doubling at least 0.8 times the monolith's | Worse than joint training by more than 0.15 bpc at 11 KB |
| **C** | **"Polymers become words"**: a tiny char module is trained; segments it predicts confidently (low surprisal) are merged into new tokens; the next-level tiny module runs on the token stream; repeat for each level | Fixed-size modules plus a growing vocabulary make the characters covered per token, and so the effective context, grow geometrically per level | Characters, then words, then phrases; the macro-freezing result carried over to language; the most faithful "exponential" | "That's BPE; the vocabulary table eats your bytes" | Medium (about 3–4 h) | At equal total bytes including vocabulary, beats the flat monolith by more than 2 seed SDs; characters per token grows at least 1.6x per level with roughly constant cost per level | No bpc advantage after 2 levels, or the vocabulary bytes dominate |
| **D** | **Word-routed mixture of tiny experts**: a shared tiny backbone plus N tiny experts, chosen by a router at word boundaries | The number of parameters grows while compute per token stays constant; this is the only credible route to 1B+ parameters | Scalable modularity; experts specialize | MoE routing collapses at tiny scale | Medium | Bpc improves monotonically as experts go from 1 to 64, and beats a dense model with the same active parameters | Router collapse (over 80% of tokens sent to one expert), or no gain past 8 experts |
| **E** | **Iterated amplify-and-distill**: a community (polymer) is distilled into a monomer, which seeds the next community; repeat | Each generation's monomer outperforms one trained directly at the same size | "Smarter as it grows" in a literal, compounding way | Distillation gains at tiny scale are small | Low | Monomer from cycle k beats one trained on data alone by more than 2 SD, and the advantage grows with k | No gain at k=2 |
| **F** | **Evolve the improver**: learned update rules and optimizer genes, tested on how well they transfer across a stream of more than 50 skills | Cost per skill falls with library size; this would be true recursion | It is the actual RSI claim | Meta-learning is too costly and noisy for one day | High | Cost per skill falls over 50 skills with a significant negative slope | Slope near zero after the first duplication gain |

## 3. Ranking: probability of a defensible positive result in about one day

1. **B, Bind-anneal-dissociate (about 60%).** Growth-then-repair has strong precedent and is cheap. It also directly refutes objection 1.
2. **C, Polymers become words (about 35%).** This is the highest-value result for the user's vision and the most original.
3. **E, Distillation cycle (about 30%).**
4. **D, Routed experts (about 25%).**
5. **A, Level 1b alone (about 15%).**
6. **F, Evolving the improver (under 10% in one day).** It remains the long-term test of the RSI claim.

**Recommendation**

- **Primary: B, run as the adversary's falsifying benchmark with A included as an arm.** It costs the same as running 1b alone, and it either rescues modular growth or kills it cleanly.
- **Backup / second half of the day: C**, which runs on whichever growth rule wins.
- **Framing for the user.** Present the RSI framing as on probation. If the ablations match full RSI, rename the language track "modular growth".

## 4. First experiment (pre-registered before running)

**Data.** tiny-shakespeare, split 90/10. Bpc is measured on a fixed 200k-character held-out slice. Uncertainty comes from bootstrap over evaluation windows plus 3 seeds (0, 1, 2).

**Byte budgets.** 1.3 KB, 5.4 KB, 11 KB and 22 KB, all int8.

**Arms**

- **(a) Level 1b.** Frozen stacked residual blocks on a d=16 stream. The embedding is shared and frozen after module 1. Frozen activations are cached.
- **(b) Joint.** The identical final architecture, trained jointly from scratch with total FLOPs matched.
- **(c) Bind-anneal.** Growth as in (a), plus function-preserving initialization of each new module (a noisy duplicate of its best parent, with output scaled to zero), plus a 200-step joint anneal of all modules after each cycle.
- **(d) Tuned monolith.** 1–2 layers, with a sweep over 3 learning rates and 2 depths; report the best.

**RSI ablations, run on (c)**

- **Full:** M=8 candidates, evolved lr and σ, lineage credit.
- **Ablated:** M=1, σ=0, fixed lr, no credit.
- **Extra arm:** M=8 with fixed lr.

**What to log**

- Bpc against bytes and against FLOPs.
- Candidate spread against the validation-noise SD.
- Slope per size-doubling.
- Word metrics: tokens of 3 or more letters, 20 samples, with confidence intervals. These are secondary.

**Pre-registered outcomes**

| Claim | Success | Kill |
|---|---|---|
| Modular growth (H1) | (c) within 0.05 bpc of (b) at all budgets, and slope at least 0.8 times (d)'s | (c) more than 0.15 bpc worse than (b) at 11 KB. Then abandon frozen or incremental growth for language and move to D or C with joint training. |
| Level 1b (H2) | (a) within 0.10 bpc of (b) | Gap above 0.25 bpc, or the gap widens from 5.4 to 22 KB. Then drop 1b. |
| RSI (H3) | Full beats ablated by more than 2 pooled SDs, and candidate spread is more than 3 times the validation noise | Otherwise, drop the RSI label for the language track. |

**Compute.** About 4 arms x 4 budgets x 3 seeds, plus ablations, comes to roughly 8–10 CPU-hours across 4 cores, or about 3 hours of wall-clock time. That leaves room for C.

**C follow-up (same day).** Freeze the level-1 char model. Mine merges from its confident spans to reach a vocabulary of 256 and then 1,024. Train a fixed-size level-2 module on the token stream. Report total bytes including the vocabulary table, characters per token per level, and bpc against the flat monolith at equal bytes.
