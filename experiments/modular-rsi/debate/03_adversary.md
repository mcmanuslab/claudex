# Adversary, round 2

## Concessions
The director is right on several points:
- Gradient learning inside each lifetime is legitimate. The test that matters is whether the outer loop beats ablations.
- Freezing modules forever was our own strawman; the user's cycle includes "dissociate".
- Matching FLOPs as well as bytes, pre-registering outcomes, using a 1 MB corpus with bootstrap confidence intervals, and putting RSI "on probation" are all correct.
- The operational exponential (unit length per level at constant cost) is the right kind of target.

## 1. B cannot be told apart from Net2Net or progressive growing
B consists of:
- function-preserving duplicate or split (Net2Net, 2015);
- growth by stacking (progressive or gradual stacking, the BERT-stacking and MSG/LiGO line of work);
- a global joint anneal;
- a population search over the anneal length (PBT).

**A positive result would re-derive a known result.** Grown-then-annealed networks match from-scratch training, often at lower FLOPs. That says "growth is cheap", not "modular RSI works".

**After a global anneal, the modules stop being modules.** Every module has co-adapted with every other. "Dissociate" is just a name unless a module is taken out and reused somewhere else, and nothing in the protocol tests that.

**The anneal touches all parameters every cycle.** Cost per cycle therefore grows with total size, which removes the "start tiny, grow cheaply" property the user cares about.

**To make B bear on the thesis, add one of two arms:**
- **Recombination:** dissociated modules, recombined into a community for a held-out domain, beat fresh modules at matched FLOPs.
- **Locality:** a local anneal that touches only the new module plus its k neighbours stays within threshold.

## 2. Thresholds and probabilities
**H1 is close to guaranteed, which makes it uninformative.** "(c) within 0.05 bpc of (b)" is the expected Net2Net outcome. My estimate is 75–85%, not 60%. The anneal FLOPs and the search FLOPs of all M candidates must also count toward (c)'s budget, or the comparison is rigged. The band between 0.05 and 0.15 bpc counts as neither success nor kill, and a "partial support" story will get told from it.

**H3 is confounded by compute.** M=8 against M=1 gives the full arm 8x the candidate FLOPs. The ablation should be M=1 trained for 8x the steps. Without that, "full beats ablated" just means "more compute wins". With 3 seeds, the pooled SD is itself very uncertain, so use 5 seeds or a paired test against the same seeds.

**C's 1.6x-per-level criterion holds by construction.** Merging tokens up to a vocabulary of 256 and then 1,024 guarantees that characters per token grow. That is an input to the experiment, not a result. Only the bpc-at-equal-bytes criterion is real.

**A's 15% is fair. E's kill threshold is good.**

## 3. Is C the better primary bet? More faithful, yes; less curriculum-built, no
C is the closest match to "chars → words → sentences", and it is the right *primary* bet for the user's vision. Its problems:

- **It already exists.** C is essentially BLT-style entropy patching (Byte Latent Transformer, 2024) or MegaByte. The novelty is small.
- **Vocabulary bytes dominate at this scale.** A 1,024-token table at d=16 is 16 KB, more than the whole 11 KB budget. Under the director's own accounting rule, C probably fails at 11 KB and could only work at 22 KB and above.
- **The exponential will saturate.** English gives about 5 characters per word. Beyond word level, phrase merges get sparse (Zipf), and a 1 MB corpus runs out of repeats by level 2–3. The geometric growth is built in by the merge schedule, then stalls. That is the same pattern as Exp 4.

## 4. Is about 3 hours of wall clock realistic? Only for compute, and only if everything works first time
**What makes it plausible.** The existing runs are cheap. The monoliths (up to 20.8k steps at d≈28) finished within minutes, according to the file times in `lang/results`.

**What the estimate leaves out:**
- **Nothing in the plan exists in code yet.** That includes:
  - the stacked residual model;
  - Net2Net initialization;
  - the cache;
  - the anneal;
  - FLOP accounting;
  - the monolith sweep;
  - bootstrap evaluation.
- **The cache has two problems.**
  - Positional embeddings plus random 16-char windows make cached hidden states depend on where a window starts, so the corpus must be cut into fixed chunks.
  - In arm (c), every anneal invalidates the whole cache, so it must be recomputed over 1 MB once per cycle.
- **The joint arm (b) cannot start early.** It needs the FLOP totals from (a) and (c), so the runs are partly sequential.
- **The run count is large.** Roughly 3 growth arms × 3 seeds, plus ablations, plus 6 monolith configs × 4 budgets × 3 seeds comes to about 100 runs.

**My estimate:** 6–10 hours including implementation and one debug rerun. C will not fit on the same day.

## 5. What I would accept as decisive evidence FOR the thesis
**The experiment: a compounding continual-acquisition test.**
- Take 10 or more distinct text domains from GitHub raw, for example Shakespeare, Alice, Python code, legal text, and French or German.
- The modular system adds and evolves tiny modules for each new domain, keeps old ones and reuses them by duplication or recombination.
- Compare it with a FLOP-matched monolith that is continually fine-tuned with replay.

**Pass criteria** (pre-registered, 5 seeds, FLOPs counted for everything):
1. **Compounding:** FLOPs to reach a target bpc on domain k fall with k, with a significant negative slope that continues *after* the first two domains.
2. **No forgetting:** the modular system loses no performance on old domains.
3. **Efficiency:** total FLOPs and bytes are no worse than the monolith's.
4. **The outer loop matters:** the evolutionary outer loop beats FLOP-matched ablations (M=1 with more steps, no lineage credit).

Passing (1) and (4) together would show that the system gets cheaper to improve as it grows, and that the selection mechanism causes it. Nothing about Net2Net or BPE predicts that result.

## Bottom line
- Run B only together with the recombination or local-anneal arm and with FLOP-fair ablations.
- Treat C as the user-facing bet, but only at 22 KB and above, and with vocabulary bytes counted.
- Make the continual-acquisition test the program's real thesis test.
