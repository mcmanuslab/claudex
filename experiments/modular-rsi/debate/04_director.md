# Director: final position

## Response to the adversary's round 2, point by point

1. **B is Net2Net plus progressive growing, and the global anneal dissolves the modules.** Conceded in full. A positive result would rediscover known work. A per-cycle anneal also costs O(total size), which breaks "grow cheaply". I am dropping B as the primary.
2. **H1 is near-guaranteed, and the 0.05–0.15 bpc band is a loophole.** Conceded. Every threshold below has no dead band: an outcome is either success or kill.
3. **H3 is confounded by compute.** Conceded. The fair ablation is M=1 trained for M times as many steps, run with paired seeds and n≥5.
4. **C's characters per token grow by construction.** Conceded: it is an input, not a result. Only bits per character at equal bytes and FLOPs counts.
   - The vocabulary bytes dominate. This is fixable but must be counted: a factorized embedding (a 1,024×4 table plus a 4→16 projection) costs about 4 KB in int8.
   - Novelty is low (it is close to the Byte Latent Transformer and MegaByte). Agreed.
   - Growth saturates beyond words. Agreed. Word level is exactly the user's next stated step, so saturation beyond it does not disqualify C as a backup.
5. **The time estimate.** Conceded. 6–10 hours including implementation is realistic. Both plans below are scoped to fit that.
6. **The adversary's decisive test.** I adopt it, with two corrections:
   - "No forgetting" holds by construction for frozen modules. It is a bookkeeping check, not evidence.
   - 10+ domains will not fit in a day once implementation is included. 8 domains with shuffled orders fit, and still allow a slope fit over k=3..8.

## Primary plan: the compounding continual-acquisition test

**Question.** Does the modular system get cheaper to improve as it grows? And is the lineage-steered mutation loop the cause?

**Domains.** 8 text domains of 100–200 KB each from GitHub raw, chosen and frozen before the run: Shakespeare, Alice, Python, JavaScript, LaTeX, Markdown READMEs, legal or constitutional text, and one non-English text. All use a shared 96-symbol ASCII alphabet.

**Modular system**

- **Base.** A frozen 2 KB char base module with d=16.
- **Per domain.** The system adds one 1 KB module that reads the library through learned per-module scalar gates. This is a polymer: the new module plus selected old modules.
- **Variation.** M=4 candidates are made by mutating duplicates of library modules. The parents are chosen by lineage credit, which goes to modules whose offspring produced gains on earlier domains. One candidate is fresh.
- **Training and selection.** Only the new module and its gates train, so compute per domain stays constant as the library grows. The winner is selected on domain-held-out bpc.

**Metric.** The cost of domain k is the FLOPs needed to reach that domain's target bpc. Each target is pre-set as the bpc a fresh 1 KB module reaches in 6,000 steps on that domain. All FLOPs are counted, including every candidate, every evaluation and the forward passes through the library.

**Controls.** All controls are FLOP-matched and use 5 seeds, where each seed is a different domain order. All comparisons are paired by seed.

- **(i) Fresh:** no library.
- **(ii) Outer-loop ablation:** M=1, inherit from the latest module, no credit, no mutation, and 4x the steps.
- **(iii) Monolith:** continually fine-tuned with 10% replay, with matched final bytes and total FLOPs.

**Pre-registered outcomes**

| Claim | Success | Kill |
|---|---|---|
| Compounding | The slope of log(cost_k / fresh cost_k) against k, over k=3..8, is negative with a 95% CI that excludes 0 | Otherwise. A one-time transfer gain at k=2 does not count. |
| RSI loop | Full beats (ii) on total FLOPs-to-target in a paired test (p<0.05 with 5 seeds) | Otherwise. Drop the label "RSI" and call it "modular transfer". |
| Modularity is worth it | Final mean bpc across all 8 domains is at most the monolith's plus 0.05 | Otherwise. |

The first two tests are not guaranteed by construction. The third could go either way, because the monolith with replay is a strong baseline.

**Compute.** About 4 hours of implementation plus 4–5 hours of compute: 4 arms × 5 seeds × 8 domains of tiny modules. If it runs late, cut to 6 domains, which still leaves a 4-point slope.

**How it serves the user's goals**

- **Tested:** tiny modules; mutation with lineage-steered RSI, against a fair ablation; polymers or communities formed by reuse; "smarter as it grows", operationalized as cheaper to improve as it grows; and an honest version of "exponential", meaning compounding returns.
- **Not tested:**
  - chars → words → sentences;
  - exponential capability growth;
  - scaling toward 1B parameters;
  - modules of about 100 bytes;
  - binding and dissociation in any strong sense, since modules are only reused, never pulled apart and recombined freely.

## Backup plan: C at word level only, mechanism-isolated

**Setup.** A level-1 char module, then token merges, then a fixed-size level-2 module on the token stream. Budgets are 22 KB and 44 KB, with the factorized vocabulary counted. Bpc is converted to bits per character. The corpus is tiny-shakespeare.

**Controls.** Both controls use matched bytes and FLOPs, 5 paired seeds.

- **(a)** A flat char monolith, with a learning-rate sweep.
- **(b)** The same hierarchy using plain BPE merges, which isolates whether merges driven by the module ("polymers become words") matter at all.

**Pre-registered outcomes**

| Claim | Success | Kill |
|---|---|---|
| Hierarchy beats flat | Beats (a) by more than 2 paired SDs at 22 KB or 44 KB | No advantage at either budget |
| Module-driven merges | At least as good as (b) | Worse than (b), which means the "polymer" mechanism adds nothing beyond BPE |

**Compute.** About 6–8 hours in total.

**How it serves the user's goals**

- **Tested:** the chars → words step, with tiny modules at each level.
- **Not tested:** RSI or lineage; growth beyond words (likely to saturate); communities.

## Bottom line

The primary plan tests the thesis: does it compound, and does the RSI loop cause it. The backup tests the language vision. Neither tests the 1B-parameter or exponential-capability claims, and we should tell the user so plainly.
