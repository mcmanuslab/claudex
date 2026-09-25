# PRIOR_ART.md — what already exists, what we reuse, what is actually new

Survey completed 2026-09-25, before any implementation. Sources are papers and
public repositories; no code was copied verbatim, so no third-party license
attaches to this directory (see **Licensing** below).

---

## 1. Closest methods

### 1.1 Function-preserving growth (the mechanism we need for Gate 1)

| Work | Operators | Function preservation | Relevance |
|---|---|---|---|
| **Net2Net** (Chen et al., 2016) | `Net2WiderNet` (split/duplicate units), `Net2DeeperNet` (insert identity layer) | Exact for ReLU MLP/conv; **broken by normalization layers** and needs noise to break weight symmetry | Origin of the whole idea. Its failure mode under LayerNorm is exactly the trap we must avoid. |
| **bert2BERT** (Chen et al., 2022) | Net2Net-style width expansion + advanced knowledge init, for Transformers | Approximate | Shows Net2Net transfers to Transformers; ~45% pretraining cost saving. |
| **Staged Training** (Shen et al., ICML 2022) — [`allenai/staged-training`](https://github.com/allenai/staged-training) | Depth doubling (stacking), width doubling | Loss-preserving **and** *training-dynamics*-preserving | The key insight we adopt: loss preservation is **not sufficient**. If the grown model's loss-decrease *rate* collapses, growth is useless. They also handle **optimizer state** across the growth boundary — the detail most reimplementations get wrong. Depth growth gave better compute savings than width (37.0% vs 22.5% at 2x). |
| **MSG — Masked Structural Growth** (Yao et al., ICLR 2024) | All dims (width, depth, heads, FFN) | **Strictly** function-preserving, independent of new-weight init | Identifies the *LayerNorm dilemma*: when you widen `d_model`, LN statistics are computed over the new dims too, so the function changes even if new weights are zero. Their fix folds a mask into the LN statistics. Up to 2.2x speedup. |
| **G_stack / "Stacking Your Transformers"** (Du et al., NeurIPS 2024 Spotlight) — [`tongxuluo/prts`](https://github.com/tongxuluo/prts) | Systematic study of 4 atomic growth operators; depthwise stacking wins | Not strictly preserving | Best current empirical guidance on *which* operator to pick. Their answer: depthwise stacking. Validated to 7B / 300B tokens (54.6% speedup). |
| **Firefly Neural Architecture Descent** (Wu et al., NeurIPS 2020) | Split neuron / add neuron / add layer, chosen greedily from a "functional neighborhood" by Taylor approximation | Preserving at the growth instant | The closest published ancestor of our **overgrow → compete → prune** cycle: it instantiates a *set* of candidate growth directions and greedily selects. Difference: Firefly selects via a first-order Taylor surrogate *at the instant of growth*; we let candidates **actually train for a developmental window** and select on measured marginal contribution. |

### 1.2 Structured pruning (the "prune" half of the cycle)

- **Movement pruning / L0 gates** (Sanh et al. 2020; Louizos et al. 2018): learned masks over structures, removed when the gate closes.
- **First-order (Taylor) importance** (Molchanov et al.; "What matters in structured pruning of LLMs", ICLR): score a structure by the first-order loss change when it is zeroed. Cheap, gradient-based.
- **"When BERT Plays the Lottery"** (Prasanna et al. 2020): direct masking-ablation of heads/FFNs; showed importance estimates are noisy and that *random* structured pruning is a surprisingly strong baseline.

> The last point drives one of our controls: **any claim that competitive pruning works must be tested against random pruning of the same budget.**

### 1.3 The behavioural target (Question 4)

- **"Grokked Transformers are Implicit Reasoners"** (Wang et al., 2024): the canonical recipe for a synthetic world of *atomic facts* + *inferred facts* derived from **latent rules**, with explicit ID vs OOD splits. Their composition/comparison split design is the direct ancestor of our hidden world.
- **CLUTRR / COGS / CFQ**: standard systematic-generalization benchmarks with controlled train/test splits. Too large and too NLP-flavoured for a CPU go/no-go, but they define what a defensible split looks like.
- **"OOD generalization via composition"** (PNAS 2025): OOD generalization in Transformers arises from *composing* attention layers — evidence that depth, specifically, is the axis that buys compositional extrapolation. This motivates making `GROW_DEPTH` our primary operator.

---

## 2. What we reuse, and how

| Component | Source of the idea | Our implementation |
|---|---|---|
| Depth growth by inserting a block that is initially the identity | Net2Net `Net2DeeperNet`; Staged Training; G_stack | **Zero-initialised residual output projections.** A pre-norm block contributes `x + attn_out + mlp_out`; setting `W_o = 0` and `W_down = 0` makes the contribution *exactly* zero. |
| MLP width growth | Net2Net `Net2WiderNet`; MSG | **New `d_ff` units with zero-initialised down-projection columns.** |
| Avoiding the LayerNorm dilemma | MSG | **We side-step it rather than solve it: we never grow `d_model`.** Both our operators leave `d_model` fixed, so LN statistics are untouched and preservation is exact to floating-point, with no masking machinery. This is MSG's problem stated in reverse — it is a deliberate scope restriction, not a novelty claim. |
| Preserving *training dynamics*, not just loss | Staged Training | Explicit optimizer-state policy at every growth event (§ `growth.py`): new parameters get zeroed Adam moments **and a short LR warmup**, because zero-moment Adam parameters otherwise take near-maximal steps and destroy the parent function on the first update. |
| Candidate generation + greedy selection | Firefly | Candidates trained for a real developmental window, then scored by **masking ablation** on validation. |
| Importance by masking ablation | Molchanov; "BERT Plays the Lottery" | Δ(val loss) when a branch gate is set to 0. |
| Random pruning as the honest control | "BERT Plays the Lottery" | `D-RANDPRUNE` ablation. |
| Atomic facts + latent-rule-derived inferred facts, ID/OOD splits | Grokked Transformers | `world.py` — but with a *latent coordinate* generator so extrapolation distance is definable, plus an unpredictable-by-construction split (see below). |

**Nothing is copied.** `allenai/staged-training` is Apache-2.0, `tongxuluo/prts` is
Apache-2.0, MSG's release is Apache-2.0 — all permissive and compatible had we
vendored code. We did not: the operators are ~40 lines each and reimplementing
them against our own `model.py` is cheaper and more auditable than adapting a
Megatron/Fairseq-shaped codebase to a 4-core CPU box. This file is the
attribution.

---

## 3. What this experiment adds that the prior work does not have

Stated narrowly, because three of the four questions are **reproductions**, and
saying so is the point:

1. **Question 1 (function-preserving growth) is a reproduction.** If it fails,
   we have a bug, not a discovery. It is a Gate, not a result.
2. **Question 2 is a small delta.** Every paper above grows on a
   *hand-designed schedule* chosen by scaling-law analysis or by fiat
   (Staged Training explicitly derives the optimal schedule *offline* from
   scaling laws). We instead trigger growth from **online evidence of
   persistent failure**, and we test it against a *random schedule with the
   same parameter trajectory* — which, as far as this survey found, no growth
   paper reports. Growth papers compare against fixed-size baselines, not
   against randomly-timed growth. That control is where the claim lives.
3. **Question 3 is a genuine, if modest, delta.** Firefly selects candidates by
   a first-order surrogate at the growth instant. We let candidates
   *differentiate* under real training before selecting. Nobody in this
   literature reports the overgrow → differentiate → prune loop, repeatedly,
   with the FLOPs of the overgrowth window honestly charged to the method.
4. **Question 4 is the only part aiming at a new phenomenon.** The growth
   literature is evaluated on perplexity and standard downstream benchmarks —
   i.e. on *fitting* — never on *extrapolating to structure that was withheld
   by construction*. The hypothesis that developmental architecture improves
   extrapolation-per-FLOP is untested. It is also the claim most likely to be
   false, which is why the whole apparatus exists.

**Prior expectation, recorded before running anything:** Questions 1 and 2 will
likely pass (they are well-supported by the literature). Question 3 is a coin
flip — the "BERT Plays the Lottery" result that random structured pruning is
competitive is a direct threat. Question 4 most likely returns **NO-GO**: the
strongest alternative explanation for any gain is simply "more parameters, and
growth acts as an optimizer perturbation / implicit LR schedule." The
experiment is built to make that alternative easy to confirm.

---

## 4. Sources

- Chen, Goodfellow, Shlens. *Net2Net: Accelerating Learning via Knowledge Transfer.* ICLR 2016.
- Chen et al. *bert2BERT: Towards Reusable Pretrained Language Models.* ACL 2022.
- Shen, Walsh et al. *Staged Training for Transformer Language Models.* ICML 2022. <https://arxiv.org/abs/2203.06211> · <https://github.com/allenai/staged-training>
- Yao et al. *Masked Structural Growth for 2x Faster Language Model Pre-training.* ICLR 2024. <https://arxiv.org/abs/2305.02869>
- Du, Luo et al. *Stacking Your Transformers: A Closer Look at Model Growth for Efficient LLM Pre-Training.* NeurIPS 2024 Spotlight. <https://arxiv.org/abs/2405.15319> · <https://github.com/tongxuluo/prts>
- Wu, Liu et al. *Firefly Neural Architecture Descent: a General Approach for Growing Neural Networks.* NeurIPS 2020. <https://arxiv.org/abs/2102.08574>
- Wang et al. *Grokked Transformers are Implicit Reasoners.* NeurIPS 2024. <https://arxiv.org/abs/2405.15071>
- Prasanna, Rogers, Rumshisky. *When BERT Plays the Lottery, All Tickets Are Winning.* EMNLP 2020.
- Sanh, Wolf, Rush. *Movement Pruning.* NeurIPS 2020.
- *Out-of-distribution generalization via composition: a lens through induction heads in Transformers.* PNAS 2025.
