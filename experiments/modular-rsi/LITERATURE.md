# Literature check: are we reinventing the wheel?

This review was done by four parallel search agents on web-verified sources. Items marked
(unverified) were cited from memory by an agent and should be checked before quoting.

## Verdict in one paragraph
Most individual pieces of this program already exist:
* step-by-step shared computation that extrapolates to longer inputs;
* frozen adapter libraries that do not forget;
* a generative mutator paired with a strict checker and an archive;
* a fast network that imitates a slow search and then guides it;
* the generator–verifier bound on self-improvement.

Our results in these areas reproduce known findings and are best treated as replications
or baselines. What appears **open** is a specific combination:
* a *small, learned* mutator that reads a *structured lineage log* and makes structural
  "model surgery" edits (copy which module, what size, what to train next) to tiny
  networks;
* trained by an imitate → correct → replace-the-parent curriculum;
* where the parent's judgment is a *noisy, biased short-horizon proxy* rather than an
  exact checker.

## What is already known (our results are replications)

| Our finding / idea | Prior work |
|---|---|
| Step-by-step composition through a shared block extrapolates from 1–3 to 16 steps; a standard transformer fails | Neural Programmer-Interpreters (Reed & de Freitas, ICLR'16, [1511.06279](https://arxiv.org/abs/1511.06279)); Neural GPU ([1511.08228](https://arxiv.org/abs/1511.08228)); Neural Data Router (Csordás et al., ICLR'22, [2110.07732](https://arxiv.org/abs/2110.07732)); **Looped Transformers for Length Generalization** (Fan et al., ICLR'25, [2409.15647](https://arxiv.org/abs/2409.15647)); scratchpads and length generalization (Anil et al. [2207.04901](https://arxiv.org/abs/2207.04901); Zhou et al. [2402.09371](https://arxiv.org/abs/2402.09371)) |
| One shared block ≥ many separate modules | Universal Transformers ([1807.03819](https://arxiv.org/abs/1807.03819)); MoEUT (NeurIPS'24, [2405.16039](https://arxiv.org/abs/2405.16039)); "Is a Modular Architecture Enough?" (Mittal et al., NeurIPS'22, [2206.02713](https://arxiv.org/abs/2206.02713)) |
| Frozen per-skill adapters do not forget; copying a related module speeds learning | Progressive Nets ([1606.04671](https://arxiv.org/abs/1606.04671)); PathNet ([1701.08734](https://arxiv.org/abs/1701.08734)); LoraHub ([2307.13269](https://arxiv.org/abs/2307.13269)); Arrow / LoRA libraries (Ostapenko et al., ICML'24, [2405.11157](https://arxiv.org/abs/2405.11157)); Mendez & Eaton (ICLR'21, [2007.07732](https://arxiv.org/abs/2007.07732)) |
| Voting communities of tiny models are parameter-inefficient on language | Unified Scaling Laws for Routed LMs (Clark et al., [2202.01169](https://arxiv.org/abs/2202.01169)); Branch-Train-Merge ([2208.03306](https://arxiv.org/abs/2208.03306)) |
| Generative "best-guess" mutator + strict verifier + archive | ELM ([2206.08896](https://arxiv.org/abs/2206.08896)); FunSearch (Nature'23); AlphaEvolve ([2506.13131](https://arxiv.org/abs/2506.13131)); Darwin Gödel Machine ([2505.22954](https://arxiv.org/abs/2505.22954)) |
| The mutator improves from its own search outcomes | EvoTune ([2504.05108](https://arxiv.org/abs/2504.05108)); SOAR (ICML'25, [2507.14172](https://arxiv.org/abs/2507.14172)); EvoPrompting ([2302.14838](https://arxiv.org/abs/2302.14838)); Promptbreeder, which mutates its own mutation prompts ([2309.16797](https://arxiv.org/abs/2309.16797)) |
| Imitate the parent search, then become the parent | **Expert Iteration** (Anthony et al., NeurIPS'17, [1705.08439](https://arxiv.org/abs/1705.08439)); AlphaZero; OptFormer, which reads trial logs and imitates and then beats tuners ([2205.13320](https://arxiv.org/abs/2205.13320)); Algorithm Distillation ([2210.14215](https://arxiv.org/abs/2210.14215)) |
| The parent corrects the child; the child surpasses an imperfect parent | DAgger ([1011.0686](https://arxiv.org/abs/1011.0686)); AggreVaTe ([1406.5979](https://arxiv.org/abs/1406.5979)); LOLS, "learning to search better than your teacher" ([1502.02206](https://arxiv.org/abs/1502.02206)) |
| A small learned mutation operator for networks | RENAS (CVPR'19, [1808.00193](https://arxiv.org/abs/1808.00193)); Learned GA / LES (Lange et al., [2304.03995](https://arxiv.org/abs/2304.03995), [2211.11260](https://arxiv.org/abs/2211.11260)) |
| "Self-improvement is limited by verification" | Mind the Gap (ICLR'25, [2412.02674](https://arxiv.org/abs/2412.02674)), where gains saturate after 2–3 rounds; Stroebl et al., where false positives cap best-of-N ([2411.17501](https://arxiv.org/abs/2411.17501)); reward overoptimization (Gao et al., [2210.10760](https://arxiv.org/abs/2210.10760)); model collapse (Shumailov et al., Nature'24) |
| The parent's noisy 150-step judgment (0.49) | **Short-horizon bias** (Wu et al., ICLR'18, [1803.02021](https://arxiv.org/abs/1803.02021)): short trials systematically favour greedy choices. This is a *bias*, not just noise, and the likely cause of our 0.49 |
| Lineage-level credit | Huxley-Gödel Machine ([2510.21614](https://arxiv.org/abs/2510.21614)): clade metaproductivity, judging a node by its descendants rather than its own fitness |
| Clonal selection / germinal center | CLONALG (de Castro & Von Zuben, IEEE TEC 2002) |
| Cultural accumulation across generations | Artificial Generational Intelligence (Cook et al., NeurIPS'24, [2406.00392](https://arxiv.org/abs/2406.00392)) |

## What appears genuinely open
1. **A small learned mutator that reads a structured lineage log** (won/lost plus a diff)
   and outputs **structural edits that compose trained weights** (copy which adapter, what
   size, what to train next). The state of the art prompts or fine-tunes *large* LLMs that
   edit *code*. HGM uses lineage only to choose what to expand, not to generate edits.
2. **The full staged handover:** imitate a costly search parent → DAgger-style correction →
   self-correction with a value head → *replace the parent*, applied to evolving neural
   modules. Each piece exists; we found no paper that combines them.
3. **Recursive parenting under a noisy, biased verifier.** Nearly all successes (AlphaZero,
   AlphaProof, AlphaDev) have exact checkers. How parent error compounds across
   generations, and whether the child inherits or averages out the parent's bias, is open.
4. **Generalization of a learned mutator to held-out task families**, as opposed to
   within-distribution sharpening.
5. **Measuring whether evolvability itself rises over generations** at small scale. This
   would be direct evidence of "learning how to improve".
6. **A discrete symbol vs continuous state vs VQ ablation between looped steps** for length
   generalization. Fan'25 and recurrent-depth models use continuous state, and DVNC
   ([2107.02367](https://arxiv.org/abs/2107.02367)) never tested length.
7. **Major transitions:** composites of tiny modules becoming new units of selection, with
   their own heredity.

## Methods to borrow (for the Stage 1 rerun)
**Making the parent reliable**
* Calibrate the short-horizon score against long-horizon rollouts on a subsample
  (AggreVaTe / Math-Shepherd style), or extrapolate learning curves (Domhan et al. 2015).
* Spend parent compute where its ranking is uncertain: re-evaluate the top-k at long
  horizon (surrogate model management, Jin 2011).
* Track proxy vs true score per generation, and stop when they diverge (Gao).
* Report the verifier's false-positive rate.

**The child**
* Imitate the parent's *values or advantages* (AggreVaTe), not its single choice.
* Add a confidence loss so the child does not copy parent noise (weak-to-strong, Burns et
  al. [2312.09390](https://arxiv.org/abs/2312.09390)).
* Train a policy head and a fitness-prediction head together (OptFormer), and pre-screen
  proposals with the predictor (Neural Predictor, [1912.00848](https://arxiv.org/abs/1912.00848)).
* Use order-invariant attention over the candidate set and log (LES/LGA). We already do this.
* Credit edits by how productive their descendants are (HGM's CMP).

**Generalization and diversity**
* Split by task family and generate many families procedurally (CoinRun, Cobbe et al.
  [1812.02341](https://arxiv.org/abs/1812.02341); POET).
* Accumulate data across generations rather than replacing it (Gerstgrasser et al.
  [2404.01413](https://arxiv.org/abs/2404.01413)) to avoid collapse.
* Use quality-diversity archives (MAP-Elites) so the self-trained mutator does not collapse
  to a few modes.

**Benchmarks to use instead of only our own**
* SCAN length split; compositional table lookup / ListOps (NDR); Fan'25 tasks; CLRS-30.

## Implications for our claims
* The composition result should be described as **a replication** of known findings (NPI,
  NDR, Fan'25), not a discovery.
* The "parenting" program should be positioned as **Expert Iteration / OptFormer applied to
  structural model edits under a noisy verifier**, with clear credit to those works.
* The most defensible novel contribution, if it works, is items 1–4 above. The highest-value
  next experiment is the one that directly tests item 3: does a child trained by a noisy,
  short-horizon parent inherit the parent's bias, and do calibration or value-imitation fix
  it?
