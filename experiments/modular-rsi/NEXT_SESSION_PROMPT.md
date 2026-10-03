# Prompt for the next Claude session

Copy everything below the line into a new Claude Code session on `mcmanuslab/claudex`.

---

You are continuing a research program on **recursive self-improvement (RSI) with tiny,
modular transformers**. Work in `experiments/modular-rsi/` on branch
`claude/recursive-self-improving-transformer-remv5e` (draft PR #4). Before doing anything,
read `README.md`, `LITERATURE.md`, `LITERATURE_DEEP.md`, `parenting/PREREG.md` and
`parenting/student.py`, `parenting/parent.py`. Don't redo work they describe.

## The big vision (keep this in view, don't oversell it)
1. **Start tiny.** Modules of about 1–50k parameters that are cheap enough to try
   thousands of variants.
2. **Hybrid architecture.** A shared, frozen core plus small per-skill adapters.
   * Knowledge lives *outside* any one module, in a library or community.
   * Old skills are never overwritten.
   * Skills compose step by step through a discrete interface, so short training
     programs become long-program ability.
3. **A learned mutator.** A small transformer reads the lineage log: which variants
   won, which lost, and what changed. It proposes *best-guess* structural edits: which
   adapter to copy, what size, what to train next. Mutation becomes targeted guessing
   rather than blind noise, with a strict verifier to catch bad guesses (the analogue of
   hallucination).
4. **A parenting curriculum for the mutator.**
   * Infant: imitate an expensive search "parent".
   * Child: propose while the parent corrects (DAgger).
   * Adolescent: act and self-correct with a value head.
   * Adult: replace the parent and train the next generation. **This is the recursion.**
5. **Scale.** If each generation of mutator makes the next skill cheaper, and that
   effect compounds, then grow modules, the library and the mutator toward much larger
   models. The 1B+ endpoint is an aspiration, not a result. Every step must earn the
   next one with a pre-registered test.

## Where things stand (honest summary; details in README.md)

**Symbolic toy domain**
* Steered evolution with Lamarckian lifetime learning beats blind evolution.
* Polymers and communities solve problems no single monomer can.
* Gene duplication plus path credit halves the cost of a new skill. That gain arrives
  once; it does not compound.
* Exp 4's "exponential" came from the curriculum, not from the system.

**Language track**
* The voting community needs about 8× more parameters than one small model.
* The RSI outer loop does not beat copy-and-train (Phase 0).
* The apparent compounding was learning-rate tuning: a fresh module at lr 0.03 beat the
  full system in all 10 orders.
* **Negative.**

**Composition track (Part B)**
* Trained on 1–3-step programs, chained modular blocks (0.97) and one shared looped
  block (1.00) run 16-step programs. A same-size transformer stays at chance.
* Modular = looped. Evolution adds nothing over gradient training.
* This is a **replication** of NPI, Neural Data Router and Looped Transformers
  (Fan'25), not a discovery.

**Parenting Stage 1** (register machine, frozen core + adapter, decision = adapter
source × rank)
* The child imitates the parent well: agreement 0.67 vs heuristic 0.26; regret 0.10
  vs 0.34.
* Calibration fails narrowly (G3: 0.586 < 0.6), so **Stage 1 fails as pre-registered.**
* On held-out opcode families the child is **worse than a probe heuristic** (regret 0.34
  vs 0.21).
* The parent itself is unreliable: its 150-step rankings correlate only 0.49 with
  1,500-step rankings. This is consistent with **short-horizon bias** (Wu et al. 2018).

**Literature verdict** (`LITERATURE.md`)
* Most pieces exist: Expert Iteration, OptFormer, DAgger/AggreVaTe/LOLS, EvoTune/SOAR,
  FunSearch/AlphaEvolve/DGM, HGM's clade metaproductivity, looped transformers, LoRA
  libraries.
* What looks open:
  1. a *small* learned mutator reading a structured lineage log and making structural
     model-surgery edits;
  2. the full imitate → correct → replace-the-parent handover for neural modules;
  3. recursive parenting under a *noisy, biased* verifier;
  4. generalization of a learned mutator to held-out task families;
  5. measuring whether evolvability rises over generations.

## Your task: the next experiment ("Stage 1b: a de-biased parent and a child that transfers")
Pre-register it in `parenting/PREREG_1b.md` and commit *before* any main run. Then
build, run, analyse and report honestly in the README, verdicts first.
`LITERATURE_DEEP.md` §"What we take into the next experiment" has the recipe and its
sources.

1. **Fix and measure the parent first.** Wu et al.: short-horizon bias is a bias, so more
   short trials will not fix it.
   * Screen candidates at 150 steps, then re-train the top 3–5 plus one random control to
     1,500 steps, and rank on those.
   * Treat ties as ties, and break saturated ties with continuous signals (held-out
     loss, margin, steps to threshold).
   * Gate: the parent's ranks correlate ≥ 0.8 (Spearman) with long-horizon ranks before
     any child is trained.
   * Report the parent's false-positive rate.
2. **Bias-inheritance test (the novel question).** Train children on the *biased*
   150-step parent and on the *de-biased* parent. Measure against **long-horizon
   truth**:
   * performance gap recovered (weak = the parent's pick, ceiling = the long-horizon
     best);
   * agreement split by whether the parent was right;
   * a bias signature: the share of picks with fast early gain but a poor 1,500-step
     outcome;
   * diversity of the choices.

   Does the child inherit, amplify or average out the bias?
3. **A better child.**
   * Imitate soft values over all candidates (ExIt tree-policy targets / AggreVaTe), not
     the argmax. Regret is primary; agreement is secondary.
   * Two heads (OptFormer):
     * a policy head;
     * a value head that predicts the **1,500-step** outcome as a calibrated
       distribution.
   * Act by drawing K proposals from the policy and re-ranking them by expected
     improvement on the value head.
   * Add the weak-to-strong confidence loss, with α ramped from 0 to 0.5–0.75.
   * LOLS-style: score a share of the child's own choices with real 1,500-step runs.
   * Early-stop on long-horizon validation, not on agreement with the parent.
   * Give it transferable features: a few-shot probe (k ≪ 150 steps), behavioural
     similarity on matched inputs, and the universe's history of earlier trials in
     context (AD).
4. **Generalization by design.** Algorithm Distillation needed hundreds of tasks; we had
   12 families.
   * Generate ≥ 100 opcode families procedurally.
   * Split train and test by family.
   * Use OptFormer-style augmentation: permute candidates, normalise per universe, drop
     feature groups, mask tokens, tag the parent type.
   * Primary gate: on held-out families, the child's regret ≤ 0.5 × the probe
     heuristic's (paired Wilcoxon p < 0.05).
5. **Only if 1–4 pass:** run Stage 2 (DAgger correction), then the recursion test.
   * The adult child, as search plus its prior (ExIt), becomes the parent for a fresh
     infant.
   * Gate each handover: the new parent must beat the old one on held-out long-horizon
     runs.
   * Use HGM clade posteriors to decide what to mutate, with the expand/evaluate rule
     N^0.6 ≥ |nodes|.
   * Pool data across generations and retrain from base (SOAR).
   * Keep the evaluator read-only to the mutator (DGM reward hacking).
   * Measure whether the cost to acquire a new skill falls generation over generation.
   * Expect saturation in about 2–3 generations (Mind the Gap) unless the long-horizon
     re-checks continue.

## Protocol rules (learned the hard way; keep them)
* **Pre-registration.** Pre-register hypotheses, thresholds and splits with no
  in-between band. Amendments are allowed only before results, and they are logged.
* **Fair baselines.** Use compute-matched baselines, including a *tuned* simple
  baseline (copy-best + fixed good lr). Simple baselines killed two of our claims.
* **Splits.** Keep separate train, selection and test splits; report test only. Watch
  for the winner's curse.
* **Statistics.** Use paired tests across seeds or orders: exact Wilcoxon for n ≤ 20,
  normal approximation above (`parenting/scipy_free.py`), plus cluster bootstrap.
* **Adversarial review.** Before a big build, run an adversarial review: an adversary
  agent, a scientific-director agent and a judge (see `debate/`). Report failures
  plainly and keep a "what went wrong" section.
* **Environment notes.**
  * CPU only, torch via plain `pip install torch`.
  * arXiv, OpenReview, HuggingFace and Gutenberg are blocked in this environment unless
    the network policy is changed; GitHub raw works.
  * Don't use `multiprocessing` fork pools with torch (they deadlocked); run seeds
    sequentially or in separate processes.
  * Never `pkill -f` a pattern that matches your own shell.
* **Git.** Commit and push to the branch above. Keep model identifiers out of commits,
  the PR and code.

Report back with: the pre-registered verdict for each gate; whether the child inherited
the parent's bias; held-out-family results versus the heuristic; and a one-paragraph
judgement of whether the program has earned the recursion test.
