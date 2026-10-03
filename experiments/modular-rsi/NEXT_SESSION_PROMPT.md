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

## Your task: the next experiment ("Stage 1b: a fixed parent and a child that transfers")
Pre-register it in `parenting/PREREG_1b.md` and commit *before* any main run. Then
build, run, analyse and report honestly in the README, verdicts first.

1. **Fix and measure the parent first.**
   * Score candidates at a longer horizon, or by repeated short trials plus a
     learning-curve extrapolation calibrated against 1,500-step outcomes on a
     subsample.
   * Treat ties as ties: the set of candidates within ε of the best are all "correct".
   * Re-evaluate the top-k at long horizon (surrogate model management).
   * Gate: Spearman correlation with the long horizon ≥ 0.8 before any child training.
   * Report the parent's false-positive rate.
2. **Bias-inheritance test (the novel question).** Train children on the *biased*
   150-step parent and on the *fixed* parent. Measure each child's regret against the
   **long-horizon truth**. Does the child inherit, amplify or average out the parent's
   bias?
3. **A better child.**
   * Imitate the parent's *values/advantages* over all candidates (AggreVaTe/OptFormer
     style), not only its argmax.
   * Add a confidence loss so the child does not copy parent noise (weak-to-strong,
     Burns et al.).
   * Add transferable features: a few-shot probe (k ≪ 150 steps per candidate) and
     behavioural similarity on matched inputs.
4. **Generalization by design.**
   * Generate many opcode families procedurally (compositions of lookups and
     arithmetic), so held-out families number ≥ 10, not 2.
   * Split train and test by family.
   * Primary gate: on held-out families, the child's regret ≤ 0.5 × the probe
     heuristic's, with a paired Wilcoxon p < 0.05.
5. **Only if 1–4 pass:** run Stage 2 (DAgger correction), then the recursion test.
   * The adult child acts as parent for a fresh infant.
   * Measure whether evolvability (cost to acquire a new skill) falls generation over
     generation.
   * Use HGM-style clade credit for which edits were productive.
   * Accumulate data across generations; do not replace it.

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
