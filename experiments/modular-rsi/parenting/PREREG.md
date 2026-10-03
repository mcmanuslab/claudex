# Pre-registration: Stage 1 ("infant"): can a tiny mutator learn to imitate its parent?

Written and committed before any Stage 1 result (only smoke and timing runs before it).
Part of the 4-stage plan: (1) infant imitates the parent, (2) the child proposes and the
parent corrects, (3) the child acts and self-corrects, (4) the child becomes the parent
(the recursion test).

## Testbed: a register machine with skills that differ in kind
* **State:** 3 registers, each holding one of 8 symbols. The context holds two random
  permutation tables A and B, new for every example.
* **Skills:** an instruction family, parameters, and register arguments (i, j):
  * `lookA` r_i = A[r_j], `lookB` r_i = B[r_j];
  * `invA` r_i = A⁻¹[r_j], `invB` r_i = B⁻¹[r_j];
  * `affine(a,b)` r_i = a·r_j + b (mod 8);
  * `add` r_i += r_j, `sub` r_i −= r_j;
  * `swap` r_i ↔ r_j.
* **Interface:** the 3 register symbols, the discrete multi-slot state between steps.
* **Hybrid model:** a shared 2-layer transformer core (d=48) over [3 register tokens,
  16 context tokens, 1 instruction token], plus a per-skill **adapter** (instruction
  embedding + low-rank residual MLP on the register tokens after layer 1, rank 4 or 16).
  The core is trained once on 8 fixed base skills together with their adapters, then
  frozen.

## The structural decision (made each time a new skill arrives)
* **Candidates:** source ∈ {fresh, copy of the adapter of any existing skill} × rank ∈ {4, 16}.
* **Outcome of a candidate:** validation loss (and accuracy) on the new skill after
  S = 300 adapter-training steps (Adam, lr 3e-3, batch 32). The core is frozen, so old
  skills are untouched.
* **Parent:** trains *every* candidate for S steps and picks the lowest validation loss.
  It is expensive but measured.

## Infant mutator (student)
* **Model:** a tiny transformer over the set of candidates. Each candidate is a token of
  cheap features:
  * fresh or copy, and rank;
  * zero-shot probe of the copied adapter on the new skill (loss and per-register accuracy);
  * the source's accuracy on its own skill;
  * lineage (times the source was copied and won before, copy depth, age);
  * the new skill's "which registers change" profile and its similarity to the source
    skill's profile.
* **Output:** a choice distribution over candidates and a predicted outcome for each.
* **Training:** imitation only (Stage 1). Cross-entropy on the parent's choice, plus
  regression on the parent's measured outcomes.
* **Data:** decision situations from skill-arrival streams ("universes"), 6 arrivals per
  universe. Universes are split train/test; test universes use skill instances not seen
  in training.

## Baselines
* **random** candidate;
* **default** (fresh, rank 16);
* **probe-best**: copy the source with the lowest zero-shot probe loss, rank 16; fresh
  if no copy beats chance.

## Gates (held-out universes; no in-between band)
* **G1 (imitation):** student top-1 agreement with the parent ≥ probe-best's agreement
  + 0.10, and ≥ 2 × the chance rate.
* **G2 (decision quality):** student's mean normalised regret ≤ 0.5 × probe-best's
  regret, and lower than random's.
  * Normalised regret = (loss of chosen − loss of parent's pick) / (worst − best loss in
    that decision).
  * Paired one-sided Wilcoxon test p < 0.05 against both.
* **G3 (calibration):** Spearman correlation between the student's predicted and actual
  candidate outcomes ≥ 0.6.
* **G4 (efficiency):** student compute (probes + training the one chosen adapter) ≤ 10%
  of the parent's (all candidates trained).
  * Expected to hold by construction; reported for completeness.

**Stage 1 passes only if G1–G3 all pass.**

## Diagnostic: is the parent a good parent?
On 24 decisions, every candidate is also trained for 1,500 steps. Reported:
* Spearman correlation between 300-step and 1,500-step outcomes;
* how often the parent's 300-step pick is within the top 2 at 1,500 steps.

If the correlation is below 0.5, the parent itself is unreliable. That is reported as a
limitation, whatever happens to G1–G3.

## Amendment 1 (design calibration, before any Stage 1 result)
The first testbed baked register arguments into each skill. New lookups on other registers
then needed rerouting inside the frozen core, and every candidate stayed near chance, so
the parent's decisions were noise. Fixed before any data collection:
* **A skill is an opcode.** Operands (i, j) are inputs to every instruction (role
  embeddings on the registers, operand embeddings on the instruction token), sampled
  per example.
* **Families:** lookA, lookB, invA, invB, lookAA (A[A[x]]), affine(a,b), add, sub, mul,
  max, min, swap.
* **Base opcodes** (core training): lookA, invA, affine(3,1), add, swap, max. The core
  reaches 100% on all six.
* **Adapters** also carry low-rank (rank 4/16) updates to the attention projections of
  both core layers.
* **S = 150 adapter steps** (was 300). Calibration showed candidate differences are
  large at 100–150 steps (e.g. new lookB: copy-of-lookA 1.00 vs fresh 0.12 after 100
  steps) and shrink by 300 as most candidates converge.
* The student's features use the updated change profile: P(target changes),
  P(other operand changes), and the 8-bin distribution of value change.

Gates, baselines, splits and the diagnostic are unchanged. The diagnostic's long run is
1,500 steps.

## Amendment 2 (split, before any data collection)
* **Training universes** (ids 0–47) never receive `invB` or `min` as a new arrival.
* **Test universes** (ids 100–111) draw arrivals from all families.

G1–G3 are evaluated on all test decisions. They are also reported separately for test
decisions whose new opcode is `invB` or `min` (opcodes never seen as arrivals in
training). That subset is descriptive only, because it is small.
* **Diagnostic universes:** ids 200–211, 2 arrivals each, with every candidate also
  trained for 1,500 steps.
