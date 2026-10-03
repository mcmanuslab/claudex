# Judge's verdict

## 1. Scorecard

**What I checked in the repo (`lang/results`, `grow_spell.py`)**

- **The 8x gap is real.** The mean RSI community reaches 2.87 bpc at 10,880 bytes. The K4 monolith reaches 2.871 bpc at 1,288 params. With steps matched at 12.8k, the 10.8 KB monolith scores 2.385, a gap of 0.49 bpc. The community also used 12x more candidate-steps.
- **The bigram claim is real.** The community scores 3.19 at 1.36 KB; the bigram table scores 3.24.
- **The candidate-spread claim is only partly right.** In the final cycles the spread is 0.003–0.004, as the adversary said. Early on it is about 0.15 (cycle 0) and about 0.03 (K=4). So selection mattered early and went inert later.
- **The seed-to-seed spread is far larger than the candidate spread.** The two RSI seeds end 0.074 bpc apart (2.907 vs 2.833).
- **RSI beats fresh by about 0.24 bpc** (2.87 vs 3.10). That exceeds the seed noise, but it remains confounded with warm start.
- **Run timings support "compute is cheap".** The monolith runs finished within minutes.

**A flaw neither side found.** `grow_spell.py` selects the winner and reports its bpc on the **same** fixed 8,192-char batch (`xv`). Reported bpc carries winner's-curse bias, and there is no untouched test set.

**Arguments that held up**

- **Adversary:** frozen greedy growth is byte-inefficient. The RSI loop is not separated from warm start plus Adam. Exp 4's exponential comes from the curriculum. The real-word metric is weak. H1 for plan B is near-guaranteed (Net2Net). The H3 ablation must be FLOP-fair. C's chars-per-token growth holds by construction. Vocabulary bytes dominate. The 3 h estimate was fantasy.
- **Director:** gradient learning within a lifetime is legitimate. "Freeze forever" was a self-made strawman. The operational exponential (unit length per level at constant cost) is a fair target. Frozen modules mean "no forgetting" holds by construction, so it is not evidence.

**Arguments that did not hold up**

- The director's 60% for B and the ~3 h compute estimate.
- The claim that per-domain compute in the primary plan "stays constant". Forward passes through a k-module library grow O(k), and that pushes *against* the compounding signal.
- The adversary's "selection does nothing", stated globally. It is true only once growth saturates.

**Blind spots both sides share (primary plan)**

1. **Ablation (ii) is a strawman.** "Inherit the latest module" is weak. The real competitor to lineage credit is a cheap heuristic: probe each library module's zero-shot bpc on the new domain, pick the best, and fine-tune it. Without that arm, a pass on "RSI loop" only means "reuse beats bad reuse".
2. **The metric can hit a floor or a ceiling.** Exp 5 already shows this: costs pinned at 1 generation, with a cap at 80. A target set at "fresh 1 KB at 6,000 steps" may be reached almost immediately once a related domain exists. The slope then flattens at the floor, and the result is a false kill.
3. **The slope confounds domain similarity with library size.** A negative slope can come from "a similar domain is probably already stored", which is coverage, not an improver improving itself. That still counts toward "smarter as it grows". It does not show recursion.
4. **Five domain orders over eight heterogeneous domains is underpowered.** Runs take minutes, so more orders are cheap.
5. **It drifts from the user's latest request.** The user asked for natural language: chars, then words, then sentences. The primary plan tests the thesis and does not deliver that skill.
6. **No one tells the user that ~100-byte modules are infeasible.** The probe already shows letter soup at 116 B. A useful monomer is about 1 KB.
7. **"Bytes" are parameter counts.** Nothing is quantized.

## 2. Decision

Ranking:

1. **The amended compounding continual-acquisition test, preceded by a 1-hour Phase 0.**
2. **Backup C** (word level, module-driven merges against a BPE control).
3. **Alternative E** (distillation cycle), which neither side developed.
4. **B**, which only rediscovers Net2Net.
5. **Level 1b**, which should be dropped. The data already argue against it, and no arm is needed.

**Run next: amended primary.**

- **P(decisive, interpretable result) ≈ 75%.**
- **P(compounding passes) ≈ 35%.**
- **P(compounding and the RSI loop both pass) ≈ 15–20%.**

I choose it because it is the only option whose positive result is not predicted by known work, and whose negative result would redirect the program.

## 3. Required amendments

**Phase 0 (≤1 h, existing code)**

- Rerun `grow_spell.py` on Alice, 4 seeds, with a split selection set and test set.
- Compare full RSI against two arms:
  - **(a)** M=1, warm start from the best-credited module, σ=0, fixed lr, 12x steps.
  - **(b)** M=12 with fixed lr.
- If full RSI fails to beat (a) by more than 2 paired SDs, the "RSI loop" hypothesis enters the main test with a prior of about 10%. Say so in advance.

**Main test**

1. **Three disjoint splits per domain.** Train, selection (used by the outer loop) and test (reported only). Pre-register domain URLs and SHA-256 hashes before anything runs.
2. **Arms, all FLOP-matched, with every candidate, evaluation and library forward pass counted.** Run 10 domain orders, paired across arms.
   - Full.
   - Fresh.
   - Inherit-latest.
   - **Probe-best-then-fine-tune** (the key ablation).
   - **Full without lineage credit** (uniform parent choice).
   - Monolith with replay.
3. **Primary metric: log FLOPs to target, with two pre-registered targets.**
   - **T1** is the bpc of a fresh 1 KB module after 6,000 steps.
   - **T2** is the bpc of a fresh 4 KB module after 6,000 steps, which avoids the floor.
   - Evaluate every 100 steps. Cap runs at 2x the fresh budget and report the censoring rate.
   - **Secondary metric:** area under the bpc-vs-FLOPs curve.
4. **Analysis.** Use a mixed-effects regression of log cost on k, with fixed effects for domain and a random effect for order.
   - **Compounding passes** if the slope over k=3..8 is below 0 with a 95% CI that excludes 0, **on T2**.
   - **The RSI loop passes** only if Full beats **probe-best** in a paired test with p<0.05.
   - There is no dead band: every other outcome counts as a kill.
5. **Floor check.** If more than 50% of k≥3 domains reach T2 within the first evaluation, the test is void. Rerun it with a stricter target that is already pre-registered.
6. **Report library-read cost separately.** Also cap the polymer to the top-r=4 library modules (by gate weight or probe) so that per-domain compute really is constant.
7. **Budget.** About 4 h to build and about 4 h to run. Cut to 6 domains before cutting orders.

## 4. What to tell the user

**What this tests**

- Whether a library of tiny modules gets **cheaper to extend** as it grows. This is the honest, measurable form of "smarter as it grows" and of compounding returns.
- Whether **mutation steered by lineage** beats simple engineering heuristics.

**What this does not test**

- Chars → words → sentences.
- Exponential capability growth.
- Scaling toward 1B or 1T parameters.
- Modules of about 100 bytes. The data say these cannot spell; about 1 KB is the minimum.
- Free binding and dissociation.

**Plain facts about current results**

- On real text, today's modular system is about 8x less byte-efficient than one ordinary small model.
- The exponential in Exp 4 comes mostly from the curriculum.

**What would change direction**

- **Compounding passes but the RSI loop fails to beat probe-best:** keep modular growth and drop "RSI" as the explanation.
- **Both pass:** this is the first real evidence for the thesis. Scale the domain count and move to C on top of it.
- **Compounding fails on T2:** modular reuse gives a one-time transfer gain, not compounding. Pivot the language goal to jointly trained hierarchies (C or E) and stop claiming modular RSI on language.
