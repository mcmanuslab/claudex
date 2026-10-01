# Pre-registration: compounding continual-acquisition test

Written and committed **before** any main run of `compound.py`, after only smoke tests
and the target calibration (`results/compound/calibration.json`). Follows the judge's
verdict in `../debate/05_judge.md`.

## Hypotheses
* **H-compound.** For a library of tiny modules, the compute needed to reach a fixed target
  on a new domain falls as the library grows, beyond a one-time transfer gain.
* **H-rsi.** The evolutionary outer loop (`full`) reaches targets with less compute than
  a cheap heuristic: probe the library, copy the best module, fine-tune (`probe`).
* **H-modular.** At the end, the library's mean test bpc over the 8 domains is within
  0.05 bpc of a continually fine-tuned single model of the same total size (`monolith`).

## Data
8 domains (`domains.py`, URLs and SHA-256 hashes in `domains_prereg.json`): shakespeare,
alice, python, javascript, latex, markdown, legal, french. 150k characters each,
96-symbol alphabet, contiguous train/selection/test = 80/10/10. Selection is used only
by the system; test is used only for reporting.

## Design
* **Module:** 1-layer char transformer, d=8, MLP 16, context 16, 1,520 parameters.
  A new module adds its logits to a gated vote of the top-4 frozen library modules
  (chosen by zero-shot selection bpc).
* **Arms:** `full`, `nocredit`, `probe`, `latest`, `fresh`, `monolith`, as defined in
  `compound.py`. The monolith has 11,936 params, close to 8 × 1,520.
* **Compute:** candidate-steps; the monolith is scaled by its parameter ratio (FLOP proxy).
  Cap is 12,000 per domain. Library forward passes are cached once per domain and
  reported separately as a constant 4 × (train windows) forward passes.
* **Cost** = candidate-steps until any candidate reaches the target on the selection split,
  checked every 100 candidate-steps. Censored at the cap; censored runs count as the cap
  in the regression, and the number of censored runs is reported.
* **Targets:** per domain, from calibration (seed 12345):
  * T1 = selection bpc of a fresh 1,520-param module after 6,000 steps;
  * **T2 (primary)** = the same for a fresh 3,968-param module.
* **Domain orders:** 10 (`order_for(0..9)`), paired across arms. Seed = order index.

## Analysis and decision rules (no in-between band)
1. **H-compound (per arm; primary arm `full`, also reported for `probe`).**
   * OLS of log(cost) on position k for k = 3..8, with domain fixed effects.
   * 95% CI by cluster bootstrap over orders (2,000 resamples).
   * **Pass** if the slope's upper CI bound < 0. Otherwise **fail**.
2. **H-rsi.** Paired over the 10 orders: log of total cost over domains k = 2..8,
   `full` vs `probe`.
   * **Pass** if `full` is lower with a one-sided Wilcoxon signed-rank p < 0.05.
   * Otherwise **fail**, and the language track is described as modular transfer, not RSI.
3. **H-modular.** Mean over orders of the final mean test bpc over the 8 domains,
   `full` vs `monolith`.
   * **Pass** if `full` ≤ `monolith` + 0.05. Otherwise **fail**.
4. **Floor/void check.** If more than half of `full`'s k ≥ 3 domains reach T2 at the first
   check (≤ 100 candidate-steps), the test is **void** and is rerun with T2 replaced by the
   selection bpc of a fresh 5,576-param module (d=20).
5. **Ceiling check.** If more than half of `full`'s k ≥ 3 domains are censored at the cap,
   H-compound is reported as **inconclusive (ceiling)**, not as pass or fail.

## Phase 0 link
If Phase 0 shows that the `full` outer loop does not beat the `inherit` ablation by more
than 2 paired SDs, H-rsi enters this test with a stated prior of about 10%.
