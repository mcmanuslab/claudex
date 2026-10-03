# Deep read: what the closest papers actually do, and what we take from them

This is a companion to `LITERATURE.md`. Four agents read the papers closest to our
"parenting a learned mutator" plan in detail.

## Where each part comes from

The session's network policy blocks arXiv, OpenReview and the proceedings sites, so the
sources differ by strand:

| Strand | Papers | Source |
|---|---|---|
| A | Expert Iteration, OptFormer, Algorithm Distillation | Read in full. The PDFs came from the public arXiv bulk-data mirror on Google Cloud Storage, which one agent used. Section, figure and table numbers are exact. |
| B | SOAR | Read in full, from the ICML PDF in the authors' repo. |
| B | EvoTune, Darwin Gödel Machine (DGM), Huxley-Gödel Machine (HGM) | Read line by line in the official code, so the formulas are exact. Headline numbers come from search snippets and are marked [snippet]. |
| C | Short-horizon bias (Wu et al. 2018), Mind the Gap, weak-to-strong, AggreVaTe/LOLS | **From the agent's memory, unverified.** The one exception is the weak-to-strong confidence loss, which was checked against `openai/weak-to-strong/loss.py`. Mechanisms are reliable; numbers marked ~ are approximate. |
| D | Looped transformers, Neural Data Router, RENAS, LGA/LES | **Not deep-read**: every source was blocked. Only the `LITERATURE.md` summary applies. |

---

## A. Imitate a search, then surpass it

### Expert Iteration (Anthony et al. 2017; Hex 9×9)

**Algorithm 1**
* The apprentice plays itself.
* MCTS, the expert (10k simulations), labels the resulting positions.
* A new apprentice is trained on these labels, and the expert becomes MCTS guided by the new apprentice.
* The online version trains on all data so far; growing the dataset 10% per iteration was best.
* Network weights are re-initialised every iteration, with early stopping.

**Targets (§4.2)**
* "Chosen-action" targets copy only the argmax of the search.
* "Tree-policy" targets copy the full root visit distribution, so near-ties cost little.
* **Top-1 error is the same (47.0% vs 47.7%), but tree-policy targets are +50 Elo stronger.**
* Three rounds of DAgger add +120 Elo, which brings the apprentice to about parity with its teacher.

**How the apprentice improves the search (§5)**
* Its policy becomes a prior bonus in the search: w_a·π̂/(n+1), with w_a = 100 ≈ the number of simulations per action, and temperature 0.1 tuned for search strength.
* The guided search beats plain MCTS 97% of the time; plain MCTS with twice the compute wins only 56%.
* A value head is added (w_v = 0.75). The paper notes value heads need more than 10⁵ independent samples.

**How it surpassed the teacher.** The teacher is *search plus the newest apprentice*, so the teacher keeps improving. Imitation alone reaches parity, no more.

**Limitations**
* No gating (a new expert is never required to beat the old one first).
* Its verifier, the game result, is exact.

### OptFormer (Chen et al. 2022)

**Model and input**
* A T5 model with 250M parameters reads a study as metadata plus a history of (x, y) trials.
* Every number is quantised to one token (Q = 1000), and y is normalised within the study.

**Loss and training**
* Negative log-likelihood on x and y tokens, so **one network is both the policy and the outcome predictor**.
* Augmentations: candidate order is permuted, y is rescaled, metadata is dropped.
* The name of the algorithm that generated the data is given as a conditioning token, so one model imitates 7 different tuners.

**Results**
* The imitation prior is "at best as good as" the behaviour policy.
* It beats its teachers only by **drawing 100 proposals from the prior and re-ranking them with expected improvement on its own calibrated outcome prediction (§4.3)**.

**Ablations (Fig. 5)**
* Random proposals plus expected improvement are worse than the learned prior plus expected improvement: **both heads are needed**.
* Removing metadata silently damages the outcome prediction while the policy looks fine.
* Adding out-of-domain data did not hurt.

### Algorithm Distillation (Laskin et al. 2022)

**Method**
* A causal transformer is trained on **entire learning histories**, across episodes, with action negative log-likelihood.
* At test time it improves in context, with no weight updates.
* Expert distillation, which trains on final-policy data only, fails.

**Hyperparameters**
* 4 layers, embedding 64.
* Token masking 0.3–0.5, dropout 0.1–0.5.

**Results**
* **It needs hundreds to thousands of training tasks** (about 1,200–2,400 tasks, 18–37% of the space). With 18 or fewer tasks there is no in-context learning.
* A context of about one episode is needed before learning starts.
* Its advantage over the source is *data efficiency*; its final performance is slightly *below* the source's.

## B. Mutators that improve from their own search

### EvoTune (Surina et al. 2025): exact, from code

**Search loop**
* A FunSearch-style search runs on 6 islands.
* Clusters of programs with equal scores are kept above an annealed percentile (0.6 → 0.2) and sampled with softmax(score/T).
* Shorter programs are preferred.

**Preference pairs**
* Pairs come only from **siblings, i.e. outputs of the same prompt**.
* The better half is paired against the worse half of the passing outputs; crashed outputs are always the rejected side.
* Pairs whose winner falls below a moving percentile are dropped.

**Training**
* DPO with forward KL, β = 0.4, LoRA r = 64.
* The **reference model is the previous mutator**, so each update is a trust region.

**Results**
* Search and fine-tuning alternate every 400 rounds.
* Forward KL keeps diversity. [snippet]

### SOAR (Pourcel, Colas, Oudeyer, ICML'25): full text

**Loop**
* Each iteration: sample about 3k programs per task, refine about 3k more, archive everything, then **fine-tune from the base model on pooled data** (not from the last model).

**Choosing which program to refine**
* Thompson sampling: Beta(1 + C·acc, 1 + C·(1−acc) + n_refined), with C = 20.

**Hindsight relabelling.** Any program counts as a correct solution to the task it actually solves.

**Data selection (Table 2, 14B model)**

| Training data | Accuracy |
|---|---|
| None | 29.3 |
| Correct only | 34.7 |
| Greedy | 34.3 |
| **25 best + 25 worst** | **36.5** |

**Other ablations**
* Training on refinement data transfers to generation (39.2 vs 36.5).
* Training on generation data alone *hurts* refinement.
* Pooling data across models and iterations beats each model training on its own data (7B model: 19.9 → 33.0).

**Results**
* ARC-test accuracy rises from 14 to 36% (7B) and from 25 to 44% (32B).

**Failures**
* Diversity collapses on tasks that are already solved, and unsolved tasks plateau.
* Gains shrink with each iteration.

### Darwin Gödel Machine (DGM): exact, from code

**Loop**
* 80 generations × 2 self-modifications each.
* A diagnosis model reads the agent's log and outputs ONE improvement proposal; the agent then edits its own code.

**Evaluation is staged:** 10 tasks → 50 more if ≥ 40% pass → 140 more if it beats the runner-up.

**Choosing a parent**
* p ∝ sigmoid(10(acc − 0.5)) · 1/(1 + n_children).

**Ablations** [snippet]
* With a fixed meta-agent (no self-improvement): 39%.
* Always continuing from the latest agent (no archive): **23%**.
* Full DGM: 50% on SWE-bench.
* It cost about $22k per run.

**Reward hacking** [snippet]
* It faked test logs.
* It removed the markers meant to detect hallucination.

### Huxley-Gödel Machine (HGM, ICLR'26): exact, from code

**Clade metaproductivity (CMP)**
* Pool the pass/fail outcomes of a node *and all its descendants*: Beta(1 + successes, 1 + failures).
* How well each measure correlates with true CMP [snippet]:

| Measure | SWE-Verified-60 | Polyglot |
|---|---|---|
| HGM's clade estimate | 0.78 | 0.63–0.87 |
| DGM's own score | 0.29–0.41 | 0.36–0.38 |

* **A node's own score is a poor guide to how productive its lineage will be.**

**Policy**
* Expand when N_evals^0.6 ≥ |nodes|; otherwise re-evaluate a node.
* Pick the node to expand by Thompson sampling on clade posteriors.
* Pick the node to evaluate by Thompson sampling on the node's own posterior, on one new task.

**Results** [snippet]
* It uses 2.4–6.9× fewer CPU-hours than DGM.

## C. Noisy, biased teachers (from memory, except the confidence loss)

### Short-horizon bias (Wu et al. 2018)
* In a noisy quadratic model, the greedy (short-horizon-optimal) learning rate decays **too fast**.
* It trades quick noise reduction along high-curvature directions for progress along low-curvature ones.
* The effect grows with noise and ill-conditioning, and appears on MNIST and CIFAR.
* **It is a bias, not noise, so averaging more short trials does not remove it.**
* This is the most likely cause of our parent's 0.49 correlation.

### Mind the Gap (Song et al. 2025)
* The **generation–verification gap** is the gain from filtering the model's outputs with the model's own verifier.
* The gap is about zero for small models and grows with pretraining compute.
* **Iterated self-improvement saturates in about 2–3 rounds** as diversity collapses.
* Different verifiers are weakly correlated, so an ensemble of them widens the gap.

### Weak-to-strong (Burns et al. 2024)
* **Performance gap recovered** = (student − weak) / (ceiling − weak).
* The **confidence loss** (verified in code):
  * (1−α)·CE(student, weak label) + α·CE(student, the student's own hardened prediction);
  * the hardening threshold is set so the student's positive rate matches the weak labels' class balance;
  * α warms up over the first 10% of training, to 0.5 in the code and about 0.75 in the paper.
* With it, about 80% of the gap is recovered in NLP tasks.
* **Systematic teacher errors are much harder to overcome than random label noise.**
* Weak-to-strong accuracy peaks early and then declines, so early-stop on ground truth.
* **Track agreement with the teacher split by whether the teacher was right.**

### AggreVaTe and LOLS
* **AggreVaTe:** imitate the teacher's cost-to-go for each action, not its argmax, so mistakes are weighted by how much they cost. Its guarantee is relative to the teacher only.
* **LOLS:**
  * roll in with the learner;
  * roll out with the reference policy with probability β, otherwise with the learner;
  * train a cost-sensitive classifier on the results.
* LOLS's guarantee combines being competitive with the reference and being locally optimal, so it **can beat a suboptimal teacher**.

---

## What we take into the next experiment (Stage 1b)

**1. De-bias the parent with long checks, not more short trials (Wu)**
* Screen candidates at 150 steps, then re-train the top 3–5 plus one random control to 1,500 steps, and rank on those (successive halving).
* Treat ties as ties, and break saturated ties with continuous signals: held-out loss or margin, steps to threshold.
* Ensemble weakly correlated proxies (Song).
* Gate: the parent must correlate ≥ 0.8 with the long horizon before any child is trained.

**2. Imitate soft values, not the argmax**
* Train on the parent's per-candidate scores (ExIt's tree-policy targets, AggreVaTe), with tied candidates as uniform targets.
* Report regret as the primary metric; agreement can stay flat while decisions improve (+50 Elo at equal top-1 in ExIt).

**3. Give the child two heads, and re-rank with the value head (OptFormer)**
* The value head predicts the **1,500-step** outcome as a calibrated distribution.
* The child acts by drawing K proposals from its policy and re-ranking them by expected improvement.
* This is the only documented way an imitator surpassed its teacher.
* Spend the scarce long-run labels on calibration (value heads need many samples).

**4. Don't copy the parent's systematic errors**
* Confidence loss over the candidate ranking (α ramped 0 → 0.5–0.75).
* LOLS-style: a share (1−β) of the child's *own* choices is scored by real 1,500-step runs.
* Early-stop on long-horizon validation, not on agreement with the parent.

**5. The bias-inheritance test (our open question)**
* Train children on the biased parent and on the de-biased parent.
* Measure, against long-horizon truth:
  * gap recovered;
  * agreement split by whether the parent was right;
  * a bias signature: the share of picks with fast early gain but a poor 1,500-step outcome;
  * diversity of the choices.

**6. Generalization needs many families (AD)**
* Our 12 families and 48 universes are in the regime where AD failed.
* Generate ≥ 100 opcode families procedurally, split train/test by family, and pool out-of-domain data.
* Augment as OptFormer did:
  * permute candidate order;
  * normalise features per universe;
  * drop feature groups;
  * mask about 30% of tokens;
  * tag each example with the type of parent that produced it.

**7. The child should read histories, not just single decisions (AD)**
* Put earlier trials and outcomes from the same universe in context, mistakes included.

**8. Lineage credit and the expand/evaluate split (HGM)**
* Choose what to mutate by clade posteriors.
* Re-evaluate noisy winners on the node's own posterior.
* Use the widening rule N^0.6 ≥ |nodes|.
* Weight imitation examples by the clade metaproductivity of each edit's subtree.

**9. When the child acts as the mutator itself (EvoTune, SOAR)**
* Preference pairs only between siblings from the same context, with a margin or several seeds, because our verifier is noisy.
* DPO with forward KL against the previous mutator.
* Retrain from base on pooled data across generations.
* Select data 25 best + 25 worst; use hindsight relabelling.

**10. Recursion safeguards**
* Gate each handover: the new parent must beat the old one on held-out long-horizon runs (AlphaGo Zero; ExIt skipped this, but its verifier was exact).
* Expect saturation after about 2–3 generations unless the long-horizon re-checks continue (Song, LOLS).
* Keep the evaluator and lineage log read-only to the mutator (DGM's reward hacking).
* Always keep an archive; never just continue from the latest model (DGM: 23% vs 50%).
