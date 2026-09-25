# pilot_inert_stack — the run that caught the absorbing-state bug

Kept deliberately. This run's fitness rose steadily (0.019 → 0.206), Class C-dev
adaptation AUC rose from ~0 to +0.155, 246 distinct architectures appeared, and every
headline curve looked like a working experiment.

It was not. Weight mutation used a purely scale-relative step (`sigma * rms(v)`), which
makes **zero an absorbing state** — and function-preserving growth deliberately starts
every block output path (`Wo`, `W2`) at exactly zero. Those tensors had an effective
mutation size of ~5e-6 and never left zero, so the attention and FFN stacks stayed pinned
at the identity for the entire run. Evolution was optimising nothing but the embeddings
and the output head.

What caught it: the effective-parameter ablation found **0 of 99 units** in the champion
doing anything, and ablating *every* block output path changed the policy by a total
variation of 0.00019. Nothing in the fitness curves, the architecture statistics or the
adaptation numbers would have revealed it.

Two things follow. First, the `no_feedback` result in `report.md` — that improvement was a
reactive prior rather than in-context learning — is exactly what a policy with no working
attention would produce, so it should not be read as a finding about evolution. Second,
this is the case for measuring effective rather than raw capacity: a system can look
healthy on every aggregate metric while most of it does nothing.

Fixed in `evolution/organism.py:_perturb_weights` by flooring the perturbation at the
scale the tensor would have had at initialisation. Regression tests:
`tests/test_neutrality.py::test_zero_tensors_can_escape_zero_under_mutation` and
`::test_evolved_lineage_actually_uses_its_transformer`.
