# Debate brief: what should the next step of the modular-RSI program be?

## The user's goal (in their words, condensed)
Prove a recursive self-improving (RSI) transformer can be created, with the twist that it is
MODULAR: start from tiny transformer "monomers" (1–5k params, "the smaller the better"), generate
thousands–millions of mutated variants, select the best, remember the lineage paths that led to
success and steer future mutation around them, let the best modules bind into "polymers" or
communities that solve bigger problems than any monomer, dissociate, repeat for n cycles.
Next experiments increase polymer/community n to solve ever larger problems — the long-run vision
is growing modularly to 1B+ (eventually 1T) parameters, efficiently, because we start tiny and modular.
The user wants to observe EXPONENTIAL growth, and insists "as it grows, it needs to get smarter".
Most recently the user asked for a scalable USEFUL skill: natural language — a tiny transformer
(~100 bytes ideal) that first concatenates characters into words, then words into sentences, and so on,
as the RSI modules evolve.

## What exists (read the code and results — do not rely only on this summary)
Repository: /home/user/claudex/experiments/modular-rsi  (README.md is the full write-up; figures/ and results/)
- modrsi/: monomer (3.9k-param 1-layer transformer on a symbolic relational-lookup world), evolve.py
  (steered mutation + Lamarckian lifetime learning, island/PBT improver selection, restarts),
  polymer.py, hierarchy.py (frozen macros), run.py (Exp 1–5), report.py.
- Exp 1–5 results (see README): steered > blind (43 vs 55 gens); polymers solve multi-step problems
  monomers can't; router community scales to n=12 programs; hierarchical macro-freezing grows solvable
  problem length ×2.54/cycle (3 → 12,717 steps) at ~constant search cost; gene duplication + path
  credit halve the cost of new skills vs scratch, but no compounding with library size shown.
  Important caveats: symbolic toy domain; exponential growth is partly built into a geometric curriculum
  of compositional problems; table-composition shortcut for macros (verified by execution up to 1,771 steps).
- lang/: char-level language track on Alice in Wonderland (28 chars). probe_size.py: a single 1-layer
  char transformer needs ~1 KB to spell common words, ~6 KB → 69% real words; 116 bytes → letter soup.
  grow_spell.py (Level 1): community of 340-byte modules that VOTE (sum logits), grown one frozen module
  per cycle with RSI candidate selection. Result at ~11 KB: RSI growth 2.87 bits/char, 0.50 real words;
  fresh-only growth 3.10 / 0.44; single model same size 2.39 / 0.82. Voting communities lose clearly
  to a monolith; real-word rate plateaus ~0.5.

## The plan under evaluation ("Level 1b")
Stacked community: each new module reads the frozen community's hidden state (a shared residual stream)
and adds to it, so new modules build on old ones (depth, not just width). Old modules stay frozen;
growth incremental; RSI candidate selection as before. Hope: close the gap to the monolith, then go on
to words → sentences.

## Constraints
4 CPU cores, no GPU, ~hours of compute per experiment, torch/numpy available. Internet: GitHub raw
files reachable; HuggingFace and Gutenberg blocked. Honest reporting matters more than flattering results.
