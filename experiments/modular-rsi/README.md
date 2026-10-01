# Modular recursive self-improvement (experiment 1)

**Goal.** Show that (1) a population of *tiny* transformer modules can improve
itself recursively: not just the solutions, but the process that produces
them; and (2) the modules can bind into polymers/communities that solve
problems no single module can, so capability grows by *recruiting more modules*
rather than by retraining one big model.

Everything here runs on a 4-core CPU in under an hour. Results, figures and
the full lineage of every variant are committed under `results/` and `figures/`.

```
pip install -r requirements.txt
python run.py main --mode steered --seed 0        # Exp 1 (also --mode blind)
python run.py curriculum --improver inherit --seed 0   # Exp 2 (inherit | fresh | blind)
python run.py scale                                # Exp 3 (needs main_steered_s0)
python run.py report                               # figures/ + results/summary.json
```

RESULTS_PLACEHOLDER
