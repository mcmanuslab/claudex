# seed/ — a go/no-go test of developmental architecture growth

**Question.** Does a transformer whose architecture grows in response to its own
failures become better at predicting structure that was deliberately withheld
from it — at matched training compute — than one whose architecture was fixed
in advance?

This is a **falsification attempt**, not a demo. A clean negative is a success.

## Read in this order

| file | what it is |
|---|---|
| `PRIOR_ART.md` | what already exists, what we reuse, and the narrow slice that is actually new |
| `EXPERIMENT_001.md` | **the frozen protocol** — groups, endpoints, go/no-go thresholds, confounders. Written before any comparison run. |
| `DESIGN.md` | why the code is shaped the way it is |
| `RESEARCH_LOG.md` | chronological, including the four pilots that were thrown away |
| `RESULTS_001.md` | results, uncertainty, and the strongest alternative explanation |

## Code

```
world.py       the hidden world: latent coordinates, rules, extrapolation buckets
model.py       tiny pre-norm decoder-only transformer whose structure can change
growth.py      growth operators, function-preservation checks, competition, pruning
controller.py  the developmental controller + the matched random-schedule control
ledger.py      hash-chained, append-only prediction ledger (contains no ground truth)
train.py       one run
metrics.py     scoring: VNPR, information gain, calibration, leakage alarm
evaluate.py    join truth, aggregate, paired statistics
visualize.py   the plots
```

## Reproduce

```bash
pip install torch matplotlib
python3 tests/test_gates.py                       # 23 assertions, incl. Gate 1
python3 run_experiment.py --config configs/exp001_cpu.json --seeds 0,1,2,3,4
python3 evaluate.py  --runs 'results/exp001/*/'
python3 visualize.py
```

## Two things to know before reading any number

**Scale.** The brief asked for 10–30M starting parameters. The container is 4
CPU cores with no GPU, where that is roughly 170x out of reach. This runs at
**0.077M → ~0.16M** parameters. `configs/exp001_paper_scale.json` holds the
requested size so the identical protocol can be launched on a GPU unchanged.
No result here should be assumed to transfer upward.

**Gate 1 is not a finding.** Function-preserving growth is a reproduction of
known work. `logit_delta == 0.0` exactly, asserted in the test suite. It is a
precondition, and if it had failed it would have meant a bug.
