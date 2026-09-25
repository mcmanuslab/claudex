#!/usr/bin/env python3
"""Empirically find the smallest module capable of useful behaviour.

The original proposal asks for ~500 / 1K / 2K / 3K / 5K / 10K parameters and is
explicit that the architecture should not be assumed viable.  Arithmetic alone
cannot answer this: it says what fits, not what works.

Protocol.  For each module size, run short independent evolutionary runs under
MVG with metabolism on, plus a seed-matched fitness-shuffled drift control, and
report the reward gap between them.  "Useful behaviour" is operationalised as
*reward reliably above the drift control*, not as reward above zero -- an
organism can score above zero by luck, and the drift control absorbs exactly
that.

All sizes run as lanes of ONE batched program, so no size gets a different code
path, and per-size wall clock differences reflect only the module shape.

    python3 scripts/module_sweep.py --generations 120
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from nemo.config import ExperimentConfig, RunConfig  # noqa: E402
from nemo.ecology.evolve import Experiment  # noqa: E402
from nemo.environments.world import calibrate  # noqa: E402
from nemo.metrics.definitions import cliffs_delta  # noqa: E402
from nemo.modules.spec import ModuleSpec  # noqa: E402

# (d_model, d_ff) -> roughly 150 / 560 / 1.2K / 2.1K / 3.3K / 4.8K / 8.4K params
SIZES = [(4, 8), (8, 16), (12, 24), (16, 32), (20, 40), (24, 48), (32, 64)]


def run_one(d_model: int, d_ff: int, replicates: int, generations: int,
            islands: int, size: int, episodes: int, lifetime: int,
            calib) -> dict:
    cfg = ExperimentConfig()
    cfg.module.d_model, cfg.module.d_ff = d_model, d_ff
    cfg.ecology.n_islands, cfg.ecology.island_size = islands, size
    cfg.ecology.n_episodes, cfg.ecology.lifetime = episodes, lifetime

    runs = []
    for r in range(replicates):
        runs.append(RunConfig(name=f"MVG_r{r}", goal_structure="MVG", seed=r))
        runs.append(RunConfig(name=f"DRIFT_r{r}", goal_structure="MVG",
                              shuffled_fitness=True, seed=r))
    ex = Experiment(cfg=cfg, runs=runs)
    ex.calib = calib

    t0 = time.time()
    tail_mvg: list[list[float]] = [[] for _ in range(replicates)]
    tail_drift: list[list[float]] = [[] for _ in range(replicates)]
    cut = int(generations * 0.75)
    for g in range(generations):
        recs = ex.run_generation(sample_q=0)
        if g >= cut:
            for rec in recs:
                rep = rec.run // 2
                (tail_mvg if rec.run % 2 == 0 else tail_drift)[rep].append(rec.mean_reward)
    dt = time.time() - t0

    mvg = np.array([np.mean(v) for v in tail_mvg if v])
    drift = np.array([np.mean(v) for v in tail_drift if v])
    spec = ModuleSpec(d_model=d_model, d_ff=d_ff,
                      max_in_degree=cfg.module.max_in_degree)
    return {
        "d_model": d_model, "d_ff": d_ff, "params": spec.n_params,
        "flops": spec.flops_forward(),
        "mvg_reward": float(mvg.mean()), "drift_reward": float(drift.mean()),
        "gap": float(mvg.mean() - drift.mean()),
        "cliffs_delta": cliffs_delta(mvg, drift),
        "seconds": dt, "s_per_generation": dt / generations,
        "n_replicates": len(mvg),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", type=int, default=120)
    ap.add_argument("--replicates", type=int, default=4)
    ap.add_argument("--islands", type=int, default=2)
    ap.add_argument("--island-size", type=int, default=16)
    ap.add_argument("--episodes", type=int, default=4)
    ap.add_argument("--lifetime", type=int, default=32)
    ap.add_argument("--out", type=Path, default=Path("results/module_sweep.json"))
    args = ap.parse_args()

    print("Calibrating environment difficulty ...", flush=True)
    calib = calibrate(16, 8, 2, steps=120, n=1024)

    print(f"\nModule size sweep: {args.replicates} MVG + {args.replicates} drift "
          f"replicates per size, {args.generations} generations\n")
    print(f"{'d_model':>8} {'params':>8} {'FLOP':>8} {'MVG':>9} {'DRIFT':>9} "
          f"{'gap':>9} {'delta':>7} {'s/gen':>7}")
    rows = []
    for d, f in SIZES:
        r = run_one(d, f, args.replicates, args.generations, args.islands,
                    args.island_size, args.episodes, args.lifetime, calib)
        rows.append(r)
        print(f"{r['d_model']:>8} {r['params']:>8,} {r['flops']:>8,} "
              f"{r['mvg_reward']:>9.4f} {r['drift_reward']:>9.4f} "
              f"{r['gap']:>9.4f} {r['cliffs_delta']:>7.2f} "
              f"{r['s_per_generation']:>7.3f}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2))

    print("\nInterpretation")
    print("  'gap' is reward above the seed-matched fitness-shuffled control.")
    print("  A size is viable when the gap is positive with a large effect")
    print("  (Cliff's delta >= 0.47).  The smallest viable size is the one to")
    print("  use: a larger module substitutes internal capacity for")
    print("  organisation, which is the thing this project is trying to")
    print("  measure the evolution of.")
    viable = [r for r in rows if r["gap"] > 0 and r["cliffs_delta"] >= 0.47]
    if viable:
        best = min(viable, key=lambda r: r["params"])
        print(f"\n  smallest viable: d_model={best['d_model']}, "
              f"{best['params']:,} parameters (delta {best['cliffs_delta']:.2f})")
    else:
        print("\n  No size reached the viability threshold at this budget.")
        print("  Increase --generations or --lifetime before concluding that")
        print("  the module architecture is at fault.")
    print(f"\nwritten -> {args.out}")


if __name__ == "__main__":
    main()
