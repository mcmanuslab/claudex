#!/usr/bin/env python3
"""Measure alien-world adaptation speed for the fossils of a run.

    python3 scripts/alien_test.py results/pilot --generations 40

This answers the question the project is actually about: does the organisation
evolution discovered make FUTURE adaptation faster?  Fitness going up during a
run does not answer it; adaptation speed on causal mechanisms the lineage has
never been selected on does.

Every number is reported against the run's own fitness-shuffled drift control,
at a matched offspring budget.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from nemo.config import ExperimentConfig, RunConfig  # noqa: E402
from nemo.ecology.alien import adapt, alien_goals  # noqa: E402
from nemo.environments import primitives as P  # noqa: E402
from nemo.environments.world import calibrate  # noqa: E402
from nemo.genome.population import init_population  # noqa: E402
from nemo.metrics.definitions import cliffs_delta  # noqa: E402


def condition_of(name: str) -> str:
    return str(name).split("_r")[0]


def load_fossil(out: Path, cfg: ExperimentConfig, path: Path):
    """Rebuild a Population from a saved snapshot."""
    z = np.load(path)
    L = z["alive"].shape[0]
    pop = init_population(cfg, L, z["run_id"], seed=0)
    for k in z.files:
        if hasattr(pop, k):
            setattr(pop, k, z[k])
    return pop


def slice_lanes(pop, lanes: np.ndarray):
    import copy as _c
    sub = _c.copy(pop)
    sub.L = len(lanes)
    for name in ("Wq", "Wk", "Wv", "Wo", "W1", "b1", "W2", "b2", "g1", "g2",
                 "wg", "bg", "alive", "src", "src_mask", "gate_b", "gate_s",
                 "E_obs", "W_act", "b_act", "meta", "innov", "gene_parent",
                 "birth_gen", "org_id", "org_parent", "run_id", "island"):
        setattr(sub, name, getattr(pop, name)[lanes].copy())
    return sub


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--generations", type=int, default=40)
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--lifetime", type=int, default=32)
    ap.add_argument("--fossil", default="final_population.npz")
    args = ap.parse_args()

    meta = json.loads((args.out / "meta.json").read_text())
    names = meta["runs"]
    cfg = ExperimentConfig()
    c = meta["config"]
    cfg.module.d_model = c["module"]["d_model"]
    cfg.module.d_ff = c["module"]["d_ff"]
    cfg.genome.n_rounds = c["genome"]["n_rounds"]
    cfg.genome.slots_per_round = c["genome"]["slots_per_round"]
    cfg.ecology.n_islands = c["ecology"]["n_islands"]
    cfg.ecology.island_size = c["ecology"]["island_size"]

    pop = load_fossil(args.out, cfg, args.out / args.fossil)
    calib = calibrate(16, 8, 2, steps=120, n=1024)
    goals = alien_goals(2)
    print(f"Alien pool: {[P.NAMES[p] for p in P.ALIEN_POOL]}")
    print(f"Testing {len(goals)} alien goals x {len(set(map(condition_of, names)))} "
          f"conditions, {args.generations} generations each\n")

    results: dict[str, list[float]] = {}
    detail = []
    n_org = cfg.ecology.n_organisms
    for ri, name in enumerate(names):
        lanes = np.arange(ri * n_org, (ri + 1) * n_org)
        if lanes.max() >= pop.L:
            continue
        sub = slice_lanes(pop, lanes)
        sub.island = np.repeat(np.arange(cfg.ecology.n_islands),
                               cfg.ecology.island_size)[:len(lanes)]
        for gi, goal in enumerate(goals):
            r = adapt(sub, cfg, RunConfig(name=name), goal, args.generations,
                      seed=1000 * ri + gi, episodes=args.episodes,
                      lifetime=args.lifetime, calib=calib)
            results.setdefault(condition_of(name), []).append(r.auc)
            detail.append({"run": name, "goal": [P.NAMES[g] for g in goal],
                           "pre": r.pre, "auc": r.auc, "births": r.births,
                           "final": r.curve[-1] if r.curve else None})
        print(f"  {name:<12} mean AUC "
              f"{np.mean(results[condition_of(name)]):+.5f}", flush=True)

    print("\n" + "=" * 68)
    print("ALIEN-WORLD ADAPTATION (normalised AUC, pre-adaptation as covariate)")
    print("=" * 68)
    print(f"\n{'condition':<10} {'mean AUC':>11} {'n':>5} {'delta vs DRIFT':>16}")
    drift = np.array(results.get("DRIFT", []))
    for cond in sorted(results):
        v = np.array(results[cond])
        d = cliffs_delta(v, drift) if len(drift) else float("nan")
        print(f"{cond:<10} {v.mean():>+11.5f} {len(v):>5} {d:>16.3f}")
    if not len(drift):
        print("\n  No DRIFT lanes in this run -- no null, so no claim.")

    dest = args.out / "alien_adaptation.json"
    dest.write_text(json.dumps(
        {"by_condition": {k: list(map(float, v)) for k, v in results.items()},
         "detail": detail}, indent=2))
    print(f"\nwritten -> {dest}")


if __name__ == "__main__":
    main()
