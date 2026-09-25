#!/usr/bin/env python3
"""Run a NEMO experiment.

    python3 scripts/run_experiment.py --preset smoke
    python3 scripts/run_experiment.py --preset pilot --out results/pilot
    NEMO_BACKEND=mlx python3 scripts/run_experiment.py --preset main

Presets map to the phases in DESIGN.md 10.  Every preset runs the SAME code
path for every condition -- conditions are lanes of one batched program, not
separate invocations -- so no result can be an artefact of one cell having been
run differently from another.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from nemo.config import ExperimentConfig, RunConfig, factorial_runs  # noqa: E402
from nemo.ecology.evolve import Experiment  # noqa: E402
from nemo.environments.world import calibrate  # noqa: E402
from nemo.modules.spec import ModuleSpec, match_monolithic  # noqa: E402
from nemo.storage.db import Store, dedup_stats  # noqa: E402


PRESETS = {
    # name:        islands, size, episodes, lifetime, generations, replicates
    "smoke":      (2,  8,  2,  16,   30, 1),
    "tiny":       (4,  8,  2,  24,  150, 2),
    "pilot":      (4, 16,  4,  32,  180, 4),
    "main":       (8, 32, 16, 512, 5000, 12),
}


def build_runs(replicates: int, subset: str) -> list[RunConfig]:
    if subset == "full":
        return factorial_runs(replicates)
    # Reduced design: the four lane-groups that carry the central contrast.
    runs: list[RunConfig] = []
    for r in range(replicates):
        runs.append(RunConfig(name=f"MVG_r{r}", goal_structure="MVG",
                              metabolism=True, duplication=True, seed=r))
        runs.append(RunConfig(name=f"RVG_r{r}", goal_structure="RVG",
                              metabolism=True, duplication=True, seed=r))
        runs.append(RunConfig(name=f"FIX_r{r}", goal_structure="FIX",
                              metabolism=True, duplication=True, seed=r))
        runs.append(RunConfig(name=f"DRIFT_r{r}", goal_structure="MVG",
                              metabolism=True, duplication=True,
                              shuffled_fitness=True, seed=r))
    return runs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=tuple(PRESETS), default="smoke")
    ap.add_argument("--subset", choices=("core", "full"), default="core")
    ap.add_argument("--control", choices=("none", "monolithic"),
                    default="none",
                    help="monolithic: one module, parameter- and "
                         "compute-matched to the modular ancestor. "
                         "Runs as a separate invocation because the "
                         "batched engine takes one module shape per run.")
    ap.add_argument("--out", default=None)
    ap.add_argument("--generations", type=int, default=None)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--p-dup", type=float, default=None,
                    help="override duplication AND deletion rate "
                         "(kept matched so the neutral genome-length "
                         "expectation stays flat)")
    ap.add_argument("--no-calibrate", action="store_true")
    ap.add_argument("--assay-every", type=int, default=0,
                    help="generations between observational assays "
                         "(0 = derive from preset)")
    args = ap.parse_args()

    islands, size, episodes, lifetime, gens, reps = PRESETS[args.preset]
    gens = args.generations or gens
    assay_every = args.assay_every or max(gens // 12, 5)
    out = Path(args.out or f"results/{args.preset}")
    out.mkdir(parents=True, exist_ok=True)

    cfg = ExperimentConfig()
    cfg.ecology.n_islands = islands
    cfg.ecology.island_size = size
    cfg.ecology.n_episodes = episodes
    cfg.ecology.lifetime = lifetime
    cfg.generations = gens
    cfg.out_dir = str(out)
    if args.p_dup is not None:
        # Matched, so the neutral expectation on genome length stays flat.
        # Raising both is the pilot's legitimate lever: at the design's 2% too
        # few duplicate pairs survive to a measurable age in a short run to
        # estimate the variance the power analysis needs (EXPERIMENTS.md D6).
        cfg.mutation.p_duplicate = args.p_dup
        cfg.mutation.p_delete = args.p_dup
    runs = build_runs(reps, args.subset)

    if args.control == "monolithic":
        base = ModuleSpec(d_model=cfg.module.d_model, d_ff=cfg.module.d_ff,
                          max_in_degree=cfg.module.max_in_degree)
        matched, info = match_monolithic(cfg.genome.n_genes_init, base)
        cfg.module.d_model, cfg.module.d_ff = matched.d_model, matched.d_ff
        cfg.genome.n_rounds, cfg.genome.slots_per_round = 1, 1
        cfg.genome.n_genes_init = 1
        for r in runs:
            r.duplication = False
            r.topology_mutable = False
        print("monolithic control, matched to "
              f"{info['target_params']:,} params / {info['target_flops']:,} FLOPs:")
        print(f"  d_model={info['d_model']} d_ff={info['d_ff']} -> "
              f"{info['matched_params']:,} params "
              f"({info['param_error']:+.1%}), "
              f"{info['matched_flops']:,} FLOPs ({info['flop_error']:+.1%})")

    ex = Experiment(cfg=cfg, runs=runs)
    if not args.no_calibrate:
        # Difficulty matching across goal structures (DESIGN.md 5.2).
        ex.calib = calibrate(cfg.environment.n_symbols, cfg.environment.n_actions,
                             cfg.environment.n_obs_channels, steps=120, n=1024)

    store = Store(out / "nemo.sqlite")
    names = [r.name for r in runs]
    meta = {
        "preset": args.preset, "subset": args.subset,
        "control": args.control, "generations": gens,
        "lanes": int(ex.pop.L), "runs": names, "config": cfg.to_dict(),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=str))

    print(f"NEMO {args.preset}: {len(runs)} runs x {cfg.ecology.n_organisms} organisms "
          f"= {ex.pop.L} lanes, E={episodes}, lifetime={lifetime}, {gens} generations")
    t0 = time.time()
    for g in range(gens):
        recs = ex.run_generation(sample_q=4 if g % 5 == 0 else 0)
        store.write_generations(recs, names)
        if ex.events:
            store.write_events(ex.events)
            ex.events.clear()
        if (g + 1) % assay_every == 0 or g == gens - 1:
            ex.assay(n_lanes_per_run=6, episodes=2, lifetime=max(16, lifetime // 2))
            store.write_pair_observations(ex.pair_obs)
            store.write_organism_observations(ex.org_obs)
            ex.pair_obs.clear()
            ex.org_obs.clear()
        if g % args.log_every == 0 or g == gens - 1:
            el = time.time() - t0
            eta = el / (g + 1) * (gens - g - 1)
            line = " | ".join(
                f"{names[r.run].split('_r')[0]} rew={r.mean_reward:+.4f} "
                f"len={r.mean_genome_len:.2f}"
                for r in recs[:4])
            print(f"  gen {g:>5}/{gens}  {line}  [{el:.0f}s elapsed, ~{eta:.0f}s left]",
                  flush=True)

    store.write_module_lineage(ex.registry)
    store.write_duplicate_pairs(ex.duplicate_pairs)
    dd = dedup_stats(ex.pop)
    (out / "summary.json").write_text(json.dumps({
        "seconds": time.time() - t0,
        "generations": gens,
        "s_per_generation": (time.time() - t0) / gens,
        "n_duplicate_pairs": len(ex.duplicate_pairs),
        "assay_every": assay_every,
        "n_modules_ever": ex.registry.next_id,
        "module_dedup": dd,
    }, indent=2))
    np.savez_compressed(out / "final_population.npz",
                        **{n: getattr(ex.pop, n) for n in
                           ("alive", "src", "src_mask", "innov", "gene_parent",
                            "birth_gen", "gate_b", "gate_s", "run_id", "island",
                            "org_id", "org_parent")})
    store.close()
    print(f"done in {time.time()-t0:.0f}s -> {out}")
    print(f"  module dedup: {dd['distinct']}/{dd['total']} distinct "
          f"({dd['distinct_fraction']:.1%})")


if __name__ == "__main__":
    main()
