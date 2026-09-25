"""Run an evolutionary experiment from a config file.

Structure of a run:

* a sealed :class:`SuiteSplit` partitions world families into A / B / C-dev / C-test;
* each island draws its worlds from its *own* overlapping subset of Class A families, so
  no single benchmark exists for the whole population to overfit;
* every ``probe_every`` generations a cohort is **frozen** (deep-copied) and evaluated on
  held-out suites under the transplant conditions in :mod:`openevo.metrics.probe`.

The probe is deliberately structured so it cannot leak: it runs on copies, returns plain
numbers, and its result is written to the database and nowhere else. No selection,
mutation, stopping or sampling decision reads it. Class C-test stays closed unless
``--open-test-set`` is passed, which is intended to happen once per pre-registered
experiment.
"""

from __future__ import annotations

import argparse
import copy
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from openevo.environments.suites import (  # noqa: E402
    Scored, SuiteSplit, build_suite, reference_profiles, reference_scores, sample_spec,
)
from openevo.evolution.organism import PhaseConfig  # noqa: E402
from openevo.evolution.population import EvoConfig, Population  # noqa: E402
from openevo.metrics.complexity import (  # noqa: E402
    effective_params, probe_batch, stack_activity,
)
from openevo.metrics.probe import CONDITIONS, run_probes  # noqa: E402
from openevo.models.genome import scale_to_params  # noqa: E402
from openevo.storage.run import RunStore  # noqa: E402

DEFAULTS = {
    "name": "pilot", "generations": 30, "seed": 0,
    "founder_params": [5000], "displaced_founder_params": None,
    "probe_every": 5, "probe_cohort": 8, "probe_worlds": 6, "probe_instances": 8,
    "probe_conditions": list(CONDITIONS), "open_test_set": False,
    "checkpoint_every": 0, "island_family_fraction": 0.6,
    "effective_params": True, "effective_cohort": 3,
    "evo": {}, "phase": {},
}


def load_config(path: str | None, overrides: list[str]) -> dict:
    cfg = json.loads(json.dumps(DEFAULTS))
    if path:
        user = json.loads(Path(path).read_text())
        for k, v in user.items():
            if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                cfg[k].update(v)
            else:
                cfg[k] = v
    for ov in overrides:
        k, _, v = ov.partition("=")
        tgt, key = (cfg, k)
        if "." in k:
            head, key = k.split(".", 1)
            tgt = cfg[head]
        try:
            tgt[key] = json.loads(v)
        except json.JSONDecodeError:
            tgt[key] = v
    return cfg


def make_world_sampler(split: SuiteSplit, cfg: dict, evo: EvoConfig):
    """Per-island, partially overlapping environment distributions.

    Each island gets its own random subset of the Class A families. The subsets overlap,
    so islands share some ecology, but no island sees the whole distribution and the
    population as a whole has no single benchmark to converge on.
    """
    rng = np.random.default_rng(cfg["seed"] + 991)
    pool = split.pool("A")
    frac = cfg["island_family_fraction"]
    n = max(4, int(round(frac * len(pool))))
    island_pools = {i: [pool[j] for j in rng.choice(len(pool), n, replace=False)]
                    for i in range(evo.n_islands)}

    def sampler(island: int, generation: int):
        r = np.random.default_rng([cfg["seed"], island, generation])
        fams = island_pools[island]
        out, tries = [], 0
        while len(out) < evo.worlds_per_island and tries < evo.worlds_per_island * 40:
            tries += 1
            fam = fams[int(r.integers(0, len(fams)))]
            spec = sample_spec(fam, r, seed=int(r.integers(0, 2**31)))
            lo, hi = reference_scores(spec, seed=spec.seed)
            if hi - lo >= 0.08:
                lb, hb = reference_profiles(spec, seed=spec.seed)
                out.append(Scored(spec, lo, hi, lb, hb))
        return out

    return sampler


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=JSON")
    ap.add_argument("--out", type=str, default="results")
    ap.add_argument("--open-test-set", action="store_true",
                    help="Open the sealed Class C-test suite. Once per experiment.")
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--force", action="store_true",
                    help="Overwrite an existing results database for this run name.")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    cfg["open_test_set"] = cfg["open_test_set"] or args.open_test_set
    evo = EvoConfig(seed=cfg["seed"], **cfg["evo"])
    phase = PhaseConfig(**cfg["phase"])
    rng = np.random.default_rng(cfg["seed"])

    split = SuiteSplit()
    probe_classes = ["A", "B", "C_dev"] + (["C_test"] if cfg["open_test_set"] else [])
    # Probe suites are drawn once, from a seed independent of the evolutionary run, so
    # every generation's cohort is measured against the same worlds.
    suites = {c: build_suite(split, c, cfg["probe_worlds"],
                             np.random.default_rng(777_000 + i))
              for i, c in enumerate(probe_classes)}

    founders = [scale_to_params(p) for p in cfg["founder_params"]]
    if cfg["displaced_founder_params"]:
        # McShea's subclade test: seed some islands well above the bulk of the size
        # distribution. If the trend is driven, displaced lineages keep growing; if it is
        # passive diffusion off the lower bound, they regress towards the bulk.
        founders += [scale_to_params(p) for p in cfg["displaced_founder_params"]]

    out_dir = Path(args.out) / cfg["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    db_path = out_dir / "run.db"
    if db_path.exists() and not (args.resume or args.force):
        raise SystemExit(
            f"{db_path} already exists. A run must not append to a previous run's "
            f"database -- the metrics would silently interleave. Pass --force to "
            f"overwrite, --resume to continue, or choose another --set name=...")
    if db_path.exists() and args.force and not args.resume:
        db_path.unlink()
    store = RunStore(db_path)
    cfg_hash = store.write_manifest(
        config=cfg, evo=evo, phase=phase,
        suite_seal=split.seal(), suite_summary=split.summary(),
        probe_classes=probe_classes,
        founder_archs=[f.to_dict() for f in founders],
    )

    pop = Population(evo, phase, founders, rng)
    if args.resume and Path(args.resume).exists():
        with open(args.resume, "rb") as fh:
            pop = pickle.load(fh)
        print(f"resumed at generation {pop.generation}")

    sampler = make_world_sampler(split, cfg, evo)
    ancestral = founders[0]
    t0 = time.time()
    if not args.quiet:
        print(f"run={cfg['name']} hash={cfg_hash} phase={phase} seal={split.seal()[:12]}")
        print(f"{'gen':>4} {'alive':>5} {'med.par':>8} {'p_min':>7} {'p_max':>8} "
              f"{'score':>7} {'gain':>6} {'grow':>5} {'spp':>4} {'qd':>4} "
              f"{'stack':>6} {'sec':>6}")

    for _ in range(cfg["generations"]):
        snap = pop.step(sampler)
        # Standing health check: is the transformer stack doing anything at all? One
        # extra forward pass. The first pilot ran to completion with an inert stack and
        # no aggregate metric noticed, so this is logged every generation.
        live = pop.living()
        if live:
            champ = max(live, key=lambda o: o.fitness)
            ob, pa, pr = probe_batch(champ.arch, np.random.default_rng(31337), batch=4)
            snap["stack_activity"] = stack_activity(champ.weights, champ.arch, ob, pa, pr)
        store.add_generation(snap)
        store.add_organisms(pop.records)
        pop.records.clear()

        if cfg["probe_every"] and pop.generation % cfg["probe_every"] == 0:
            live = pop.living()
            k = min(cfg["probe_cohort"], len(live))
            order = np.argsort([-o.fitness for o in live])
            cohort = [copy.deepcopy(live[i]) for i in order[:k]]   # frozen copies
            for cls, cond, m in run_probes(
                cohort, suites, ancestral, np.random.default_rng(4242),
                tuple(cfg["probe_conditions"]), cfg["probe_instances"]):
                store.add_probe(pop.generation, cls, cond, m)
            # Effective (ablation-surviving) parameters, so that a rise in raw parameter
            # count can be distinguished from bloat -- PREREGISTRATION.md H3 criterion 4.
            if cfg["effective_params"]:
                eff = []
                for o in cohort[: cfg["effective_cohort"]]:
                    ob, pa, pr = probe_batch(o.arch, np.random.default_rng(31337))
                    eff.append(effective_params(o.weights, o.arch, ob, pa, pr))
                if eff:
                    store.add_probe(pop.generation, "-", "effective_params",
                                    {k: float(np.mean([e[k] for e in eff]))
                                     for k in eff[0]})
        store.commit()

        if cfg["checkpoint_every"] and pop.generation % cfg["checkpoint_every"] == 0:
            with open(out_dir / "checkpoint.pkl", "wb") as fh:
                pickle.dump(pop, fh)

        if not args.quiet:
            print(f"{snap['generation']:>4} {snap['n_alive']:>5} "
                  f"{snap['params_median']:>8.0f} {snap['params_min']:>7.0f} "
                  f"{snap['params_max']:>8.0f} {snap['score_mean']:>7.3f} "
                  f"{snap['gain_mean']:>6.3f} {snap['gene_p_growth_bias']:>5.2f} "
                  f"{snap['n_species']:>4} {snap['archive_coverage']:>4} "
                  f"{snap.get('stack_activity', 0.0):>6.3f} {time.time() - t0:>6.1f}")

    store.commit()
    store.close()
    print(f"\ndone in {time.time() - t0:.1f}s -> {out_dir}/run.db "
          f"({pop.total_births} births, {pop.total_flops / 1e9:.1f} GFLOP accounted)")


if __name__ == "__main__":
    main()
