#!/usr/bin/env python3
"""Backend benchmark for the M3 Ultra.

This script measures the five hardware constants that `scripts/compute_budget.py`
assumes, and attaches a decision rule to each.  It CANNOT be run in the
development container (Linux, no Apple Silicon); it is written to be run on the
target machine:

    NEMO_BACKEND=mlx  python3 scripts/bench_backend.py --device gpu
    NEMO_BACKEND=mlx  python3 scripts/bench_backend.py --device cpu
    NEMO_BACKEND=numpy python3 scripts/bench_backend.py

What it measures, and why each one matters
------------------------------------------
B1  per-dispatch overhead
    THE number to measure first.  The model is 4x sensitive to it across the
    plausible 5-100 us range, and it decides whether the pilot is dispatch-
    bound or FLOP-bound.  Measured as the slope of (time vs number of trivially
    small sequential kernels).

B2  batched-GEMM throughput at module scale
    The (L, S, E, d) @ (L, S, d, d) shape the rollout actually issues, swept
    over L.  Gives the achievable fraction of peak and locates the
    dispatch/FLOP crossover.

B3  gather/scatter cost
    Message routing is take_along_axis + concatenate.  If routing costs more
    than the module compute it dominates, and the round structure needs
    rethinking.

B4  end-to-end rollout step
    The real thing: `nemo.organisms.execute.step` on a real Population.
    Reported as module-steps/sec, which is the only throughput number that
    matters for planning generations.

B5  mutation + copy throughput
    Host-side births per second.  If this exceeds ~10% of rollout time the
    birth path needs vectorising too.

Decision rules are printed with the results.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402


def timeit(fn, n_warm: int = 3, n_rep: int = 10) -> float:
    for _ in range(n_warm):
        fn()
    ts = []
    for _ in range(n_rep):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def sync(B):
    """Force completion.  MLX is lazy, so without this every timing is a lie."""
    def _f(*xs):
        B.eval_(*xs)
    return _f


def bench_dispatch(B, xp) -> dict:
    """B1: slope of time vs kernel count on deliberately tiny work."""
    x = xp.array(np.random.randn(64, 64).astype(np.float32))
    out = {}
    for n_ops in (1, 10, 50, 200):
        def run(n=n_ops):
            y = x
            for _ in range(n):
                y = y + 1.0
            B.eval_(y)
        out[n_ops] = timeit(run)
    ns = np.array(sorted(out))
    ts = np.array([out[n] for n in ns])
    slope = float(np.polyfit(ns, ts, 1)[0])
    return {"per_dispatch_us": slope * 1e6, "raw": {int(k): v * 1e6 for k, v in out.items()}}


def bench_batched_gemm(B, xp, d: int = 16, S: int = 4, E: int = 8) -> dict:
    """B2: the exact GEMM shape the rollout issues, swept over lane count."""
    res = {}
    for L in (256, 1024, 4096, 16384, 65536):
        h = xp.array(np.random.randn(L, S, E, d).astype(np.float32))
        W = xp.array(np.random.randn(L, S, d, d).astype(np.float32))

        def run():
            B.eval_(xp.matmul(h, W))

        t = timeit(run, n_rep=5)
        flops = 2 * L * S * E * d * d
        res[L] = {"ms": t * 1e3, "gflops": flops / t / 1e9}
    return res


def bench_gather(B, xp, d: int = 16, G: int = 16, K: int = 4, E: int = 8) -> dict:
    res = {}
    n_slots = G + 3
    for L in (1024, 16384, 65536):
        buf = xp.array(np.random.randn(L, n_slots, E, d).astype(np.float32))
        idx = xp.array(np.random.randint(0, n_slots, size=(L, G, K)).astype(np.int32))

        def run():
            B.eval_(B.gather_slots(buf, idx))

        res[L] = timeit(run, n_rep=5) * 1e3
    return res


def bench_rollout(cfg_lanes: int, lifetime: int, episodes: int) -> dict:
    from nemo.config import ExperimentConfig
    from nemo.genome.population import init_population
    from nemo.organisms.execute import new_state, step

    cfg = ExperimentConfig()
    cfg.ecology.n_episodes = episodes
    pop = init_population(cfg, cfg_lanes, np.zeros(cfg_lanes))
    st = new_state(pop, episodes)
    rng = np.random.default_rng(0)
    obs = rng.integers(0, cfg.environment.n_symbols,
                       size=(cfg_lanes, cfg.environment.n_obs_channels, episodes))

    t0 = time.perf_counter()
    for _ in range(lifetime):
        step(pop, st, obs, cfg.gate_threshold)
    dt = time.perf_counter() - t0
    module_steps = cfg_lanes * episodes * int(pop.alive.sum(1).mean()) * lifetime
    return {"seconds": dt, "s_per_timestep": dt / lifetime,
            "module_steps_per_s": module_steps / dt,
            "projected_s_per_generation": dt}


def bench_births(n: int = 2000) -> dict:
    from nemo.config import ExperimentConfig, RunConfig
    from nemo.genome.population import init_population
    from nemo.mutation.operators import InnovationRegistry, mutate

    cfg = ExperimentConfig()
    pop = init_population(cfg, 64, np.zeros(64))
    reg = InnovationRegistry(next_id=8)
    rng = np.random.default_rng(0)
    run = RunConfig()
    t0 = time.perf_counter()
    for i in range(n):
        mutate(pop, i % 64, rng, cfg.mutation, run, reg, i)
    dt = time.perf_counter() - t0
    return {"births_per_s": n / dt, "us_per_birth": dt / n * 1e6}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    ap.add_argument("--lanes", type=int, default=4096)
    ap.add_argument("--lifetime", type=int, default=64)
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--skip-heavy", action="store_true")
    args = ap.parse_args()

    if os.environ.get("NEMO_BACKEND", "numpy") == "mlx":  # pragma: no cover
        import mlx.core as mx
        mx.set_default_device(mx.gpu if args.device == "gpu" else mx.cpu)

    from nemo import backend as B

    print("=" * 74)
    print(f"NEMO backend benchmark -- backend={B.NAME} device={args.device}")
    print("=" * 74)

    print("\n[B1] per-dispatch overhead")
    d1 = bench_dispatch(B, B.xp)
    print(f"  measured: {d1['per_dispatch_us']:.2f} us/dispatch")
    print(f"  raw (us per batch): {d1['raw']}")
    print("  DECISION: model assumes 25 us.  <10 us -> raise episodes/lanes until")
    print("            FLOP-bound.  >50 us -> reduce exec rounds R from 4 to 2, or")
    print("            fuse the per-round kernels, before scaling the experiment.")

    print("\n[B2] batched GEMM at module scale (L,S,E,d)@(L,S,d,d)")
    d2 = bench_batched_gemm(B, B.xp)
    print(f"  {'lanes':>8} {'ms':>9} {'GFLOP/s':>10}")
    for L, r in d2.items():
        print(f"  {L:>8,} {r['ms']:>9.3f} {r['gflops']:>10.1f}")
    print("  DECISION: pick the smallest lane count on the throughput plateau.")
    print("            Lanes below the plateau are free; above it they cost linearly.")

    print("\n[B3] gather/scatter (message routing)")
    d3 = bench_gather(B, B.xp)
    for L, ms in d3.items():
        print(f"  lanes {L:>7,}: {ms:>8.3f} ms")
    print("  DECISION: if routing > 50% of [B2] at the same lane count, the")
    print("            round structure is the bottleneck, not the modules.")

    if not args.skip_heavy:
        print("\n[B4] end-to-end rollout step")
        d4 = bench_rollout(args.lanes, args.lifetime, args.episodes)
        print(f"  {args.lanes:,} lanes x {args.episodes} episodes x {args.lifetime} steps")
        print(f"  {d4['seconds']:.3f} s   {d4['s_per_timestep']*1e3:.3f} ms/timestep")
        print(f"  {d4['module_steps_per_s']:,.0f} module-steps/s")
        print("  DECISION: generations/hour = 3600 / (s_per_timestep * lifetime).")
        print("            Compare against scripts/compute_budget.py section 11.")

    print("\n[B5] host-side birth throughput")
    d5 = bench_births()
    print(f"  {d5['births_per_s']:,.0f} births/s  ({d5['us_per_birth']:.1f} us/birth)")
    print("  DECISION: a generation replaces ~n_organisms/4 lanes.  If birth time")
    print("            exceeds 10% of [B4], vectorise the birth path too.")
    print()


if __name__ == "__main__":
    main()
