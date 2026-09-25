"""Stage 1: decide the backend on the target machine, by measurement.

This repository's reference implementation is NumPy, and every number in `DESIGN.md`
measured on the development container is NumPy on CPU. No Apple Silicon was available
there, so the claim that MLX wins is a *hypothesis* derived from an arithmetic-intensity
argument, not a result. This script settles it.

The argument it is testing: a 5K-parameter organism's 64-step context is about 2,176
kernel dispatches carrying roughly 428 FLOP each. That is dispatch-bound by about four
orders of magnitude, so what matters is not kernel quality but how many operations the
backend can fuse and how many organisms ride in one launch. MLX builds a graph and
evaluates lazily; PyTorch MPS dispatches eagerly, one command buffer per op. If the
argument is right, MLX's advantage should *grow* with batch size and the gap should be
largest at the smallest model sizes.

Run:  python scripts/bench_backend.py --backends numpy mlx torch
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from openevo.environments.worlds import CONTEXT  # noqa: E402
from openevo.models.genome import scale_to_params  # noqa: E402
from openevo.models.transformer import (  # noqa: E402
    init_params, new_kv_cache, rollout_step,
)


def bench_numpy(arch, n_org, n_inst, reps=3):
    p = init_params(arch, np.random.default_rng(0), n=n_org)
    temps = np.ones(n_org, dtype=np.float32)  # noqa: F841 - parity with the evo path
    best = float("inf")
    for _ in range(reps):
        kv = new_kv_cache(p, arch, n_inst, CONTEXT)
        obs = np.zeros((n_org, n_inst), dtype=np.int64)
        t0 = time.perf_counter()
        for _ in range(CONTEXT):
            rollout_step(p, arch, kv, obs, obs, obs)
        best = min(best, time.perf_counter() - t0)
    return best


def bench_mlx(arch, n_org, n_inst, reps=3):
    """Mirror of the NumPy rollout in MLX, evaluated once per timestep.

    Deliberately keeps a single `mx.eval` per timestep rather than per op: the whole point
    is to let MLX fuse everything inside a step into as few Metal submissions as possible.
    """
    import mlx.core as mx

    d, H, hd, f, L = arch.d_model, arch.n_heads, arch.head_dim, arch.d_ff, arch.n_blocks
    rng = np.random.default_rng(0)

    def arr(*shape):
        return mx.array(rng.normal(0, 0.2, shape).astype(np.float32))

    W = {"E": arr(n_org, arch.n_obs, d), "head": arr(n_org, d, arch.n_act)}
    for b in range(L):
        W[f"q{b}"], W[f"k{b}"], W[f"v{b}"] = arr(n_org, d, H * hd), arr(n_org, d, H * hd), arr(n_org, d, H * hd)
        W[f"o{b}"] = arr(n_org, H * hd, d)
        W[f"w1{b}"], W[f"w2{b}"] = arr(n_org, d, f), arr(n_org, f, d)

    best = float("inf")
    for _ in range(reps):
        K = [mx.zeros((n_org, n_inst, H, CONTEXT, hd)) for _ in range(L)]
        V = [mx.zeros((n_org, n_inst, H, CONTEXT, hd)) for _ in range(L)]
        idx = mx.zeros((n_org, n_inst), dtype=mx.int32)
        mx.eval(K, V, idx, list(W.values()))
        t0 = time.perf_counter()
        for t in range(CONTEXT):
            x = mx.take_along_axis(W["E"][:, None], idx[..., None, None], axis=2)[:, :, 0]
            for b in range(L):
                q = (x[:, :, None] @ W[f"q{b}"][:, None]).reshape(n_org, n_inst, H, 1, hd)
                k = (x[:, :, None] @ W[f"k{b}"][:, None]).reshape(n_org, n_inst, H, 1, hd)
                v = (x[:, :, None] @ W[f"v{b}"][:, None]).reshape(n_org, n_inst, H, 1, hd)
                K[b][:, :, :, t:t + 1] = k
                V[b][:, :, :, t:t + 1] = v
                s = (q @ K[b][:, :, :, :t + 1].transpose(0, 1, 2, 4, 3)) / np.sqrt(hd)
                ctx = (mx.softmax(s, axis=-1) @ V[b][:, :, :, :t + 1]).reshape(n_org, n_inst, H * hd)
                x = x + (ctx[:, :, None] @ W[f"o{b}"][:, None])[:, :, 0]
                h = mx.maximum(x[:, :, None] @ W[f"w1{b}"][:, None], 0)
                x = x + (h @ W[f"w2{b}"][:, None])[:, :, 0]
            out = x[:, :, None] @ W["head"][:, None]
            mx.eval(out, K[-1], V[-1])
        best = min(best, time.perf_counter() - t0)
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backends", nargs="+", default=["numpy", "mlx"])
    ap.add_argument("--sizes", type=int, nargs="+", default=[1000, 5000, 20000, 80000])
    ap.add_argument("--organisms", type=int, nargs="+", default=[8, 32, 128, 512])
    ap.add_argument("--instances", type=int, default=8)
    args = ap.parse_args()

    runners = {"numpy": bench_numpy, "mlx": bench_mlx}
    print(f"context={CONTEXT}, instances/organism={args.instances}\n")
    print(f"{'backend':>8} {'params':>8} {'orgs':>6} {'s/batch':>9} "
          f"{'contexts/s':>12} {'GFLOP/s':>10}")
    for name in args.backends:
        fn = runners.get(name)
        if fn is None:
            print(f"{name:>8}  (no runner)")
            continue
        for target in args.sizes:
            arch = scale_to_params(target)
            for n_org in args.organisms:
                try:
                    dt = fn(arch, n_org, args.instances)
                except Exception as e:                        # pragma: no cover
                    print(f"{name:>8} {arch.n_params:>8} {n_org:>6}  unavailable: "
                          f"{type(e).__name__}: {e}")
                    break
                ctx = n_org * args.instances
                fl = ctx * arch.flops_forward(CONTEXT)
                print(f"{name:>8} {arch.n_params:>8} {n_org:>6} {dt:>9.4f} "
                      f"{ctx / dt:>12,.0f} {fl / dt / 1e9:>10.2f}")
    print("\nDecision rule: adopt the backend with the highest contexts/s at the largest "
          "batch that fits, at the smallest model size -- that is where the evolutionary "
          "run actually spends its time.")


if __name__ == "__main__":
    main()
