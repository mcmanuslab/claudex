"""Stage 3: establish the minimum viable transformer size empirically.

Probes *representational capacity*, which is a precondition for everything else: if a
scale cannot express a good policy even when handed supervision, no amount of evolution
will find one. Each scale is behaviour-cloned against the reference policy on Class A
worlds (with epsilon-greedy noise, so the state distribution is not purely on-policy for
the teacher), then scored by actually acting in held-out instances of those worlds.

The answer sets the ancestral scale ladder for the evolutionary runs. It is deliberately
run before, and independently of, any selection experiment.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from openevo.environments.suites import SuiteSplit, build_suite  # noqa: E402
from openevo.environments.worlds import (  # noqa: E402
    CONTEXT, N_ACT, WorldBatch, quantise_reward,
)
from openevo.evolution.evaluate import evaluate_group  # noqa: E402
from openevo.models.genome import ArchGenome, scale_to_params  # noqa: E402
from openevo.models.transformer import (  # noqa: E402
    backward, forward_full, init_params, new_kv_cache, rollout_step,
)


def teacher_contexts(arch: ArchGenome, worlds, n_inst, rng, eps: float = 0.25):
    """Roll out the reference policy with epsilon noise; return tokens and labels."""
    O, A, R, Y = [], [], [], []
    for spec, _, _ in worlds:
        wb = WorldBatch(spec, n_inst, rng)
        pa = np.zeros(n_inst, dtype=np.int64)
        pr = np.zeros(n_inst, dtype=np.int64)
        o_s, a_s, r_s, y_s = [], [], [], []
        for _ in range(CONTEXT):
            obs = wb.observe()
            tgt = wb.oracle_action()
            act = np.where(rng.random(n_inst) < eps, rng.integers(0, N_ACT, n_inst), tgt)
            o_s.append(obs); a_s.append(pa); r_s.append(pr); y_s.append(tgt)
            rew = wb.step(act)
            pa, pr = act, quantise_reward(rew)
        O.append(np.stack(o_s, 1)); A.append(np.stack(a_s, 1))
        R.append(np.stack(r_s, 1)); Y.append(np.stack(y_s, 1))
    return (np.concatenate(O)[None], np.concatenate(A)[None],
            np.concatenate(R)[None], np.concatenate(Y)[None])


def adam_train(arch, tokens, steps, lr, rng, batch=64):
    obs, pa, pr, y = tokens
    w = init_params(arch, rng, n=1)
    m = {k: np.zeros_like(v) for k, v in w.items()}
    v = {k: np.zeros_like(x) for k, x in w.items()}
    n = obs.shape[1]
    curve = []
    for step in range(1, steps + 1):
        idx = rng.integers(0, n, batch)
        ob, a, r, t = obs[:, idx], pa[:, idx], pr[:, idx], y[:, idx]
        logits, cache = forward_full(w, arch, ob, a, r, want_cache=True)
        z = logits - logits.max(axis=-1, keepdims=True)
        p = np.exp(z); p /= p.sum(axis=-1, keepdims=True)
        loss = -np.log(np.take_along_axis(p, t[..., None], -1) + 1e-9).mean()
        curve.append(float(loss))
        d = (p - np.eye(N_ACT, dtype=np.float32)[t]) / (t.size)
        g = backward(w, arch, cache, d.astype(np.float32), ob, a, r)
        for k in w:
            m[k] = 0.9 * m[k] + 0.1 * g[k]
            v[k] = 0.999 * v[k] + 0.001 * g[k] ** 2
            mh = m[k] / (1 - 0.9 ** step); vh = v[k] / (1 - 0.999 ** step)
            w[k] = (w[k] - lr * mh / (np.sqrt(vh) + 1e-8)).astype(w[k].dtype)
    return w, curve


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", type=int, nargs="+",
                    default=[800, 1500, 3000, 5000, 12000, 20000, 45000, 80000])
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--worlds", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--variants", type=int, nargs="+", default=[1, 0],
                    help="Environmental stability regimes to test: 1 = one fixed "
                         "instance (representational capacity), 0 = a fresh instance "
                         "every context (must be inferred in context).")
    ap.add_argument("--out", type=str, default="results/min_viable_size.json")
    args = ap.parse_args()

    split = SuiteSplit()
    rows = []
    for nv in args.variants:
        regime = ("stable (n_variants=1): representational capacity" if nv == 1
                  else f"variable (n_variants={nv}): must infer the instance in context")
        print(f"\n=== {regime} ===")
        train_worlds = [replace(s, spec=s.spec.with_variants(nv)) for s in
                        build_suite(split, "A", args.worlds, np.random.default_rng(args.seed + 1))]
        held_out = [replace(s, spec=s.spec.with_variants(nv)) for s in
                    build_suite(split, "A", args.worlds, np.random.default_rng(args.seed + 99))]
        # Scored on the *training* worlds as well as held-out ones. With n_variants=1 a
        # world's mapping is fixed, so a held-out world presents a mapping the organism
        # has no way to know and no feedback channel it was ever trained to use. Scoring
        # only held-out worlds there would measure generalisation, not capacity, and
        # would report every architecture as non-viable.
        rows += _sweep(args, train_worlds, train_worlds, nv, "train")
        rows += _sweep(args, train_worlds, held_out, nv, "heldout")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {args.out}")


def _sweep(args, train_worlds, test_worlds, nv, split_name):
    rows = []
    print(f"-- scored on {split_name} worlds --")
    print(f"{'target':>7} {'params':>7} {'d_model':>8} {'CE loss':>9} "
          f"{'trained':>9} {'random':>8} {'sec':>6}")
    for target in args.targets:
        arch = scale_to_params(target)
        rng = np.random.default_rng(args.seed)
        t0 = time.time()
        toks = teacher_contexts(arch, train_worlds, 24, rng)
        w, curve = adam_train(arch, toks, args.steps, args.lr, rng)
        temps = np.full(1, 0.35, dtype=np.float32)   # near-greedy for the capacity probe
        trained = evaluate_group([w], arch, test_worlds, temps,
                                 np.random.default_rng(7), n_inst=24)
        base = init_params(arch, np.random.default_rng(123), n=1)
        rand = evaluate_group([base], arch, test_worlds, temps,
                              np.random.default_rng(7), n_inst=24)
        dt = time.time() - t0
        row = dict(n_variants=nv, scored_on=split_name, target=target, params=arch.n_params, d_model=arch.d_model,
                   ce_final=float(np.mean(curve[-20:])),
                   trained=float(trained["score"][0]), random=float(rand["score"][0]),
                   gain=float(trained["gain"][0]), seconds=round(dt, 1))
        rows.append(row)
        print(f"{target:>7} {arch.n_params:>7} {arch.d_model:>8} {row['ce_final']:>9.3f} "
              f"{row['trained']:>9.3f} {row['random']:>8.3f} {dt:>6.1f}")
    return rows


if __name__ == "__main__":
    main()
