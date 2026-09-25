"""Batched evaluation of a population against a suite of worlds.

Organisms are grouped by architecture signature and each group is rolled out as a single
batched graph, against world instances that are *shared* across the group with common
random numbers. Two separate adaptation channels are measured:

* **fast / in-context** -- the context window is scored in blocks, and the improvement
  from the first block to the last is adaptation with no weight change at all. This is
  the primary evolvability metric precisely because it has no learning-rate to confound
  it: there is no lifetime hyperparameter that could be tuned to the evaluation protocol.
* **slow / in-weights** -- bounded gradient descent on the organism's own trajectory
  (:func:`lifetime_learn`), which does have a heritable learning rate, and is therefore
  reported separately rather than mixed into the same number.

Compute is charged from :meth:`ArchGenome.flops_forward`, an analytic count, never from
wall-clock. Backend inefficiency (padding, bucketing, dispatch overhead) can therefore
never leak into fitness.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from ..environments.suites import Scored, normalise  # noqa: F401
from ..environments.worlds import (
    CONTEXT, N_ACT, N_BLOCKS_EVAL, WorldBatch, WorldSpec, quantise_reward,
)
from ..models.genome import ArchGenome
from ..storage.scheduler import max_group
from ..models.transformer import (
    Params, backward, forward_full, new_kv_cache, rollout_step, stack, unstack,
)

def _sample(logits: np.ndarray, temps: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Temperature sampling with common random numbers shared across organisms."""
    z = logits / temps[:, None, None]
    z = z - z.max(axis=-1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(axis=-1, keepdims=True)
    return (np.cumsum(p, axis=-1) < u[None, :, None]).sum(axis=-1).clip(0, N_ACT - 1)


def rollout(weights: Params, arch: ArchGenome, spec: WorldSpec, n_inst: int,
            copies: int, temps: np.ndarray, rng: np.random.Generator,
            *, record: bool = False, feedback: bool = True) -> dict[str, np.ndarray]:
    """One context of interaction. Returns per-block reward and, optionally, the trace.

    With ``feedback=False`` the previous-action and previous-reward tokens are held at
    zero: the organism still sees observations but is blind to the consequences of what
    it did. Any within-context improvement that survives this ablation was not in-context
    learning -- it was a good reactive prior. This is the causal test that separates the
    two, and it is the reason `no_feedback` is one of the probe conditions.
    """
    wb = WorldBatch(spec, n_inst, rng, copies=copies)
    kv = new_kv_cache(weights, arch, n_inst, CONTEXT)
    prev_a = np.zeros((copies, n_inst), dtype=np.int64)
    prev_r = np.zeros((copies, n_inst), dtype=np.int64)
    rews = np.zeros((CONTEXT, copies, n_inst), dtype=np.float32)
    o_log = np.zeros((copies, n_inst, CONTEXT), dtype=np.int64)
    a_log = np.zeros_like(o_log)
    r_log = np.zeros_like(o_log)
    acts = np.zeros((CONTEXT, copies, n_inst), dtype=np.int64)
    obs_seq = np.zeros_like(acts)

    for t in range(CONTEXT):
        obs = wb.observe().reshape(copies, n_inst)
        in_a = prev_a if feedback else np.zeros_like(prev_a)
        in_r = prev_r if feedback else np.zeros_like(prev_r)
        logits = rollout_step(weights, arch, kv, obs, in_a, in_r)
        a = _sample(logits, temps, rng.random(n_inst))
        r = wb.step(a.reshape(-1)).reshape(copies, n_inst)
        if record:
            o_log[:, :, t], a_log[:, :, t], r_log[:, :, t] = obs, in_a, in_r
        obs_seq[t], acts[t], rews[t] = obs, a, r
        prev_a, prev_r = a, quantise_reward(r)

    blocks = rews.reshape(N_BLOCKS_EVAL, CONTEXT // N_BLOCKS_EVAL, copies, n_inst)
    out = {
        "per_block": blocks.mean(axis=(1, 3)).T,          # (copies, n_blocks)
        "per_step": rews.mean(axis=(0, 2)),               # (copies,)
        "actions": acts.transpose(1, 2, 0),               # (copies, n_inst, T)
        "obs": obs_seq.transpose(1, 2, 0),
    }
    if record:
        out |= {"tok_obs": o_log, "tok_prev_a": a_log, "tok_prev_r": r_log,
                "rew": rews.transpose(1, 2, 0)}
    return out


def _behaviour(actions: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """Fitness-independent behavioural descriptor, (copies, 3).

    (action entropy, action repetition rate, stimulus dependence). Used for the
    quality-diversity archive and for novelty, so it deliberately says nothing about how
    well the organism did -- only about *what it does*.
    """
    P = actions.shape[0]
    flat_a = actions.reshape(P, -1)
    counts = np.stack([(flat_a == a).mean(axis=1) for a in range(N_ACT)], axis=1)
    ent = -(counts * np.log(counts + 1e-9)).sum(axis=1) / np.log(N_ACT)
    repeat = (actions[:, :, 1:] == actions[:, :, :-1]).mean(axis=(1, 2))
    # Stimulus dependence: how much the action distribution varies with the observation.
    dep = np.zeros(P)
    for p in range(P):
        o, a = obs[p].reshape(-1), actions[p].reshape(-1)
        tab = np.zeros((32, N_ACT))
        np.add.at(tab, (o, a), 1.0)
        row = tab.sum(axis=1, keepdims=True)
        seen = row[:, 0] > 0
        if seen.sum() > 1:
            cond = tab[seen] / row[seen]
            marg = tab[seen].sum(axis=0) / tab[seen].sum()
            dep[p] = float(np.abs(cond - marg).sum(axis=1).mean() / 2)
    return np.stack([ent, repeat, dep], axis=1)


def evaluate_group(weights_list: list[Params], arch: ArchGenome,
                   worlds: list[Scored], temps: np.ndarray, rng: np.random.Generator,
                   n_inst: int, credit: np.ndarray | None = None,
                   feedback: bool = True,
                   mem_budget: int = 2_000_000_000) -> dict[str, np.ndarray]:
    """Evaluate one architecture group over `worlds`.

    `credit` is a (n_organisms, n_worlds) boolean mask: each organism is scored only on
    its own ecological sample, so no single global benchmark exists for the whole
    population to overfit to.
    """
    P, W = len(weights_list), len(worlds)
    # Split the group if the batched activations would exceed the memory budget. The
    # accounted FLOPs are unaffected: chunking is an implementation detail, and fitness
    # is charged from the analytic model, not from how the work was scheduled.
    cap = max_group(arch, n_inst, CONTEXT, mem_budget)
    if P > cap:
        parts = [evaluate_group(weights_list[a:b], arch, worlds, temps[a:b], rng, n_inst,
                                None if credit is None else credit[a:b], feedback,
                                mem_budget)
                 for a, b in [(i, min(P, i + cap)) for i in range(0, P, cap)]]
        return {k: np.concatenate([p[k] for p in parts], axis=0) for k in parts[0]}
    # each entry carries a leading axis of size 1; strip it before batching
    weights = stack([unstack(w, 0) for w in weights_list])
    norm_blocks = np.zeros((P, W, N_BLOCKS_EVAL))
    beh = np.zeros((P, W, 3))
    for j, (spec, lo, hi) in enumerate(worlds):
        out = rollout(weights, arch, spec, n_inst, P, temps, rng, feedback=feedback)
        span = max(1e-6, hi - lo)
        norm_blocks[:, j] = np.clip((out["per_block"] - lo) / span, -0.5, 1.5)
        beh[:, j] = _behaviour(out["actions"], out["obs"])
    m = np.ones((P, W), dtype=bool) if credit is None else credit
    wts = m / np.maximum(1, m.sum(axis=1, keepdims=True))
    blocks = np.einsum("pw,pwb->pb", wts, norm_blocks)
    n_credit = m.sum(axis=1)
    flops = n_credit * n_inst * arch.flops_forward(CONTEXT)
    return {
        "blocks": blocks,                                  # (P, n_blocks)
        "score": blocks.mean(axis=1),                      # adaptation AUC over context
        "gain": blocks[:, -1] - blocks[:, 0],              # in-context adaptation
        "final": blocks[:, -1],
        "behaviour": np.einsum("pw,pwk->pk", wts, beh),
        "flops": flops.astype(np.int64),
        "per_world": norm_blocks.mean(axis=2),
    }


def group_by_arch(items: list) -> dict[tuple, list[int]]:
    g: dict[tuple, list[int]] = defaultdict(list)
    for i, o in enumerate(items):
        g[o.arch.signature()].append(i)
    return dict(g)


# ------------------------------------------------------------ slow channel (SGD)
def lifetime_learn(weights: Params, arch: ArchGenome, spec: WorldSpec, n_inst: int,
                   lr: float, steps: int, temps: np.ndarray,
                   rng: np.random.Generator) -> tuple[Params, int, list[float]]:
    """Bounded gradient learning on the organism's own experience (REINFORCE).

    Returns (updated weights, accounted FLOPs, per-step mean reward). Separated from the
    in-context channel so the two can be compared, and so that the heritable learning
    rate confounds only the slow number.
    """
    w = {k: v.copy() for k, v in weights.items()}
    copies = next(iter(w.values())).shape[0]
    total_flops, curve = 0, []
    baseline = np.zeros(copies, dtype=np.float32)
    for _ in range(steps):
        tr = rollout(w, arch, spec, n_inst, copies, temps, rng, record=True)
        curve.append(float(tr["per_step"].mean()))
        logits, cache = forward_full(w, arch, tr["tok_obs"], tr["tok_prev_a"],
                                     tr["tok_prev_r"], want_cache=True)
        z = logits - logits.max(axis=-1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(axis=-1, keepdims=True)
        onehot = np.eye(N_ACT, dtype=np.float32)[tr["actions"]]
        # Return-to-go with a moving per-organism baseline keeps the estimator sane at
        # these tiny batch sizes.
        g = np.cumsum(tr["rew"][:, :, ::-1], axis=2)[:, :, ::-1]
        adv = g - baseline[:, None, None]
        adv = adv / (np.abs(adv).mean(axis=(1, 2), keepdims=True) + 1e-6)
        dlogits = ((p - onehot) * adv[..., None]).astype(np.float32)
        grads = backward(w, arch, cache, dlogits, tr["tok_obs"], tr["tok_prev_a"],
                         tr["tok_prev_r"])
        for k in w:
            gk = grads[k]
            w[k] = (w[k] - lr * gk / (np.abs(gk).mean() + 1e-8)).astype(w[k].dtype)
        baseline = 0.8 * baseline + 0.2 * tr["rew"].mean(axis=(1, 2))
        total_flops += n_inst * arch.flops_train_step(CONTEXT)
    return w, total_flops, curve
