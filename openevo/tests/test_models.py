"""Model-layer invariants: exact accounting, exact gradients, exact morphisms."""

from __future__ import annotations

import numpy as np
import pytest

from openevo.models import transformer as tr
from openevo.models.genome import ArchGenome, scale_to_params
from openevo.models.morphisms import (
    GROWTH_OPS, OP_FIELD, SHRINK_OPS, apply_morphism, snap_genome,
)

ARCHS = [
    ArchGenome(d_model=10, n_heads=2, head_dim=4, d_ff=12, n_blocks=1),
    ArchGenome(d_model=24, n_heads=3, head_dim=8, d_ff=47, n_blocks=2),
    ArchGenome(d_model=37, n_heads=4, head_dim=8, d_ff=30, n_blocks=1),
]


def _tokens(g, rng, P=2, B=3, T=12):
    return (rng.integers(0, g.n_obs, (P, B, T)),
            rng.integers(0, g.n_act, (P, B, T)),
            rng.integers(0, g.n_rew, (P, B, T)))


@pytest.mark.parametrize("g", ARCHS)
def test_param_count_is_exact(g):
    """The analytic count must match the tensors byte for byte: fitness depends on it."""
    p = tr.init_params(g, np.random.default_rng(0), n=3)
    assert tr.count_params(p) == g.n_params


@pytest.mark.parametrize("target", [900, 5000, 20000, 80000, 300000])
def test_scale_ladder_reaches_target(target):
    g = scale_to_params(target)
    assert g.n_params >= target
    assert g.valid()


def test_flops_monotonic_in_size():
    small, big = scale_to_params(5000), scale_to_params(80000)
    assert big.flops_forward(64) > small.flops_forward(64)
    assert small.flops_train_step(64) == 3 * small.flops_forward(64)


@pytest.mark.parametrize("g", ARCHS)
def test_incremental_rollout_matches_full_forward(g):
    """KV-cached acting and teacher-forced learning must see the same function."""
    rng = np.random.default_rng(1)
    p = tr.init_params(g, rng, n=2)
    for k in list(p):
        if k.endswith(("Wo", "W2")):
            p[k] = rng.normal(0, 0.4, p[k].shape).astype(tr.DTYPE)
    obs, pa, pr = _tokens(g, rng)
    full, _ = tr.forward_full(p, g, obs, pa, pr)
    kv = tr.new_kv_cache(p, g, obs.shape[1], obs.shape[2])
    inc = np.stack([tr.rollout_step(p, g, kv, obs[:, :, t], pa[:, :, t], pr[:, :, t])
                    for t in range(obs.shape[2])], axis=2)
    assert np.abs(inc - full).max() < 1e-5


def test_backward_matches_finite_differences(monkeypatch):
    """Full reverse-mode gradient check in float64."""
    monkeypatch.setattr(tr, "DTYPE", np.float64)
    g = ArchGenome(d_model=10, n_heads=2, head_dim=4, d_ff=12, n_blocks=2,
                   n_obs=6, n_act=3, n_rew=3)
    rng = np.random.default_rng(7)
    p = tr.init_params(g, rng, n=2)
    for k in list(p):
        if k.endswith(("Wo", "W2")):
            p[k] = rng.normal(0, 0.4, p[k].shape)
        if k.endswith(("g1", "g2")) or k == "g_final":
            p[k] = p[k] + rng.normal(0, 0.2, p[k].shape)
    obs, pa, pr = _tokens(g, rng, P=2, B=2, T=5)
    W = rng.normal(0, 1, (2, 2, 5, g.n_act))

    def loss(pp):
        return float(np.sum(tr.forward_full(pp, g, obs, pa, pr)[0] * W))

    _, cache = tr.forward_full(p, g, obs, pa, pr, want_cache=True)
    grads = tr.backward(p, g, cache, W.copy(), obs, pa, pr)
    eps, worst = 1e-6, 0.0
    for k in tr.param_keys(g):
        flat, gflat = p[k].reshape(-1), grads[k].reshape(-1)
        for i in rng.choice(flat.size, size=min(8, flat.size), replace=False):
            o = flat[i]
            flat[i] = o + eps; lp = loss(p)
            flat[i] = o - eps; lm = loss(p)
            flat[i] = o
            num, ana = (lp - lm) / (2 * eps), gflat[i]
            worst = max(worst, abs(num - ana) / max(1e-8, abs(num) + abs(ana)))
    assert worst < 1e-5, f"worst relative gradient error {worst:.2e}"


@pytest.mark.parametrize("g", ARCHS)
@pytest.mark.parametrize("op", GROWTH_OPS)
def test_growth_is_exactly_function_preserving(g, op):
    """At noise=0 a growth mutation must not change the organism's behaviour at all."""
    g = snap_genome(g)
    rng = np.random.default_rng(3)
    p = tr.init_params(g, rng, n=2)
    for k in list(p):
        if k.endswith(("Wo", "W2")):
            p[k] = rng.normal(0, 0.4, p[k].shape).astype(tr.DTYPE)
    obs, pa, pr = _tokens(g, rng)
    base, _ = tr.forward_full(p, g, obs, pa, pr)
    res = apply_morphism(p, g, op, np.random.default_rng(1), noise=0.0)
    if res is None:
        pytest.skip("ladder edge")
    q, g2 = res
    assert tr.count_params(q) == g2.n_params
    out, _ = tr.forward_full(q, g2, obs, pa, pr)
    assert np.abs(out - base).max() < 1e-4


@pytest.mark.parametrize("g", ARCHS)
@pytest.mark.parametrize("grow,shrink", list(zip(GROWTH_OPS, SHRINK_OPS)))
def test_grow_then_shrink_is_identity_on_the_ladder(g, grow, shrink):
    """Grow and shrink must be exact inverses, or neutral drift acquires a direction."""
    g = snap_genome(g)
    rng = np.random.default_rng(3)
    p = tr.init_params(g, rng, n=1)
    r1 = apply_morphism(p, g, grow, np.random.default_rng(1), noise=0.0)
    if r1 is None:
        pytest.skip("ladder edge")
    r2 = apply_morphism(r1[0], r1[1], shrink, np.random.default_rng(1), noise=0.0)
    assert r2 is not None
    assert r2[1].signature() == g.signature()
    assert tr.count_params(r2[0]) == g.n_params


def test_every_op_moves_exactly_one_ladder_rung():
    g = snap_genome(ArchGenome(d_model=24, n_heads=3, head_dim=8, d_ff=47, n_blocks=2))
    p = tr.init_params(g, np.random.default_rng(0), n=1)
    for op, (fieldname, direction) in OP_FIELD.items():
        res = apply_morphism(p, g, op, np.random.default_rng(0), noise=0.0)
        if res is None:
            continue
        before, after = getattr(g, fieldname), getattr(res[1], fieldname)
        assert (after > before) == (direction > 0)
        for other in ("d_model", "n_heads", "head_dim", "d_ff", "n_blocks"):
            if other != fieldname:
                assert getattr(res[1], other) == getattr(g, other), \
                    f"{op} changed {other} as a side effect"
