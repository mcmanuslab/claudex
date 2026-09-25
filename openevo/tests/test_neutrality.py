"""The complexity-growth null: mutation must not have a direction of its own.

If the mutation operators are even slightly biased towards growth, a population under
*no selection at all* will drift towards larger architectures, and the headline result of
this study ("complexity increased spontaneously") would be an artefact of the operator
set. These tests measure that bias directly rather than assuming it away.
"""

from __future__ import annotations

import numpy as np
import pytest

from openevo.evolution.organism import (
    DEFAULT_GENES, PhaseConfig, founder, reproduce,
)
from openevo.models.genome import scale_to_params
from openevo.models.morphisms import LADDERS, rung


def _rung_sum(arch) -> int:
    return sum(rung(f, getattr(arch, f)) for f in LADDERS)


def _neutral_lineages(n_lineages=60, n_steps=60, start=5000, seed=0,
                      growth_bias=0.5, phase=None):
    """Independent lineages reproducing with no selection whatsoever."""
    phase = phase or PhaseConfig(structural_mutation=True)
    genes = dict(DEFAULT_GENES, p_structural=0.6, p_growth_bias=growth_bias)
    arch = scale_to_params(start)
    d_rung, d_logp, n_events = [], [], 0
    for li in range(n_lineages):
        rng = np.random.default_rng(seed * 1000 + li)
        o = founder(arch, rng, genes=genes)
        r0, p0 = _rung_sum(o.arch), np.log(o.n_params)
        for gen in range(n_steps):
            o = reproduce(o, rng, phase, gen)
            n_events += len(o.mutations)
        d_rung.append(_rung_sum(o.arch) - r0)
        d_logp.append(np.log(o.n_params) - p0)
    return np.array(d_rung, dtype=float), np.array(d_logp), n_events


def test_neutral_ladder_walk_is_unbiased():
    """On the rung ladder the walk must be symmetric: this is true by construction."""
    d_rung, _, n_events = _neutral_lineages()
    assert n_events > 500, "not enough structural mutations to test"
    se = d_rung.std(ddof=1) / np.sqrt(len(d_rung))
    z = d_rung.mean() / max(se, 1e-9)
    assert abs(z) < 3.0, (
        f"neutral drift on the ladder is biased: mean={d_rung.mean():.3f} rungs, z={z:.2f}")


def test_neutral_parameter_count_drift_is_small():
    """Parameter count is a nonlinear function of the dimensions, so perfect symmetry on
    the ladder does not guarantee perfect symmetry in parameters. The residual bias is
    *measured* here and must stay far below the effect sizes the study reports."""
    _, d_logp, _ = _neutral_lineages()
    se = d_logp.std(ddof=1) / np.sqrt(len(d_logp))
    per_gen = d_logp.mean() / 60
    assert abs(per_gen) < 0.01, (
        f"neutral parameter drift {per_gen:+.5f} nats/generation "
        f"(total {d_logp.mean():+.3f} +- {1.96 * se:.3f})")


def test_biased_operator_set_is_detectable():
    """Sanity check on the test itself: a deliberately biased operator set must fail the
    same check that the real one passes. A null test that cannot fail is worthless."""
    d_rung, _, _ = _neutral_lineages(growth_bias=0.8)
    se = d_rung.std(ddof=1) / np.sqrt(len(d_rung))
    assert d_rung.mean() / max(se, 1e-9) > 3.0, "growth bias of 0.8 went undetected"


def test_growth_bias_gene_drifts_symmetrically_about_half():
    """The heritable growth-bias gene mutates on the logit scale, so under drift its
    median must stay at 0.5 -- otherwise the gene's own parameterisation would create
    the trend the experiment is looking for."""
    phase = PhaseConfig(heritable_mutation=True)
    finals = []
    for li in range(200):
        rng = np.random.default_rng(4000 + li)
        o = founder(scale_to_params(5000), rng,
                    genes=dict(DEFAULT_GENES, p_structural=0.0))
        for gen in range(40):
            o = reproduce(o, rng, phase, gen)
        finals.append(o.gene("p_growth_bias"))
    logits = np.log(np.clip(finals, 1e-6, 1 - 1e-6) / (1 - np.clip(finals, 1e-6, 1 - 1e-6)))
    se = logits.std(ddof=1) / np.sqrt(len(logits))
    assert abs(logits.mean() / max(se, 1e-9)) < 3.0, (
        f"growth-bias gene drifts: median={np.median(finals):.3f}")


@pytest.mark.parametrize("mode", ["lamarckian", "darwinian", "partial"])
def test_weight_inheritance_modes_keep_tensors_consistent(mode):
    from openevo.models.transformer import count_params
    phase = PhaseConfig(weight_inheritance=mode)
    rng = np.random.default_rng(11)
    o = founder(scale_to_params(5000), rng)
    for gen in range(25):
        o = reproduce(o, rng, phase, gen)
        assert count_params(o.weights) == o.arch.n_params, f"{mode} desynced at gen {gen}"
