"""World-suite invariants: class isolation, paired evaluation, stability semantics."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from openevo.environments.suites import (
    MIN_GAP, SuiteSplit, build_suite, family_is_alien, normalise, reference_scores,
)
from openevo.environments.worlds import (
    ALIEN, CONTEXT, N_ACT, WorldBatch, WorldSpec,
)

SPLIT = SuiteSplit()


def test_classes_are_disjoint_families():
    pools = {c: set(SPLIT.pool(c)) for c in ("A", "B", "C_dev", "C_test")}
    for a, b in itertools.combinations(pools, 2):
        assert not (pools[a] & pools[b]), f"{a} and {b} share world families"


def test_alien_mechanisms_never_appear_outside_class_c():
    """The whole held-out design rests on this: a Class A or B world must not contain a
    mechanism that Class C is supposed to be testing."""
    for cls in ("A", "B"):
        for fam in SPLIT.pool(cls):
            lat, om, core, rm = fam
            assert not ({lat, core, *om, *rm} & ALIEN), f"{cls} family {fam} is alien"
    for cls in ("C_dev", "C_test"):
        for fam in SPLIT.pool(cls):
            assert family_is_alien(fam)


def test_class_b_shares_vocabulary_with_class_a():
    """Class B must be a *recombination* of familiar mechanisms, not new ones."""
    vocab_a = set()
    for lat, om, core, rm in SPLIT.pool("A"):
        vocab_a |= {lat, core, *om, *rm}
    for lat, om, core, rm in SPLIT.pool("B"):
        assert {lat, core, *om, *rm} <= vocab_a


def test_split_is_deterministic_and_sealed():
    assert SuiteSplit().seal() == SuiteSplit().seal()
    assert SuiteSplit(seed=1).seal() != SuiteSplit(seed=2).seal()


@pytest.mark.parametrize("cls", ["A", "B", "C_dev", "C_test"])
def test_sampled_worlds_are_discriminative(cls):
    """Every world must separate a good policy from a random one, or it contributes
    nothing but noise -- and, since the filter is applied identically to every class,
    it also keeps difficulty comparable across A/B/C."""
    for spec, lo, hi in build_suite(SPLIT, cls, 5, np.random.default_rng(3)):
        assert hi - lo >= MIN_GAP
        assert normalise(lo, lo, hi) == pytest.approx(0.0, abs=1e-6)
        assert normalise(hi, lo, hi) == pytest.approx(1.0, abs=1e-6)


def test_common_random_numbers_pair_organisms():
    """Identical action sequences must produce identical trajectories across organism
    copies. Without this, between-organism variance swamps the selection signal."""
    spec = WorldSpec("action_driven", ("masked", "distractor"), "match", ("delay",),
                     k=6, seed=5)
    P, B = 3, 8
    wb = WorldBatch(spec, B, np.random.default_rng(0), copies=P)
    for _ in range(CONTEXT):
        o = wb.observe().reshape(P, B)
        assert (o == o[0]).all()
        wb.step(np.tile(np.arange(B) % N_ACT, P))


@pytest.mark.parametrize("nv,expected", [(1, 1), (2, 2), (4, 4), (0, 8)])
def test_n_variants_controls_environmental_stability(nv, expected):
    spec = WorldSpec("action_driven", (), "match", (), k=6, seed=3, n_variants=nv)
    wb = WorldBatch(spec, 8, np.random.default_rng(0))
    assert len({tuple(r) for r in wb.state["target"]}) == expected


def test_interface_is_identical_across_all_classes():
    """Observation alphabet, action count and context length must not vary with class,
    or 'adaptation on novel worlds' would be measuring interface shift."""
    for cls in ("A", "B", "C_dev", "C_test"):
        for spec, _, _ in build_suite(SPLIT, cls, 3, np.random.default_rng(9)):
            wb = WorldBatch(spec, 4, np.random.default_rng(0))
            for _ in range(CONTEXT):
                obs = wb.observe()
                assert obs.shape == (4,) and obs.min() >= 0 and obs.max() < 32
                wb.step(np.random.default_rng(0).integers(0, N_ACT, 4))


def test_reference_scores_are_reproducible():
    spec = WorldSpec("cycle", ("scramble",), "match", (), k=6, seed=11)
    assert reference_scores(spec, seed=1) == reference_scores(spec, seed=1)
