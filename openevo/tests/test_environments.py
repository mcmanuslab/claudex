"""World-suite invariants: class isolation, paired evaluation, stability semantics."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from openevo.environments.suites import (
    MIN_GAP, REF_N, SuiteSplit, build_suite, family_is_alien, normalise,
    reference_scores,
)
from openevo.environments.worlds import (
    ALIEN, CONTEXT, N_ACT, N_BLOCKS_EVAL, WorldBatch, WorldSpec,
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


def test_per_block_references_are_populated_and_bracket_the_whole_context_pair():
    """The in-context gain metric is normalised per block, so those references must exist
    and must be consistent with the whole-context pair."""
    for w in build_suite(SPLIT, "A", 5, np.random.default_rng(12)):
        assert len(w.lo_blocks) == N_BLOCKS_EVAL and len(w.hi_blocks) == N_BLOCKS_EVAL
        assert min(w.lo_blocks) - 1e-6 <= w.lo <= max(w.lo_blocks) + 1e-6
        assert min(w.hi_blocks) - 1e-6 <= w.hi <= max(w.hi_blocks) + 1e-6
        spec, lo, hi = w                      # three-way unpacking still works
        assert spec is w.spec and lo == w.lo and hi == w.hi


def _random_blocks(spec, n, seed):
    rng = np.random.default_rng(seed)
    wb = WorldBatch(spec, n, rng)
    rew = []
    for _ in range(CONTEXT):
        wb.observe()
        rew.append(wb.step(rng.integers(0, N_ACT, n)))
    return np.stack(rew).reshape(N_BLOCKS_EVAL, CONTEXT // N_BLOCKS_EVAL, n).mean(axis=(1, 2))


def test_per_block_references_reproduce_the_random_policy_exactly():
    """Plumbing check: replaying the reference rollout must normalise to exactly zero in
    every block, or the stored profiles do not describe the policy they claim to."""
    for w in build_suite(SPLIT, "A", 4, np.random.default_rng(12)):
        r = _random_blocks(w.spec, REF_N, w.spec.seed)
        assert np.allclose(r, np.asarray(w.lo_blocks), atol=1e-6)


def test_per_block_normalisation_reduces_random_policy_gain_bias():
    """Out of sample: a world that is intrinsically easier late in the context must not
    masquerade as in-context adaptation. Compared on mean absolute gain across a suite,
    since a single world's estimate is dominated by sampling noise."""
    worlds = build_suite(SPLIT, "B", 24, np.random.default_rng(11))
    whole, per = [], []
    for w in worlds:
        r = _random_blocks(w.spec, REF_N, w.spec.seed + 7919)   # out-of-sample seed
        n1 = (r - w.lo) / max(1e-6, w.hi - w.lo)
        lb = np.asarray(w.lo_blocks)
        n2 = (r - lb) / np.maximum(1e-6, np.asarray(w.hi_blocks) - lb)
        whole.append(n1[-1] - n1[0])
        per.append(n2[-1] - n2[0])
    assert np.mean(np.abs(per)) < np.mean(np.abs(whole)), (
        f"per-block mean |gain| {np.mean(np.abs(per)):.4f} did not improve on "
        f"whole-context {np.mean(np.abs(whole)):.4f}")
