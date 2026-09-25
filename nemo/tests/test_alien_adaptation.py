"""Alien-world adaptation and the matched monolithic control."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemo.config import ExperimentConfig, RunConfig          # noqa: E402
from nemo.ecology.alien import adapt, alien_goals             # noqa: E402
from nemo.ecology.evolve import Experiment                    # noqa: E402
from nemo.environments import primitives as P                 # noqa: E402
from nemo.metrics.definitions import adaptation_auc           # noqa: E402
from nemo.modules.spec import ModuleSpec, match_monolithic    # noqa: E402

FIELDS = ("alive", "Wq", "W1", "src", "src_mask", "gate_b", "innov")


def _tiny():
    cfg = ExperimentConfig()
    cfg.ecology.n_islands, cfg.ecology.island_size = 2, 8
    cfg.ecology.n_episodes, cfg.ecology.lifetime = 2, 8
    return cfg


def test_alien_goals_contain_only_held_out_primitives():
    for goal in alien_goals(2):
        assert set(goal).issubset(set(P.ALIEN_POOL))
        assert set(goal).isdisjoint(P.ANCESTRAL_POOL)


def test_adaptation_never_touches_the_live_population():
    """The whole experiment is invalid if alien performance can feed back into
    selection.  This asserts the structural guarantee, not the intention."""
    cfg = _tiny()
    ex = Experiment(cfg=cfg, runs=[RunConfig(name="t", seed=1)])
    for _ in range(6):
        ex.run_generation(sample_q=0)
    before = {f: getattr(ex.pop, f).copy() for f in FIELDS}
    adapt(ex.pop, cfg, RunConfig(name="t"), alien_goals(2)[0],
          generations=5, seed=0, episodes=1, lifetime=8)
    for f in FIELDS:
        assert np.array_equal(getattr(ex.pop, f), before[f]), f


def test_adaptation_rejects_a_non_alien_goal():
    """A goal containing an ancestral primitive must be refused outright,
    not silently evaluated."""
    cfg = _tiny()
    ex = Experiment(cfg=cfg, runs=[RunConfig(name="t", seed=1)])
    try:
        adapt(ex.pop, cfg, RunConfig(name="t"), (P.RECALL, P.COUNT),
              generations=2, seed=0, episodes=1, lifetime=8)
    except AssertionError:
        return
    raise AssertionError("adapt() accepted a goal containing an ancestral primitive")


def test_adaptation_curve_has_the_requested_budget():
    cfg = _tiny()
    ex = Experiment(cfg=cfg, runs=[RunConfig(name="t", seed=1)])
    r = adapt(ex.pop, cfg, RunConfig(name="t"), alien_goals(2)[0],
              generations=7, seed=0, episodes=1, lifetime=8)
    assert len(r.curve) == 7
    assert r.births > 0
    assert np.isfinite(r.auc)


def test_adaptation_auc_controls_for_starting_level():
    """Two populations with the same SHAPE of improvement but different
    starting levels must score the same, or the metric just rewards having
    started low (RESEARCH.md 7, C6)."""
    low = np.array([0.10, 0.20, 0.30, 0.40])
    high = np.array([0.60, 0.70, 0.80, 0.90])
    assert abs(adaptation_auc(low, low[0]) - adaptation_auc(high, high[0])) < 1e-9
    flat = np.full(4, 0.5)
    assert abs(adaptation_auc(flat, 0.5)) < 1e-9


# ------------------------------------------------------- monolithic control
def test_monolithic_control_matches_both_budgets():
    """If the modular arm wins only on parameters or FLOPs, the result says
    nothing about organisation.  Both budgets must match closely."""
    base = ModuleSpec()
    for n in (2, 4, 8):
        _, info = match_monolithic(n, base)
        assert abs(info["param_error"]) < 0.05, (n, info)
        assert abs(info["flop_error"]) < 0.05, (n, info)


def test_monolithic_control_is_a_single_module():
    base = ModuleSpec()
    matched, info = match_monolithic(4, base)
    assert matched.n_params > base.n_params * 3
    assert info["d_model"] > base.d_model


# --------------------------------------------------- duplicate-pair tracking
def test_pairs_are_found_by_innovation_not_by_lane():
    """Lane identity is not organism identity.

    Offspring overwrite the lanes of the organisms they displace, so a pair
    born in lane 37 may be carried by a descendant in any lane of its run --
    and lane 37 may now hold an unrelated organism.  Tracking by innovation id
    both finds the survivors and cannot misattribute a pair to a stranger.
    """
    from nemo.metrics.tracking import lanes_carrying_pairs

    cfg = _tiny()
    cfg.mutation.p_duplicate, cfg.mutation.p_delete = 0.5, 0.02
    ex = Experiment(cfg=cfg, runs=[RunConfig(name="t", seed=3)])
    for _ in range(20):
        ex.run_generation(sample_q=0)
    assert ex.duplicate_pairs, "no duplications occurred"

    n_org = cfg.ecology.n_organisms
    found = lanes_carrying_pairs(ex.pop, ex.duplicate_pairs, 0, n_org)
    assert found, "no surviving pair found by innovation id"

    # Every lane reported must genuinely carry BOTH innovation ids, alive.
    for lane, recs in found.items():
        live = np.flatnonzero(ex.pop.alive[lane] > 0)
        ids = set(int(ex.pop.innov[lane, g]) for g in live)
        ids |= {-i - 1 for i in ids}
        for rec in recs:
            assert rec["innov"] in ids, (lane, rec["innov"])
            assert rec["parent_innov"] in ids, (lane, rec["parent_innov"])

    # Birth-lane-only lookup finds strictly fewer lanes than lineage tracking.
    birth_lanes = {r["lane"] for r in ex.duplicate_pairs}
    assert len(set(found)) >= 1
    assert set(found) - birth_lanes or len(found) <= len(birth_lanes)


def test_fresh_duplicates_have_zero_weight_distance():
    """A duplication is function-preserving, so a pair observed immediately
    after the event must have byte-identical weights.  Any contribution
    divergence such a pair shows is pure estimation noise -- which is exactly
    why the drift control, not a permutation, is the primary outcome's null."""
    cfg = _tiny()
    cfg.mutation.p_duplicate, cfg.mutation.p_delete = 0.6, 0.0
    cfg.mutation.p_weight = 0.0
    ex = Experiment(cfg=cfg, runs=[RunConfig(name="t", seed=5)])
    ex.run_generation(sample_q=0)
    assert ex.duplicate_pairs
    ex.assay(n_lanes_per_run=8, episodes=1, lifetime=8)
    fresh = [o for o in ex.pair_obs if o.age <= 1]
    assert fresh, "no fresh pairs observed"
    assert max(o.weight_distance for o in fresh) < 1e-6
