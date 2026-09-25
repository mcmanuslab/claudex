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
