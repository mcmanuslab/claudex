"""Alien-world performance must never reach selection.

DESIGN.md 5.3 promises strict separation between the environment classes.  A
promise in a document is not an enforcement mechanism; this is.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemo.config import ExperimentConfig, RunConfig          # noqa: E402
from nemo.ecology.evolve import Experiment                    # noqa: E402
from nemo.environments import primitives as P                 # noqa: E402
from nemo.environments.world import GoalSchedule              # noqa: E402


def _tiny_cfg():
    cfg = ExperimentConfig()
    cfg.ecology.n_islands = 2
    cfg.ecology.island_size = 4
    cfg.ecology.n_episodes = 2
    cfg.ecology.lifetime = 8
    return cfg


def test_pools_are_disjoint():
    assert set(P.ANCESTRAL_POOL).isdisjoint(P.ALIEN_POOL)
    assert set(P.MODIFIERS).isdisjoint(set(P.ANCESTRAL_POOL) | set(P.ALIEN_POOL))
    assert len(set(P.ANCESTRAL_POOL) | set(P.ALIEN_POOL) | set(P.MODIFIERS)) == 10


def test_every_scored_channel_has_a_usable_calibrated_range():
    """A channel narrower than MIN_RANGE cannot be normalised without
    amplifying noise, and its ceiling becomes reachable by a degenerate
    constant-action policy.  The smoke test caught exactly that failure mode in
    a penalty-only DECOY channel, where the fitness-shuffled drift control
    appeared to improve from 0.01 to 0.42."""
    from nemo.environments.world import MIN_RANGE, calibrate

    c = calibrate(16, 8, 2, steps=120, n=1024)
    for prim in set(P.ANCESTRAL_POOL) | set(P.ALIEN_POOL):
        rng_ = c.ceiling[prim] - c.baseline[prim]
        assert rng_ >= MIN_RANGE, (P.NAMES[prim], rng_)


def test_no_alien_primitive_is_ever_selected_on():
    """Across every goal structure and a long horizon of generations, no goal
    used for reproductive fitness may contain an alien primitive."""
    for structure in ("MVG", "RVG", "FIX"):
        for seed in range(6):
            sch = GoalSchedule.build(structure, 3, 5, 6, seed=seed)
            for gen in range(0, 400, 5):
                goal = sch.goal_at(gen)
                assert set(goal).isdisjoint(P.ALIEN_POOL), (structure, seed, gen, goal)


def test_selection_reward_excludes_alien_channels():
    """The reward that reaches selection must be completely insensitive to the
    alien channels.

    Even if an alien primitive's reward is computed -- it always is, the
    primitives are vectorised together -- the `active` mask must render it
    invisible to fitness.  The check is behavioural rather than structural:
    perturb the alien channels arbitrarily and require the selection signal to
    be bit-for-bit unchanged.
    """
    from nemo.ecology.evolve import aggregate_reward

    cfg = _tiny_cfg()
    ex = Experiment(cfg=cfg, runs=[RunConfig(name="t", goal_structure="MVG", seed=0)])
    reward, flops, active, per_prim = ex.evaluate()
    goal = ex._goals[0]
    assert set(goal).isdisjoint(P.ALIEN_POOL)

    act = np.zeros_like(per_prim)
    act[:, list(goal)] = 1.0
    recomputed = aggregate_reward(per_prim, act, cfg.environment.aggregation,
                                  cfg.environment.conj_weight)
    assert np.allclose(reward, recomputed, atol=1e-6)

    for perturbation in (1000.0, -1000.0, np.nan):
        poisoned = per_prim.copy()
        poisoned[:, list(P.ALIEN_POOL)] = perturbation
        still = aggregate_reward(poisoned, act, cfg.environment.aggregation,
                                 cfg.environment.conj_weight)
        assert np.allclose(reward, still, atol=1e-6, equal_nan=False), perturbation


def test_assays_do_not_mutate_the_population():
    """Ablation assays are observational.  They must not touch the live
    population, or they would leak into selection through the back door."""
    from nemo.metrics.assays import ablate_gene, contribution_matrix

    cfg = _tiny_cfg()
    ex = Experiment(cfg=cfg, runs=[RunConfig(name="t", seed=0)])
    before = {n: getattr(ex.pop, n).copy()
              for n in ("alive", "src", "src_mask", "Wq", "W1")}
    abl = ablate_gene(ex.pop, 0)
    assert abl.alive[:, 0].sum() == 0
    rng = np.random.default_rng(0)
    params = rng.integers(1, 8, size=(ex.pop.L, 10)).astype(np.int32)
    contribution_matrix(ex.pop, (0, 1, 2), params, 1, 4, cfg, rng)
    for n, v in before.items():
        assert np.array_equal(getattr(ex.pop, n), v), f"assay mutated {n}"
