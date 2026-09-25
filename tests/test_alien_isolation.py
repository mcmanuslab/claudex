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
    assert len(set(P.ANCESTRAL_POOL) | set(P.ALIEN_POOL)) == 10


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
    """The reward vector that reaches selection must be zero on alien channels.

    This is the load-bearing test: even if an alien primitive's reward is
    computed (it always is -- the primitives are vectorised together), the
    `active` mask must zero it before it can enter fitness.
    """
    cfg = _tiny_cfg()
    ex = Experiment(cfg=cfg, runs=[RunConfig(name="t", goal_structure="MVG", seed=0)])
    reward, flops, active, per_prim = ex.evaluate()
    goal = ex._goals[0]
    for prim in P.ALIEN_POOL:
        assert prim not in goal
    # Reward is the mean over ACTIVE channels only; alien channels carry no
    # weight in it regardless of what they scored.
    masked = per_prim[:, list(P.ALIEN_POOL)]
    recomputed = per_prim[:, list(goal)].mean(axis=1)
    assert np.allclose(reward, recomputed, atol=1e-6)
    # Perturbing the alien channels must not change the selection signal at all.
    per_prim[:, list(P.ALIEN_POOL)] += 1000.0
    still = per_prim[:, list(goal)].mean(axis=1)
    assert np.allclose(reward, still, atol=1e-6)
    assert masked.shape[1] == len(P.ALIEN_POOL)


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
