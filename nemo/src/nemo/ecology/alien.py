"""Alien-world adaptation: the evolvability measurement.

DESIGN.md calls this the headline test.  The question is not "does fitness
increase" but "does the organisation evolution discovered make FUTURE
adaptation faster".

Protocol
--------
1. Take a fossil -- a snapshot of a population at generation g.
2. Deep-copy it.  Nothing here may touch the live experiment.
3. Evaluate it, unchanged, on a world built only from ALIEN primitives
   (COUNT, IRREV, DRIFT) -- causal mechanisms never used for reproductive
   fitness in any condition.  That is the pre-adaptation fitness.
4. Evolve the copy on the alien world for a fixed offspring budget.
5. Report the normalised area under the adaptation curve, with pre-adaptation
   fitness as a covariate.

Confounder control (RESEARCH.md 7, C6): a population that starts lower has
more room to improve, so raw gain regresses to the mean.  `adaptation_auc`
subtracts the pre-adaptation level, and every comparison is at a matched
offspring budget -- the same number of births, not the same wall clock, so a
larger organism does not get more adaptation for being slower.

Isolation is structural, not procedural: this module builds its goals from
`ALIEN_POOL` and is never imported by the selection path.
`tests/test_alien_isolation.py` asserts that no alien primitive can enter a
reproductive-fitness goal.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np

from ..config import ExperimentConfig, RunConfig
from ..environments import primitives as P
from ..environments.world import Calibration
from ..genome.population import Population
from ..metrics.assays import rollout_rewards
from ..metrics.definitions import adaptation_auc
from ..modules.spec import ModuleSpec
from ..mutation.operators import InnovationRegistry, mutate
from ..selection.islands import step_generation


@dataclass
class AdaptationResult:
    run: int
    alien_goal: tuple[int, ...]
    pre: float
    curve: list[float]
    auc: float
    births: int


def alien_goals(arity: int = 2) -> list[tuple[int, ...]]:
    """Combinations drawn only from the held-out pool."""
    import itertools
    return list(itertools.combinations(P.ALIEN_POOL, min(arity, len(P.ALIEN_POOL))))


def _alien_reward(pop: Population, goal: tuple[int, ...], params: np.ndarray,
                  cfg: ExperimentConfig, episodes: int, lifetime: int,
                  rng: np.random.Generator, calib: Calibration | None) -> np.ndarray:
    per_prim = rollout_rewards(pop, goal, params, episodes, lifetime, cfg, rng)
    if calib is not None:
        per_prim = calib.normalise(per_prim)
    return per_prim[:, list(goal)].mean(axis=1)


def adapt(pop_slice: Population, cfg: ExperimentConfig, run: RunConfig,
          goal: tuple[int, ...], generations: int, seed: int,
          episodes: int = 2, lifetime: int = 32,
          calib: Calibration | None = None) -> AdaptationResult:
    """Evolve a COPY of `pop_slice` on an alien world and return the curve."""
    assert set(goal).issubset(set(P.ALIEN_POOL)), "goal is not alien"
    pop = copy.deepcopy(pop_slice)
    rng = np.random.default_rng(seed)
    reg = InnovationRegistry(next_id=int(max(pop.innov.max(), 0)) + 1)
    spec = ModuleSpec(d_model=cfg.module.d_model, d_ff=cfg.module.d_ff,
                      max_in_degree=cfg.module.max_in_degree)
    params = np.repeat(rng.integers(1, 8, size=(1, 10)).astype(np.int32),
                       pop.L, axis=0)

    pre = float(_alien_reward(pop, goal, params, cfg, episodes, lifetime,
                              rng, calib).mean())
    curve, births = [], 0
    for _ in range(generations):
        r = _alien_reward(pop, goal, params, cfg, episodes, lifetime, rng, calib)
        curve.append(float(r.mean()))
        flops = np.full(pop.L, float(spec.n_params))
        parent, _ = step_generation(
            r, flops, pop.island, cfg.ecology.n_islands, cfg.ecology.island_size,
            cfg.ecology.tournament, run.metabolism, rng)
        for local, par in enumerate(parent):
            if par < 0:
                continue
            _copy_lane(pop, int(par), local)
            mutate(pop, local, rng, cfg.mutation, run, reg, 0)
            births += 1
    return AdaptationResult(
        run=int(pop.run_id[0]) if pop.L else -1, alien_goal=goal, pre=pre,
        curve=curve, auc=adaptation_auc(np.array(curve), pre), births=births)


def _copy_lane(pop: Population, src: int, dst: int) -> None:
    for name in ("Wq", "Wk", "Wv", "Wo", "W1", "b1", "W2", "b2", "g1", "g2",
                 "wg", "bg", "alive", "src", "src_mask", "gate_b", "gate_s",
                 "E_obs", "W_act", "b_act", "meta", "innov", "gene_parent",
                 "birth_gen"):
        arr = getattr(pop, name)
        arr[dst] = arr[src]
