"""Population mechanics: the archive, the compute budget, and lifetime learning."""

from __future__ import annotations

import numpy as np
import pytest

from openevo.environments.suites import SuiteSplit, build_suite
from openevo.evolution.organism import PhaseConfig, founder
from openevo.evolution.population import Archive, EvoConfig, Population
from openevo.models.genome import scale_to_params
from openevo.models.transformer import count_params


def _pop(**kw):
    cfg = EvoConfig(n_islands=2, island_capacity=6, worlds_per_island=3,
                    credited_worlds=2, instances_per_world=4,
                    max_births_per_island=6, **kw)
    return Population(cfg, PhaseConfig(heritable_mutation=True),
                      [scale_to_params(2000)], np.random.default_rng(0))


def _sampler(split=None):
    split = split or SuiteSplit()
    worlds = build_suite(split, "A", 3, np.random.default_rng(4))
    return lambda island, generation: worlds


def test_archive_fossil_survives_the_organism_it_copied():
    """`cull` frees the weights of everything it kills. If the archive aliased a live
    organism it would silently end up holding empty tensors."""
    a = Archive()
    o = founder(scale_to_params(2000), np.random.default_rng(0))
    o.behaviour, o.fitness = (0.5, 0.1, 0.2), 1.0
    assert a.insert(o)
    o.weights = {}                       # what cull() does on death
    fossil = next(iter(a.cells.values()))
    assert count_params(fossil.weights) == fossil.arch.n_params


def test_archive_keeps_the_better_occupant_of_a_cell():
    a = Archive()
    lo = founder(scale_to_params(2000), np.random.default_rng(0))
    hi = founder(scale_to_params(2000), np.random.default_rng(1))
    lo.behaviour = hi.behaviour = (0.5, 0.0, 0.0)
    lo.fitness, hi.fitness = 0.1, 0.9
    a.insert(lo)
    assert a.insert(hi) and a.coverage() == 1
    assert next(iter(a.cells.values())).fitness == pytest.approx(0.9)
    assert not a.insert(lo)


def test_dead_organisms_free_their_weights_but_keep_their_record():
    pop = _pop()
    pop.step(_sampler())
    dead = [r for r in pop.records if not r["alive"]]
    assert dead, "nothing died in the first generation"
    for r in dead:
        assert r["death_reason"] and r["params"] > 0 and r["sig"]
    for o in pop.living():
        assert count_params(o.weights) == o.arch.n_params


def test_compute_budget_limits_births_not_fitness():
    """A tight budget must reduce the *number* of offspring, leaving fitness untouched.

    The budget is derived from the population's own cost model rather than hard-coded,
    so the test keeps testing the mechanism if the architecture or FLOP model changes.
    """
    generous = _pop(flop_budget_per_island=10**15)
    per_child = generous._estimate_flops(generous.living()[0])
    tight = _pop(flop_budget_per_island=int(2.5 * per_child))
    n_gen = len(generous.breed(generous.islands[0]))
    n_tight = len(tight.breed(tight.islands[0]))
    assert n_gen == generous.cfg.max_births_per_island
    assert n_tight <= 3, f"budget of 2.5 children yielded {n_tight}"
    assert n_tight < n_gen, f"budget had no effect ({n_tight} vs {n_gen})"


def test_larger_organisms_consume_more_budget():
    pop = _pop()
    small = founder(scale_to_params(2000), np.random.default_rng(0))
    big = founder(scale_to_params(60000), np.random.default_rng(0))
    assert pop._estimate_flops(big) > 5 * pop._estimate_flops(small)


def test_parameter_ceiling_is_enforced_before_evaluation():
    pop = _pop(max_params=3000)
    for _ in range(4):
        pop.step(_sampler())
    assert all(o.n_params <= 3000 for o in pop.living())


def test_lifetime_learning_changes_weights_and_is_charged_for():
    pop = _pop(lifetime_steps=2)
    worlds = build_suite(SuiteSplit(), "A", 2, np.random.default_rng(4))
    kid = founder(scale_to_params(2000), np.random.default_rng(3))
    before = {k: v.copy() for k, v in kid.weights.items()}
    pop.develop([kid], worlds)
    assert kid.eval_flops > 0, "lifetime learning was not accounted"
    assert any(not np.array_equal(kid.weights[k], before[k]) for k in before)
    assert count_params(kid.weights) == kid.arch.n_params


def test_neutral_and_selected_arms_share_demography():
    """The null is only a null if everything except fitness is identical."""
    sel = Population(EvoConfig(n_islands=2, island_capacity=6, worlds_per_island=3,
                               credited_worlds=2, instances_per_world=4),
                     PhaseConfig(), [scale_to_params(2000)], np.random.default_rng(0))
    neu = Population(EvoConfig(n_islands=2, island_capacity=6, worlds_per_island=3,
                               credited_worlds=2, instances_per_world=4),
                     PhaseConfig(neutral=True), [scale_to_params(2000)],
                     np.random.default_rng(0))
    a, b = sel.step(_sampler()), neu.step(_sampler())
    assert a["n_alive"] == b["n_alive"]
    assert a["total_births"] == b["total_births"]
