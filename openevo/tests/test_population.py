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


def test_effective_params_detects_dead_capacity():
    """Raw parameter count cannot distinguish complexity from bloat; this must."""
    from openevo.metrics.complexity import effective_params, probe_batch
    from openevo.models.transformer import init_params
    arch = scale_to_params(2000)
    rng = np.random.default_rng(0)
    w = init_params(arch, rng, n=1)
    for k in list(w):                 # de-zero the output paths so the test is real
        if k.endswith(("Wo", "W2")):
            w[k] = rng.normal(0, 0.3, w[k].shape).astype(np.float32)
    ob, pa, pr = probe_batch(arch, np.random.default_rng(7))
    live = effective_params(w, arch, ob, pa, pr)
    assert live["effective"] <= live["raw"]
    assert live["fraction"] > 0.5, "a healthy organism should be mostly effective"

    dead = {k: v.copy() for k, v in w.items()}
    for b in range(arch.n_blocks):    # silence half the FFN units
        dead[f"b{b}.W2"][0, : arch.d_ff // 2, :] = 0.0
    bloated = effective_params(dead, arch, ob, pa, pr)
    assert bloated["effective"] < live["effective"]
    assert bloated["n_effective"] < live["n_effective"]


def test_recombination_produces_a_valid_child_from_compatible_parents():
    from openevo.evolution.organism import recombine
    rng = np.random.default_rng(2)
    arch = scale_to_params(2000)
    a, b = founder(arch, rng), founder(arch, rng)
    child = recombine(a, b, rng, PhaseConfig(recombination=True), generation=1)
    assert child is not None
    assert count_params(child.weights) == child.arch.n_params
    assert set(child.parents) == {a.oid, b.oid}
    # every module must come intact from one parent or the other, never be blended
    for k, v in child.weights.items():
        assert np.array_equal(v, a.weights[k]) or np.array_equal(v, b.weights[k]), k


def test_recombination_refuses_incompatible_architectures():
    """Crossover between differently-shaped networks is meaningless, not merely risky."""
    from openevo.evolution.organism import recombine
    rng = np.random.default_rng(2)
    a = founder(scale_to_params(2000), rng)
    b = founder(scale_to_params(40000), rng)
    assert recombine(a, b, rng, PhaseConfig(recombination=True), generation=1) is None


def test_checkpoint_round_trips():
    """A long run must survive interruption without losing lineage state."""
    import pickle
    pop = _pop()
    s = _sampler()
    pop.step(s)
    pop.records.clear()
    blob = pickle.dumps(pop)
    restored = pickle.loads(blob)
    assert restored.generation == pop.generation
    assert [o.oid for o in restored.living()] == [o.oid for o in pop.living()]
    assert restored.total_births == pop.total_births
    restored.step(s)                     # must keep running after a restore
    assert restored.generation == pop.generation + 1
    assert all(count_params(o.weights) == o.arch.n_params for o in restored.living())


def test_every_ancestral_scale_is_seeded_into_every_island():
    """The subclade test only discriminates driven from passive trends if displaced
    lineages compete with the bulk. One founder size per island does not achieve that."""
    small, big = scale_to_params(2000), scale_to_params(40000)
    cfg = EvoConfig(n_islands=3, island_capacity=6)
    pop = Population(cfg, PhaseConfig(), [small, big], np.random.default_rng(0))
    for isl in pop.islands:
        sizes = {o.n_params for o in isl.members}
        assert sizes == {small.n_params, big.n_params}, \
            f"island {isl.idx} is monomorphic: {sizes}"
