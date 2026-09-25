"""Held-out isolation: the probe must not be able to influence evolution.

Keeping alien-world scores out of the fitness function is necessary but not sufficient.
Three further leaks are possible and are tested here: the probe mutating the organisms it
measures, the evolution code being able to read probe results at all, and the probe
suites being drawn from the same stream as the evolutionary worlds.
"""

from __future__ import annotations

import copy
import inspect

import numpy as np

from openevo.environments.suites import SuiteSplit, build_suite
from openevo.evolution import population as population_mod
from openevo.evolution.organism import PhaseConfig, founder
from openevo.metrics.probe import CONDITIONS, run_probes
from openevo.models.genome import scale_to_params
from openevo.models.transformer import count_params


def test_evolution_module_cannot_read_probe_results():
    """Structural guarantee: nothing in the selection path imports the probe module."""
    src = inspect.getsource(population_mod)
    assert "probe" not in src, "population.py references the probe machinery"
    assert "C_test" not in src and "C_dev" not in src, \
        "population.py references a held-out world class"


def test_probe_does_not_mutate_the_organisms_it_measures():
    split = SuiteSplit()
    suites = {c: build_suite(split, c, 2, np.random.default_rng(5))
              for c in ("A", "C_dev")}
    arch = scale_to_params(5000)
    cohort = [founder(arch, np.random.default_rng(i)) for i in range(3)]
    before = [copy.deepcopy(o.weights) for o in cohort]
    before_genes = [dict(o.genes) for o in cohort]
    run_probes(cohort, suites, arch, np.random.default_rng(0), CONDITIONS, n_inst=4)
    for o, w0, g0 in zip(cohort, before, before_genes):
        assert o.genes == g0
        assert count_params(o.weights) == o.arch.n_params
        for k in w0:
            assert np.array_equal(o.weights[k], w0[k]), f"probe mutated {k}"


def test_probe_reports_every_condition_on_every_class():
    split = SuiteSplit()
    suites = {c: build_suite(split, c, 2, np.random.default_rng(6))
              for c in ("A", "B", "C_dev")}
    arch = scale_to_params(5000)
    cohort = [founder(arch, np.random.default_rng(i)) for i in range(2)]
    out = run_probes(cohort, suites, arch, np.random.default_rng(0), CONDITIONS, n_inst=4)
    got = {(cls, cond) for cls, cond, _ in out}
    assert got == {(c, k) for c in suites for k in CONDITIONS}


def test_no_feedback_condition_actually_blinds_the_organism():
    """The ablation has to bite, or it proves nothing when a result survives it."""
    from openevo.evolution.evaluate import rollout
    from openevo.models.transformer import init_params
    arch = scale_to_params(5000)
    w = init_params(arch, np.random.default_rng(0), n=1)
    spec = build_suite(SuiteSplit(), "A", 1, np.random.default_rng(2))[0].spec
    temps = np.ones(1, dtype=np.float32)
    on = rollout(w, arch, spec, 4, 1, temps, np.random.default_rng(0), record=True,
                 feedback=True)
    off = rollout(w, arch, spec, 4, 1, temps, np.random.default_rng(0), record=True,
                  feedback=False)
    assert off["tok_prev_a"].max() == 0 and off["tok_prev_r"].max() == 0
    assert on["tok_prev_a"].max() > 0, "feedback channel was empty even when enabled"


def test_neutral_phase_makes_fitness_uninformative():
    from openevo.evolution.population import EvoConfig, Population
    arch = scale_to_params(5000)
    pop = Population(EvoConfig(n_islands=1, island_capacity=4), PhaseConfig(neutral=True),
                     [arch], np.random.default_rng(0))
    o = pop.living()[0]
    o.fitness_components = {"score": 0.99, "gain": 0.0, "final": 0.0, "flops": 1}
    vals = {pop._fitness(o) for _ in range(20)}
    assert len(vals) > 1, "neutral fitness is not randomised"
    assert all(0.0 <= v <= 1.0 for v in vals)
