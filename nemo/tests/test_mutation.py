"""Mutation-operator invariants.

The scientific claims depend on these operators meaning what DESIGN.md says
they mean -- in particular that duplication is function-preserving and that
regulatory mutation changes WHEN a module fires without changing WHAT it
computes.  If those are not true, the factorial measures nothing.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemo.config import ExperimentConfig, MutationConfig, RunConfig   # noqa: E402
from nemo.genome.population import init_population                     # noqa: E402
from nemo.mutation.operators import InnovationRegistry, mutate         # noqa: E402
from nemo.organisms.execute import new_state, step                     # noqa: E402

WEIGHTS = ("Wq", "Wk", "Wv", "Wo", "W1", "b1", "W2", "b2", "g1", "g2", "wg")


def _pop(n=2):
    cfg = ExperimentConfig()
    return cfg, init_population(cfg, n, np.zeros(n))


def _only(**rates):
    mc = MutationConfig(p_weight=0, p_regulatory=0, p_wiring=0, p_duplicate=0,
                        p_delete=0, p_subsystem_dup=0, p_encapsulate=0)
    for k, v in rates.items():
        setattr(mc, k, v)
    return mc


def test_duplication_is_function_preserving():
    """A fresh duplicate must be byte-identical to its source, so the event is
    initially neutral.  If duplication perturbed weights, every duplication
    would also be a weight mutation and the D factor would be confounded."""
    cfg, pop = _pop()
    reg = InnovationRegistry(next_id=8)
    before_len = pop.alive[0].sum()
    ev = []
    rng = np.random.default_rng(3)
    for _ in range(200):
        ev = mutate(pop, 0, rng, _only(p_duplicate=1.0), RunConfig(), reg, 0)
        if any(e.kind == "duplicate" for e in ev):
            break
    dup = [e for e in ev if e.kind == "duplicate"][0]
    for name in WEIGHTS:
        arr = getattr(pop, name)
        assert np.array_equal(arr[0, dup.gene], arr[0, dup.src_gene]), name
    assert pop.alive[0].sum() == before_len + 1
    assert pop.gene_parent[0, dup.gene] == dup.parent_innov
    assert pop.innov[0, dup.gene] != pop.innov[0, dup.src_gene]


def _modules_by_innov(pop, lane):
    """Map innovation id -> the module's weight tensors.

    Keyed by innovation id, not by slot: the `regulatory_round` operator
    relocates a module to a different execution band, which legitimately
    changes the slot-indexed arrays while leaving the module itself untouched.
    The invariant that matters is about the module, not about where it sits.
    """
    out = {}
    for g in np.flatnonzero(pop.alive[lane] > 0):
        key = int(pop.innov[lane, g])
        out[key] = tuple(getattr(pop, n)[lane, g].copy() for n in WEIGHTS)
    return out


def test_regulatory_mutation_does_not_change_computation():
    """Regulatory mutation may change WHEN a module fires -- its gate, or which
    execution band it sits in -- but must leave every weight that determines
    WHAT it computes bit-identical.  This separability is what makes the
    operator-level factorial in DESIGN.md 3.1 mean anything."""
    cfg, pop = _pop()
    reg = InnovationRegistry(next_id=8)
    rng = np.random.default_rng(1)
    seen = {"regulatory_gate": 0, "regulatory_round": 0}
    for _ in range(600):
        before = _modules_by_innov(pop, 0)
        evs = mutate(pop, 0, rng, _only(p_regulatory=1.0), RunConfig(), reg, 0)
        if not evs:
            continue
        after = _modules_by_innov(pop, 0)
        for e in evs:
            seen[e.kind] = seen.get(e.kind, 0) + 1
        # Same set of modules, and every one of them numerically unchanged.
        assert set(before) == set(after), "regulatory mutation changed the module set"
        for innov, tensors in before.items():
            for name, b, a in zip(WEIGHTS, tensors, after[innov]):
                assert np.array_equal(b, a), f"{name} changed for module {innov}"
    assert seen["regulatory_gate"] > 0 and seen["regulatory_round"] > 0, seen


def test_regulatory_round_move_preserves_genome_length():
    """Relocating a module between execution bands must not create or destroy
    a gene -- otherwise the R factor would silently be a duplication factor."""
    cfg, pop = _pop()
    reg = InnovationRegistry(next_id=8)
    rng = np.random.default_rng(5)
    for _ in range(400):
        n_before = pop.alive[0].sum()
        evs = mutate(pop, 0, rng, _only(p_regulatory=1.0), RunConfig(), reg, 0)
        if any(e.kind == "regulatory_round" for e in evs):
            assert pop.alive[0].sum() == n_before


def test_deletion_never_leaves_a_dangling_edge():
    """Every source index must point at a live gene, a sensor, or the null
    slot.  A dangling edge would read a stale buffer and silently corrupt the
    phenotype."""
    cfg, pop = _pop()
    reg = InnovationRegistry(next_id=8)
    rng = np.random.default_rng(0)
    mc = _only(p_duplicate=0.5, p_delete=0.5, p_wiring=0.5)
    for gen in range(300):
        mutate(pop, 0, rng, mc, RunConfig(), reg, gen)
        for g in np.flatnonzero(pop.alive[0] > 0):
            for s, m in zip(pop.src[0, g], pop.src_mask[0, g]):
                if m > 0:
                    assert 0 <= s < pop.n_slots
                    if s > pop.n_sensor_slots:
                        gene = int(s) - 1 - pop.n_sensor_slots
                        assert pop.alive[0, gene] > 0, f"edge to dead gene {gene}"


def test_genome_never_exceeds_capacity():
    cfg, pop = _pop()
    reg = InnovationRegistry(next_id=8)
    rng = np.random.default_rng(0)
    for gen in range(400):
        mutate(pop, 0, rng, _only(p_duplicate=1.0, p_subsystem_dup=0.5),
               RunConfig(), reg, gen)
        assert pop.alive[0].sum() <= pop.G


def test_genome_never_empties():
    cfg, pop = _pop()
    reg = InnovationRegistry(next_id=8)
    rng = np.random.default_rng(0)
    for gen in range(400):
        mutate(pop, 0, rng, _only(p_delete=1.0), RunConfig(), reg, gen)
        assert pop.alive[0].sum() >= 1


def test_duplication_disabled_produces_no_duplications():
    """The D=off cell of the factorial must genuinely disable the operator."""
    cfg, pop = _pop()
    reg = InnovationRegistry(next_id=8)
    rng = np.random.default_rng(0)
    run = RunConfig(duplication=False)
    kinds = set()
    for gen in range(500):
        for e in mutate(pop, 0, rng, _only(p_duplicate=1.0, p_subsystem_dup=1.0),
                        run, reg, gen):
            kinds.add(e.kind)
    assert "duplicate" not in kinds and "subsystem_dup" not in kinds
    assert pop.alive[0].sum() == 4


def test_frozen_conditions_are_actually_frozen():
    cfg, pop = _pop()
    reg = InnovationRegistry(next_id=8)
    rng = np.random.default_rng(0)
    src0 = pop.src[0].copy()
    for gen in range(300):
        mutate(pop, 0, rng, _only(p_wiring=1.0), RunConfig(topology_mutable=False),
               reg, gen)
    assert np.array_equal(pop.src[0], src0)

    gb0 = pop.gate_b[0].copy()
    for gen in range(300):
        mutate(pop, 0, rng, _only(p_regulatory=1.0),
               RunConfig(regulation_mutable=False), reg, gen)
    assert np.array_equal(pop.gate_b[0], gb0)


def test_mutated_organism_still_executes_finitely():
    """A genome that survives 500 mutations must still produce finite output.
    Evolution will find every representational crack there is."""
    cfg, pop = _pop(n=4)
    reg = InnovationRegistry(next_id=8)
    rng = np.random.default_rng(7)
    mc = MutationConfig(p_weight=0.9, p_regulatory=0.5, p_wiring=0.5,
                        p_duplicate=0.2, p_delete=0.2, p_subsystem_dup=0.1,
                        p_encapsulate=0.1)
    for gen in range(500):
        for lane in range(4):
            mutate(pop, lane, rng, mc, RunConfig(), reg, gen)
    st = new_state(pop, 2)
    obs = rng.integers(0, cfg.environment.n_symbols, size=(4, 2, 2))
    for _ in range(20):
        out = step(pop, st, obs)
    assert np.isfinite(out).all()


def test_heritable_rates_stay_in_bounds():
    cfg, pop = _pop()
    reg = InnovationRegistry(next_id=8)
    rng = np.random.default_rng(0)
    mc = MutationConfig(heritable=True, meta_sigma=1.0)
    for gen in range(500):
        mutate(pop, 0, rng, mc, RunConfig(), reg, gen)
    assert np.isfinite(pop.meta).all() and np.abs(pop.meta).max() <= 4.0
