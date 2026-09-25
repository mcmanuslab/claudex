"""Mutation operators (DESIGN.md 8).

The genome representation splits computation (`module weights`), wiring
(`src`) and timing (`round`/`gate`), so operators 1, 2 and 3 are causally
separable: a regulatory mutation changes WHEN a module fires without touching
what it computes or what it reads.  That separability is what makes the
operator-level factorial and the claim "regulatory mutation, specifically, did
this" possible at all.

All operators act in place on a child lane of a Population.  They are written
as host-side NumPy because they run once per birth, not once per timestep.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import MutationConfig, RunConfig
from ..genome.population import Population

# Heritable meta-parameter order (log space).
META_KEYS = ("p_weight", "p_regulatory", "p_wiring", "p_duplicate",
             "p_delete", "p_subsystem_dup", "p_encapsulate", "weight_sigma")


@dataclass
class InnovationRegistry:
    """NEAT-style innovation ids: homology for recombination, and the module
    phylogeny that is a primary scientific output (DESIGN.md 9)."""

    next_id: int = 0
    parent_of: dict[int, int] = field(default_factory=dict)
    born_at: dict[int, int] = field(default_factory=dict)

    def new(self, parent: int = -1, generation: int = 0) -> int:
        i = self.next_id
        self.next_id += 1
        self.parent_of[i] = parent
        self.born_at[i] = generation
        return i


@dataclass
class MutationEvent:
    lane: int
    kind: str
    gene: int = -1
    src_gene: int = -1
    innov: int = -1
    parent_innov: int = -1


def _rates(pop: Population, lane: int, mc: MutationConfig) -> dict[str, float]:
    """Effective rates for this lane: bootstrap defaults, scaled by the lane's
    heritable meta-parameters when the evolvable-mutation-rate condition is on."""
    base = {k: getattr(mc, k) for k in META_KEYS}
    if not mc.heritable:
        return base
    lo, hi = mc.meta_bounds
    return {k: float(np.clip(base[k] * np.exp(pop.meta[lane, i]), lo, hi))
            for i, k in enumerate(META_KEYS)}


def _free_slot_in_band(pop: Population, lane: int, band: int,
                       rng: np.random.Generator) -> int:
    S = pop.cfg.genome.slots_per_round
    free = [g for g in range(band * S, (band + 1) * S) if pop.alive[lane, g] == 0]
    return int(rng.choice(free)) if free else -1


def _live_genes(pop: Population, lane: int) -> np.ndarray:
    return np.flatnonzero(pop.alive[lane] > 0)


def _copy_module(pop: Population, lane: int, src: int, dst: int) -> None:
    for name in ("Wq", "Wk", "Wv", "Wo", "W1", "b1", "W2", "b2",
                 "g1", "g2", "wg"):
        arr = getattr(pop, name)
        arr[lane, dst] = arr[lane, src]
    pop.bg[lane, dst] = pop.bg[lane, src]
    pop.gate_b[lane, dst] = pop.gate_b[lane, src]
    pop.gate_s[lane, dst] = pop.gate_s[lane, src]
    pop.src[lane, dst] = pop.src[lane, src]
    pop.src_mask[lane, dst] = pop.src_mask[lane, src]


def _valid_sources(pop: Population, lane: int, gene: int) -> list[int]:
    """Slots gene `gene` may read: sensors, plus genes in strictly earlier
    bands, plus itself (recurrence).  Restricting to earlier bands keeps the
    per-timestep graph acyclic so round order is meaningful; recurrence across
    timesteps is available via SELF.
    """
    S = pop.cfg.genome.slots_per_round
    band = gene // S
    out = [1 + c for c in range(pop.n_sensor_slots)]
    for g in _live_genes(pop, lane):
        if g // S < band or g == gene:
            out.append(pop.gene_slot(int(g)))
    return out


def mutate(pop: Population, lane: int, rng: np.random.Generator,
           mc: MutationConfig, run: RunConfig, reg: InnovationRegistry,
           generation: int) -> list[MutationEvent]:
    """Apply one birth's worth of mutations to `lane`.  Returns the events."""
    r = _rates(pop, lane, mc)
    events: list[MutationEvent] = []
    live = _live_genes(pop, lane)
    if len(live) == 0:
        return events

    # --- 1. weight perturbation -------------------------------------------
    if rng.random() < r["p_weight"]:
        g = int(rng.choice(live))
        s = r["weight_sigma"]
        for name in ("Wq", "Wk", "Wv", "Wo", "W1", "W2", "wg", "g1", "g2"):
            arr = getattr(pop, name)
            arr[lane, g] += rng.standard_normal(arr[lane, g].shape).astype(np.float32) * s
        events.append(MutationEvent(lane, "weight", gene=g))

    # --- 2. regulatory mutation (WHEN, not what) --------------------------
    if run.regulation_mutable and rng.random() < r["p_regulatory"]:
        g = int(rng.choice(live))
        if rng.random() < 0.5:
            pop.gate_b[lane, g] += rng.standard_normal() * 0.4
            pop.gate_s[lane, g] = float(np.clip(
                pop.gate_s[lane, g] * np.exp(rng.standard_normal() * 0.2), 0.1, 10.0))
            kind = "regulatory_gate"
        else:
            band = int(rng.integers(0, pop.cfg.genome.n_rounds))
            dst = _free_slot_in_band(pop, lane, band, rng)
            if dst >= 0 and dst != g:
                _copy_module(pop, lane, g, dst)
                pop.innov[lane, dst] = pop.innov[lane, g]
                pop.gene_parent[lane, dst] = pop.gene_parent[lane, g]
                pop.birth_gen[lane, dst] = pop.birth_gen[lane, g]
                pop.alive[lane, dst] = 1.0
                _retire(pop, lane, g)
                g = dst
            kind = "regulatory_round"
        events.append(MutationEvent(lane, kind, gene=g))

    # --- 3. wiring mutation -----------------------------------------------
    if run.topology_mutable and rng.random() < r["p_wiring"]:
        g = int(rng.choice(live))
        cand = _valid_sources(pop, lane, g)
        slot = int(rng.integers(0, pop.K))
        if rng.random() < 0.25 and pop.src_mask[lane, g].sum() > 1:
            pop.src[lane, g, slot] = 0
            pop.src_mask[lane, g, slot] = 0.0
        elif cand:
            pop.src[lane, g, slot] = int(rng.choice(cand))
            pop.src_mask[lane, g, slot] = 1.0
        events.append(MutationEvent(lane, "wiring", gene=g))

    # --- 4. gene duplication ----------------------------------------------
    if run.duplication and rng.random() < r["p_duplicate"]:
        g = int(rng.choice(live))
        band = g // pop.cfg.genome.slots_per_round
        dst = _free_slot_in_band(pop, lane, band, rng)
        if dst < 0:
            for b in rng.permutation(pop.cfg.genome.n_rounds):
                dst = _free_slot_in_band(pop, lane, int(b), rng)
                if dst >= 0:
                    break
        if dst >= 0:
            _copy_module(pop, lane, g, dst)       # exact copy: function-preserving
            parent_innov = int(pop.innov[lane, g])
            new_innov = reg.new(parent_innov, generation)
            pop.innov[lane, dst] = new_innov
            pop.gene_parent[lane, dst] = parent_innov
            pop.birth_gen[lane, dst] = generation
            pop.alive[lane, dst] = 1.0
            events.append(MutationEvent(lane, "duplicate", gene=dst, src_gene=g,
                                        innov=new_innov, parent_innov=parent_innov))

    # --- 5. gene deletion --------------------------------------------------
    live = _live_genes(pop, lane)
    if len(live) > 1 and rng.random() < r["p_delete"]:
        g = int(rng.choice(live))
        _retire(pop, lane, g)
        _repair_dangling(pop, lane, pop.gene_slot(g))
        events.append(MutationEvent(lane, "delete", gene=g))

    # --- 6. subsystem duplication ------------------------------------------
    if run.duplication and rng.random() < r["p_subsystem_dup"]:
        events += _duplicate_subsystem(pop, lane, rng, reg, generation)

    # --- 7. encapsulation (the designated hierarchy operator) --------------
    if rng.random() < r["p_encapsulate"]:
        events += _encapsulate(pop, lane, rng)

    # --- heritable meta-parameters -----------------------------------------
    if mc.heritable:
        pop.meta[lane] += rng.standard_normal(pop.meta.shape[1]).astype(np.float32) * mc.meta_sigma
        pop.meta[lane] = np.clip(pop.meta[lane], -4.0, 4.0)

    return events


def _retire(pop: Population, lane: int, g: int) -> None:
    pop.alive[lane, g] = 0.0
    pop.innov[lane, g] = -1
    pop.gene_parent[lane, g] = -1
    pop.src_mask[lane, g] = 0.0
    pop.src[lane, g] = 0


def _repair_dangling(pop: Population, lane: int, dead_slot: int) -> None:
    """Edges pointing at a deleted gene are redirected to the null slot, which
    is permanently zero.  Deletion is therefore always expressible, never a
    crash, and the lost contribution is exactly what selection sees."""
    hit = pop.src[lane] == dead_slot
    pop.src[lane][hit] = 0
    pop.src_mask[lane][hit] = 0.0


def _duplicate_subsystem(pop: Population, lane: int, rng: np.random.Generator,
                         reg: InnovationRegistry, generation: int) -> list[MutationEvent]:
    """Copy a connected pair (a gene and one of its live gene inputs) together,
    preserving the internal edge.  The smallest operator that can duplicate
    *organisation* rather than a single computation."""
    live = _live_genes(pop, lane)
    if len(live) < 2:
        return []
    g = int(rng.choice(live))
    parents = [int(s) for s, m in zip(pop.src[lane, g], pop.src_mask[lane, g])
               if m > 0 and s > pop.n_sensor_slots]
    if not parents:
        return []
    up_slot = int(rng.choice(parents))
    up = up_slot - 1 - pop.n_sensor_slots
    if pop.alive[lane, up] == 0:
        return []

    evs: list[MutationEvent] = []
    mapping: dict[int, int] = {}
    for orig in (up, g):
        band = orig // pop.cfg.genome.slots_per_round
        dst = _free_slot_in_band(pop, lane, band, rng)
        if dst < 0:
            return evs
        _copy_module(pop, lane, orig, dst)
        pi = int(pop.innov[lane, orig])
        ni = reg.new(pi, generation)
        pop.innov[lane, dst] = ni
        pop.gene_parent[lane, dst] = pi
        pop.birth_gen[lane, dst] = generation
        pop.alive[lane, dst] = 1.0
        mapping[orig] = dst
        evs.append(MutationEvent(lane, "subsystem_dup", gene=dst, src_gene=orig,
                                 innov=ni, parent_innov=pi))
    # Rewire the copy's internal edge to point at the copied upstream gene.
    dg, du = mapping[g], mapping[up]
    hit = pop.src[lane, dg] == pop.gene_slot(up)
    pop.src[lane, dg][hit] = pop.gene_slot(du)
    return evs


def _encapsulate(pop: Population, lane: int, rng: np.random.Generator) -> list[MutationEvent]:
    """Freeze a connected subgraph into a reusable composite.

    Marked by a negative innovation id on the member genes, which makes the
    group inherited and duplicated as a unit and excluded from independent
    weight mutation.  Prior art: modular CGP (Walker & Miller 2008), Modular
    NEAT (Reisinger et al. 2004), TPG (Kelly & Heywood 2018).  Without an
    operator of this kind nothing in the design can produce a level above
    "module", so a null on hierarchy would be structural rather than
    biological (RESEARCH.md 7, C9).
    """
    live = _live_genes(pop, lane)
    if len(live) < 2:
        return []
    g = int(rng.choice(live))
    parents = [int(s) for s, m in zip(pop.src[lane, g], pop.src_mask[lane, g])
               if m > 0 and s > pop.n_sensor_slots]
    if not parents:
        return []
    up = int(rng.choice(parents)) - 1 - pop.n_sensor_slots
    if pop.alive[lane, up] == 0:
        return []
    for m in (g, up):
        if pop.innov[lane, m] >= 0:
            pop.innov[lane, m] = -pop.innov[lane, m] - 1   # negative => encapsulated
    return [MutationEvent(lane, "encapsulate", gene=g, src_gene=up)]


def is_encapsulated(innov: np.ndarray) -> np.ndarray:
    return innov < -1
