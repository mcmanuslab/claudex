"""Dense population state.

Every organism in every replicate run is a *lane* in one set of dense arrays
(DESIGN.md 7.1).  There is no per-organism Python object in the hot path; the
modelled cost of the alternative is 3.7 h/generation versus 0.41 s/generation.

Slot layout of the message buffer, per lane:

    0                      : null slot, permanently zero (dangling edges point here)
    1 .. C                 : sensor slots, written each timestep
    C+1 .. C+G_max         : one slot per gene, holding that module's state

Genes are partitioned into R round-bands of S slots each (G_max = R*S).  Gene
slot g executes in round g // S.  A "regulatory mutation" that changes *when* a
module fires is therefore a move to a free slot in another band, which keeps
timing separable from wiring and from weights (DESIGN.md 4.1).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import ExperimentConfig

NULL_SLOT = 0


@dataclass
class Population:
    """Dense arrays over L lanes.  Weight arrays are float32; genome structure
    arrays are int32/float32; lineage arrays are host-side int64."""

    cfg: ExperimentConfig
    L: int

    # --- module weights, (L, G, ...) ---
    Wq: np.ndarray; Wk: np.ndarray; Wv: np.ndarray; Wo: np.ndarray
    W1: np.ndarray; b1: np.ndarray; W2: np.ndarray; b2: np.ndarray
    g1: np.ndarray; g2: np.ndarray
    wg: np.ndarray; bg: np.ndarray

    # --- genome structure ---
    alive: np.ndarray       # (L, G) float32 0/1
    src: np.ndarray         # (L, G, K) int32, slot indices
    src_mask: np.ndarray    # (L, G, K) float32
    gate_b: np.ndarray      # (L, G) float32
    gate_s: np.ndarray      # (L, G) float32

    # --- interface ---
    E_obs: np.ndarray       # (L, V, d)
    W_act: np.ndarray       # (L, d, A)
    b_act: np.ndarray       # (L, A)

    # --- heritable mutation meta-parameters, (L, 8) log-space ---
    meta: np.ndarray

    # --- lineage bookkeeping (host side, never in the rollout) ---
    innov: np.ndarray       # (L, G) int64, module innovation id; -1 if empty
    gene_parent: np.ndarray  # (L, G) int64, innov of the gene duplicated from
    birth_gen: np.ndarray   # (L, G) int64, generation the gene appeared
    org_id: np.ndarray      # (L,) int64
    org_parent: np.ndarray  # (L,) int64
    run_id: np.ndarray      # (L,) int32, which replicate run / condition
    island: np.ndarray      # (L,) int32

    @property
    def G(self) -> int:
        return self.cfg.genome.g_max

    @property
    def d(self) -> int:
        return self.cfg.module.d_model

    @property
    def K(self) -> int:
        return self.cfg.module.max_in_degree

    @property
    def n_sensor_slots(self) -> int:
        return self.cfg.environment.n_obs_channels

    @property
    def n_slots(self) -> int:
        return 1 + self.n_sensor_slots + self.G

    def gene_slot(self, g: int) -> int:
        """Message-buffer slot holding gene g's output."""
        return 1 + self.n_sensor_slots + g

    def genome_length(self) -> np.ndarray:
        return self.alive.sum(axis=1).astype(np.int64)


def _init_weights(rng: np.random.Generator, shape, fan_in: int) -> np.ndarray:
    """Near-identity-preserving init: small enough that a freshly duplicated
    module's output does not blow up when its copy is rewired."""
    scale = 1.0 / np.sqrt(max(fan_in, 1))
    return (rng.standard_normal(shape) * scale).astype(np.float32)


def init_population(cfg: ExperimentConfig, L: int, run_id: np.ndarray,
                    seed: int = 0) -> Population:
    """Create L ancestral organisms.

    The ancestor is a chain: gene 0 reads the sensors, gene i reads gene i-1,
    the last gene drives the actuator.  A chain rather than a random graph so
    that any later branching is unambiguously an evolved event.
    """
    rng = np.random.default_rng(seed)
    gc, mc, ec = cfg.genome, cfg.module, cfg.environment
    G, d, dff, K = gc.g_max, mc.d_model, mc.d_ff, mc.max_in_degree
    S = gc.slots_per_round

    def W(shape, fan_in):
        return _init_weights(rng, (L, G) + shape, fan_in)

    pop = Population(
        cfg=cfg, L=L,
        Wq=W((d, d), d), Wk=W((d, d), d), Wv=W((d, d), d), Wo=W((d, d), d),
        W1=W((d, dff), d), b1=np.zeros((L, G, dff), np.float32),
        W2=W((dff, d), dff), b2=np.zeros((L, G, d), np.float32),
        g1=np.ones((L, G, d), np.float32), g2=np.ones((L, G, d), np.float32),
        wg=W((d,), d), bg=np.zeros((L, G), np.float32),
        alive=np.zeros((L, G), np.float32),
        src=np.zeros((L, G, K), np.int32),
        src_mask=np.zeros((L, G, K), np.float32),
        gate_b=np.full((L, G), 1.0, np.float32),   # ancestor fires by default
        gate_s=np.ones((L, G), np.float32),
        E_obs=_init_weights(rng, (L, ec.n_symbols, d), d),
        W_act=_init_weights(rng, (L, d, ec.n_actions), d),
        b_act=np.zeros((L, ec.n_actions), np.float32),
        meta=np.zeros((L, 8), np.float32),
        innov=np.full((L, G), -1, np.int64),
        gene_parent=np.full((L, G), -1, np.int64),
        birth_gen=np.zeros((L, G), np.int64),
        org_id=np.arange(L, dtype=np.int64),
        org_parent=np.full(L, -1, np.int64),
        run_id=run_id.astype(np.int32),
        island=np.zeros(L, np.int32),
    )

    # Ancestral chain: one gene per round-band, so the chain respects round order.
    n0 = min(gc.n_genes_init, gc.n_rounds)
    for i in range(n0):
        g = i * S                       # first slot of band i
        pop.alive[:, g] = 1.0
        pop.innov[:, g] = i
        if i == 0:
            for c in range(min(K, pop.n_sensor_slots)):
                pop.src[:, g, c] = 1 + c
                pop.src_mask[:, g, c] = 1.0
        else:
            pop.src[:, g, 0] = pop.gene_slot((i - 1) * S)
            pop.src_mask[:, g, 0] = 1.0
    island_size = cfg.ecology.island_size
    pop.island = ((np.arange(L) % cfg.ecology.n_organisms) // island_size).astype(np.int32)
    return pop


def output_gene_slots(pop: Population) -> np.ndarray:
    """Slots whose output drives the actuator: the live genes of the last band."""
    S = pop.cfg.genome.slots_per_round
    R = pop.cfg.genome.n_rounds
    return np.arange((R - 1) * S, R * S)
