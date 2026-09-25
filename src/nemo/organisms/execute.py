"""Population-vectorised organism execution.

One call steps every organism, in every replicate run, on every parallel
episode, through one environment timestep.  Per timestep the work is
R round-bands x ~8 fused kernels = 32 dispatches, independent of how many
organisms there are (DESIGN.md 7.1).

A note on honesty about metabolic cost: because all slots are computed and
masked, the *simulator* spends compute on genes whose gate did not fire.  The
metabolic cost that enters fitness is not the simulator's cost -- it is the
exact analytic FLOP count of the phenotype, derived from the realised gate
pattern and in-degree via ModuleSpec.flops_forward.  That is the quantity the
organism is charged for, and it is exact for the phenotype even though the
implementation is deliberately wasteful.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .. import backend as B
from ..genome.population import Population, output_gene_slots
from ..modules.spec import ModuleSpec


@dataclass
class RolloutState:
    buf: np.ndarray          # (L, n_slots, E, d) message buffer
    gate_fired: np.ndarray   # (L, G) accumulated count of firings
    steps: int


def new_state(pop: Population, E: int) -> RolloutState:
    return RolloutState(
        buf=np.zeros((pop.L, pop.n_slots, E, pop.d), np.float32),
        gate_fired=np.zeros((pop.L, pop.G), np.float64),
        steps=0,
    )


def _round_band(pop: Population, r: int) -> tuple[int, int]:
    S = pop.cfg.genome.slots_per_round
    return r * S, (r + 1) * S


def step(pop: Population, st: RolloutState, obs: np.ndarray,
         gate_threshold: float = 0.5) -> np.ndarray:
    """Advance one environment timestep.

    obs : (L, C, E) int32 observation symbols, C = n_obs_channels
    returns action logits (L, E, A)
    """
    L, G, d, K = pop.L, pop.G, pop.d, pop.K
    E = st.buf.shape[2]
    S = pop.cfg.genome.slots_per_round
    C = pop.n_sensor_slots

    # ---- sensory input: symbol -> latent message, once per organism ----
    # E_obs (L, V, d); obs (L, C, E) -> sens (L, C, E, d)
    sens = pop.E_obs[np.arange(L)[:, None, None], obs.astype(np.int64)]
    buf = B.set_slots(st.buf, sens.astype(np.float32), 1, C)

    # ---- R sequential message-passing rounds ----
    for r in range(pop.cfg.genome.n_rounds):
        lo, hi = _round_band(pop, r)
        sl0 = pop.gene_slot(lo)

        h = buf[:, sl0:sl0 + S]                                  # (L,S,E,d) own state
        msgs = B.gather_slots(buf, pop.src[:, lo:hi])            # (L,S,K,E,d)
        msgs = msgs * pop.src_mask[:, lo:hi, :, None, None]
        # (L,S,E,K,d): put the message axis last-but-one so scores and context
        # are real GEMMs.  The obvious (q[:,:,None]*k).sum(-1) spelling
        # materialises an (L,S,K,E,d) product and then reduces it, and profiled
        # at roughly half of total rollout time.
        m = np.ascontiguousarray(msgs.transpose(0, 1, 3, 2, 4))
        mf = m.reshape(L, S, E * K, d)

        q = np.matmul(h, pop.Wq[:, lo:hi])                       # (L,S,E,d)
        k = np.matmul(mf, pop.Wk[:, lo:hi]).reshape(L, S, E, K, d)
        v = np.matmul(mf, pop.Wv[:, lo:hi]).reshape(L, S, E, K, d)

        scores = np.matmul(k, q[..., None])[..., 0] / np.sqrt(d)  # (L,S,E,K)
        neg = (1.0 - pop.src_mask[:, lo:hi, None, :]) * -1e9
        attn = B.softmax(scores + neg, axis=-1)
        # A gene with no live inputs must contribute nothing, not a uniform mix.
        has_in = (pop.src_mask[:, lo:hi].sum(-1) > 0).astype(np.float32)[:, :, None, None]
        ctx = np.matmul(attn[:, :, :, None, :], v)[:, :, :, 0, :] * has_in
        ctx = np.matmul(ctx, pop.Wo[:, lo:hi])

        h1 = B.rms_norm(h + ctx, pop.g1[:, lo:hi, None, :])
        ff = np.matmul(B.relu(np.matmul(h1, pop.W1[:, lo:hi])
                                    + pop.b1[:, lo:hi, None, :]),
                       pop.W2[:, lo:hi]) + pop.b2[:, lo:hi, None, :]
        h2 = B.rms_norm(h1 + ff, pop.g2[:, lo:hi, None, :])

        logit = (h1 * pop.wg[:, lo:hi, None, :]).sum(-1) + pop.bg[:, lo:hi, None]
        gate = B.sigmoid(pop.gate_s[:, lo:hi, None] * logit
                         + pop.gate_b[:, lo:hi, None])           # (L,S,E)
        gate = gate * pop.alive[:, lo:hi, None]

        # Regulatory semantics: a gene that does not fire keeps its previous
        # state and emits nothing new.  This is WHEN, decoupled from WHAT.
        new = h2 * gate[..., None] + h * (1.0 - gate[..., None])
        buf = B.set_slots(buf, new, sl0, S)

        st.gate_fired[:, lo:hi] += (gate > gate_threshold).mean(axis=2)

    st.buf = buf
    st.steps += 1

    # ---- motor output: mean of the final band's live gene states ----
    out_slots = output_gene_slots(pop)
    sl0 = pop.gene_slot(int(out_slots[0]))
    tail = buf[:, sl0:sl0 + len(out_slots)]                       # (L,S,E,d)
    w = pop.alive[:, out_slots][:, :, None, None]
    denom = np.maximum(w.sum(axis=1, keepdims=True), 1e-6)
    pooled = (tail * w).sum(axis=1) / denom[:, 0]                 # (L,E,d)
    return np.matmul(pooled, pop.W_act) + pop.b_act[:, None, :]


def metabolic_flops(pop: Population, st: RolloutState, spec: ModuleSpec) -> np.ndarray:
    """Exact phenotype FLOPs charged to each organism over the rollout.

    Charged per gene per firing, at that gene's realised in-degree.  A dormant
    gene pays only the presence cost (DESIGN.md 6), which is accounted
    separately by the caller.
    """
    in_deg = pop.src_mask.sum(axis=2).astype(np.int64)             # (L, G)
    per_fire = np.zeros_like(in_deg, dtype=np.float64)
    for k in range(pop.K + 1):
        per_fire = np.where(in_deg == k, float(spec.flops_forward(k)), per_fire)
    return (per_fire * st.gate_fired * pop.alive).sum(axis=1)


def active_modules(pop: Population, st: RolloutState, frac: float = 0.05) -> np.ndarray:
    """Count of genes whose gate fired on >= `frac` of lifetime timesteps.

    This is the operational definition of "active module" in DESIGN.md 2.
    """
    if st.steps == 0:
        return np.zeros(pop.L, np.int64)
    rate = st.gate_fired / st.steps
    return ((rate >= frac) * pop.alive).sum(axis=1).astype(np.int64)
