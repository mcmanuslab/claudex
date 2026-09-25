"""Causal primitives: the shared subgoal basis (DESIGN.md 5.1).

Each primitive is a small, tensor-expressible finite-state mechanism over a
symbol alphabet.  Worlds are *combinations* of primitives bound to observation
channels, which is what makes Modularly Varying Goals constructible: under MVG
the same primitives recur in different combinations, which is precisely
Kashtan & Alon's (2005) construction and the strongest known driver of the
evolution of modularity.

Every primitive steps the whole population at once with no Python branching on
per-lane values, so a rollout is one tensor graph with no host sync
(DESIGN.md 7.2).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Primitive ids.  ANCESTRAL_POOL / ALIEN_POOL enforce the environment-class
# separation of DESIGN.md 5.3.
RECALL, XOR, SWITCH, DECOY, GATE, NOISE, COUNT, IRREV, DRIFT, DELAY = range(10)

NAMES = {
    RECALL: "RECALL", XOR: "XOR", SWITCH: "SWITCH", DECOY: "DECOY",
    GATE: "GATE", NOISE: "NOISE", COUNT: "COUNT", IRREV: "IRREV",
    DRIFT: "DRIFT", DELAY: "DELAY",
}

# NOISE is a MODIFIER, not a subgoal: it corrupts observations and has no
# reward channel of its own.  It is therefore excluded from both pools and
# applied to whatever world is active.  Putting it in the subgoal basis gave it
# a calibrated range of ~0, which `calibrate` now refuses.
MODIFIERS = (NOISE,)

# The shared subgoal basis used for reproductive fitness.  C(6,3) = 20 goals.
ANCESTRAL_POOL = (RECALL, XOR, SWITCH, DECOY, GATE, DELAY)

# Never used for reproductive fitness in any condition, under any goal
# structure.  Held out for the alien-adaptation evolvability test.
ALIEN_POOL = (COUNT, IRREV, DRIFT)

assert set(ANCESTRAL_POOL).isdisjoint(ALIEN_POOL)
assert set(MODIFIERS).isdisjoint(set(ANCESTRAL_POOL) | set(ALIEN_POOL))


@dataclass
class PrimitiveState:
    """Per-lane, per-episode hidden state for all primitives at once.

    Kept as one struct of dense arrays so a world can activate any subset
    without changing shapes.
    """

    history: np.ndarray     # (N, H) int32 rolling symbol history
    counter: np.ndarray     # (N,) int32
    ctx: np.ndarray         # (N,) int32 latent context for SWITCH/GATE
    closed: np.ndarray      # (N,) float32 IRREV branch closed flag
    drift: np.ndarray       # (N,) int32 DRIFT phase
    pending: np.ndarray     # (N, D) float32 DELAY reward queue


HISTORY = 8
DELAY_D = 4


def init_state(n: int, rng: np.random.Generator) -> PrimitiveState:
    return PrimitiveState(
        history=np.zeros((n, HISTORY), np.int32),
        counter=np.zeros(n, np.int32),
        ctx=rng.integers(0, 2, size=n).astype(np.int32),
        closed=np.zeros(n, np.float32),
        drift=np.zeros(n, np.int32),
        pending=np.zeros((n, DELAY_D), np.float32),
    )


def observe(st: PrimitiveState, active: np.ndarray, params: np.ndarray,
            n_symbols: int, n_channels: int,
            rng: np.random.Generator) -> np.ndarray:
    """Emit this timestep's observation.

    active : (N, P) float32 -- which primitives are live for this lane's world
    params : (N, P) int32   -- per-primitive parameter (k, m, channel, ...)
    returns (N, C) int32
    """
    n = st.history.shape[0]
    cue = rng.integers(1, n_symbols, size=(n, n_channels)).astype(np.int32)

    # GATE: a hidden context decides which channel carries signal; the other is
    # filled with a distractor.  Partial observability.
    g = (active[:, GATE] * (st.ctx == 1))[:, None]
    cue = np.where(g > 0, cue[:, ::-1], cue).astype(np.int32)

    # SWITCH: the context symbol is *shown* on channel 0's low bit, so the
    # organism can in principle learn to condition on it.  Regulation is
    # learnable, not hidden.
    s = active[:, SWITCH]
    cue[:, 0] = np.where(s > 0, (cue[:, 0] & ~1) | st.ctx, cue[:, 0]).astype(np.int32)

    # NOISE: independent symbol corruption, always on (it is a modifier, not a
    # subgoal), at a level carried in `params`.
    p_noise = params[:, NOISE].astype(np.float32) / 100.0
    corrupt = rng.random((n, n_channels)) < p_noise[:, None]
    cue = np.where(corrupt, rng.integers(1, n_symbols, size=(n, n_channels)), cue).astype(np.int32)

    st.history = np.concatenate([cue[:, :1], st.history[:, :-1]], axis=1).astype(np.int32)
    return cue


def reward(st: PrimitiveState, active: np.ndarray, params: np.ndarray,
           action: np.ndarray, n_actions: int, n_symbols: int) -> np.ndarray:
    """Per-primitive reward, returned as (N, P) so specialisation can be
    attributed to individual subgoal channels (DESIGN.md 2, `Specialisation`).
    """
    n, P = active.shape
    out = np.zeros((n, P), np.float32)
    a = action.astype(np.int32)

    # RECALL(k): correct action encodes the symbol seen k steps ago.
    k = np.clip(params[:, RECALL], 1, HISTORY - 1).astype(np.int64)
    target = np.take_along_axis(st.history, k[:, None], axis=1)[:, 0]
    out[:, RECALL] = (a == (target % n_actions)).astype(np.float32)

    # XOR(a,b): parity of the low bits of the two channels.
    parity = (st.history[:, 0] & 1) ^ (st.ctx & 1)
    out[:, XOR] = (a % 2 == parity).astype(np.float32)

    # SWITCH(c): the context flips which mapping is correct.
    base = st.history[:, 0] % n_actions
    alt = (base + n_actions // 2) % n_actions
    want = np.where(st.ctx == 1, alt, base)
    out[:, SWITCH] = (a == want).astype(np.float32)

    # DECOY(p): a real task with a deceptive trap.  Reward for tracking the
    # cue, a large penalty for one specific action.
    #
    # This channel must carry a positive component.  A penalty-only channel has
    # a calibrated range of (baseline, 0) -- about 0.06 wide -- so normalising
    # by it amplifies noise ~16x, and its "ceiling" is reachable by a degenerate
    # constant-action policy that simply never emits the trap action.  The smoke
    # test caught exactly that: the fitness-shuffled drift control appeared to
    # improve from 0.01 to 0.42 purely through this channel.
    decoy_a = params[:, DECOY] % n_actions
    decoy_want = (st.history[:, 1] + 1) % n_actions
    out[:, DECOY] = np.where(a == decoy_a, -1.0,
                             (a == decoy_want).astype(np.float32)).astype(np.float32)

    # GATE(h): reward for tracking the live channel.
    out[:, GATE] = (a % 2 == st.ctx).astype(np.float32)

    # NOISE contributes no reward channel of its own; it degrades the others.
    out[:, NOISE] = 0.0

    # --- alien primitives (never used for reproductive fitness) ---
    m = np.clip(params[:, COUNT], 2, 8)
    out[:, COUNT] = ((st.counter % m == 0) == (a == 0)).astype(np.float32)

    # IRREV(a): one action permanently closes a branch; the task is solvable
    # only while the branch is open.  Positive component for the same reason as
    # DECOY.
    irrev_a = params[:, IRREV] % n_actions
    out[:, IRREV] = np.where(st.closed > 0, -0.25,
                             (a == (st.history[:, 0] % n_actions)).astype(np.float32)
                             ).astype(np.float32)

    phase_want = (st.history[:, 0] + st.drift) % n_actions
    out[:, DRIFT] = (a == phase_want).astype(np.float32)

    out[:, DELAY] = st.pending[:, 0]

    # --- state update ---
    st.counter = (st.counter + 1).astype(np.int32)
    flip_ctx = (st.counter % np.maximum(params[:, SWITCH], 2) == 0)
    st.ctx = np.where(flip_ctx, 1 - st.ctx, st.ctx).astype(np.int32)
    st.closed = np.maximum(st.closed, (a == irrev_a).astype(np.float32))
    st.drift = np.where(st.counter % 64 == 0, st.drift + 1, st.drift).astype(np.int32)
    earned = (a == (st.history[:, 0] % n_actions)).astype(np.float32)
    st.pending = np.concatenate([st.pending[:, 1:], earned[:, None]], axis=1).astype(np.float32)

    return out * active
