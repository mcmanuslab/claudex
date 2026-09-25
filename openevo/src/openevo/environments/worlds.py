"""A compositional family of tiny partially-observable sequential worlds.

Three design commitments, each of which exists to remove a confounder.

**1. Mechanisms, not difficulty knobs.** A world is a composition of a *latent
dynamics*, a chain of *observation channels* and a chain of *reward couplings*. Two
worlds differ because their causal structure differs, not because one has more noise.
"Alien" mechanisms (see :data:`ALIEN`) are distinct causal structures, not harder
settings of familiar ones.

**2. One fixed interface, forever.** Every world in the study uses the same observation
alphabet, the same action count, the same reward quantisation and the same context
length. If held-out worlds changed the interface, "adaptation speed on novel worlds"
would be measuring interface shift rather than adaptation.

**3. What must be *learned* is the instance, not the class.** A :class:`WorldSpec`'s
structure (which mechanisms) is fixed, while its *instance parameters* -- which action
each latent rewards, which symbol codes it, the transition table -- are redrawn from
``seed`` for every context. So evolution can learn the mechanism distribution, but the
organism must infer this instance in context. That is the quantity the adaptation
metrics measure, and it is what makes an "alien" world a test of the inference
machinery rather than of memorised answers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

# --------------------------------------------------------------- fixed interface
N_OBS = 32          # 0 = null/masked, 1..15 = cue symbols, 16..31 = distractor symbols
N_CUE = 15
CUE0 = 1
DIST0 = 16
N_DIST = 16
N_ACT = 5           # action 0 doubles as "wait" for action-cost worlds
N_REW_BINS = 4      # reward quantisation fed back as an input token
CONTEXT = 64        # steps per lifetime context; also the transformer's sequence length
N_BLOCKS_EVAL = 4   # context is scored in 4 equal blocks to expose within-context gains

REW_EDGES = np.array([-0.25, 0.25, 0.75], dtype=np.float32)

# ------------------------------------------------------------------- mechanisms
LATENT_ANCESTRAL = ("cycle", "randwalk", "action_driven", "switch")
LATENT_ALIEN = ("parity_history", "modadd")
OBS_ANCESTRAL = ("direct", "masked", "scramble", "distractor")
OBS_ALIEN = ("aggregate", "mimic")
REW_ANCESTRAL = ("match", "delay", "sparse", "deceptive", "action_cost")
REW_ALIEN = ("xor_compose", "reversal_after")

ALIEN = frozenset(LATENT_ALIEN + OBS_ALIEN + REW_ALIEN)
# Reward *cores* produce the base signal; the rest are modifiers layered on top.
REW_CORES = frozenset({"match", "xor_compose", "reversal_after"})


@dataclass(frozen=True)
class WorldSpec:
    """A world's causal structure plus the seed that fixes one instance of it."""

    latent: str
    obs_mods: tuple[str, ...]
    rew_core: str
    rew_mods: tuple[str, ...]
    k: int = 6
    seed: int = 0
    params: tuple[tuple[str, float], ...] = ()
    n_variants: int = 0
    """How many distinct instance-parameter draws exist for this world.

    This is the *environmental stability* knob, and it decides where adaptation has to
    live. With ``n_variants == 1`` every context uses the same mapping, so the answer is
    stable across evolutionary time and can be assimilated into the weights -- the
    Baldwin/genetic-assimilation regime. With ``n_variants == 0`` (unbounded) every
    context draws a fresh mapping and nothing about the specific instance is heritable,
    so the only route to reward is inferring it in context. Sweeping this parameter turns
    "does adaptation migrate from weights into context?" from an observation into a
    controlled experiment.
    """

    @property
    def family(self) -> tuple[str, tuple[str, ...], str, tuple[str, ...]]:
        """Structure identity, ignoring the instance seed. Class membership is on this."""
        return (self.latent, self.obs_mods, self.rew_core, self.rew_mods)

    @property
    def mechanisms(self) -> frozenset[str]:
        return frozenset((self.latent, self.rew_core, *self.obs_mods, *self.rew_mods))

    @property
    def is_alien(self) -> bool:
        return bool(self.mechanisms & ALIEN)

    @property
    def depth(self) -> int:
        """Compositional depth: how many mechanisms are stacked."""
        return 2 + len(self.obs_mods) + len(self.rew_mods)

    def p(self, name: str, default: float) -> float:
        return dict(self.params).get(name, default)

    def with_seed(self, seed: int) -> "WorldSpec":
        return WorldSpec(self.latent, self.obs_mods, self.rew_core, self.rew_mods,
                         self.k, int(seed), self.params, self.n_variants)

    def with_variants(self, n_variants: int) -> "WorldSpec":
        return WorldSpec(self.latent, self.obs_mods, self.rew_core, self.rew_mods,
                         self.k, self.seed, self.params, int(n_variants))

    def label(self) -> str:
        return (f"{self.latent}|{'+'.join(self.obs_mods) or '-'}"
                f"|{self.rew_core}|{'+'.join(self.rew_mods) or '-'}|k{self.k}")


def quantise_reward(r: np.ndarray) -> np.ndarray:
    return np.searchsorted(REW_EDGES, r).astype(np.int64)


# ------------------------------------------------------------------- simulator
@dataclass
class WorldBatch:
    """`n` independent instances of one :class:`WorldSpec`, stepped in lockstep.

    Instances differ only in their seed-derived parameters, so a whole population can be
    rolled out against the same structure in a single batched call.
    """

    spec: WorldSpec
    n_inst: int
    rng: np.random.Generator
    copies: int = 1
    state: dict[str, Any] = field(default_factory=dict)

    @property
    def n(self) -> int:
        """Total parallel streams: `copies` organisms x `n_inst` world instances."""
        return self.copies * self.n_inst

    def _tile(self, a: np.ndarray) -> np.ndarray:
        """Replicate a per-instance array across organism copies.

        Layout is copy-major (index ``p * n_inst + b``) so callers can reshape to
        ``(copies, n_inst)``. Every organism therefore faces *the same* world instances,
        and -- because the per-step stochastic draws are tiled too -- the same noise
        realisations. This common-random-numbers pairing is what makes a selection
        signal visible at small population sizes.
        """
        return np.concatenate([a] * self.copies, axis=0) if self.copies > 1 else a

    def _u(self) -> np.ndarray:
        return self._tile(self.state["rng"].random(self.n_inst))

    def _ri(self, hi: int) -> np.ndarray:
        return self._tile(self.state["rng"].integers(0, hi, self.n_inst))

    def __post_init__(self) -> None:
        s, n, k = self.spec, self.n_inst, self.spec.k
        r = np.random.default_rng([s.seed, n, hash(s.family) % (2**31)])
        st: dict[str, Any] = {}
        # Instance parameters: constant within a context, redrawn per context. These are
        # exactly what the organism has to infer in context.
        st["rng"] = r
        self.state = st
        tile = self._tile
        # Instance parameters come from a pool of `n_variants` draws (0 = one per
        # instance, i.e. unbounded variety), then are assigned round-robin.
        nv = n if s.n_variants <= 0 else min(n, max(1, s.n_variants))
        vr = np.random.default_rng([s.seed, 0xA11CE])
        pick = np.arange(n) % nv

        def variant(a: np.ndarray) -> np.ndarray:
            return tile(a[pick])

        st["target"] = variant(vr.integers(0, N_ACT, size=(nv, k)))
        st["code"] = variant(CUE0 + vr.integers(0, N_CUE, size=(nv, k)))
        st["perm"] = variant(np.argsort(vr.random((nv, N_CUE)), axis=1))
        st["trans"] = variant(vr.integers(0, k, size=(nv, k, N_ACT)))
        st["w"] = variant(vr.integers(1, max(2, k), size=(nv, N_ACT)))
        st["target2"] = variant(vr.integers(0, N_ACT, size=(nv, k)))
        st["h"] = tile(r.integers(0, k, size=n))
        st["h2"] = tile(r.integers(0, k, size=n))
        st["t"] = 0
        n = self.n
        hist = max(2, int(s.p("hist_k", 3)))
        st["hist"] = hist
        st["act_hist"] = np.zeros((n, hist), dtype=np.int64)
        st["lat_hist"] = np.zeros((n, hist), dtype=np.int64)
        delay = max(1, int(s.p("delay", 2)))
        st["delay"] = delay
        st["pending"] = np.zeros((n, delay + 1), dtype=np.float32)
        st["acc"] = np.zeros(n, dtype=np.float32)
        st["n_correct"] = np.zeros(n, dtype=np.int64)
        st["flipped"] = np.zeros(n, dtype=bool)
        st["since_cheat"] = np.full(n, 99, dtype=np.int64)
        self.state = st

    # ---- observation -------------------------------------------------------
    def observe(self) -> np.ndarray:
        s, st = self.spec, self.state
        n, k = self.n, s.k  # noqa: F841 - n kept for symmetry with step()
        rows = np.arange(n)
        h = st["h"]
        if "aggregate" in s.obs_mods:
            # The cue codes an *integral* of recent latents, so a memoryless policy
            # cannot recover it from the current observation alone.
            h_eff = (st["lat_hist"].sum(axis=1) + h) % k
        else:
            h_eff = h
        sym = st["code"][rows, h_eff]
        if "scramble" in s.obs_mods:
            sym = CUE0 + st["perm"][rows, sym - CUE0]
        if "mimic" in s.obs_mods:
            # A distractor that imitates the cue from `hist` steps ago: a misleading
            # correlate that a naive learner will latch onto.
            lag = st["lat_hist"][:, 0]
            fake = DIST0 + (st["code"][rows, lag] - CUE0) % N_DIST
            hit = self._u() < s.p("mimic_p", 0.3)
            sym = np.where(hit, fake, sym)
        if "distractor" in s.obs_mods:
            hit = self._u() < s.p("distractor_p", 0.25)
            sym = np.where(hit, DIST0 + self._ri(N_DIST), sym)
        if "masked" in s.obs_mods:
            hit = self._u() < s.p("mask_p", 0.3)
            sym = np.where(hit, 0, sym)
        return sym.astype(np.int64)

    # ---- transition + reward ----------------------------------------------
    def step(self, action: np.ndarray) -> np.ndarray:
        """Apply `action` (n,) and return the reward the agent *observes* this step."""
        s, st = self.spec, self.state
        n, k = self.n, s.k
        rows = np.arange(n)
        h, h2 = st["h"], st["h2"]

        # ---- base reward -------------------------------------------------
        tgt = st["target"][rows, h]
        if s.rew_core == "xor_compose":
            # Two independent latent factors; only their XOR determines the answer.
            combo = (st["target"][rows, h] ^ st["target2"][rows, h2]) % N_ACT
            tgt = combo
        elif s.rew_core == "reversal_after":
            # The rule inverts once the agent has been right `flip_at` times: a
            # contingent, performance-triggered reversal.
            flip_at = int(s.p("flip_at", 8))
            st["flipped"] |= st["n_correct"] >= flip_at
            tgt = np.where(st["flipped"], (N_ACT - 1) - tgt, tgt)
        correct = action == tgt
        r = correct.astype(np.float32)
        if s.rew_core == "reversal_after":
            st["n_correct"] += correct

        # ---- reward modifiers -------------------------------------------
        if "deceptive" in s.rew_mods:
            # Action 0 pays a reliable pittance but poisons the next few real rewards.
            cheat = action == 0
            r = np.where(cheat, np.float32(s.p("cheat_r", 0.3)), r)
            poisoned = (~cheat) & (st["since_cheat"] <= int(s.p("poison_k", 3)))
            r = np.where(poisoned, np.float32(0.0), r)
            st["since_cheat"] = np.where(cheat, 0, st["since_cheat"] + 1)
        if "action_cost" in s.rew_mods:
            r = r - np.where(action != 0, np.float32(s.p("act_cost", 0.1)), np.float32(0.0))
        if "delay" in s.rew_mods:
            d = st["delay"]
            st["pending"][:, st["t"] % (d + 1)] = r
            r = st["pending"][:, (st["t"] - d) % (d + 1)].copy()
        if "sparse" in s.rew_mods:
            every = max(2, int(s.p("sparse_n", 4)))
            st["acc"] += r
            pay = (st["t"] + 1) % every == 0
            r = np.where(pay, st["acc"], np.float32(0.0))
            if pay:
                st["acc"] = np.zeros(n, dtype=np.float32)

        # ---- latent transition ------------------------------------------
        st["act_hist"] = np.roll(st["act_hist"], -1, axis=1)
        st["act_hist"][:, -1] = action
        st["lat_hist"] = np.roll(st["lat_hist"], -1, axis=1)
        st["lat_hist"][:, -1] = h

        if s.latent == "cycle":
            nh = (h + 1) % k
        elif s.latent == "randwalk":
            move = self._u() < s.p("walk_p", 0.7)
            d = np.where(self._u() < 0.5, -1, 1)
            nh = np.where(move, np.clip(h + d, 0, k - 1), h)
        elif s.latent == "action_driven":
            nh = st["trans"][rows, h, action]
        elif s.latent == "switch":
            period = max(2, int(s.p("period", 8)))
            regime = (st["t"] // period) % 2
            nh = np.where(regime == 0, (h + 1) % k, st["trans"][rows, h, action])
        elif s.latent == "parity_history":
            nh = (st["act_hist"].sum(axis=1) % 2) * (k // 2) + (h % max(1, k // 2))
            nh = nh % k
        elif s.latent == "modadd":
            nh = (h + st["w"][rows, action]) % k
        else:  # pragma: no cover
            raise ValueError(s.latent)
        st["h"] = nh.astype(np.int64)
        if s.rew_core == "xor_compose":
            st["h2"] = ((h2 + 1 + (action % 2)) % k).astype(np.int64)
        st["t"] += 1
        return r.astype(np.float32)

    def oracle_action(self) -> np.ndarray:
        """Reference-high action: knows the latent and the instance mapping.

        Not provably optimal under `deceptive`/`action_cost`/`reversal_after`, which is
        why it is used only as a *consistent* normalisation reference -- the same policy
        is applied identically across every world class.
        """
        s, st = self.spec, self.state
        rows = np.arange(self.n)
        tgt = st["target"][rows, st["h"]]
        if s.rew_core == "xor_compose":
            tgt = (tgt ^ st["target2"][rows, st["h2"]]) % N_ACT
        elif s.rew_core == "reversal_after":
            tgt = np.where(st["flipped"], (N_ACT - 1) - tgt, tgt)
        return tgt.astype(np.int64)
