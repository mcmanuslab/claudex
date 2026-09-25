"""Worlds: combinations of causal primitives, and the three goal structures.

DESIGN.md 5.2.  The distinction that matters scientifically:

  MVG  goal = a size-`arity` subset of a FIXED basis of `mvg_basis` primitives,
       switched among a fixed set of combinations.  The same subgoals recur in
       different combinations -- Kashtan & Alon's (2005) construction.
  RVG  goal = `arity` primitives resampled independently from the whole
       ancestral pool at each switch.  Same switching RATE, same difficulty
       distribution, NO shared basis.  This is the control that separates
       "modular variation" from "variation", and it is what the original
       proposal's "overlapping but nonidentical world samples" actually is.
  FIX  one combination for the entire run.

Difficulty is matched across goal structures by normalising each primitive's
reward against its empirically measured random-policy baseline (`calibrate`),
so a world's score is "fraction of the achievable gap closed", not raw reward.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from . import primitives as P


@dataclass
class Calibration:
    baseline: np.ndarray    # (P,) random-policy mean reward
    ceiling: np.ndarray     # (P,) oracle mean reward

    def normalise(self, r: np.ndarray) -> np.ndarray:
        gap = np.maximum(self.ceiling - self.baseline, 1e-3)
        return (r - self.baseline) / gap


def calibrate(n_symbols: int, n_actions: int, n_channels: int,
              steps: int = 400, n: int = 4096, seed: int = 0) -> Calibration:
    """Measure each primitive's random-policy baseline and its ceiling.

    The ceiling is measured with a per-primitive oracle rather than assumed to
    be 1.0, because DECOY and IRREV have negative-only channels and COUNT's
    optimum is not 1.
    """
    rng = np.random.default_rng(seed)
    active = np.ones((n, 10), np.float32)
    params = rng.integers(1, 8, size=(n, 10)).astype(np.int32)

    base = np.zeros(10)
    st = P.init_state(n, rng)
    for _ in range(steps):
        P.observe(st, active, params, n_symbols, n_channels, rng)
        base += P.reward(st, active, params,
                         rng.integers(0, n_actions, size=n), n_actions, n_symbols).mean(0)
    base /= steps

    # Oracle: for each primitive, act optimally for THAT primitive alone.
    ceil = np.zeros(10)
    for prim in range(10):
        st = P.init_state(n, rng)
        tot = 0.0
        for _ in range(steps):
            P.observe(st, active, params, n_symbols, n_channels, rng)
            a = _oracle(prim, st, params, n_actions)
            tot += P.reward(st, active, params, a, n_actions, n_symbols)[:, prim].mean()
        ceil[prim] = tot / steps
    return Calibration(baseline=base, ceiling=np.maximum(ceil, base + 1e-3))


def _oracle(prim: int, st: P.PrimitiveState, params: np.ndarray, n_actions: int) -> np.ndarray:
    n = st.history.shape[0]
    if prim == P.RECALL:
        k = np.clip(params[:, P.RECALL], 1, P.HISTORY - 1).astype(np.int64)
        return np.take_along_axis(st.history, k[:, None], axis=1)[:, 0] % n_actions
    if prim == P.XOR:
        return ((st.history[:, 0] & 1) ^ (st.ctx & 1)).astype(np.int64)
    if prim == P.SWITCH:
        base = st.history[:, 0] % n_actions
        return np.where(st.ctx == 1, (base + n_actions // 2) % n_actions, base)
    if prim == P.DECOY:
        return (params[:, P.DECOY] % n_actions + 1) % n_actions
    if prim == P.GATE:
        return st.ctx.astype(np.int64)
    if prim == P.COUNT:
        m = np.clip(params[:, P.COUNT], 2, 8)
        return np.where(st.counter % m == 0, 0, 1).astype(np.int64)
    if prim == P.IRREV:
        return (params[:, P.IRREV] % n_actions + 1) % n_actions
    if prim == P.DRIFT:
        return (st.history[:, 0] + st.drift) % n_actions
    if prim == P.DELAY:
        return st.history[:, 0] % n_actions
    return np.zeros(n, np.int64)


def combinations(pool: tuple[int, ...], arity: int) -> list[tuple[int, ...]]:
    return list(itertools.combinations(pool, arity))


@dataclass
class GoalSchedule:
    """Which primitives, with which parameters, are active at each generation.

    MVG and RVG are matched on everything except *subgoal recurrence*:

      - both cycle through the SAME list of primitive combinations, in the same
        order, at the same switching rate, so task difficulty and goal
        diversity are identical by construction;
      - under MVG the per-primitive parameters (RECALL's lag, SWITCH's period,
        DECOY's action, ...) are FIXED for the whole run, so a given subgoal --
        e.g. RECALL(k=3) -- is a stable, reusable target that recurs in many
        different combinations;
      - under RVG the parameters are RESAMPLED at every switch, so no subgoal
        ever recurs identically and there is nothing stable to build a module
        around.

    That single difference is the variable Kashtan & Alon (2005) isolate.  Any
    modularity difference between these two conditions cannot be attributed to
    difficulty, to goal diversity, or to how often the goal changes.
    """

    structure: str                  # MVG | RVG | FIX
    pool: tuple[int, ...]
    arity: int
    switch_every: int
    seed: int
    _combos: list[tuple[int, ...]]

    @classmethod
    def build(cls, structure: str, arity: int, switch_every: int,
              mvg_basis: int, seed: int) -> "GoalSchedule":
        pool = tuple(P.ANCESTRAL_POOL[:mvg_basis])
        combos = combinations(pool, arity)
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(combos))
        combos = [combos[i] for i in order]
        return cls(structure, pool, arity, switch_every, seed, combos)

    def epoch_at(self, generation: int) -> int:
        return generation // max(self.switch_every, 1)

    def goal_at(self, generation: int) -> tuple[int, ...]:
        if self.structure == "FIX":
            return self._combos[0]
        return self._combos[self.epoch_at(generation) % len(self._combos)]

    def params_at(self, generation: int, n: int) -> np.ndarray:
        """Per-primitive parameters, (n, 10) int32.

        Fixed for the run under MVG and FIX; resampled per epoch under RVG.
        """
        epoch = 0 if self.structure in ("MVG", "FIX") else self.epoch_at(generation) + 1
        rng = np.random.default_rng((self.seed * 1_000_003 + epoch) % (2 ** 63))
        base = rng.integers(1, 8, size=(1, 10)).astype(np.int32)
        return np.repeat(base, n, axis=0)

    def n_goals(self) -> int:
        return 1 if self.structure == "FIX" else len(self._combos)


def active_matrix(goal: tuple[int, ...], n: int) -> np.ndarray:
    a = np.zeros((n, 10), np.float32)
    for g in goal:
        a[:, g] = 1.0
    return a
