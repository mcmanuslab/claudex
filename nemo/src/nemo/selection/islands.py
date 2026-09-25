"""Island ecology and selection (DESIGN.md 6).

Selection is local within islands and rank-based on a two-objective Pareto
front of (performance, -metabolic cost).  A scalar `reward - lambda*compute`
would silently fix the exchange rate between performance and parsimony; the
original proposal is right that this is the wrong knob to pin arbitrarily, and
the metabolic factor of the experiment would then be a lambda sweep rather
than a clean on/off.
"""

from __future__ import annotations

import numpy as np


def pareto_rank(objectives: np.ndarray) -> np.ndarray:
    """Non-dominated sorting rank (0 = front).  objectives: (n, m), maximised."""
    n = objectives.shape[0]
    rank = np.zeros(n, np.int32)
    remaining = np.ones(n, bool)
    cur = 0
    while remaining.any():
        idx = np.flatnonzero(remaining)
        o = objectives[idx]
        # i is dominated if some j is >= on all objectives and > on one.
        ge = (o[None, :, :] >= o[:, None, :]).all(axis=2)
        gt = (o[None, :, :] > o[:, None, :]).any(axis=2)
        dominated = (ge & gt).any(axis=1)
        front = idx[~dominated]
        if front.size == 0:                      # numerical tie safety valve
            front = idx
        rank[front] = cur
        remaining[front] = False
        cur += 1
    return rank


def crowding_distance(objectives: np.ndarray) -> np.ndarray:
    """NSGA-II crowding, used to break ties within a front so that selection
    does not collapse onto one corner of the performance/cost trade-off."""
    n, m = objectives.shape
    if n <= 2:
        return np.full(n, np.inf)
    dist = np.zeros(n)
    for j in range(m):
        order = np.argsort(objectives[:, j])
        vals = objectives[order, j]
        dist[order[0]] = dist[order[-1]] = np.inf
        span = max(vals[-1] - vals[0], 1e-9)
        dist[order[1:-1]] += (vals[2:] - vals[:-2]) / span
    return dist


def fitness_key(reward: np.ndarray, compute: np.ndarray,
                metabolism: bool,
                min_criterion_pct: float = 50.0) -> tuple[np.ndarray, np.ndarray]:
    """Return (rank, tiebreak); lower rank is better, higher tiebreak is better.

    With metabolism off this is plain reward rank.

    With metabolism on it is a Pareto rank on (reward, -compute) **restricted
    to organisms that clear a minimal performance criterion**.

    Why the criterion is necessary, and it is not a detail.  On a two-objective
    front the cheapest organism is ALWAYS non-dominated -- nothing can beat it
    on cost -- so a bare Pareto rank hands rank 0 to the most degenerate
    genome in the population regardless of how badly it performs.  The pilot
    showed exactly that: with metabolism on, genomes collapsed from 4 genes to
    ~1 within 40 generations while the fitness-shuffled drift control stayed at
    3.7.  A one-gene organism has no organisation to measure, so the
    experiment's central question becomes unaskable.

    The minimal criterion (Brant & Stanley 2017; used in POET) makes cheapness
    pay only among organisms that actually perform: below the reward floor, an
    organism is ranked after every eligible one no matter how cheap it is.
    That is also the behaviour the original proposal asks for -- redundancy
    should survive when its benefits justify its metabolic cost, rather than
    being competed away by whatever is smallest.
    """
    if not metabolism:
        order = np.argsort(np.argsort(-reward))
        return order.astype(np.int32), reward

    n = reward.shape[0]
    floor = np.percentile(reward, min_criterion_pct)
    eligible = reward >= floor
    if eligible.sum() < 2:                      # degenerate population
        eligible = np.ones(n, bool)

    rank = np.zeros(n, np.int32)
    tie = np.zeros(n, float)

    idx = np.flatnonzero(eligible)
    obj = np.stack([reward[idx], -compute[idx]], axis=1)
    rank[idx] = pareto_rank(obj)
    tie[idx] = crowding_distance(obj)

    bad = np.flatnonzero(~eligible)
    if bad.size:
        offset = int(rank[idx].max()) + 1
        rank[bad] = offset + np.argsort(np.argsort(-reward[bad])).astype(np.int32)
        tie[bad] = reward[bad]
    return rank, tie
def tournament_select(rank: np.ndarray, tiebreak: np.ndarray, k: int,
                      rng: np.random.Generator, n_picks: int) -> np.ndarray:
    """k-way tournament within one island."""
    n = rank.shape[0]
    cand = rng.integers(0, n, size=(n_picks, k))
    r = rank[cand]
    t = tiebreak[cand]
    # Lowest rank wins; higher crowding distance breaks ties.
    key = r.astype(np.float64) - 1e-9 * np.clip(t, -1e6, 1e6)
    return cand[np.arange(n_picks), np.argmin(key, axis=1)]


def step_generation(reward: np.ndarray, compute: np.ndarray, island: np.ndarray,
                    n_islands: int, island_size: int, tournament: int,
                    metabolism: bool, rng: np.random.Generator,
                    shuffled: bool = False,
                    min_criterion_pct: float = 50.0
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Decide who reproduces and who dies, per island.

    Returns (parent_of_slot, survived) where parent_of_slot[i] is the index of
    the organism whose offspring fills slot i, or -1 if slot i's occupant
    survives unchanged.  Population size is constant: offspring displace
    tournament losers, so there is no exponential growth.

    `shuffled` implements the fitness-shuffled drift control -- fitness is
    permuted within island, breaking the link between phenotype and
    reproductive success while leaving every other dynamic identical.  This is
    the null against which genome growth, duplication rate and every modularity
    metric must be compared (DESIGN.md 3.2).
    """
    n = reward.shape[0]
    parent = np.full(n, -1, np.int64)
    survived = np.ones(n, bool)

    for isl in range(n_islands):
        idx = np.flatnonzero(island == isl)
        if idx.size == 0:
            continue
        r, c = reward[idx], compute[idx]
        if shuffled:
            perm = rng.permutation(idx.size)
            r, c = r[perm], c[perm]
        rank, tb = fitness_key(r, c, metabolism, min_criterion_pct)

        n_replace = max(1, idx.size // 4)
        # Losers: worst by (rank, -tiebreak).
        key = rank.astype(np.float64) - 1e-9 * np.clip(tb, -1e6, 1e6)
        losers = idx[np.argsort(-key)[:n_replace]]
        winners = idx[tournament_select(rank, tb, tournament, rng, n_replace)]
        parent[losers] = winners
        survived[losers] = False
    return parent, survived


def migrate(island: np.ndarray, n_islands: int, migrants: int,
            rng: np.random.Generator) -> np.ndarray:
    """Occasional stepping-stone migration.  No global champion colonisation:
    migrants are chosen at random, not by fitness."""
    island = island.copy()
    for isl in range(n_islands):
        idx = np.flatnonzero(island == isl)
        if idx.size <= migrants:
            continue
        movers = rng.choice(idx, size=migrants, replace=False)
        island[movers] = (isl + 1) % n_islands
    return island
