"""Observational assays: ablation, silencing, message corruption.

These are *never* used for reproductive fitness (the original proposal is right
about this, and `tests/test_alien_isolation.py` enforces it).  They are run on
copies of the population, at intervals, purely to measure what evolved.
"""

from __future__ import annotations

import copy

import numpy as np

from ..genome.population import Population
from ..organisms.execute import new_state, step
from ..environments import primitives as P


def _clone(pop: Population) -> Population:
    return copy.deepcopy(pop)


def rollout_rewards(pop: Population, goal: tuple[int, ...], params: np.ndarray,
                    n_episodes: int, lifetime: int, cfg, rng: np.random.Generator,
                    ) -> np.ndarray:
    """Run a rollout and return per-lane, per-primitive mean reward: (L, 10)."""
    ec = cfg.environment
    L, E = pop.L, n_episodes
    st = new_state(pop, E)
    n = L * E
    ps = P.init_state(n, rng)
    active = np.zeros((n, 10), np.float32)
    for g in goal:
        active[:, g] = 1.0
    prm = np.repeat(params, E, axis=0) if params.shape[0] == L else \
        np.repeat(params[:1], n, axis=0)

    total = np.zeros((n, 10), np.float64)
    for _ in range(lifetime):
        obs = P.observe(ps, active, prm, ec.n_symbols, ec.n_obs_channels, rng)
        obs_l = obs.reshape(L, E, ec.n_obs_channels).transpose(0, 2, 1)
        logits = step(pop, st, obs_l, cfg.gate_threshold)
        action = np.argmax(logits, axis=-1).reshape(n)
        total += P.reward(ps, active, prm, action, ec.n_actions, ec.n_symbols)
    return (total / lifetime).reshape(L, E, 10).mean(axis=1)


def ablate_gene(pop: Population, gene: int) -> Population:
    """Silence one gene: it is present in the genome but emits nothing."""
    p = _clone(pop)
    p.alive[:, gene] = 0.0
    p.src_mask[:, gene] = 0.0
    hit = p.src == p.gene_slot(gene)
    p.src[hit] = 0
    p.src_mask[hit] = 0.0
    return p


def contribution_matrix(pop: Population, goal: tuple[int, ...], params: np.ndarray,
                        n_episodes: int, lifetime: int, cfg,
                        rng: np.random.Generator) -> np.ndarray:
    """Causal contribution of every gene to every subgoal: (L, G, n_goal).

    contribution[l, g, k] = baseline reward on subgoal k, minus reward on
    subgoal k with gene g silenced.  This is the quantity all of
    specialisation, division of labour and the primary outcome are built on.
    """
    base = rollout_rewards(pop, goal, params, n_episodes, lifetime, cfg,
                           np.random.default_rng(rng.integers(1 << 31)))
    out = np.zeros((pop.L, pop.G, len(goal)))
    cols = list(goal)
    for g in range(pop.G):
        if pop.alive[:, g].sum() == 0:
            continue
        abl = rollout_rewards(ablate_gene(pop, g), goal, params, n_episodes,
                              lifetime, cfg, np.random.default_rng(rng.integers(1 << 31)))
        out[:, g, :] = (base[:, cols] - abl[:, cols])
    return out


def ablation_curve(pop: Population, goal: tuple[int, ...], params: np.ndarray,
                   n_episodes: int, lifetime: int, cfg, rng: np.random.Generator,
                   fractions=(0.0, 0.125, 0.25, 0.5), draws: int = 8
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Fitness under random ablation of a fraction of live genes.

    Returns (fractions, fitness) with fitness averaged over `draws` random
    ablation sets per fraction.  Feeds `robustness_auc`.
    """
    fr = np.asarray(fractions, dtype=float)
    out = np.zeros((len(fr), pop.L))
    for fi, f in enumerate(fr):
        acc = np.zeros(pop.L)
        for _ in range(draws):
            p = _clone(pop)
            for lane in range(pop.L):
                live = np.flatnonzero(p.alive[lane] > 0)
                k = int(round(f * len(live)))
                if k:
                    for g in rng.choice(live, size=k, replace=False):
                        p.alive[lane, int(g)] = 0.0
                        p.src_mask[lane, int(g)] = 0.0
            r = rollout_rewards(p, goal, params, n_episodes, lifetime, cfg,
                                np.random.default_rng(rng.integers(1 << 31)))
            acc += r[:, list(goal)].sum(axis=1)
        out[fi] = acc / draws
    return fr, out


def joint_ablation(pop: Population, goal, params, n_episodes, lifetime, cfg,
                   rng, genes: list[int]) -> np.ndarray:
    """Pairwise joint-ablation matrix over `genes`, for the redundancy test."""
    n = len(genes)
    base = rollout_rewards(pop, goal, params, n_episodes, lifetime, cfg,
                           np.random.default_rng(rng.integers(1 << 31)))
    base_s = base[:, list(goal)].sum(axis=1)
    joint = np.zeros((pop.L, n, n))
    for a in range(n):
        for b in range(a + 1, n):
            p = ablate_gene(ablate_gene(pop, genes[a]), genes[b])
            r = rollout_rewards(p, goal, params, n_episodes, lifetime, cfg,
                                np.random.default_rng(rng.integers(1 << 31)))
            loss = (base_s - r[:, list(goal)].sum(axis=1)) / np.maximum(np.abs(base_s), 1e-9)
            joint[:, a, b] = joint[:, b, a] = loss
    return joint


def realised_graph(pop: Population, lane: int) -> np.ndarray:
    """Adjacency over live genes, for Q_str.  Sensor edges are excluded: they
    are shared by construction and would depress modularity uniformly."""
    live = np.flatnonzero(pop.alive[lane] > 0)
    index = {int(pop.gene_slot(int(g))): i for i, g in enumerate(live)}
    adj = np.zeros((len(live), len(live)))
    for i, g in enumerate(live):
        for s, m in zip(pop.src[lane, g], pop.src_mask[lane, g]):
            if m > 0 and int(s) in index:
                j = index[int(s)]
                if j != i:
                    adj[j, i] = 1.0
    return adj
