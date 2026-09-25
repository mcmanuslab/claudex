"""Periodic assays that feed the primary outcome measure.

The experiment records every duplication event as it happens (lane, generation,
the new gene's innovation id and its parent's).  This module revisits those
pairs later and asks the question the project exists to answer: have the two
copies diverged in what they causally contribute, more than the fitness-shuffled
drift control's duplicate pairs do at the same age?

Run periodically, never every generation -- the ablation assay costs one extra
rollout per gene.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..genome.population import Population
from ..metrics.assays import contribution_matrix, realised_graph
from ..metrics.definitions import (division_of_labour, pair_divergence,
                                   q_structural, specialisation)


@dataclass
class PairObservation:
    run: int
    lane: int
    born_generation: int
    observed_generation: int
    age: int
    innov: int
    parent_innov: int
    divergence: float
    weight_distance: float
    both_alive: int


@dataclass
class OrganismObservation:
    run: int
    lane: int
    generation: int
    n_live: int
    q_str: float
    mean_specialisation: float
    division_of_labour: float


WEIGHTS = ("Wq", "Wk", "Wv", "Wo", "W1", "W2", "wg")


def _slot_of_innov(pop: Population, lane: int, innov: int) -> int:
    """Find the live gene carrying this innovation id, or -1.

    Encapsulated genes carry a negated innovation id (see
    mutation.operators._encapsulate), so both spellings are checked.
    """
    row = pop.innov[lane]
    hit = np.flatnonzero(((row == innov) | (row == -innov - 1)) & (pop.alive[lane] > 0))
    return int(hit[0]) if len(hit) else -1


def _weight_distance(pop: Population, lane: int, a: int, b: int) -> float:
    """Normalised L2 between two modules' weights.

    Reported alongside contribution divergence because the two can come apart:
    weights can drift a long way with no change in causal contribution, and
    that difference is exactly what separates neutral drift from functional
    divergence.
    """
    num = 0.0
    den = 0.0
    for name in WEIGHTS:
        arr = getattr(pop, name)
        d = arr[lane, a] - arr[lane, b]
        num += float((d * d).sum())
        den += float((arr[lane, a] ** 2).sum() + (arr[lane, b] ** 2).sum())
    return float(np.sqrt(num) / max(np.sqrt(den * 0.5), 1e-9))


def assay_duplicate_pairs(pop: Population, duplicate_pairs: list[dict],
                          contrib: np.ndarray, generation: int,
                          min_age: int = 1) -> list[PairObservation]:
    """Measure divergence for every tracked duplicate pair still expressed.

    `contrib` is (L, G, n_subgoals) from `contribution_matrix`.
    """
    out: list[PairObservation] = []
    for rec in duplicate_pairs:
        age = generation - rec["generation"]
        if age < min_age:
            continue
        lane = rec["lane"]
        if lane >= pop.L:
            continue
        child = _slot_of_innov(pop, lane, rec["innov"])
        parent = _slot_of_innov(pop, lane, rec["parent_innov"])
        both = int(child >= 0 and parent >= 0)
        if not both:
            continue
        div = pair_divergence(contrib[lane], child, parent)
        out.append(PairObservation(
            run=rec["run"], lane=lane, born_generation=rec["generation"],
            observed_generation=generation, age=age, innov=rec["innov"],
            parent_innov=rec["parent_innov"], divergence=float(div),
            weight_distance=_weight_distance(pop, lane, child, parent),
            both_alive=both,
        ))
    return out


def assay_organisms(pop: Population, contrib: np.ndarray, lanes: np.ndarray,
                    generation: int) -> list[OrganismObservation]:
    out: list[OrganismObservation] = []
    for lane in lanes:
        lane = int(lane)
        live = np.flatnonzero(pop.alive[lane] > 0)
        if len(live) < 2:
            continue
        c = contrib[lane][live]
        out.append(OrganismObservation(
            run=int(pop.run_id[lane]), lane=lane, generation=generation,
            n_live=len(live),
            q_str=float(q_structural(realised_graph(pop, lane))),
            mean_specialisation=float(specialisation(c).mean()),
            division_of_labour=float(division_of_labour(c)),
        ))
    return out


def lanes_carrying_pairs(pop: Population, records: list[dict],
                         lo: int, hi: int) -> dict[int, list[dict]]:
    """Find every lane in [lo, hi) where BOTH copies of a tracked pair are live.

    A duplication record stores the lane the event happened in, but lane
    identity is not organism identity: offspring overwrite the lanes of the
    organisms they displace, and a descendant carrying the pair may end up in
    any lane of its run.  Looking only in the birth lane therefore both misses
    most surviving pairs and would silently attribute a pair to whatever
    unrelated organism now occupies that lane.

    Searching by innovation id instead is correct by construction: an
    innovation id is unique to one module lineage, so a lane that carries both
    ids carries that pair, wherever it drifted to.
    """
    live = pop.alive[lo:hi] > 0
    innov = pop.innov[lo:hi]
    out: dict[int, list[dict]] = {}
    for rec in records:
        a, b = rec["innov"], rec["parent_innov"]
        # Encapsulated genes carry a negated id (mutation.operators._encapsulate).
        has_a = ((innov == a) | (innov == -a - 1)) & live
        has_b = ((innov == b) | (innov == -b - 1)) & live
        both = np.flatnonzero(has_a.any(axis=1) & has_b.any(axis=1))
        for local in both:
            out.setdefault(lo + int(local), []).append(rec)
    return out


def run_assay(experiment, n_lanes_per_run: int = 8,
              episodes: int = 2, lifetime: int = 24
              ) -> tuple[list[PairObservation], list[OrganismObservation]]:
    """One full assay pass over the experiment's current population.

    Lanes are sampled *preferentially from those carrying a tracked duplicate
    pair*, padded out with random lanes so the organism-level statistics stay
    representative.  Without that targeting almost every sampled lane carries
    no observable pair and the primary outcome has no data.

    Uses a shorter lifetime than selection does: the assay measures the SHAPE
    of each module's contribution, not fitness, and it costs one rollout per
    gene.
    """
    pop = experiment.pop
    cfg = experiment.cfg
    rng = np.random.default_rng(experiment.generation * 7919 + 13)
    n_org = cfg.ecology.n_organisms

    pairs: list[PairObservation] = []
    orgs: list[OrganismObservation] = []
    by_run: dict[int, list[dict]] = {}
    for rec in experiment.duplicate_pairs:
        if experiment.generation - rec["generation"] >= 1:
            by_run.setdefault(rec["run"], []).append(rec)

    for ri, sch in enumerate(experiment.schedules):
        lo, hi = ri * n_org, (ri + 1) * n_org
        carrying = lanes_carrying_pairs(pop, by_run.get(ri, []), lo, hi)

        want = list(carrying)
        rng.shuffle(want)
        want = want[:n_lanes_per_run]
        if len(want) < n_lanes_per_run:
            rest = [l for l in range(lo, hi) if l not in set(want)]
            want += list(rng.permutation(rest)[:n_lanes_per_run - len(want)])
        lanes = np.array(sorted(want), dtype=int)

        goal = sch.goal_at(experiment.generation)
        params = sch.params_at(experiment.generation, len(lanes))
        sub = _slice_population(pop, lanes)
        contrib_sub = contribution_matrix(sub, goal, params, episodes, lifetime,
                                          cfg, rng)
        contrib = np.zeros((pop.L, pop.G, len(goal)))
        contrib[lanes] = contrib_sub

        orgs += assay_organisms(pop, contrib, lanes, experiment.generation)
        for lane in lanes:
            for rec in carrying.get(int(lane), []):
                obs = assay_duplicate_pairs(pop, [dict(rec, lane=int(lane))],
                                            contrib, experiment.generation)
                pairs += obs
    return pairs, orgs


def _slice_population(pop: Population, lanes: np.ndarray) -> Population:
    """A view of the population restricted to `lanes`, for cheap assays."""
    import copy

    sub = copy.copy(pop)
    sub.L = len(lanes)
    for name in ("Wq", "Wk", "Wv", "Wo", "W1", "b1", "W2", "b2", "g1", "g2",
                 "wg", "bg", "alive", "src", "src_mask", "gate_b", "gate_s",
                 "E_obs", "W_act", "b_act", "meta", "innov", "gene_parent",
                 "birth_gen", "org_id", "org_parent", "run_id", "island"):
        setattr(sub, name, getattr(pop, name)[lanes])
    return sub
