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


def run_assay(experiment, n_lanes_per_run: int = 8,
              episodes: int = 2, lifetime: int = 24
              ) -> tuple[list[PairObservation], list[OrganismObservation]]:
    """One full assay pass over the experiment's current population.

    Uses a shorter lifetime than selection does: the assay is measuring the
    *shape* of each module's contribution, not estimating fitness precisely,
    and it costs one rollout per gene.
    """
    pop = experiment.pop
    cfg = experiment.cfg
    rng = np.random.default_rng(experiment.generation * 7919 + 13)
    n_org = cfg.ecology.n_organisms

    pairs: list[PairObservation] = []
    orgs: list[OrganismObservation] = []
    by_run: dict[int, list[dict]] = {}
    for rec in experiment.duplicate_pairs:
        by_run.setdefault(rec["run"], []).append(rec)

    for ri, sch in enumerate(experiment.schedules):
        lo = ri * n_org
        lanes = lo + rng.permutation(n_org)[:n_lanes_per_run]
        goal = sch.goal_at(experiment.generation)
        params = sch.params_at(experiment.generation, len(lanes))

        sub = _slice_population(pop, lanes)
        contrib_sub = contribution_matrix(sub, goal, params, episodes, lifetime,
                                          cfg, rng)
        contrib = np.zeros((pop.L, pop.G, len(goal)))
        contrib[lanes] = contrib_sub
        orgs += assay_organisms(pop, contrib, lanes, experiment.generation)
        lane_set = set(int(x) for x in lanes)
        pairs += assay_duplicate_pairs(
            pop, [r for r in by_run.get(ri, []) if r["lane"] in lane_set],
            contrib, experiment.generation)
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
