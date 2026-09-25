"""Islands, ecological compute budgets, selection, and the information archive.

**Compute as a resource, not a penalty.** The brief proposed fitness of the form
``performance - lambda*flops``. The trouble is that lambda is a free parameter that
*determines the answer*: pick it small and architectures grow, pick it large and they
shrink, and the experiment reports the experimenter's choice back to them. Here compute
is instead a finite per-generation resource. An island may spend ``flop_budget`` FLOPs
on offspring each generation, and every offspring is charged the accounted cost of its
own evaluation, so a larger organism does not pay a fitness tax -- it simply means fewer
offspring fit in the generation. Growth is favoured only when the return *per unit of
compute* justifies it, and the interesting experiment becomes sweeping the budget to
find where that threshold lies. ``scalarised`` and ``pareto`` remain available as
ablations, precisely so the lambda-dependence can be demonstrated rather than assumed.

**Active population vs information archive.** The MAP-Elites archive records what was
discovered, indexed by (log2 parameter count, behaviour), but by default it does *not*
feed organisms back into reproduction. Protecting size niches from extinction would
manufacture the size diversity the study is trying to observe. ``archive_reentry`` turns
that protection on as an explicit experimental condition.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..environments.suites import Scored
from ..models.genome import ArchGenome
from .evaluate import evaluate_group, group_by_arch
from .organism import Organism, PhaseConfig, founder, recombine, reproduce

SELECTION_MODES = ("ecological", "scalarised", "pareto", "global_topk")


@dataclass
class EvoConfig:
    n_islands: int = 8
    island_capacity: int = 24
    tournament_k: int = 3
    worlds_per_island: int = 10
    credited_worlds: int = 7        # each organism is scored on a random subset
    instances_per_world: int = 8
    flop_budget_per_island: int = 3_000_000_000
    max_births_per_island: int = 24
    migration_interval: int = 10
    migrants: int = 1
    selection: str = "ecological"
    lambda_compute: float = 0.0     # only used by `scalarised`
    archive_reentry: bool = False
    novelty_weight: float = 0.0
    resample_worlds_every: int = 1
    max_params: int = 1_000_000     # hard ceiling enforced by the scheduler
    seed: int = 0


@dataclass
class Archive:
    """MAP-Elites grid over (log2 params, action entropy). Science, not selection."""

    size_bins: int = 24
    beh_bins: int = 8
    cells: dict[tuple[int, int], Organism] = field(default_factory=dict)
    n_insertions: int = 0

    def key(self, o: Organism) -> tuple[int, int]:
        s = int(np.clip(np.log2(max(1, o.n_params)) * 2, 0, self.size_bins - 1))
        b = int(np.clip((o.behaviour[0] if o.behaviour else 0.0) * self.beh_bins,
                        0, self.beh_bins - 1))
        return s, b

    def insert(self, o: Organism) -> bool:
        k = self.key(o)
        cur = self.cells.get(k)
        if cur is None or o.fitness > cur.fitness:
            self.cells[k] = o
            self.n_insertions += 1
            return True
        return False

    def coverage(self) -> int:
        return len(self.cells)

    def qd_score(self) -> float:
        return float(sum(max(0.0, o.fitness) for o in self.cells.values()))


class Island:
    def __init__(self, idx: int, members: list[Organism]) -> None:
        self.idx = idx
        self.members = members
        self.worlds: list[Scored] = []
        self.flops_spent = 0
        self.extinctions = 0


class Population:
    def __init__(self, cfg: EvoConfig, phase: PhaseConfig,
                 founders: list[ArchGenome], rng: np.random.Generator) -> None:
        self.cfg, self.phase, self.rng = cfg, phase, rng
        self.generation = 0
        self.archive = Archive()
        self.islands: list[Island] = []
        for i in range(cfg.n_islands):
            arch = founders[i % len(founders)]
            members = [founder(arch, rng, island=i) for _ in range(cfg.island_capacity)]
            self.islands.append(Island(i, members))
        self.records: list[dict] = []
        self.total_births = cfg.n_islands * cfg.island_capacity
        self.total_flops = 0

    # ------------------------------------------------------------- evaluation
    def evaluate(self, members: list[Organism], worlds: list[Scored]) -> None:
        """Score `members` on `worlds`, each on its own credited subset."""
        if not members:
            return
        cfg = self.cfg
        for sig, idxs in group_by_arch(members).items():
            group = [members[i] for i in idxs]
            credit = np.zeros((len(group), len(worlds)), dtype=bool)
            n_cred = min(cfg.credited_worlds, len(worlds))
            for r in range(len(group)):
                credit[r, self.rng.choice(len(worlds), n_cred, replace=False)] = True
            temps = np.array([o.gene("temperature") for o in group], dtype=np.float32)
            res = evaluate_group([o.weights for o in group], group[0].arch, worlds,
                                 temps, self.rng, cfg.instances_per_world, credit)
            for r, o in enumerate(group):
                o.fitness_components = {
                    "score": float(res["score"][r]),
                    "gain": float(res["gain"][r]),
                    "final": float(res["final"][r]),
                    "flops": int(res["flops"][r]),
                }
                o.behaviour = tuple(float(x) for x in res["behaviour"][r])
                o.eval_flops = int(res["flops"][r])
                o.fitness = self._fitness(o)
                self.total_flops += o.eval_flops

    def _fitness(self, o: Organism) -> float:
        if self.phase.neutral:
            # Neutral-drift null: identical operators and demography, fitness carries no
            # information. Any trend that survives this is an artefact of the mechanics.
            return float(self.rng.random())
        s = o.fitness_components["score"]
        if self.cfg.selection == "scalarised":
            s -= self.cfg.lambda_compute * np.log10(max(1, o.eval_flops))
        return float(s)

    # ------------------------------------------------------------ reproduction
    def _tournament(self, members: list[Organism]) -> Organism:
        k = min(self.cfg.tournament_k, len(members))
        cand = [members[i] for i in self.rng.choice(len(members), k, replace=False)]
        return max(cand, key=lambda o: o.fitness)

    def _estimate_flops(self, o: Organism) -> int:
        from ..environments.worlds import CONTEXT
        return (self.cfg.credited_worlds * self.cfg.instances_per_world
                * o.arch.flops_forward(CONTEXT))

    def breed(self, isl: Island) -> list[Organism]:
        """Produce offspring until the island's generation compute budget is spent.

        This is where architecture size acquires a cost: a bigger child consumes more of
        the island's finite budget, so the island simply gets fewer children that
        generation. No exchange rate between performance and FLOPs is ever chosen.
        """
        cfg = self.cfg
        pool = list(isl.members)
        if cfg.archive_reentry and self.archive.cells:
            pool += list(self.archive.cells.values())
        kids: list[Organism] = []
        spent = 0
        while len(kids) < cfg.max_births_per_island:
            parent = self._tournament(pool)
            if self.phase.recombination and len(pool) > 1 and self.rng.random() < 0.3:
                mate = self._tournament(pool)
                child = recombine(parent, mate, self.rng, self.phase, self.generation)
                if child is None:
                    child = reproduce(parent, self.rng, self.phase, self.generation)
            else:
                child = reproduce(parent, self.rng, self.phase, self.generation)
            if child.arch.n_params > cfg.max_params:
                child.death_reason = "resource_ceiling"
                self.records.append(_record(child, self.generation, alive=False))
                continue
            cost = self._estimate_flops(child)
            if spent + cost > cfg.flop_budget_per_island:
                break
            spent += cost
            kids.append(child)
        isl.flops_spent = spent
        self.total_births += len(kids)
        return kids

    # ----------------------------------------------------------------- culling
    def cull(self, isl: Island, kids: list[Organism]) -> None:
        cfg = self.cfg
        pool = isl.members + kids
        for o in pool:
            self.archive.insert(o)
        if cfg.selection == "pareto":
            keep = _pareto_truncate(pool, cfg.island_capacity)
        else:
            keep_idx = np.argsort([-o.fitness for o in pool])[: cfg.island_capacity]
            keep = [pool[i] for i in keep_idx]
        keep_ids = {id(o) for o in keep}
        for o in pool:
            if id(o) not in keep_ids:
                o.death_reason = "selection"
                isl.extinctions += 1
                self.records.append(_record(o, self.generation, alive=False))
                o.weights = {}          # free tensors; metadata survives in `records`
        for o in keep:
            o.age += 1
        isl.members = keep

    def migrate(self) -> None:
        cfg = self.cfg
        if cfg.n_islands < 2 or self.generation % cfg.migration_interval:
            return
        for i, isl in enumerate(self.islands):
            for _ in range(cfg.migrants):
                if len(isl.members) <= 1:
                    break
                j = int(self.rng.integers(0, cfg.n_islands))
                if j == i:
                    continue
                mover = self._tournament(isl.members)
                isl.members.remove(mover)
                mover.island = j
                self.islands[j].members.append(mover)

    # -------------------------------------------------------------- main step
    def step(self, world_sampler) -> dict:
        cfg = self.cfg
        self.generation += 1
        for isl in self.islands:
            if not isl.worlds or self.generation % cfg.resample_worlds_every == 0:
                isl.worlds = world_sampler(isl.idx, self.generation)
        for isl in self.islands:
            self.evaluate(isl.members, isl.worlds)
            kids = self.breed(isl)
            self.evaluate(kids, isl.worlds)
            self.cull(isl, kids)
        self.migrate()
        for isl in self.islands:
            for o in isl.members:
                self.records.append(_record(o, self.generation, alive=True))
        return self.snapshot()

    def living(self) -> list[Organism]:
        return [o for isl in self.islands for o in isl.members]

    def snapshot(self) -> dict:
        live = self.living()
        params = np.array([o.n_params for o in live], dtype=float)
        fit = np.array([o.fitness_components.get("score", 0.0) for o in live])
        gain = np.array([o.fitness_components.get("gain", 0.0) for o in live])
        genes = {k: float(np.median([o.gene(k) for o in live]))
                 for k in ("p_growth_bias", "p_structural", "temperature", "lr",
                           "weight_sigma", "meta_sigma")}
        sigs = {o.arch.signature() for o in live}
        return {
            "generation": self.generation,
            "n_alive": len(live),
            "params_median": float(np.median(params)),
            "params_mean": float(params.mean()),
            "params_min": float(params.min()),
            "params_max": float(params.max()),
            "params_log_mean": float(np.mean(np.log(params))),
            "params_log_sd": float(np.std(np.log(params))),
            "score_mean": float(fit.mean()),
            "score_max": float(fit.max()),
            "gain_mean": float(gain.mean()),
            "n_species": len(sigs),
            "archive_coverage": self.archive.coverage(),
            "qd_score": self.archive.qd_score(),
            "total_births": self.total_births,
            "total_flops": self.total_flops,
            **{f"gene_{k}": v for k, v in genes.items()},
        }


def _pareto_truncate(pool: list[Organism], n: int) -> list[Organism]:
    """Non-dominated sorting on (performance, -compute), filling by rank."""
    pts = np.array([[o.fitness_components.get("score", 0.0), -float(o.eval_flops)]
                    for o in pool])
    remaining = list(range(len(pool)))
    keep: list[Organism] = []
    while remaining and len(keep) < n:
        front = [i for i in remaining
                 if not any(np.all(pts[j] >= pts[i]) and np.any(pts[j] > pts[i])
                            for j in remaining if j != i)]
        front = front or list(remaining)
        front.sort(key=lambda i: -pts[i][0])
        for i in front:
            if len(keep) < n:
                keep.append(pool[i])
            remaining.remove(i)
    return keep


def _record(o: Organism, gen: int, alive: bool) -> dict:
    """Compact, permanent metadata. Kept even when the weight tensors are freed."""
    return {
        "oid": o.oid, "gen": gen, "birth_gen": o.birth_gen, "island": o.island,
        "parents": ",".join(map(str, o.parents)), "lineage_root": o.lineage_root,
        "params": o.n_params, "sig": "x".join(map(str, o.arch.signature())),
        "d_model": o.arch.d_model, "n_heads": o.arch.n_heads,
        "d_ff": o.arch.d_ff, "n_blocks": o.arch.n_blocks,
        "mutations": ",".join(o.mutations), "n_struct_events": o.n_struct_events,
        "fitness": o.fitness, "score": o.fitness_components.get("score", 0.0),
        "gain": o.fitness_components.get("gain", 0.0),
        "eval_flops": o.eval_flops, "age": o.age, "alive": int(alive),
        "death_reason": o.death_reason,
        "beh0": o.behaviour[0] if o.behaviour else 0.0,
        "beh1": o.behaviour[1] if len(o.behaviour) > 1 else 0.0,
        "beh2": o.behaviour[2] if len(o.behaviour) > 2 else 0.0,
        **{f"gene_{k}": o.gene(k) for k in
           ("p_growth_bias", "p_structural", "temperature", "lr", "weight_sigma",
            "meta_sigma", "morph_noise")},
    }
