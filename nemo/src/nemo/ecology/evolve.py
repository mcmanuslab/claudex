"""The evolution loop.

One `run_generation` call advances every replicate run in the batch by one
generation.  Replicate runs and factorial cells are *lanes*, not separate
processes: the modelled cost of 12 replicates is the same as 1 until the FLOP
roof (RESEARCH.md 5), so statistical power is close to free and only
lifetime x generations is serial.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np

from ..config import ExperimentConfig, RunConfig
from ..environments import primitives as P
from ..environments.world import Calibration, GoalSchedule
from ..genome.population import Population, init_population
from ..metrics.assays import realised_graph
from ..metrics.definitions import q_structural
from ..metrics.tracking import run_assay
from ..modules.spec import ModuleSpec
from ..mutation.operators import InnovationRegistry, mutate
from ..organisms.execute import active_modules, metabolic_flops, new_state, step
from ..selection.islands import migrate, step_generation


def aggregate_reward(per_prim: np.ndarray, active: np.ndarray,
                     mode: str = "conjunctive", conj_weight: float = 0.75
                     ) -> np.ndarray:
    """Combine per-subgoal scores into one reproductive-fitness number.

    per_prim : (L, 10) calibrated score per primitive
    active   : (L, 10) which primitives are in this lane's goal

    "mean" is the obvious choice and it is the wrong one.  Under an arithmetic
    mean an organism can score well by handling whichever subgoal is easiest
    and ignoring the others, so there is no pressure for one part of the
    organism to do one job and another part a different job -- which is exactly
    the pressure this experiment is trying to detect the consequences of.  The
    pilot showed the effect plainly: under a mean, the MVG cell reached the
    HIGHEST reward with the SMALLEST genome (1.1 genes, 0.5 of them active).

    "conjunctive" mixes a geometric mean over the active subgoals into the
    arithmetic one.  The geometric term is zero unless every subgoal is handled
    above its random baseline, so subgoals become genuine sub-problems; the
    arithmetic term is kept at a minority weight so early populations, which
    are below baseline everywhere, still have a gradient to climb.
    """
    n_act = np.maximum(active.sum(axis=1), 1)
    # Select rather than multiply.  `per_prim * active` would let a NaN in an
    # INACTIVE channel reach fitness (NaN * 0 == NaN), which would breach the
    # alien-channel isolation guarantee the moment any primitive misbehaved.
    # `where` makes that impossible rather than merely unlikely.
    live = np.where(active > 0, per_prim, 0.0)
    arith = live.sum(axis=1) / n_act
    if mode == "mean":
        return arith
    pos = np.clip(np.where(active > 0, per_prim, 1.0), 0.0, 1.0)
    # Geometric mean over ACTIVE channels only: an inactive channel contributes
    # a factor of 1 by construction, not a factor of 0.
    log_terms = np.where(active > 0, np.log(pos + 1e-6), 0.0)
    geo = np.exp(log_terms.sum(axis=1) / n_act)
    return (1.0 - conj_weight) * arith + conj_weight * geo


@dataclass
class GenerationRecord:
    generation: int
    run: int
    mean_reward: float
    max_reward: float
    mean_genome_len: float
    mean_active: float
    mean_flops: float
    n_duplications: int
    n_deletions: int
    n_encapsulations: int
    mean_q_str: float
    goal: tuple[int, ...]


@dataclass
class Experiment:
    cfg: ExperimentConfig
    runs: list[RunConfig]
    pop: Population = field(init=False)
    schedules: list[GoalSchedule] = field(init=False)
    registry: InnovationRegistry = field(init=False)
    calib: Calibration | None = None
    generation: int = 0
    duplicate_pairs: list[dict] = field(default_factory=list)
    records: list[GenerationRecord] = field(default_factory=list)
    events: list[tuple] = field(default_factory=list)
    pair_obs: list = field(default_factory=list)
    org_obs: list = field(default_factory=list)

    def __post_init__(self) -> None:
        n_org = self.cfg.ecology.n_organisms
        L = len(self.runs) * n_org
        run_id = np.repeat(np.arange(len(self.runs)), n_org)
        self.pop = init_population(self.cfg, L, run_id, seed=self.runs[0].seed)
        self.schedules = [
            GoalSchedule.build(r.goal_structure, self.cfg.environment.goal_arity,
                               self.cfg.environment.switch_every,
                               self.cfg.environment.mvg_basis, seed=r.seed)
            for r in self.runs
        ]
        self.registry = InnovationRegistry(next_id=self.cfg.genome.n_genes_init)
        self.spec = ModuleSpec(d_model=self.cfg.module.d_model,
                               d_ff=self.cfg.module.d_ff,
                               max_in_degree=self.cfg.module.max_in_degree)
        self._rng = np.random.default_rng(self.runs[0].seed + 12345)

    # -- rollout ---------------------------------------------------------
    def evaluate(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Returns (reward, compute_flops, active_count, per_primitive_reward)."""
        cfg, ec = self.cfg, self.cfg.environment
        L, E = self.pop.L, cfg.ecology.n_episodes
        n = L * E
        rng = self._rng

        active = np.zeros((n, 10), np.float32)
        params = np.zeros((n, 10), np.int32)
        n_org = cfg.ecology.n_organisms
        goals: list[tuple[int, ...]] = []
        for ri, sch in enumerate(self.schedules):
            goal = sch.goal_at(self.generation)
            goals.append(goal)
            lo, hi = ri * n_org * E, (ri + 1) * n_org * E
            for g in goal:
                active[lo:hi, g] = 1.0
            params[lo:hi] = sch.params_at(self.generation, hi - lo)
        self._goals = goals

        st = new_state(self.pop, E)
        ps = P.init_state(n, rng)
        total = np.zeros((n, 10), np.float64)
        for _ in range(cfg.ecology.lifetime):
            obs = P.observe(ps, active, params, ec.n_symbols, ec.n_obs_channels, rng)
            obs_l = obs.reshape(L, E, ec.n_obs_channels).transpose(0, 2, 1)
            logits = step(self.pop, st, obs_l, cfg.gate_threshold)
            action = np.argmax(logits, axis=-1).reshape(n)
            total += P.reward(ps, active, params, action, ec.n_actions, ec.n_symbols)
        per_prim = (total / cfg.ecology.lifetime).reshape(L, E, 10).mean(axis=1)

        if self.calib is not None:
            per_prim = self.calib.normalise(per_prim)
        act_l = active.reshape(L, E, 10)[:, 0, :]
        reward = aggregate_reward(per_prim, act_l, ec.aggregation, ec.conj_weight)

        flops = metabolic_flops(self.pop, st, self.spec)
        # Presence cost: a dormant gene still costs something to carry, but far
        # less than an executed one (DESIGN.md 6).
        flops = flops + 0.01 * self.pop.alive.sum(axis=1) * self.spec.n_params
        return reward, flops, active_modules(self.pop, st), per_prim

    # -- one generation ---------------------------------------------------
    def run_generation(self, sample_q: int = 4) -> list[GenerationRecord]:
        cfg = self.cfg
        reward, flops, n_active, _ = self.evaluate()
        n_org = cfg.ecology.n_organisms
        rng = self._rng
        recs: list[GenerationRecord] = []

        for ri, run in enumerate(self.runs):
            lo, hi = ri * n_org, (ri + 1) * n_org
            sl = slice(lo, hi)
            parent, _ = step_generation(
                reward[sl], flops[sl], self.pop.island[sl],
                cfg.ecology.n_islands, cfg.ecology.island_size,
                cfg.ecology.tournament, run.metabolism, rng,
                shuffled=run.shuffled_fitness,
            )
            n_dup = n_del = n_enc = 0
            for local, par in enumerate(parent):
                if par < 0:
                    continue
                child, src = lo + local, lo + int(par)
                self._copy_organism(src, child)
                evs = mutate(self.pop, child, rng, cfg.mutation, run,
                             self.registry, self.generation)
                for e in evs:
                    if e.kind == "duplicate":
                        n_dup += 1
                        self.duplicate_pairs.append({
                            "run": ri, "lane": child, "generation": self.generation,
                            "gene": e.gene, "src_gene": e.src_gene,
                            "innov": e.innov, "parent_innov": e.parent_innov,
                        })
                    elif e.kind == "delete":
                        n_del += 1
                    elif e.kind == "encapsulate":
                        n_enc += 1
                    self.events.append((self.generation, ri, child, e.kind, e.gene))

            q = 0.0
            if sample_q:
                lanes = rng.choice(np.arange(lo, hi), size=min(sample_q, n_org),
                                   replace=False)
                q = float(np.mean([q_structural(realised_graph(self.pop, int(l)))
                                   for l in lanes]))
            recs.append(GenerationRecord(
                generation=self.generation, run=ri,
                mean_reward=float(reward[sl].mean()), max_reward=float(reward[sl].max()),
                mean_genome_len=float(self.pop.alive[sl].sum(axis=1).mean()),
                mean_active=float(n_active[sl].mean()), mean_flops=float(flops[sl].mean()),
                n_duplications=n_dup, n_deletions=n_del, n_encapsulations=n_enc,
                mean_q_str=q, goal=self._goals[ri],
            ))

        if (self.generation + 1) % cfg.ecology.migration_interval == 0:
            for ri in range(len(self.runs)):
                sl = slice(ri * n_org, (ri + 1) * n_org)
                self.pop.island[sl] = migrate(self.pop.island[sl], cfg.ecology.n_islands,
                                              cfg.ecology.migrants, rng)
        self.generation += 1
        self.records.extend(recs)
        return recs

    def assay(self, n_lanes_per_run: int = 8, episodes: int = 2,
              lifetime: int = 24) -> None:
        """Periodic observational assay (never used for selection).

        Costs one extra rollout per gene, so it runs on a sample of lanes at a
        cadence set by the caller, not every generation.
        """
        pairs, orgs = run_assay(self, n_lanes_per_run=n_lanes_per_run,
                                episodes=episodes, lifetime=lifetime)
        self.pair_obs.extend(pairs)
        self.org_obs.extend(orgs)

    def _copy_organism(self, src: int, dst: int) -> None:
        for name in ("Wq", "Wk", "Wv", "Wo", "W1", "b1", "W2", "b2", "g1", "g2",
                     "wg", "bg", "alive", "src", "src_mask", "gate_b", "gate_s",
                     "E_obs", "W_act", "b_act", "meta", "innov", "gene_parent",
                     "birth_gen"):
            arr = getattr(self.pop, name)
            arr[dst] = arr[src]
        self.pop.org_parent[dst] = self.pop.org_id[src]
        self.pop.org_id[dst] = int(self.pop.org_id.max()) + 1
