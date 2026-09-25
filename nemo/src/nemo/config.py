"""Experiment configuration.

Defaults are the phase-2 pilot from DESIGN.md 10.  Every number here that
differs from the original proposal is annotated with the reason.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Literal

GoalStructure = Literal["MVG", "RVG", "FIX"]
RewardAggregation = Literal["conjunctive", "mean"]
LearnMode = Literal["none", "reset", "lamarckian", "partial"]


@dataclass
class ModuleConfig:
    d_model: int = 16
    d_ff: int = 32
    max_in_degree: int = 4          # K


@dataclass
class GenomeConfig:
    n_rounds: int = 4               # R: sequential message-passing rounds
    slots_per_round: int = 4        # S: genes per round-band
    n_genes_init: int = 4           # ancestral genome length
    # G_max = n_rounds * slots_per_round.  Fixed capacity with masking, because
    # mx.compile is shape-dependent (DESIGN.md 7.2).

    @property
    def g_max(self) -> int:
        return self.n_rounds * self.slots_per_round


@dataclass
class MutationConfig:
    """Bootstrap rates.  DESIGN.md 8.2 disagrees with the proposal on one
    number: duplication starts at 2%, not 5%, with deletion matched, so the
    neutral expectation on genome length is flat rather than upward."""

    p_weight: float = 0.70
    p_regulatory: float = 0.09
    p_wiring: float = 0.15
    p_duplicate: float = 0.02
    p_delete: float = 0.02
    p_subsystem_dup: float = 0.01
    p_encapsulate: float = 0.01
    weight_sigma: float = 0.08
    heritable: bool = False         # evolvable-mutation-rate condition
    meta_sigma: float = 0.15        # log-normal step on heritable rates
    meta_bounds: tuple[float, float] = (1e-4, 0.5)

    def structural_mass(self) -> float:
        return self.p_duplicate + self.p_delete + self.p_subsystem_dup + self.p_encapsulate


@dataclass
class EcologyConfig:
    n_islands: int = 8
    island_size: int = 32
    n_episodes: int = 8             # E: arithmetic-intensity lever (DESIGN.md 6)
    lifetime: int = 256
    tournament: int = 4
    offspring_per_event: int = 2
    migration_interval: int = 20
    migrants: int = 1

    @property
    def n_organisms(self) -> int:
        return self.n_islands * self.island_size


@dataclass
class EnvironmentConfig:
    n_symbols: int = 16
    n_actions: int = 8
    n_obs_channels: int = 2
    goal_arity: int = 3             # primitives active per world
    mvg_basis: int = 6              # size of the shared subgoal basis under MVG
    switch_every: int = 50          # generations between goal switches
    noise: float = 0.05
    aggregation: RewardAggregation = "conjunctive"
    # "conjunctive": every active subgoal must be handled, so subgoals are
    #   genuine sub-problems and division of labour can pay.
    # "mean": the arithmetic mean.  Kept as an explicit CONTROL, not as a
    #   default -- the pilot showed that under a mean, one module specialises
    #   on the easiest channel, ignores the rest, and still scores best, which
    #   removes the only reason for an organism to be modular at all.
    conj_weight: float = 0.75       # weight on the conjunctive term


@dataclass
class RunConfig:
    """One lane-group of the batched experiment: a single condition."""

    name: str = "pilot"
    goal_structure: GoalStructure = "MVG"
    metabolism: bool = True
    duplication: bool = True
    regulation_mutable: bool = True
    topology_mutable: bool = True
    learn: LearnMode = "none"       # phase 1 has no lifetime learning
    shuffled_fitness: bool = False  # the drift control
    global_selection: bool = False
    seed: int = 0


@dataclass
class ExperimentConfig:
    module: ModuleConfig = field(default_factory=ModuleConfig)
    genome: GenomeConfig = field(default_factory=GenomeConfig)
    mutation: MutationConfig = field(default_factory=MutationConfig)
    ecology: EcologyConfig = field(default_factory=EcologyConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    runs: list[RunConfig] = field(default_factory=list)
    generations: int = 1000
    metabolic_lambda: float = 1.0   # only sets Pareto tie-breaking scale
    gate_threshold: float = 0.5
    mem_soft_gb: float = 32.0       # DESIGN.md 7.4 -- not 350
    mem_hard_gb: float = 64.0
    out_dir: str = "results/pilot"

    def to_dict(self) -> dict:
        return asdict(self)


def factorial_runs(replicates: int = 12) -> list[RunConfig]:
    """The phase-1 factorial from DESIGN.md 3.1, plus the drift control.

    3 goal structures x 2 metabolism x 2 duplication x `replicates`, and one
    fitness-shuffled drift lane-group per replicate.

    The drift control shares its *seed* with the MVG/M-on/D-on replicate it
    controls for, so the two see the identical goal schedule, the identical
    subgoal parameters and the identical ancestral population.  Without that
    pairing the control differs from the cell in goal sequence as well as in
    selection, and the comparison stops being a control.
    """
    runs: list[RunConfig] = []
    seed_of: dict[int, int] = {}
    seed = 0
    for goal in ("MVG", "RVG", "FIX"):
        for metab in (True, False):
            for dup in (True, False):
                for r in range(replicates):
                    if goal == "MVG" and metab and dup:
                        seed_of[r] = seed
                    runs.append(RunConfig(
                        name=f"{goal}_M{int(metab)}_D{int(dup)}_r{r}",
                        goal_structure=goal, metabolism=metab,
                        duplication=dup, seed=seed,
                    ))
                    seed += 1
    for r in range(replicates):
        runs.append(RunConfig(
            name=f"DRIFT_r{r}", goal_structure="MVG", metabolism=True,
            duplication=True, shuffled_fitness=True, seed=seed_of[r],
        ))
    return runs
