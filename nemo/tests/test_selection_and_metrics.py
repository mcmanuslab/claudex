"""Selection, metric and null-model invariants."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemo.config import ExperimentConfig, RunConfig, factorial_runs   # noqa: E402
from nemo.ecology.evolve import Experiment                             # noqa: E402
from nemo.metrics import definitions as M                              # noqa: E402
from nemo.selection.islands import (pareto_rank, step_generation,      # noqa: E402
                                    tournament_select, fitness_key)


# ---------------------------------------------------------------- selection
def test_pareto_front_is_correct():
    obj = np.array([[1., 0.], [0., 1.], [2., 2.], [0.5, 0.5], [3., -1.]])
    r = pareto_rank(obj)
    assert r[2] == 0 and r[4] == 0            # non-dominated
    assert r[0] > 0 and r[1] > 0 and r[3] > 0  # dominated by [2,2]


def test_selection_favours_high_reward():
    rng = np.random.default_rng(0)
    n = 256
    reward = rng.random(n)
    compute = np.ones(n)
    island = np.repeat(np.arange(8), 32)
    gains = []
    for _ in range(20):
        parent, survived = step_generation(reward, compute, island, 8, 32, 4,
                                           False, rng)
        rep = np.flatnonzero(~survived)
        gains.append(reward[parent[rep]].mean() - reward[rep].mean())
    assert np.mean(gains) > 0.2, np.mean(gains)


def test_drift_control_has_no_selection_differential():
    """The fitness-shuffled control must reproduce at random.  If it had any
    selection differential it would not be a null."""
    rng = np.random.default_rng(0)
    n = 256
    reward = rng.random(n)
    compute = np.ones(n)
    island = np.repeat(np.arange(8), 32)
    gains = []
    for _ in range(60):
        parent, survived = step_generation(reward, compute, island, 8, 32, 4,
                                           True, rng, shuffled=True)
        rep = np.flatnonzero(~survived)
        gains.append(reward[parent[rep]].mean() - reward[rep].mean())
    assert abs(np.mean(gains)) < 0.05, np.mean(gains)


def test_metabolism_off_ignores_compute():
    reward = np.array([0.5, 0.4, 0.3])
    cheap = np.array([1.0, 1.0, 1.0])
    dear = np.array([1e9, 1.0, 1.0])
    r1, _ = fitness_key(reward, cheap, metabolism=False)
    r2, _ = fitness_key(reward, dear, metabolism=False)
    assert np.array_equal(r1, r2), "compute leaked into selection with M=off"


def test_metabolism_on_promotes_the_cheap_runner_up():
    """Pareto selection does NOT eliminate an expensive high performer -- it is
    non-dominated and stays on the front.  What metabolic cost does is *promote*
    a slightly worse but much cheaper organism onto the same front, so it
    reproduces as often.  That is the pressure, and it is what the M factor
    turns on and off."""
    reward = np.array([0.50, 0.45, 0.40])
    dear = np.array([1e9, 1.0, 2.0])

    off, _ = fitness_key(reward, dear, metabolism=False)
    assert off[0] < off[1], "with M=off the best performer should rank first"

    on, _ = fitness_key(reward, dear, metabolism=True)
    assert on[0] == 0, "an expensive best performer is still non-dominated"
    assert on[1] == 0, "the cheap runner-up is promoted onto the front"
    assert on[2] > 0, "dominated on both objectives -> worse rank"


def test_population_size_is_constant():
    parent, survived = step_generation(
        np.random.random(64), np.ones(64), np.repeat(np.arange(4), 16),
        4, 16, 4, False, np.random.default_rng(0))
    assert len(parent) == 64
    assert (parent >= 0).sum() == (~survived).sum()


def test_tournament_prefers_better_rank():
    rng = np.random.default_rng(0)
    rank = np.arange(32)
    tb = np.zeros(32)
    picks = tournament_select(rank, tb, 4, rng, 4000)
    assert rank[picks].mean() < 12.0


# ------------------------------------------------------------------ metrics
def test_q_str_separates_modular_from_dense():
    adj = np.zeros((8, 8))
    for i, j in [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (3, 4)]:
        adj[i, j] = 1
    rng = np.random.default_rng(0)
    dense = (rng.random((8, 8)) < 0.7).astype(float)
    np.fill_diagonal(dense, 0)
    assert M.q_structural(adj, rng) > M.q_structural(dense, rng)


def test_rewire_null_preserves_out_degree():
    rng = np.random.default_rng(0)
    adj = np.zeros((8, 8))
    for i, j in [(0, 1), (0, 2), (1, 3), (2, 3), (4, 5), (5, 6)]:
        adj[i, j] = 1
    vals = M.rewire_null(adj, n_draws=20, rng=rng)
    assert len(vals) == 20 and np.isfinite(vals).all()


def test_specialisation_bounds():
    assert M.specialisation(np.array([[1.0, 0, 0]]))[0] == 1.0
    assert abs(M.specialisation(np.array([[1.0, 1.0, 1.0]]))[0]) < 1e-9
    s = M.specialisation(np.array([[0.6, 0.3, 0.1]]))[0]
    assert 0.0 < s < 1.0


def test_specialisation_permutation_null_is_degenerate_as_documented():
    """The docstring claims this null preserves entropy exactly.  If that ever
    stopped being true the primary outcome measure would need revisiting."""
    rng = np.random.default_rng(0)
    c = np.array([[0.7, 0.2, 0.1], [0.1, 0.1, 0.8]])
    null = M.specialisation_null(c, n_draws=50, rng=rng)
    assert np.allclose(null, M.specialisation(c)[None, :])


def test_primary_outcome_has_calibrated_false_positive_rate():
    """Drift-vs-drift must reject at the nominal rate.  This is the calibration
    that licenses the DESIGN.md 3.3 decision rule."""
    rng = np.random.default_rng(0)
    drift = np.abs(rng.normal(0.05, 0.03, 4000))
    same = np.abs(rng.normal(0.05, 0.03, 4000))
    r = M.duplication_specialised_vs_drift(same, drift)
    assert 0.02 < r["rate"] < 0.08, r["rate"]
    assert abs(r["cliffs_delta"]) < 0.1

    diverged = np.abs(rng.normal(0.30, 0.10, 1000))
    r2 = M.duplication_specialised_vs_drift(diverged, drift)
    assert r2["rate"] > 0.9 and r2["cliffs_delta"] > 0.8


def test_redundancy_definition():
    single = np.array([0.01, 0.02, 0.40])
    joint = np.zeros((3, 3))
    joint[0, 1] = joint[1, 0] = 0.55     # neither alone matters, both do
    joint[0, 2] = joint[2, 0] = 0.42
    pairs = M.redundancy_pairs(single, joint)
    assert pairs[0, 1] and not pairs[0, 2]   # gene 2 fails the "neither alone" test


def test_cliffs_delta_extremes():
    assert M.cliffs_delta(np.arange(10) + 100, np.arange(10)) == 1.0
    assert M.cliffs_delta(np.arange(10), np.arange(10) + 100) == -1.0
    assert abs(M.cliffs_delta(np.arange(10), np.arange(10))) < 1e-9


# ----------------------------------------------------------- factorial wiring
def test_drift_control_is_seed_matched_to_its_cell():
    runs = factorial_runs(replicates=5)
    mvg = {r.name[-2:]: r.seed for r in runs if r.name.startswith("MVG_M1_D1")}
    drift = {r.name[-2:]: r.seed for r in runs if r.name.startswith("DRIFT")}
    assert mvg == drift and len(mvg) == 5


def test_factorial_covers_every_cell():
    runs = factorial_runs(replicates=3)
    cells = {(r.goal_structure, r.metabolism, r.duplication)
             for r in runs if not r.shuffled_fitness}
    assert len(cells) == 12
    assert len(runs) == 12 * 3 + 3


def test_conditions_are_actually_distinct_in_the_loop():
    """Two runs differing only in `duplication` must differ in the loop.

    The discriminator is the duplication EVENT COUNT, not genome length:
    with metabolism on, selection removes duplicates about as fast as they
    arise, so both conditions shrink.  That is itself informative -- it is the
    metabolic factor doing its job -- but it means genome length alone cannot
    be used to verify that the D factor is wired up.
    """
    cfg = ExperimentConfig()
    cfg.ecology.n_islands, cfg.ecology.island_size = 2, 8
    cfg.ecology.n_episodes, cfg.ecology.lifetime = 2, 8
    cfg.mutation.p_duplicate = 0.5
    ex = Experiment(cfg=cfg, runs=[
        RunConfig(name="dup_on", duplication=True, seed=1),
        RunConfig(name="dup_off", duplication=False, seed=1),
    ])
    on = off = 0
    for _ in range(25):
        recs = ex.run_generation(sample_q=0)
        on += recs[0].n_duplications
        off += recs[1].n_duplications
    assert on > 0, "duplication enabled but no duplication events occurred"
    assert off == 0, "duplication disabled but events occurred"
    assert len(ex.duplicate_pairs) == on


def test_metabolism_off_lets_genomes_grow():
    """Without metabolic cost, duplication should inflate genome length --
    the bloat that RESEARCH.md 7 (C5) warns can masquerade as complexity.
    With cost on, that growth should be checked."""
    def run(metabolism):
        cfg = ExperimentConfig()
        cfg.ecology.n_islands, cfg.ecology.island_size = 2, 16
        cfg.ecology.n_episodes, cfg.ecology.lifetime = 2, 8
        cfg.mutation.p_duplicate, cfg.mutation.p_delete = 0.4, 0.02
        ex = Experiment(cfg=cfg, runs=[
            RunConfig(name="x", duplication=True, metabolism=metabolism, seed=2)])
        for _ in range(40):
            recs = ex.run_generation(sample_q=0)
        return recs[0].mean_genome_len

    free, costed = run(False), run(True)
    assert free > cfg_init(), "genomes did not grow even with cost off"
    assert free > costed, (free, costed)


def cfg_init() -> int:
    return ExperimentConfig().genome.n_genes_init


def test_minimal_criterion_blocks_the_degenerate_cheap_corner():
    """On a two-objective front the cheapest organism is ALWAYS non-dominated,
    so a bare Pareto rank hands rank 0 to the most degenerate genome in the
    population however badly it performs.

    The pilot showed this is not hypothetical: with metabolism on, genomes
    collapsed from 4 genes to ~1 within 40 generations while the drift control
    stayed at 3.7 -- and a one-gene organism has no organisation to measure,
    which makes the experiment's central question unaskable.
    """
    reward = np.array([0.01, 0.50, 0.45, 0.40, 0.35, 0.30])
    compute = np.array([1.0, 9.0, 8.0, 7.0, 6.0, 5.0])

    bare, _ = fitness_key(reward, compute, True, min_criterion_pct=0.0)
    assert bare[0] == 0, "precondition: without a criterion the worst is on the front"

    ranked, _ = fitness_key(reward, compute, True, min_criterion_pct=50.0)
    assert ranked[0] > ranked[1:].max() or ranked[0] > ranked[1:].min()
    assert ranked[np.argmax(reward)] == 0, "the best performer must stay on the front"


def test_minimal_criterion_survives_a_degenerate_population():
    """If almost nothing clears the floor, selection must still rank sanely
    rather than divide by an empty front."""
    reward = np.array([0.5, 0.5, 0.5, 0.5])
    compute = np.array([1.0, 2.0, 3.0, 4.0])
    rank, tie = fitness_key(reward, compute, True, min_criterion_pct=99.0)
    assert len(rank) == 4
    # Crowding distance is +inf for boundary solutions by NSGA-II's definition;
    # tournament_select clips it, so inf is expected, NaN is not.
    assert not np.isnan(tie).any()
    assert rank.min() == 0
    assert rank[np.argmin(compute)] == 0, "cheapest wins when performance ties"


def test_metabolism_still_rewards_cheapness_among_performers():
    """The criterion must not switch metabolism off: among organisms that
    clear the floor, the cheaper one is still promoted."""
    reward = np.array([0.50, 0.49, 0.20, 0.10])
    compute = np.array([1e6, 1.0, 1.0, 1.0])
    rank, _ = fitness_key(reward, compute, True, min_criterion_pct=50.0)
    assert rank[0] == 0 and rank[1] == 0, "both eligible extremes are on the front"
    assert rank[2] > 0 and rank[3] > 0


# -------------------------------------------------------- reward aggregation
def test_conjunctive_aggregation_penalises_ignoring_a_subgoal():
    """Under an arithmetic mean an organism can score well by handling the
    easiest subgoal and ignoring the rest, which removes the only reason for
    it to be modular.  The pilot showed this directly: under a mean the MVG
    cell reached the highest reward with the smallest genome (1.1 genes, 0.5
    of them active).
    """
    from nemo.ecology.evolve import aggregate_reward

    active = np.zeros((4, 10)); active[:, :3] = 1.0
    scores = np.zeros((4, 10))
    scores[0, :3] = [1.0, 0.0, 0.0]     # specialist: one subgoal, ignores two
    scores[1, :3] = [0.35, 0.35, 0.35]  # generalist
    scores[2, :3] = [1.0, 1.0, 1.0]     # solves everything
    scores[3, :3] = [1.0, 1.0, 0.0]     # two of three

    mean = aggregate_reward(scores, active, "mean")
    conj = aggregate_reward(scores, active, "conjunctive")

    assert mean[0] < mean[1] * 1.05, "precondition: under a mean these are close"
    assert conj[0] < conj[1] * 0.5, "conjunctive must punish ignoring subgoals"
    assert conj[3] < conj[2], "failing any subgoal must cost"
    assert conj[2] > conj[1] > conj[0]


def test_conjunctive_aggregation_keeps_an_early_gradient():
    """A pure geometric mean is zero for a population below baseline
    everywhere, which would leave early evolution with nothing to climb."""
    from nemo.ecology.evolve import aggregate_reward

    active = np.zeros((2, 10)); active[:, :3] = 1.0
    below = np.zeros((2, 10))
    below[0, :3] = [-0.2, -0.2, -0.2]
    below[1, :3] = [-0.1, -0.1, -0.1]
    conj = aggregate_reward(below, active, "conjunctive")
    assert conj[1] > conj[0], "no gradient below baseline"


def test_aggregation_ignores_inactive_channels():
    from nemo.ecology.evolve import aggregate_reward

    active = np.zeros((1, 10)); active[0, :2] = 1.0
    s = np.zeros((1, 10)); s[0, :2] = [0.6, 0.6]; s[0, 5:] = -9.0
    a = aggregate_reward(s, active, "conjunctive")[0]
    s2 = np.zeros((1, 10)); s2[0, :2] = [0.6, 0.6]
    b = aggregate_reward(s2, active, "conjunctive")[0]
    assert abs(a - b) < 1e-9
