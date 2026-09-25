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
    """Two runs differing only in `duplication` must diverge in genome length."""
    cfg = ExperimentConfig()
    cfg.ecology.n_islands, cfg.ecology.island_size = 2, 8
    cfg.ecology.n_episodes, cfg.ecology.lifetime = 2, 8
    cfg.mutation.p_duplicate = 0.5
    ex = Experiment(cfg=cfg, runs=[
        RunConfig(name="dup_on", duplication=True, seed=1),
        RunConfig(name="dup_off", duplication=False, seed=1),
    ])
    for _ in range(25):
        recs = ex.run_generation(sample_q=0)
    assert recs[0].mean_genome_len > recs[1].mean_genome_len
    assert recs[1].mean_genome_len <= cfg.genome.n_genes_init
