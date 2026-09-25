"""Operational metrics.

Every biological term this project uses has exactly one computational meaning,
and it is defined here (DESIGN.md 2).  No claim anywhere in the project may use
one of these words outside this module's definition.

Each metric that could be inflated by an artefact carries its own null model:

  * structural modularity      -> degree-preserving rewiring (controls sparsity)
  * specialisation             -> permutation over subgoal labels
  * division of labour         -> the same permutation null
  * functional modularity      -> phase-shuffled activity traces
  * regulatory organisation    -> circular shift of the context stream

Reporting a metric without its null is not permitted: a sparser graph scores
higher on Newman modularity for purely structural reasons, and two randomly
initialised modules always differ.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-12


# --------------------------------------------------------------------------
# graph modularity
# --------------------------------------------------------------------------
def newman_q(adj: np.ndarray, communities: np.ndarray) -> float:
    """Newman modularity Q of an undirected graph under a given partition."""
    a = ((adj + adj.T) > 0).astype(np.float64)
    np.fill_diagonal(a, 0.0)
    m2 = a.sum()
    if m2 < EPS:
        return 0.0
    k = a.sum(axis=1)
    same = communities[:, None] == communities[None, :]
    return float(((a - np.outer(k, k) / m2) * same).sum() / m2)


def _greedy_partition(adj: np.ndarray, n_try: int = 8,
                      rng: np.random.Generator | None = None) -> np.ndarray:
    """Best of several randomised greedy 2..4-way partitions.

    Deliberately simple: the graphs here have at most G_max=16 nodes, so an
    exhaustive-ish greedy search is exact enough and has no library dependency.
    """
    rng = rng or np.random.default_rng(0)
    n = adj.shape[0]
    best_q, best_c = -1.0, np.zeros(n, np.int64)
    for n_comm in (2, 3, 4):
        if n_comm > n:
            break
        for _ in range(n_try):
            c = rng.integers(0, n_comm, size=n)
            improved = True
            while improved:
                improved = False
                for i in rng.permutation(n):
                    cur = c[i]
                    qs = []
                    for t in range(n_comm):
                        c[i] = t
                        qs.append(newman_q(adj, c))
                    c[i] = int(np.argmax(qs))
                    if c[i] != cur:
                        improved = True
            q = newman_q(adj, c)
            if q > best_q:
                best_q, best_c = q, c.copy()
    return best_c


def q_structural(adj: np.ndarray, rng: np.random.Generator | None = None) -> float:
    """Q_str: modularity of the realised message-passing graph."""
    if adj.shape[0] < 2:
        return 0.0
    return newman_q(adj, _greedy_partition(adj, rng=rng))


def rewire_null(adj: np.ndarray, n_draws: int = 200,
                rng: np.random.Generator | None = None) -> np.ndarray:
    """Degree-preserving rewired null for Q_str.

    This is what controls the sparsity confound (RESEARCH.md 7, C3): metabolic
    cost selects for fewer active modules, and sparser graphs score higher on Q
    for reasons that have nothing to do with modular organisation.
    """
    rng = rng or np.random.default_rng(0)
    n = adj.shape[0]
    edges = np.argwhere(adj > 0)
    if len(edges) < 2 or n < 2:
        return np.zeros(n_draws)
    out = np.empty(n_draws)
    for d in range(n_draws):
        e = edges.copy()
        for _ in range(4 * len(e)):
            i, j = rng.integers(0, len(e), size=2)
            if i == j:
                continue
            a, b = e[i].copy(), e[j].copy()
            a[1], b[1] = b[1], a[1]          # swap heads, preserving out-degree
            if a[0] == a[1] or b[0] == b[1]:
                continue
            e[i], e[j] = a, b
        m = np.zeros_like(adj)
        m[e[:, 0], e[:, 1]] = 1.0
        out[d] = q_structural(m, rng=rng)
    return out


def q_functional(activity: np.ndarray, threshold: float = 0.3,
                 rng: np.random.Generator | None = None) -> float:
    """Q_cor: modularity of the thresholded module-activity correlation matrix.

    Reported ALONGSIDE Q_str, never instead of it: structural modularity does
    not in general guarantee functional specialisation
    (Nature Communications 15, 2024).

    activity : (T, n_modules)
    """
    if activity.shape[1] < 2:
        return 0.0
    a = activity - activity.mean(axis=0, keepdims=True)
    sd = a.std(axis=0) + EPS
    c = (a.T @ a) / (activity.shape[0] * np.outer(sd, sd))
    adj = (np.abs(c) > threshold).astype(np.float64)
    np.fill_diagonal(adj, 0.0)
    return q_structural(adj, rng=rng)


# --------------------------------------------------------------------------
# specialisation, redundancy, division of labour
# --------------------------------------------------------------------------
def contribution_simplex(contrib: np.ndarray) -> np.ndarray:
    """Normalise a module's per-subgoal ablation cost to a probability vector.

    contrib : (n_modules, n_subgoals) fitness drop under single ablation.
    Negative drops (ablation helped) are clipped to zero: a module that hurts
    a subgoal is not specialised *for* it.
    """
    c = np.clip(contrib, 0.0, None)
    s = c.sum(axis=1, keepdims=True)
    return np.where(s > EPS, c / np.maximum(s, EPS), 1.0 / c.shape[1])


def specialisation(contrib: np.ndarray) -> np.ndarray:
    """S_i = 1 - H(c_i)/log K.  1 = wholly specialised, 0 = wholly general."""
    p = contribution_simplex(contrib)
    k = p.shape[1]
    if k < 2:
        return np.zeros(p.shape[0])
    h = -(p * np.log(p + EPS)).sum(axis=1)
    return np.clip(1.0 - h / np.log(k), 0.0, 1.0)


def specialisation_null(contrib: np.ndarray, n_draws: int = 1000,
                        rng: np.random.Generator | None = None) -> np.ndarray:
    """Independent within-row permutation of subgoal labels.

    IMPORTANT, and easy to get wrong: this null is **degenerate for per-module
    S_i**.  Permuting the entries of a module's contribution vector preserves
    its entropy exactly, so E[S_i under this null] == S_i.  It therefore says
    nothing about whether a single module is specialised.

    What it *is* valid for is any statistic that depends on the *alignment*
    between modules -- pairwise JS divergence, and division of labour.  Those
    are what the primary outcome measure uses, so this is the null that matters
    (see `pair_divergence_null`).

    For per-module S_i the correct comparison is against a matched control
    distribution -- the fitness-shuffled drift control, or the parameter-matched
    monolithic control -- not against a permutation.  Use
    `specialisation_vs_control`.
    """
    rng = rng or np.random.default_rng(0)
    out = np.empty((n_draws, contrib.shape[0]))
    for d in range(n_draws):
        perm = np.stack([rng.permutation(contrib.shape[1])
                         for _ in range(contrib.shape[0])])
        out[d] = specialisation(np.take_along_axis(contrib, perm, axis=1))
    return out


def specialisation_vs_control(observed: np.ndarray, control: np.ndarray) -> dict:
    """Compare observed per-module S_i against a matched control distribution.

    `control` is S_i drawn from the fitness-shuffled drift runs (or from the
    parameter-matched monolithic control).  Returns the effect size and the
    empirical exceedance probability, which together are what DESIGN.md 3.3's
    decision rule is evaluated on.
    """
    obs = np.asarray(observed).ravel()
    ctl = np.asarray(control).ravel()
    return {
        "observed_mean": float(obs.mean()) if obs.size else 0.0,
        "control_mean": float(ctl.mean()) if ctl.size else 0.0,
        "cliffs_delta": cliffs_delta(obs, ctl),
        "p_exceed": float((ctl[None, :] >= obs[:, None]).mean()) if obs.size and ctl.size else 1.0,
    }


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    m = 0.5 * (p + q)
    def kl(a, b):
        return float((a * np.log((a + EPS) / (b + EPS))).sum())
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def division_of_labour(contrib: np.ndarray) -> float:
    """Mean pairwise JS divergence between modules' contribution vectors."""
    p = contribution_simplex(contrib)
    n = p.shape[0]
    if n < 2:
        return 0.0
    tot, cnt = 0.0, 0
    for i in range(n):
        for j in range(i + 1, n):
            tot += js_divergence(p[i], p[j])
            cnt += 1
    return tot / max(cnt, 1)


def redundancy_pairs(single: np.ndarray, joint: np.ndarray,
                     single_tol: float = 0.05, joint_min: float = 0.20) -> np.ndarray:
    """Pairs (i,j) where neither alone matters but both together do.

    single : (n,) fractional fitness loss from ablating i alone
    joint  : (n, n) fractional fitness loss from ablating i and j together
    """
    n = single.shape[0]
    ok = np.zeros((n, n), bool)
    for i in range(n):
        for j in range(i + 1, n):
            if single[i] < single_tol and single[j] < single_tol and joint[i, j] > joint_min:
                ok[i, j] = ok[j, i] = True
    return ok


# --------------------------------------------------------------------------
# regulation, robustness, evolvability
# --------------------------------------------------------------------------
def mutual_information(x: np.ndarray, y: np.ndarray, bins: int = 8) -> float:
    """I(X;Y) in bits from a joint histogram."""
    h, _, _ = np.histogram2d(x, y, bins=bins)
    p = h / max(h.sum(), EPS)
    px, py = p.sum(1, keepdims=True), p.sum(0, keepdims=True)
    nz = p > 0
    return float((p[nz] * np.log2(p[nz] / (px @ py)[nz] + EPS)).sum())


def regulatory_organisation(context: np.ndarray, gates: np.ndarray,
                            rng: np.random.Generator | None = None) -> np.ndarray:
    """I(context ; gate_i) minus the circular-shift baseline, in bits.

    context : (T,)   subgoal-phase stream
    gates   : (T, n) per-module gate outputs
    """
    rng = rng or np.random.default_rng(0)
    out = np.empty(gates.shape[1])
    for i in range(gates.shape[1]):
        mi = mutual_information(context, gates[:, i])
        shifts = [mutual_information(np.roll(context, int(s)), gates[:, i])
                  for s in rng.integers(1, len(context), size=16)]
        out[i] = mi - float(np.mean(shifts))
    return out


def robustness_auc(fractions: np.ndarray, fitness: np.ndarray) -> float:
    """Area under the fitness-vs-ablation-fraction curve, normalised so that
    1.0 = no degradation and 0.0 = total collapse at the first ablation."""
    if len(fractions) < 2:
        return 0.0
    f = fitness / max(abs(fitness[0]), EPS)
    return float(np.trapezoid(f, fractions) / max(fractions[-1] - fractions[0], EPS))


def adaptation_auc(curve: np.ndarray, pre: float) -> float:
    """Normalised area under an alien-world adaptation curve.

    `pre` (pre-adaptation fitness) is subtracted, which is the covariate that
    controls regression to the mean (RESEARCH.md 7, C6).  Runs must be compared
    at a matched offspring budget, i.e. equal len(curve).
    """
    if len(curve) < 2:
        return 0.0
    gain = curve - pre
    return float(np.trapezoid(gain, np.linspace(0, 1, len(curve))))


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Non-parametric effect size.  |delta| >= 0.47 is 'large' -- the
    pre-registered threshold in DESIGN.md 3.3."""
    a, b = np.asarray(a), np.asarray(b)
    if a.size == 0 or b.size == 0:
        return 0.0
    gt = (a[:, None] > b[None, :]).sum()
    lt = (a[:, None] < b[None, :]).sum()
    return float((gt - lt) / (a.size * b.size))


def pair_divergence(contrib: np.ndarray, i: int, j: int) -> float:
    """JS divergence between two modules' contribution vectors."""
    p = contribution_simplex(contrib)
    return js_divergence(p[i], p[j])


def pair_divergence_null(contrib: np.ndarray, i: int, j: int,
                         n_draws: int = 1000,
                         rng: np.random.Generator | None = None) -> np.ndarray:
    """Null distribution for the PRIMARY OUTCOME MEASURE.

    A duplication counts as a duplication->specialisation event when the two
    surviving copies' contribution vectors diverge beyond the 95th percentile
    of this null (DESIGN.md 2).  The null permutes each module's subgoal labels
    independently, which preserves each module's contribution magnitudes and
    its entropy while destroying any alignment between the two -- exactly the
    quantity the claim is about.
    """
    rng = rng or np.random.default_rng(0)
    out = np.empty(n_draws)
    for d in range(n_draws):
        c = contrib.copy()
        c[i] = c[i][rng.permutation(c.shape[1])]
        c[j] = c[j][rng.permutation(c.shape[1])]
        out[d] = pair_divergence(c, i, j)
    return out


def duplication_specialised(contrib: np.ndarray, i: int, j: int,
                            percentile: float = 95.0,
                            n_draws: int = 1000,
                            rng: np.random.Generator | None = None) -> tuple[bool, float, float]:
    """Permutation-based test that a duplicate pair has diverged.

    Returns (is_event, observed_divergence, null_threshold).

    KNOWN LIMITATION, and the reason this is NOT the primary criterion: when
    contributions are strongly concentrated, permuting a module's subgoal
    labels produces another strongly concentrated vector, so the null itself
    piles up at maximum divergence and the test loses all power.  Concretely,
    for two perfectly one-hot contribution vectors over K channels the null
    attains the maximum with probability (K-1)/K, so the empirical p-value is
    ~0.83 at K=6 no matter how cleanly the two modules have specialised.

    That is the null behaving correctly -- with few channels you genuinely
    cannot distinguish "specialised on different subgoals" from "assigned to
    different subgoals by chance" -- but it means the permutation null answers
    the wrong question.  Use `duplication_specialised_vs_drift`.
    """
    obs = pair_divergence(contrib, i, j)
    null = pair_divergence_null(contrib, i, j, n_draws=n_draws, rng=rng)
    thr = float(np.percentile(null, percentile))
    return bool(obs > thr), float(obs), thr


def duplication_specialised_vs_drift(observed: np.ndarray, drift: np.ndarray,
                                     percentile: float = 95.0) -> dict:
    """PRIMARY OUTCOME MEASURE (DESIGN.md 2, 3.3).

    A duplicate pair counts as a duplication->specialisation event when its
    contribution-vector JS divergence, measured at a matched age since
    duplication, exceeds the `percentile`-th percentile of the divergence
    distribution of duplicate pairs in the **fitness-shuffled drift control**.

    Why this null and not a permutation: the question is not "are these two
    modules distinguishable" (two randomly initialised modules always are) but
    "have they diverged more than drift alone produces".  The drift control
    runs the identical machinery -- same duplication rate, same mutation
    operators, same environments, same lifetimes -- with fitness permuted
    within island, so it supplies exactly the neutral-divergence distribution
    this claim needs.  It also, unlike a permutation null, keeps its power when
    contributions are concentrated.

    observed : (n_pairs,) divergences from an experimental cell
    drift    : (n_pairs,) divergences from the matched drift control
    """
    observed = np.asarray(observed, dtype=float).ravel()
    drift = np.asarray(drift, dtype=float).ravel()
    if drift.size == 0:
        return {"threshold": float("nan"), "n_events": 0, "rate": 0.0,
                "cliffs_delta": 0.0, "n_pairs": int(observed.size)}
    thr = float(np.percentile(drift, percentile))
    events = observed > thr
    return {
        "threshold": thr,
        "n_events": int(events.sum()),
        "n_pairs": int(observed.size),
        "rate": float(events.mean()) if observed.size else 0.0,
        "cliffs_delta": cliffs_delta(observed, drift),
        "observed_median": float(np.median(observed)) if observed.size else 0.0,
        "drift_median": float(np.median(drift)),
    }
