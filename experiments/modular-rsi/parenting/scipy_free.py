"""One-sided Wilcoxon signed-rank test (H1: median < 0) without scipy.
Exact null distribution for n <= 20; normal approximation (with tie correction) above."""
import itertools, math
import numpy as np


def wilcoxon_signed_rank_one_sided(d):
    d = np.asarray([x for x in d if x != 0], float)
    n = len(d)
    if n == 0:
        return 1.0
    absd = np.abs(d)
    ranks = np.argsort(np.argsort(absd)) + 1.0
    for v in np.unique(absd):
        m = absd == v
        ranks[m] = ranks[m].mean()
    w_pos = ranks[d > 0].sum()
    if n <= 20:
        sums = [sum(r for r, s in zip(ranks, signs) if s) for signs in itertools.product([0, 1], repeat=n)]
        return float(np.mean(np.array(sums) <= w_pos + 1e-9))
    mean = n * (n + 1) / 4
    _, counts = np.unique(absd, return_counts=True)
    var = n * (n + 1) * (2 * n + 1) / 24 - (counts ** 3 - counts).sum() / 48
    z = (w_pos - mean + 0.5) / math.sqrt(var)            # continuity correction
    return float(0.5 * (1 + math.erf(z / math.sqrt(2))))
