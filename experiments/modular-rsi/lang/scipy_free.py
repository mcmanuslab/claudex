"""Exact one-sided Wilcoxon signed-rank test (H1: median < 0) without scipy."""
import itertools
import numpy as np


def wilcoxon_signed_rank_one_sided(d):
    d = np.asarray([x for x in d if x != 0], float)
    n = len(d)
    if n == 0:
        return 1.0
    ranks = np.argsort(np.argsort(np.abs(d))) + 1.0
    # average ties
    absd = np.abs(d)
    for v in np.unique(absd):
        m = absd == v
        ranks[m] = ranks[m].mean()
    w_pos = ranks[d > 0].sum()
    # exact null distribution of W+ over all sign assignments (n <= 20 is fine)
    sums = [sum(r for r, s in zip(ranks, signs) if s) for signs in itertools.product([0, 1], repeat=n)]
    return float(np.mean(np.array(sums) <= w_pos + 1e-9))
