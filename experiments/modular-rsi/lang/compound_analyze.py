"""Pre-registered analysis for compound.py (see compound_prereg.md). Committed before
the main runs."""
import glob, json, os
import numpy as np
from scipy_free import wilcoxon_signed_rank_one_sided

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "compound")
CAP = 12_000


def load(arm):
    return {json.load(open(f))["order"]: json.load(open(f)) for f in glob.glob(os.path.join(OUT, f"{arm}_o*.json"))}


def costs(run):
    return [(x["k"] + 1, x["domain"], x["cost"] if x["cost"] is not None else CAP, x["cost"] is None) for x in run["log"]]


def slope_fe(rows):
    """OLS log(cost) ~ k + domain fixed effects; rows = (order, k, domain, cost)."""
    doms = sorted({r[2] for r in rows})
    X = np.array([[r[1]] + [1.0 if r[2] == d else 0.0 for d in doms] for r in rows])
    y = np.log([r[3] for r in rows])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    return beta[0]


def h_compound(runs, kmin=3, B=2000, seed=0):
    rows = [(o, k, d, c) for o, r in runs.items() for (k, d, c, _) in costs(r) if k >= kmin]
    est = slope_fe(rows)
    orders = sorted(runs)
    rng = np.random.default_rng(seed); bs = []
    for _ in range(B):
        pick = rng.choice(orders, len(orders))
        bs.append(slope_fe([r for o in pick for r in rows if r[0] == o]))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    cens = np.mean([c for r in runs.values() for (k, d, _, c) in costs(r) if k >= kmin])
    floor = np.mean([cc <= 100 for r in runs.values() for (k, d, cc, c) in costs(r) if k >= kmin])
    verdict = ("VOID (floor)" if floor > 0.5 else "INCONCLUSIVE (ceiling)" if cens > 0.5
               else "PASS" if hi < 0 else "FAIL")
    return dict(slope=float(est), ci=[float(lo), float(hi)], censored_frac=float(cens), floor_frac=float(floor),
                n_orders=len(runs), verdict=verdict)


def h_rsi(full, probe):
    common = sorted(set(full) & set(probe))
    tot = lambda r: np.log(sum(c for (k, d, c, _) in costs(r) if k >= 2))
    diff = np.array([tot(full[o]) - tot(probe[o]) for o in common])
    p = wilcoxon_signed_rank_one_sided(diff)        # H1: full < probe
    return dict(n=len(common), mean_log_ratio=float(diff.mean()), p_one_sided=float(p),
                verdict="PASS" if (p < 0.05 and diff.mean() < 0) else "FAIL")


def h_modular(full, mono):
    common = sorted(set(full) & set(mono))
    f = [np.mean([x["test_bpc"] for x in full[o]["log"]]) for o in common]
    m = [np.mean(list(mono[o]["final_test"].values())) for o in common]
    return dict(n=len(common), full_mean_test=float(np.mean(f)), monolith_mean_test=float(np.mean(m)),
                verdict="PASS" if np.mean(f) <= np.mean(m) + 0.05 else "FAIL")


def main():
    arms = {a: load(a) for a in ("full", "nocredit", "probe", "latest", "fresh", "monolith")}
    res = {"H_compound": {a: h_compound(arms[a]) for a in ("full", "probe", "nocredit", "latest", "fresh") if arms[a]}}
    if arms["full"] and arms["probe"]:
        res["H_rsi"] = h_rsi(arms["full"], arms["probe"])
    if arms["full"] and arms["monolith"]:
        res["H_modular"] = h_modular(arms["full"], arms["monolith"])
    res["mean_cost_by_k"] = {a: [float(np.mean([costs(r)[k][2] for r in arms[a].values()])) for k in range(8)]
                             for a in arms if arms[a]}
    json.dump(res, open(os.path.join(OUT, "analysis.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
