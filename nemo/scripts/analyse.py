#!/usr/bin/env python3
"""Analyse a NEMO run and evaluate the pre-registered decision rule.

    python3 scripts/analyse.py results/pilot

Everything here compares against a null.  A metric reported without its null is
not a result -- a sparser graph scores higher on Newman modularity for purely
structural reasons, two randomly initialised modules always differ, and genome
length random-walks upward under near-neutral duplication.  The
fitness-shuffled drift control supplies the null for the dynamics; the
degree-preserving rewired graph supplies it for structure.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from nemo.metrics.definitions import (cliffs_delta,  # noqa: E402
                                      duplication_specialised_vs_drift,
                                      q_structural, rewire_null)


def load(out: Path):
    con = sqlite3.connect(str(out / "nemo.sqlite"))
    cur = con.execute(
        "SELECT generation, run, run_name, mean_reward, mean_genome_len, "
        "mean_active, mean_flops, n_duplications, n_deletions, mean_q_str "
        "FROM generations ORDER BY generation")
    rows = cur.fetchall()
    con.close()
    cols = ["generation", "run", "run_name", "mean_reward", "mean_genome_len",
            "mean_active", "mean_flops", "n_duplications", "n_deletions", "mean_q_str"]
    return {c: np.array([r[i] for r in rows],
                        dtype=object if c == "run_name" else float)
            for i, c in enumerate(cols)}


def condition_of(name: str) -> str:
    return str(name).split("_r")[0]


def tail_mean(d, cond: str, field: str, frac: float = 0.25) -> np.ndarray:
    """Per-replicate mean of `field` over the final `frac` of generations."""
    out = []
    names = {str(n) for n in d["run_name"]}
    for nm in sorted(n for n in names if condition_of(n) == cond):
        m = np.array([str(x) == nm for x in d["run_name"]])
        g = d["generation"][m]
        v = d[field][m]
        if len(g) == 0:
            continue
        cut = g.max() * (1 - frac)
        sel = v[g >= cut]
        sel = sel[np.isfinite(sel)]
        if len(sel):
            out.append(float(sel.mean()))
    return np.array(out)


def mannwhitney_u_p(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sided Mann-Whitney U via the normal approximation.

    Written out rather than pulled from scipy so the analysis has no dependency
    beyond NumPy.  With n=12 per cell the normal approximation is adequate; the
    effect size (Cliff's delta) is the primary number regardless.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    n1, n2 = len(a), len(b)
    if n1 == 0 or n2 == 0:
        return 1.0
    allv = np.concatenate([a, b])
    order = allv.argsort()
    ranks = np.empty(len(allv), float)
    ranks[order] = np.arange(1, len(allv) + 1)
    # average ties
    for v in np.unique(allv):
        idx = np.flatnonzero(allv == v)
        if len(idx) > 1:
            ranks[idx] = ranks[idx].mean()
    u1 = ranks[:n1].sum() - n1 * (n1 + 1) / 2
    mu = n1 * n2 / 2
    sd = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    if sd == 0:
        return 1.0
    z = (u1 - mu) / sd
    # two-sided normal tail
    return float(2 * 0.5 * (1 - _erf(abs(z) / np.sqrt(2))))


def _erf(x: float) -> float:
    import math
    return math.erf(x)


def holm_bonferroni(pvals: dict[str, float], alpha: float = 0.01) -> dict[str, bool]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, rejected_so_far = {}, True
    for i, (k, p) in enumerate(items):
        thresh = alpha / (m - i)
        rejected_so_far = rejected_so_far and (p <= thresh)
        out[k] = rejected_so_far
    return out


def structural_modularity_vs_null(npz: Path, n_draws: int = 60,
                                  max_lanes: int = 120) -> dict:
    """Q_str of evolved organisms against their own degree-preserving rewiring."""
    if not npz.exists():
        return {}
    z = np.load(npz)
    alive, src, mask, run_id = z["alive"], z["src"], z["src_mask"], z["run_id"]
    rng = np.random.default_rng(0)
    per_run: dict[int, list[tuple[float, float]]] = {}
    lanes = rng.permutation(alive.shape[0])[:max_lanes]
    n_sensor = 2
    for lane in lanes:
        live = np.flatnonzero(alive[lane] > 0)
        if len(live) < 3:
            continue
        index = {int(1 + n_sensor + g): i for i, g in enumerate(live)}
        adj = np.zeros((len(live), len(live)))
        for i, g in enumerate(live):
            for s, m in zip(src[lane, g], mask[lane, g]):
                if m > 0 and int(s) in index and index[int(s)] != i:
                    adj[index[int(s)], i] = 1.0
        if adj.sum() < 2:
            continue
        q = q_structural(adj, rng)
        null = rewire_null(adj, n_draws=n_draws, rng=rng)
        per_run.setdefault(int(run_id[lane]), []).append((q, float(null.mean())))
    return {str(k): {"q_mean": float(np.mean([a for a, _ in v])),
                     "null_mean": float(np.mean([b for _, b in v])),
                     "n": len(v)}
            for k, v in per_run.items()}


def primary_outcome(out: Path, names: list[str], min_age: int = 20) -> dict:
    """Duplication -> specialisation event rate, per condition.

    A duplicate pair counts as an event when its contribution divergence
    exceeds the 95th percentile of the drift control's divergences at
    comparable age.  Pairs younger than `min_age` generations are excluded:
    a fresh duplicate is byte-identical by construction, so including it would
    dilute the rate with events that had no opportunity to diverge.
    """
    con = sqlite3.connect(str(out / "nemo.sqlite"))
    try:
        rows = con.execute(
            "SELECT run, age, divergence, weight_distance FROM pair_observations "
            "WHERE both_alive = 1 AND age >= ?", (min_age,)).fetchall()
    except sqlite3.OperationalError:
        return {}
    finally:
        con.close()
    if not rows:
        return {}

    by: dict[str, list[tuple[float, float]]] = {}
    for run, _age, div, wd in rows:
        by.setdefault(condition_of(names[int(run)]), []).append((float(div), float(wd)))
    if "DRIFT" not in by:
        return {"error": "no DRIFT pairs observed -- no null, so no claim"}

    drift = np.array([d for d, _ in by["DRIFT"]])
    res = {}
    for c, vals in sorted(by.items()):
        obs = np.array([d for d, _ in vals])
        wds = np.array([w for _, w in vals])
        r = duplication_specialised_vs_drift(obs, drift)
        r["median_weight_distance"] = float(np.median(wds))
        r["frac_identical_weights"] = float((wds < 1e-6).mean())
        res[c] = r
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--alpha", type=float, default=0.01)
    args = ap.parse_args()

    d = load(args.out)
    conds = sorted({condition_of(n) for n in d["run_name"]})
    print("=" * 76)
    print(f"NEMO analysis: {args.out}")
    print("=" * 76)

    fields = [("mean_reward", "reward"), ("mean_genome_len", "genome len"),
              ("mean_active", "active mods"), ("mean_flops", "metab. flops"),
              ("mean_q_str", "Q_str")]

    print(f"\n{'condition':<10} " + " ".join(f"{lbl:>13}" for _, lbl in fields) + f"{'n':>4}")
    data: dict[str, dict[str, np.ndarray]] = {}
    for c in conds:
        data[c] = {f: tail_mean(d, c, f) for f, _ in fields}
        row = " ".join(f"{np.mean(data[c][f]):>13.4f}" if len(data[c][f]) else f"{'--':>13}"
                       for f, _ in fields)
        print(f"{c:<10} {row}{len(data[c]['mean_reward']):>4}")

    if "DRIFT" not in data:
        print("\nNo DRIFT control present -- no null, so no claim can be made.")
        return

    print("\n" + "=" * 76)
    print("PRE-REGISTERED CONTRASTS vs the fitness-shuffled drift control")
    print("DESIGN.md 3.3: Cliff's delta >= 0.47 (large) AND Holm-adjusted p < %.2f"
          % args.alpha)
    print("=" * 76)

    results, pvals = {}, {}
    for c in conds:
        if c == "DRIFT":
            continue
        for f, lbl in fields:
            a, b = data[c][f], data["DRIFT"][f]
            if len(a) < 2 or len(b) < 2:
                continue
            key = f"{c}:{lbl}"
            delta = cliffs_delta(a, b)
            p = mannwhitney_u_p(a, b)
            results[key] = {"cliffs_delta": delta, "p": p,
                            "cell_mean": float(a.mean()), "drift_mean": float(b.mean()),
                            "n": len(a)}
            pvals[key] = p

    rejected = holm_bonferroni(pvals, args.alpha)
    print(f"\n{'contrast':<28} {'cell':>10} {'drift':>10} {'delta':>8} {'p':>9}  verdict")
    for key, r in results.items():
        large = abs(r["cliffs_delta"]) >= 0.47
        sig = rejected.get(key, False)
        verdict = "SUPPORTED" if (large and sig) else (
            "large, n.s." if large else ("sig., small" if sig else "null"))
        print(f"{key:<28} {r['cell_mean']:>10.4f} {r['drift_mean']:>10.4f} "
              f"{r['cliffs_delta']:>8.3f} {r['p']:>9.4f}  {verdict}")

    print("\n" + "=" * 76)
    print("PRIMARY OUTCOME: duplication -> specialisation event rate")
    print("(divergence above the 95th percentile of the drift control's pairs)")
    print("=" * 76)
    names_all = json.loads((args.out / "meta.json").read_text())["runs"]
    po = primary_outcome(args.out, names_all)
    if not po:
        print("\n  No duplicate pairs survived to the minimum age -- nothing to test.")
    elif "error" in po:
        print(f"\n  {po['error']}")
    else:
        print(f"\n{'condition':<10} {'pairs':>6} {'events':>7} {'rate':>7} "
              f"{'delta':>8} {'med div':>9} {'med wdist':>10} {'identical':>10}")
        for c, r in po.items():
            print(f"{c:<10} {r['n_pairs']:>6} {r['n_events']:>7} {r['rate']:>7.3f} "
                  f"{r['cliffs_delta']:>8.3f} {r['observed_median']:>9.4f} "
                  f"{r['median_weight_distance']:>10.4f} "
                  f"{r['frac_identical_weights']:>9.1%}")
        print("\n  DRIFT's own rate is 0.05 by construction -- that is the")
        print("  calibration, not a result.  'identical' is the fraction of pairs")
        print("  whose WEIGHTS are byte-identical: any divergence those show is")
        print("  pure estimation noise, and is why the drift null is required.")

    print("\n" + "=" * 76)
    print("STRUCTURAL MODULARITY vs degree-preserving rewired null")
    print("(controls the sparsity confound: sparser graphs score higher on Q)")
    print("=" * 76)
    sm = structural_modularity_vs_null(args.out / "final_population.npz")
    if sm:
        names = json.loads((args.out / "meta.json").read_text())["runs"]
        print(f"\n{'condition':<10} {'Q_str':>9} {'rewired null':>13} {'excess':>9} {'n':>5}")
        agg: dict[str, list] = {}
        for run_idx, v in sm.items():
            agg.setdefault(condition_of(names[int(run_idx)]), []).append(v)
        for c, vs in sorted(agg.items()):
            q = np.mean([x["q_mean"] for x in vs])
            nl = np.mean([x["null_mean"] for x in vs])
            print(f"{c:<10} {q:>9.4f} {nl:>13.4f} {q-nl:>9.4f} {sum(x['n'] for x in vs):>5}")
        print("\nExcess <= 0 means the evolved graphs are no more modular than")
        print("degree-matched random graphs, whatever the raw Q_str looks like.")

    (args.out / "analysis.json").write_text(json.dumps(
        {"contrasts": results, "holm_rejected": rejected,
         "primary_outcome": po, "structural": sm}, indent=2))
    print(f"\nwritten -> {args.out/'analysis.json'}")


if __name__ == "__main__":
    main()
