"""Pre-registered analysis for the composition test (see PREREG.md)."""
import glob, json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
LENS = (1, 2, 3, 4, 6, 8, 12, 16)


def load(part="A"):
    files = [f for f in glob.glob(os.path.join(OUT, "*.json")) if "analysis" not in f]
    files = [f for f in files if ("_st30000" in f) == (part == "B")]
    rows = [json.load(open(f)) for f in files]
    best = {}
    for sysname in ("modular", "looped", "transformer"):
        R = [r for r in rows if r["system"] == sysname]
        lrs = sorted({r["lr"] for r in R})
        lr = max(lrs, key=lambda l: np.mean([r["val"] for r in R if r["lr"] == l]))
        best[sysname] = {r["seed"]: r for r in R if r["lr"] == lr}
    best["evolved"] = {r["seed"]: r for r in rows if r["system"] == "evolved"}
    return best


def paired(a, b, key):
    s = sorted(set(a) & set(b))
    d = np.array([a[i][key] - b[i][key] for i in s])
    return float(d.mean()), float(d.std(ddof=1)) if len(d) > 1 else 0.0, len(s)


def main(part="A"):
    best = load(part)
    table = {k: {L: float(np.mean([r[f"hard_{L}"] for r in v.values()])) for L in LENS} for k, v in best.items() if v}
    res = dict(lr={k: v[next(iter(v))].get("lr") for k, v in best.items() if v}, hard_acc=table)
    m, l, t = best["modular"], best["looped"], best["transformer"]
    dl, sl, _ = paired(m, l, "hard_12"); dt, st, _ = paired(m, t, "hard_12")
    res["H1"] = dict(mod_minus_looped=dl, sd_l=sl, mod_minus_transformer=dt, sd_t=st,
                     verdict="PASS" if (dl >= 0.15 and dt >= 0.15 and dl > 2 * sl and dt > 2 * st) else "FAIL",
                     no_advantage_over_looped=dl < 0.05)
    res["H2"] = dict(modular_len16=table["modular"][16], verdict="PASS" if table["modular"][16] >= 0.80 else "FAIL")
    if best["evolved"]:
        de, se, n = paired(best["evolved"], m, "hard_12")
        res["H3"] = dict(evolved_minus_modular=de, sd=se, n=n, verdict="PASS" if (de > 0 and de > 2 * se) else "FAIL")
    conv = {k: [s for s, r in v.items() if r["val"] >= 0.95] for k, v in best.items() if v}
    res["converged_seeds"] = {k: f"{len(c)}/{len(best[k])}" for k, c in conv.items()}
    cm = [s for s in conv.get("modular", []) if s in conv.get("looped", [])]
    if cm:
        res["H1_converged_only_descriptive"] = dict(
            seeds=cm, modular_12=float(np.mean([m[s]["hard_12"] for s in cm])),
            looped_12=float(np.mean([l[s]["hard_12"] for s in cm])))
    res["per_seed_12"] = {k: {s: round(r["hard_12"], 3) for s, r in v.items()} for k, v in best.items() if v}
    json.dump(res, open(os.path.join(OUT, f"analysis{'' if part == 'A' else '_B'}.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "A")
