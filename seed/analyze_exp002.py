"""analyze_exp002.py -- read the three follow-up arms.

Arm A is the one that decides things. Every unit in the sweep got exactly the
same number of steps to integrate, so any trend against BIRTH STEP is a pure
earliness effect with the time confound removed.
"""
from __future__ import annotations
import glob, json, math, os
from collections import defaultdict

import metrics as M
from world import DIST_NAMES, World, WorldSpec
import ledger as L


def unit_effect(run: dict, uid: str) -> float:
    per = run.get("final_ablation_by_relation", {}).get(uid, {})
    return sum(v for v in per.values() if v > 0)


def holdout_primary(d: str) -> float:
    cfg = json.load(open(os.path.join(d, "run.json")))["config"]
    w = World(WorldSpec(seed=cfg["world_seed"], n_tranches=cfg["n_ages"]))
    recs = L.read(os.path.join(d, "ledger.jsonl"))
    age = max(r["age"] for r in recs if r.get("kind") == "prediction")
    rows = [r for r in recs if r.get("kind") == "prediction" and r["age"] == age]
    return M.summarize(M.score_records(w, rows, age), "holdout")


def pearson(xs, ys):
    n = len(xs)
    if n < 3: return float("nan")
    mx, my = sum(xs)/n, sum(ys)/n
    num = sum((a-mx)*(b-my) for a, b in zip(xs, ys))
    den = (sum((a-mx)**2 for a in xs) * sum((b-my)**2 for b in ys)) ** .5
    return num/den if den else float("nan")


def arm_a(root):
    by_birth = defaultdict(list)
    for d in sorted(glob.glob(f"{root}/sweep_b*_s*/")):
        r = json.load(open(os.path.join(d, "run.json")))
        if not r["events"]: continue
        e = r["events"][0]
        uid = e["survivors"][0]
        s = holdout_primary(d)
        by_birth[e["step"]].append({
            "effect": unit_effect(r, uid),
            "primary": s["PRIMARY_extrapolative_accuracy"],
            "d3": s["per_bucket"].get(DIST_NAMES[3], {}).get("accuracy", float("nan")),
            "steps": r["compute"]["steps"], "birth": e["step"],
            "dlogit": e["logit_delta"],
        })
    if not by_birth: return
    print("\n=== ARM A: growth-time sweep "
          "(every unit gets the SAME 2400 steps to integrate) ===\n")
    print(f"{'birth step':>11}{'n':>4}{'unit ablation effect':>22}{'primary':>10}{'D3':>8}{'max dlogit':>12}")
    xs, ys = [], []
    for b in sorted(by_birth):
        v = by_birth[b]; n = len(v)
        eff = sum(x["effect"] for x in v)/n
        print(f"{b:>11}{n:>4}{eff:>22.4f}"
              f"{sum(x['primary'] for x in v)/n:>10.4f}"
              f"{sum(x['d3'] for x in v)/n:>8.3f}"
              f"{max(x['dlogit'] for x in v):>12.1e}")
        for x in v:
            xs.append(x["birth"]); ys.append(x["effect"])
    r = pearson(xs, ys)
    print(f"\n  pearson(birth step, unit ablation effect) = {r:+.3f}   n={len(xs)}")
    print("  time-to-integrate is held constant, so this is EARLINESS alone.")
    print(f"  {'-> earliness matters: later-born units integrate worse'  if r < -0.3 else ''}"
          f"{'-> earliness does not matter: time was the confound' if abs(r) <= 0.3 else ''}"
          f"{'-> later is BETTER (unexpected)' if r > 0.3 else ''}")


def arm_b(root):
    rows = defaultdict(list)
    gens = defaultdict(list)
    for d in sorted(glob.glob(f"{root}/long_*_s*/")):
        r = json.load(open(os.path.join(d, "run.json")))
        g = r["config"]["group"]
        s = holdout_primary(d)
        rows[g].append({"primary": s["PRIMARY_extrapolative_accuracy"],
                        "d3": s["per_bucket"].get(DIST_NAMES[3], {}).get("accuracy", float("nan")),
                        "events": len(r["events"]),
                        "params": r["final_arch"]["params_total"],
                        "steps": r["compute"]["steps"]})
        for i, e in enumerate(r["events"], 1):
            for uid in e["survivors"]:
                gens[i].append(unit_effect(r, uid))
    if not rows: return
    print("\n=== ARM B: long-horizon recursion (2x budget, up to 10 cycles) ===\n")
    print(f"{'group':>7}{'n':>4}{'primary':>10}{'D3':>8}{'events':>8}{'params':>9}{'steps':>8}")
    for g in sorted(rows):
        v = rows[g]; n = len(v)
        print(f"{g:>7}{n:>4}{sum(x['primary'] for x in v)/n:>10.4f}"
              f"{sum(x['d3'] for x in v)/n:>8.3f}{sum(x['events'] for x in v)/n:>8.1f}"
              f"{sum(x['params'] for x in v)/n/1e3:>8.0f}k{sum(x['steps'] for x in v)/n:>8.0f}")
    if gens:
        print(f"\n  does each generation carry more than the last?")
        print(f"  {'generation':>11}{'n':>4}{'mean ablation effect':>22}")
        for i in sorted(gens):
            print(f"  {i:>11}{len(gens[i]):>4}{sum(gens[i])/len(gens[i]):>22.4f}")


def arm_c(root):
    got = []
    for d in sorted(glob.glob(f"{root}/staged_s*/")):
        r = json.load(open(os.path.join(d, "run.json")))
        s = holdout_primary(d)
        got.append({"primary": s["PRIMARY_extrapolative_accuracy"],
                    "d3": s["per_bucket"].get(DIST_NAMES[3], {}).get("accuracy", float("nan")),
                    "params": r["final_arch"]["params_total"],
                    "steps": r["compute"]["steps"], "flops": r["compute"]["flops"],
                    "events": len(r["events"])})
    if not got: return
    n = len(got)
    print("\n=== ARM C: staged expression of a pre-allocated architecture ===\n")
    print(f"  S STAGED  n={n}  primary={sum(x['primary'] for x in got)/n:.4f}"
          f"  D3={sum(x['d3'] for x in got)/n:.3f}"
          f"  params={sum(x['params'] for x in got)/n/1e3:.0f}k"
          f"  steps={sum(x['steps'] for x in got)/n:.0f}"
          f"  flops={sum(x['flops'] for x in got)/n:.3e}"
          f"  units created mid-run={sum(x['events'] for x in got)}")
    try:
        e1 = json.load(open("results/exp001/summary.json"))["aggregate"]["groups"]
        print("\n  for comparison, Experiment 001 at the same FLOPs budget:")
        for g in ("A", "B", "C", "D"):
            if g in e1:
                print(f"    {g:<14} primary={e1[g]['primary_mean']:.4f}"
                      f"  D3={e1[g]['buckets'][DIST_NAMES[3]]:.3f}"
                      f"  params={e1[g]['params_final']/1e3:.0f}k"
                      f"  steps={e1[g]['steps']:.0f}")
    except Exception:
        pass


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default="results/exp002")
    a = ap.parse_args()
    arm_a(a.root); arm_b(a.root); arm_c(a.root)
