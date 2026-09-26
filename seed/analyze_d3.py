"""
analyze_d3.py -- why is D3 BELOW chance?

The distant bucket is HEAVIER(e1, e2) where exactly one argument sits at the
withheld period. Mass increases with period, so a period-7 entity is heavier
than almost everything. If a model had simply failed to learn anything about
these entities we would expect ~chance. Getting 0.21 means it is reliably
predicting the WRONG direction, which is a much more specific claim about what
it learned, and worth pinning down.

Hypothesis: the model never learns `mass = f(period, group)` as a composable
rule. It learns a per-entity mass, which for entities appearing in no HEAVIER
fact is simply untrained -- and an untrained mass reads as LIGHT. So the model
answers "the unfamiliar one is lighter", which is exactly backwards.

Prediction if true: accuracy should be near 0 when the distant entity is
genuinely the heavier one, and near 1 when it is not -- split by argument
position and by whether the distant entity really is heavier.
"""
from __future__ import annotations

import argparse, json, os
from collections import defaultdict

import ledger as L
from world import DISTANT_PERIOD, World, WorldSpec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="results/exp001/A_s0")
    a = ap.parse_args()
    cfg = json.load(open(os.path.join(a.run, "run.json")))["config"]
    w = World(WorldSpec(seed=cfg["world_seed"], n_tranches=cfg["n_ages"]))
    recs = L.read(os.path.join(a.run, "ledger.jsonl"))
    final_age = max(r["age"] for r in recs if r.get("kind") == "prediction")
    truth = {f.fid: f for f in w.facts}

    groups = defaultdict(lambda: [0, 0])
    conf = defaultdict(list)
    for r in recs:
        if r.get("kind") != "prediction" or r["age"] != final_age or r["dist"] != 3:
            continue
        f = truth[r["fid"]]
        d1 = w.period[f.a1] == DISTANT_PERIOD
        d2 = f.a2 >= 0 and w.period[f.a2] == DISTANT_PERIOD
        if d1 == d2:
            continue
        distant_is_heavier = (w.mass[f.a1] > w.mass[f.a2]) == d1
        key = ("distant=arg1" if d1 else "distant=arg2",
               "distant truly heavier" if distant_is_heavier else "distant truly lighter")
        ok = int(r["pred_token"] == w.vocab.ans(f.answer))
        groups[key][0] += ok
        groups[key][1] += 1
        conf[key].append(r["conf"])

    print(f"run={a.run}  age={final_age}\n")
    print(f"{'case':<48}{'n':>6}{'acc':>8}{'mean conf':>11}")
    print("-" * 73)
    for k in sorted(groups):
        c, n = groups[k]
        print(f"{k[0] + ', ' + k[1]:<48}{n:>6}{c/n:>8.3f}"
              f"{sum(conf[k])/len(conf[k]):>11.3f}")

    heavier = sum(groups[k][0] for k in groups if "truly heavier" in k[1])
    heavier_n = sum(groups[k][1] for k in groups if "truly heavier" in k[1])
    lighter = sum(groups[k][0] for k in groups if "truly lighter" in k[1])
    lighter_n = sum(groups[k][1] for k in groups if "truly lighter" in k[1])
    print("-" * 73)
    print(f"{'distant entity is TRULY HEAVIER':<48}{heavier_n:>6}{heavier/max(1,heavier_n):>8.3f}")
    print(f"{'distant entity is TRULY LIGHTER':<48}{lighter_n:>6}{lighter/max(1,lighter_n):>8.3f}")
    print(f"\nIf the model treats ungrounded entities as LIGHT, the first row "
          f"collapses\ntoward 0 and the second rises toward 1. "
          f"Observed gap: {lighter/max(1,lighter_n) - heavier/max(1,heavier_n):+.3f}")


if __name__ == "__main__":
    main()
