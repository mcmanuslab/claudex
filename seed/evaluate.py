"""
evaluate.py -- read ledgers, join ground truth, produce the comparison tables.

Nothing in here writes to a ledger and nothing in train.py reads ground truth.
That separation is the guarantee behind every number in RESULTS_001.md.
"""

from __future__ import annotations

import argparse, glob, json, os
from collections import defaultdict

import ledger as L
import metrics as M
from world import World, WorldSpec, DIST_NAMES


def load_run(run_dir: str) -> dict:
    cfg = json.load(open(os.path.join(run_dir, "run.json")))
    recs = L.read(os.path.join(run_dir, "ledger.jsonl"))
    ok, msg = L.verify(os.path.join(run_dir, "ledger.jsonl"))
    return {"run": cfg, "recs": recs, "ledger_ok": ok, "ledger_msg": msg,
            "dir": run_dir}


def evaluate_run(run_dir: str) -> dict:
    r = load_run(run_dir)
    cfg = r["run"]["config"]
    world = World(WorldSpec(seed=cfg["world_seed"], n_tranches=cfg["n_ages"]))
    ages = sorted({x["age"] for x in r["recs"] if x.get("kind") == "prediction"})
    final_age = max(ages) if ages else 0

    # checkpoint metadata (compute at each age)
    ck = {x["age"]: x for x in r["recs"] if x.get("kind") == "checkpoint"}

    per_age = {}
    for a in ages:
        rows = [x for x in r["recs"] if x.get("kind") == "prediction" and x["age"] == a]
        scored = M.score_records(world, rows, age_for_prior=a)
        per_age[a] = {
            "holdout": M.summarize(scored, "holdout"),
            # predictions about pool facts not yet revealed at age a: these get
            # scored when the next tranche lands. This is the developmental
            # "is it becoming a better guesser" curve.
            "unrevealed_pool": M.summarize(scored, "pool"),
            "by_relation": M.by_relation(scored, "holdout"),
            "flops": ck.get(a, {}).get("flops"),
            "tokens": ck.get(a, {}).get("tokens"),
            "params": ck.get(a, {}).get("params"),
            "params_active": ck.get(a, {}).get("params_active"),
            "n_blocks": ck.get(a, {}).get("n_blocks"),
            "step": ck.get(a, {}).get("step"),
        }
    return {
        "dir": run_dir, "group": cfg["group"], "seed": cfg["seed"],
        "world_seed": cfg["world_seed"],
        "ledger_ok": r["ledger_ok"], "ledger_msg": r["ledger_msg"],
        "per_age": per_age, "final_age": final_age,
        "final": per_age.get(final_age, {}),
        "compute": r["run"]["compute"],
        "forgetting": forgetting_analysis(r["run"]),
        "specialization": specialization_analysis(r["run"]),
        "final_arch": r["run"]["final_arch"],
        "events": r["run"]["events"],
        "n_events": len(r["run"]["events"]),
    }


def forgetting_analysis(run: dict, horizon_steps: int = 500,
                        min_val_facts: int = 30) -> dict:
    """Catastrophic forgetting, measured against the quantity the frozen
    protocol names: ACCURACY on already-mastered categories.

    Two guards matter. First, a category that was never learned cannot be
    forgotten, so we restrict to categories already above 90% at the moment of
    growth. Second, the small relations have only a handful of validation facts
    (SOLUBLE has ~2), where a single flipped example moves the number by 50
    points; those are excluded by `min_val_facts`. Without that guard the
    measure is dominated by noise on categories nobody was measuring.
    """
    trace, events = run["trace"], run["events"]
    empty = {"n_events": 0, "max_mastered_accuracy_drop": 0.0,
             "max_logit_delta_at_growth": 0.0, "per_event": [],
             "measurable": False}
    if not events:
        return empty
    out = []
    for ev in events:
        s0 = ev["step"]
        at = min((t for t in trace if t["step"] >= s0), key=lambda t: t["step"], default=None)
        after = min((t for t in trace if t["step"] >= s0 + horizon_steps),
                    key=lambda t: t["step"], default=None)
        if not at or not after or not at.get("per_cat_acc"):
            continue
        ns = at.get("per_cat_n", {})
        mastered = [c for c, v in at["per_cat_acc"].items()
                    if v >= 0.90 and ns.get(c, 0) >= min_val_facts]
        drops = {c: at["per_cat_acc"][c] - after["per_cat_acc"].get(c, at["per_cat_acc"][c])
                 for c in mastered}
        out.append({"step": s0, "kind": ev["kind"],
                    "logit_delta_at_growth": ev["logit_delta"],
                    "mastered_categories": mastered,
                    "accuracy_drop_after_%d_steps" % horizon_steps:
                        {k: round(v, 4) for k, v in drops.items()},
                    "worst_drop": round(max(drops.values()), 4) if drops else 0.0})
    if not out:
        return empty
    return {
        "n_events": len(out),
        "max_mastered_accuracy_drop": max(e["worst_drop"] for e in out),
        "max_logit_delta_at_growth": max(e["logit_delta_at_growth"] for e in out),
        "measurable": any(e["mastered_categories"] for e in out),
        "per_event": out,
    }


def specialization_analysis(run: dict) -> dict:
    """Did surviving units take on specific jobs, or just add generic capacity?

    For each unit we take its per-relation ablation cost and compute a
    normalised concentration: the share of its total ablation cost carried by
    its single largest relation. 1/n_relations means perfectly generic.
    """
    abl = run.get("final_ablation_by_relation") or {}
    survivors = {u for e in run["events"] for u in e["survivors"]}
    out = {}
    for uid, per_rel in abl.items():
        pos = {r: v for r, v in per_rel.items() if v > 0}
        tot = sum(pos.values())
        if tot <= 0:
            out[uid] = {"grown": uid in survivors, "concentration": None,
                        "top_relation": None, "total_effect": round(sum(per_rel.values()), 5)}
            continue
        top = max(pos, key=pos.get)
        out[uid] = {
            "grown": uid in survivors,
            "concentration": round(pos[top] / tot, 3),
            "top_relation": top,
            "total_effect": round(tot, 5),
            "per_relation": {k: v for k, v in sorted(per_rel.items(), key=lambda kv: -kv[1])[:4]},
        }
    n_rel = max(1, len(next(iter(abl.values()), {})))

    # The decisive number. High "concentration" on a negligible effect is not
    # specialisation, it is concentration of noise -- so we report how much
    # functional weight the grown units carry RELATIVE to the units that were
    # there from the start. If this ratio is ~0, growth added capacity that
    # never became load-bearing, and no selection rule applied to it could
    # matter.
    grown_eff = [v["total_effect"] for k, v in out.items()
                 if v["grown"] and v["total_effect"] and v["total_effect"] > 0]
    orig_eff = [v["total_effect"] for k, v in out.items()
                if not v["grown"] and v["total_effect"] and v["total_effect"] > 0
                and not any(k.startswith(p) for p in ("G", "B"))]
    return {
        "uniform_baseline_concentration": round(1.0 / n_rel, 3),
        "grown_total_effect": round(sum(grown_eff), 6),
        "original_total_effect": round(sum(orig_eff), 6),
        "grown_share_of_effect": (round(sum(grown_eff) / (sum(grown_eff) + sum(orig_eff)), 6)
                                  if (grown_eff or orig_eff) else None),
        "units": out,
    }


def aggregate(evals: list[dict]) -> dict:
    by_group = defaultdict(list)
    for e in evals:
        by_group[e["group"]].append(e)

    table = {}
    for g, runs in sorted(by_group.items()):
        prim = [r["final"]["holdout"]["PRIMARY_extrapolative_accuracy"] for r in runs]
        table[g] = {
            "n_seeds": len(runs),
            "primary_mean": M._mean(prim),
            "primary_per_seed": prim,
            "primary_sd": (sum((x - M._mean(prim)) ** 2 for x in prim) / max(1, len(prim) - 1)) ** 0.5
                          if len(prim) > 1 else 0.0,
            "buckets": {
                DIST_NAMES[d]: M._mean(
                    r["final"]["holdout"]["per_bucket"].get(DIST_NAMES[d], {}).get("accuracy", float("nan"))
                    for r in runs)
                for d in range(5)
            },
            "info_gain_bits": M._mean(r["final"]["holdout"]["overall_info_gain_bits"] for r in runs),
            "brier": M._mean(r["final"]["holdout"]["overall_brier"] for r in runs),
            "ece": M._mean(r["final"]["holdout"]["overall_ece"] for r in runs),
            "leakage_z": M._mean(r["final"]["holdout"]["leakage_z"] for r in runs),
            "leakage_flag": any(r["final"]["holdout"]["leakage_flag"] for r in runs),
            "flops": M._mean(r["compute"]["flops"] for r in runs),
            "tokens": M._mean(r["compute"]["tokens"] for r in runs),
            "steps": M._mean(r["compute"]["steps"] for r in runs),
            "wall_s": M._mean(r["compute"]["wall_s"] for r in runs),
            "peak_rss_mb": M._mean(r["compute"]["peak_rss_mb"] for r in runs),
            "params_final": M._mean(r["final_arch"]["params_total"] for r in runs),
            "params_active": M._mean(r["final_arch"]["params_active"] for r in runs),
            "inference_flops_per_token": M._mean(r["compute"]["inference_flops_per_token"] for r in runs),
            "n_events": M._mean(r["n_events"] for r in runs),
            "ledger_ok": all(r["ledger_ok"] for r in runs),
            "grown_share_of_ablation_effect": M._mean(
                [r["specialization"]["grown_share_of_effect"] for r in runs
                 if r["specialization"].get("grown_share_of_effect") is not None] or [float("nan")]),
            "max_mastered_accuracy_drop": max(
                (r["forgetting"]["max_mastered_accuracy_drop"] for r in runs), default=0.0),
            "max_logit_delta_at_growth": max(
                (r["forgetting"]["max_logit_delta_at_growth"] for r in runs), default=0.0),
        }

    # paired comparisons across seeds
    comps = {}
    def series(g):
        return [r["final"]["holdout"]["PRIMARY_extrapolative_accuracy"]
                for r in sorted(by_group.get(g, []), key=lambda x: x["seed"])]
    for a, b in [("D", "A"), ("D", "B"), ("D", "C"), ("D", "R"),
                 ("C", "A"), ("C", "R"), ("D", "D_RANDPRUNE"), ("B", "A")]:
        sa, sb = series(a), series(b)
        if sa and sb and len(sa) == len(sb):
            comps[f"{a} - {b}"] = M.paired_bootstrap(sa, sb)

    # extrapolation per FLOP
    for g, v in table.items():
        v["extrapolation_per_petaflop"] = (v["primary_mean"] / (v["flops"] / 1e15)
                                           if v["flops"] else float("nan"))
    return {"groups": table, "paired": comps}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results/exp001/*/")
    ap.add_argument("--out", default="results/exp001/summary.json")
    a = ap.parse_args()
    dirs = sorted(d for d in glob.glob(a.runs)
                  if os.path.exists(os.path.join(d, "run.json")))
    evals = [evaluate_run(d) for d in dirs]
    agg = aggregate(evals)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump({"runs": evals, "aggregate": agg}, open(a.out, "w"), indent=1)

    # The raw ledgers are ~35MB each and are not committed. The manifest keeps
    # the part that carries the guarantee: how many predictions each run
    # committed, and the terminal hash of its chain. Re-running the same seed
    # must reproduce the same terminal hash.
    manifest = {}
    for d in dirs:
        path = os.path.join(d, "ledger.jsonl")
        if not os.path.exists(path):
            continue
        ok, msg = L.verify(path)
        recs = L.read(path)
        manifest[os.path.basename(d.rstrip("/"))] = {
            "records": len(recs),
            "predictions": sum(1 for r in recs if r.get("kind") == "prediction"),
            "chain_verified": ok,
            "terminal_hash": recs[-1]["_hash"] if recs else None,
            "bytes": os.path.getsize(path),
        }
    json.dump(manifest, open(os.path.join(os.path.dirname(a.out),
                                          "ledger_manifest.json"), "w"), indent=1)
    print(f"\nledger manifest: {len(manifest)} runs, "
          f"all chains verified = {all(v['chain_verified'] for v in manifest.values())}")

    t = agg["groups"]
    print(f"\n{'grp':<12}{'seeds':>6}{'PRIMARY':>10}{'sd':>7}{'D0':>7}{'D1':>7}{'D2':>7}"
          f"{'D3':>7}{'D4':>7}{'IG':>7}{'ECE':>7}{'PFLOP':>8}{'params':>9}{'lk_z':>6}")
    print("-" * 110)
    for g in ["A", "B", "C", "D", "D_RANDPRUNE", "R"]:
        if g not in t:
            continue
        v = t[g]
        b = v["buckets"]
        print(f"{g:<12}{v['n_seeds']:>6}{v['primary_mean']:>10.4f}{v['primary_sd']:>7.4f}"
              + "".join(f"{b.get(DIST_NAMES[d], float('nan')):>7.3f}" for d in range(5))
              + f"{v['info_gain_bits']:>7.3f}{v['ece']:>7.3f}"
              + f"{v['flops']/1e15:>8.3f}{v['params_final']/1e6:>9.3f}{v['leakage_z']:>6.1f}")
    print("\nforgetting check (worst ACCURACY drop on already-mastered categories "
          "within 500 steps of a growth event; frozen threshold = 0.02):")
    for g in ["C", "D", "D_RANDPRUNE", "R"]:
        if g in t:
            v = t[g]["max_mastered_accuracy_drop"]
            print(f"  {g:<12} worst_drop={v:+.4f} {'FAIL' if v > 0.02 else 'pass'}   "
                  f"logit_delta_at_growth={t[g]['max_logit_delta_at_growth']:.2e}")
    print("\nis the grown capacity load-bearing? (share of total positive "
          "ablation effect carried by units added during training):")
    for g in ["C", "D", "D_RANDPRUNE", "R"]:
        if g in t and t[g]["grown_share_of_ablation_effect"] == t[g]["grown_share_of_ablation_effect"]:
            print(f"  {g:<12} grown units carry "
                  f"{100*t[g]['grown_share_of_ablation_effect']:.4f}% of the model's "
                  f"functional weight")

    print("\npaired differences in PRIMARY (across seeds):")
    for k, v in agg["paired"].items():
        if v:
            print(f"  {k:<18} diff={v['mean_diff']:+.4f} "
                  f"CI95=[{v['ci95'][0]:+.4f},{v['ci95'][1]:+.4f}] "
                  f"d={v['cohens_d']:+.2f} wins={v['wins']}/{v['n_pairs']}")


if __name__ == "__main__":
    main()
