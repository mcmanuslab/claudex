"""report.py -- emit the results tables as markdown, straight from summary.json.

RESULTS_001.md quotes numbers. Transcribing them by hand is how papers end up
with a table that disagrees with its own figure, so the tables are generated.
"""
from __future__ import annotations

import argparse, json

from world import DIST_NAMES

ORDER = ["A", "B", "C", "D", "D_RANDPRUNE", "R"]
NAMES = {"A": "A FIXED SMALL", "B": "B FIXED LARGE", "C": "C GROWTH ONLY",
         "D": "D DEVELOPMENTAL", "D_RANDPRUNE": "D-RANDPRUNE", "R": "R RANDOM DEV"}
# majority-class baselines, from the world generator (world_seed-independent
# to within a few thousandths); printed so every accuracy has its null beside it
BASE = {0: 0.340, 1: 0.287, 2: 0.501, 3: 0.497, 4: 0.545}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="results/exp001/summary.json")
    a = ap.parse_args()
    s = json.load(open(a.summary))
    t = s["aggregate"]["groups"]
    gs = [g for g in ORDER if g in t]

    print("### Primary endpoint and per-bucket accuracy\n")
    print("| group | params | steps | PRIMARY (D1-D3) | sd | D0 | D1 | D2 | D3 | D4 (alarm) |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for g in gs:
        v = t[g]; b = v["buckets"]
        print(f"| **{NAMES[g]}** | {v['params_final']/1e3:.0f}k | {v['steps']:,.0f} | "
              f"**{v['primary_mean']:.4f}** | {v['primary_sd']:.4f} | "
              + " | ".join(f"{b.get(DIST_NAMES[d], float('nan')):.3f}" for d in range(5)) + " |")
    print(f"| _majority baseline_ | | | _{(BASE[1]+BASE[2]+BASE[3])/3:.3f}_ | | "
          + " | ".join(f"_{BASE[d]:.3f}_" for d in range(5)) + " |")

    print("\n### Compute accounting (all groups at an identical FLOPs budget)\n")
    print("| group | train FLOPs | steps | tokens | wall s | peak RSS MB | "
          "inference FLOPs/token | accuracy / PFLOP |")
    print("|---|---|---|---|---|---|---|---|")
    for g in gs:
        v = t[g]
        print(f"| {NAMES[g]} | {v['flops']:.3e} | {v['steps']:,.0f} | {v['tokens']/1e6:.2f}M | "
              f"{v['wall_s']:.0f} | {v['peak_rss_mb']:.0f} | {v['inference_flops_per_token']:,.0f} | "
              f"{v['extrapolation_per_petaflop']:.1f} |")

    print("\n### Calibration and novelty\n")
    print("| group | info gain (bits/pred) | Brier | ECE | D4 leakage z | leakage flag |")
    print("|---|---|---|---|---|---|")
    for g in gs:
        v = t[g]
        print(f"| {NAMES[g]} | {v['info_gain_bits']:.3f} | {v['brier']:.4f} | {v['ece']:.4f} | "
              f"{v['leakage_z']:+.2f} | {'**YES**' if v['leakage_flag'] else 'no'} |")

    print("\n### Paired differences in the primary endpoint, across seeds\n")
    print("| comparison | mean diff | 95% CI (paired bootstrap) | Cohen's d | seeds won |")
    print("|---|---|---|---|---|")
    for k, v in s["aggregate"]["paired"].items():
        if v:
            print(f"| {k} | {v['mean_diff']:+.4f} | [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}] | "
                  f"{v['cohens_d']:+.2f} | {v['wins']}/{v['n_pairs']} |")

    print("\n### Per-seed primary endpoint\n")
    print("| group | " + " | ".join(f"seed {i}" for i in range(len(t[gs[0]]['primary_per_seed']))) + " |")
    print("|---" * (1 + len(t[gs[0]]['primary_per_seed'])) + "|")
    for g in gs:
        print(f"| {NAMES[g]} | " + " | ".join(f"{x:.4f}" for x in t[g]["primary_per_seed"]) + " |")

    print("\n### Is the grown capacity load-bearing?\n")
    print("| group | growth events | grown share of total ablation effect | "
          "worst accuracy drop after growth |")
    print("|---|---|---|---|")
    for g in gs:
        v = t[g]
        sh = v.get("grown_share_of_ablation_effect")
        shs = "n/a (no growth)" if sh != sh else f"{100*sh:.4f}%"
        print(f"| {NAMES[g]} | {v['n_events']:.1f} | {shs} | "
              f"{v.get('max_mastered_accuracy_drop', 0.0):+.4f} |")


if __name__ == "__main__":
    main()
