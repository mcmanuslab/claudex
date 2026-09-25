"""visualize.py -- the plots the experiment was designed to produce."""

from __future__ import annotations

import argparse, glob, json, os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from world import DIST_NAMES

GROUP_STYLE = {
    "A": ("FIXED SMALL", "#7a7a7a", "o", "-"),
    "B": ("FIXED LARGE", "#3b6ea5", "s", "-"),
    "C": ("GROWTH ONLY", "#d98b3a", "^", "-"),
    "D": ("DEVELOPMENTAL", "#b5495b", "D", "-"),
    "D_RANDPRUNE": ("DEV (random prune)", "#8d6bb5", "v", "--"),
    "R": ("RANDOM DEV", "#4e8c6a", "P", "--"),
}
ORDER = ["A", "B", "C", "D", "D_RANDPRUNE", "R"]


def _style(ax, title, xl, yl):
    ax.set_title(title, fontsize=10, loc="left", fontweight="bold")
    ax.set_xlabel(xl, fontsize=8); ax.set_ylabel(yl, fontsize=8)
    ax.tick_params(labelsize=7); ax.grid(alpha=.25, lw=.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def critical_plot(summary: dict, out: str) -> None:
    """THE plot: validated extrapolation ability vs training compute."""
    runs = summary["runs"]
    by = defaultdict(lambda: defaultdict(list))
    for r in runs:
        for a, v in r["per_age"].items():
            f = v.get("flops")
            p = v["holdout"]["PRIMARY_extrapolative_accuracy"]
            if f:
                by[r["group"]][int(a)].append((f, p))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    for g in ORDER:
        if g not in by:
            continue
        label, c, m, ls = GROUP_STYLE[g]
        ages = sorted(by[g])
        xs = [sum(x for x, _ in by[g][a]) / len(by[g][a]) for a in ages]
        ys = [sum(y for _, y in by[g][a]) / len(by[g][a]) for a in ages]
        sd = [(sum((y - ys[i]) ** 2 for _, y in by[g][a]) / max(1, len(by[g][a]) - 1)) ** .5
              for i, a in enumerate(ages)]
        ax.plot(xs, ys, ls, color=c, marker=m, ms=4, lw=1.6, label=label)
        ax.fill_between(xs, [a - b for a, b in zip(ys, sd)],
                        [a + b for a, b in zip(ys, sd)], color=c, alpha=.13, lw=0)
    _style(ax, "Validated extrapolation vs training compute",
           "cumulative training FLOPs", "extrapolative accuracy (D1-D3)")
    ax.set_xscale("log"); ax.legend(fontsize=7, frameon=False)

    ax = axes[1]
    for g in ORDER:
        if g not in by:
            continue
        label, c, m, ls = GROUP_STYLE[g]
        ages = sorted(by[g])
        ys = [sum(y for _, y in by[g][a]) / len(by[g][a]) for a in ages]
        ax.plot(ages, ys, ls, color=c, marker=m, ms=4, lw=1.6, label=label)
    _style(ax, "Developmental age vs accuracy on still-unseen facts",
           "developmental age (revelation epoch)", "extrapolative accuracy (D1-D3)")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def bucket_plot(summary: dict, out: str) -> None:
    t = summary["aggregate"]["groups"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    names = [DIST_NAMES[d] for d in range(5)]
    w = 0.8 / max(1, len([g for g in ORDER if g in t]))
    for i, g in enumerate([g for g in ORDER if g in t]):
        label, c, _, _ = GROUP_STYLE[g]
        vals = [t[g]["buckets"].get(n, float("nan")) for n in names]
        ax.bar([x + i * w for x in range(5)], vals, w, color=c, label=label)
    ax.axhline(0.5, color="k", ls=":", lw=.8)
    ax.set_xticks([x + 0.4 for x in range(5)])
    ax.set_xticklabels(["D0\ninterp", "D1\nnear", "D2\ncompos", "D3\ndistant",
                        "D4\nunsupported"], fontsize=7)
    _style(ax, "Accuracy by extrapolation distance", "", "holdout accuracy")
    ax.legend(fontsize=7, frameon=False, ncol=2)
    ax.text(4.0, 0.52, "chance\n(leakage alarm)", fontsize=6, color="k")

    ax = axes[1]
    for g in [g for g in ORDER if g in t]:
        label, c, m, _ = GROUP_STYLE[g]
        ax.scatter(t[g]["overall_ece"], t[g]["info_gain_bits"], s=70, color=c,
                   marker=m, label=label)
        ax.annotate(label, (t[g]["overall_ece"], t[g]["info_gain_bits"]),
                    fontsize=6, xytext=(4, 4), textcoords="offset points")
    _style(ax, "Pareto: novelty vs calibration  (up-left is better)",
           "expected calibration error", "validated info gain (bits/prediction)")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def lineage_plot(run_json: str, out: str) -> None:
    """Growth lineage tree for one developmental run."""
    r = json.load(open(run_json))
    events, arch = r["events"], r["final_arch"]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    survivors = {u for e in events for u in e["survivors"]}
    ymax = max(1, len(events))
    for i, e in enumerate(events):
        x = e["step"]
        ax.axvline(x, color="#cccccc", lw=.8, zorder=0)
        ax.text(x, ymax + .55, f"{e['kind'].replace('GROW_','').replace('ADD_','')}\n@{e['site']}",
                fontsize=6, ha="center", color="#444")
        for j, uid in enumerate(e["added_uids"]):
            alive = uid in e["survivors"]
            sc = e["prune_scores"].get("marginal_val_loss_increase_when_masked", {})
            y = i + 1 + (j - len(e["added_uids"]) / 2) * 0.16
            ax.scatter([x], [y], s=46 if alive else 26,
                       color="#b5495b" if alive else "#dddddd",
                       edgecolor="#b5495b" if alive else "#bbbbbb", zorder=3)
            lbl = f"{uid.split('@')[0]}"
            if uid in sc:
                lbl += f"  Δ{sc[uid]:+.3f}"
            ax.text(x + max(60, 0.01 * x), y, lbl, fontsize=5.5,
                    color="#b5495b" if alive else "#999999", va="center")
            if alive:
                ax.plot([x, r["compute"]["steps"]], [y, y], color="#b5495b",
                        lw=1.2, alpha=.65, zorder=2)
    _style(ax, f"Growth lineage  (filled = survived competition, greyed = pruned; "
               f"Δ = val-loss increase when masked)",
           "training step", "growth event")
    ax.set_ylim(0.2, ymax + 1.4); ax.set_yticks(range(1, ymax + 1))
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def trajectory_plot(summary: dict, out: str) -> None:
    runs = summary["runs"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    by = defaultdict(list)
    for r in runs:
        by[r["group"]].append(r)
    ax = axes[0]
    for g in ORDER:
        for r in by.get(g, [])[:1]:
            label, c, m, ls = GROUP_STYLE[g]
            ages = sorted(int(a) for a in r["per_age"])
            ax.plot([r["per_age"][a]["flops"] for a in map(str, ages)] if
                    isinstance(next(iter(r["per_age"])), str) else
                    [r["per_age"][a]["flops"] for a in ages],
                    [r["per_age"][a]["params"] for a in ages] if not
                    isinstance(next(iter(r["per_age"])), str) else
                    [r["per_age"][str(a)]["params"] for a in ages],
                    ls, color=c, marker=m, ms=3.5, lw=1.5, label=label)
    _style(ax, "Parameter trajectory", "cumulative FLOPs", "total parameters")
    ax.set_xscale("log"); ax.legend(fontsize=6, frameon=False)

    t = summary["aggregate"]["groups"]
    ax = axes[1]
    gs = [g for g in ORDER if g in t]
    ax.bar(range(len(gs)), [t[g]["extrapolation_per_petaflop"] for g in gs],
           color=[GROUP_STYLE[g][1] for g in gs])
    ax.set_xticks(range(len(gs)))
    ax.set_xticklabels([GROUP_STYLE[g][0] for g in gs], rotation=25, fontsize=6, ha="right")
    _style(ax, "Extrapolation per PFLOP", "", "accuracy / PFLOP")

    ax = axes[2]
    for g in gs:
        label, c, m, _ = GROUP_STYLE[g]
        ax.scatter(t[g]["params_final"] / 1e6, t[g]["primary_mean"], s=70,
                   color=c, marker=m)
        ax.annotate(label, (t[g]["params_final"] / 1e6, t[g]["primary_mean"]),
                    fontsize=6, xytext=(4, 3), textcoords="offset points")
    _style(ax, "Is it just more parameters?", "final parameters (M)",
           "extrapolative accuracy")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="results/exp001/summary.json")
    ap.add_argument("--outdir", default="results/exp001/figs")
    a = ap.parse_args()
    s = json.load(open(a.summary))
    os.makedirs(a.outdir, exist_ok=True)
    critical_plot(s, os.path.join(a.outdir, "fig1_critical.png"))
    bucket_plot(s, os.path.join(a.outdir, "fig2_buckets_pareto.png"))
    trajectory_plot(s, os.path.join(a.outdir, "fig3_compute.png"))
    for r in s["runs"]:
        if r["group"] == "D" and r["seed"] == 0:
            lineage_plot(os.path.join(r["dir"], "run.json"),
                         os.path.join(a.outdir, "fig4_lineage.png"))
            break
    print("wrote figures to", a.outdir)


if __name__ == "__main__":
    main()
