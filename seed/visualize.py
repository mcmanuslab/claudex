"""visualize.py -- the plots the experiment was designed to produce."""

from __future__ import annotations

import argparse, glob, json, os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from world import DIST_NAMES

# Palette from the data-viz reference instance, validated with
# scripts/validate_palette.js --mode light:
#   lightness band PASS, chroma floor PASS,
#   CVD separation PASS (worst adjacent aqua<->orange dE 9.2 deutan),
#   normal-vision floor PASS (27.6).
# The contrast WARN on aqua/yellow is relieved as the skill requires: every
# series is directly labelled and every figure has a table equivalent in
# RESULTS_001.md. Marker shape and dash pattern carry identity as well as hue,
# so the chart never relies on color alone.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#8a8985"

GROUP_STYLE = {
    # group: (label, color, marker, linestyle)
    "A": ("FIXED SMALL", "#3a3a38", "o", (0, (4, 2))),   # neutral reference baseline
    "B": ("FIXED LARGE", "#2a78d6", "s", "-"),
    "C": ("GROWTH ONLY", "#eb6834", "^", "-"),
    "D": ("DEVELOPMENTAL", "#1baf7a", "D", "-"),
    "D_RANDPRUNE": ("DEV (random prune)", "#4a3aa7", "v", (0, (5, 2))),
    "R": ("RANDOM DEV", "#eda100", "P", (0, (1, 1.6))),
}
ORDER = ["A", "B", "C", "D", "D_RANDPRUNE", "R"]


def _style(ax, title, xl, yl, sub=None):
    ax.set_title(title, fontsize=10.5, loc="left", fontweight="bold", color=INK,
                 pad=14 if sub else 6)
    if sub:
        ax.annotate(sub, xy=(0, 1.012), xycoords="axes fraction", fontsize=7.5,
                    color=INK_2, va="bottom", linespacing=1.4)
    ax.set_xlabel(xl, fontsize=8, color=INK_2)
    ax.set_ylabel(yl, fontsize=8, color=INK_2)
    ax.tick_params(labelsize=7, colors=INK_2, length=3)
    ax.grid(alpha=.22, lw=.5, color=MUTED)
    ax.set_axisbelow(True)
    ax.set_facecolor(SURFACE)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(MUTED); ax.spines[sp].set_linewidth(.8)


def _fig(*a, **k):
    f, ax = plt.subplots(*a, **k)
    f.patch.set_facecolor(SURFACE)
    return f, ax


def _legend(ax, **k):
    lg = ax.legend(fontsize=7, frameon=False, labelcolor=INK_2, **k)
    return lg


def _label_ends(ax, items, x, dy_min=0.042):
    """Direct-label each series at its right-hand end, pushed apart vertically
    so converging lines do not stack their labels on top of each other."""
    items = sorted(items, key=lambda it: it[1])
    lo, hi = ax.get_ylim()
    span = hi - lo
    # NB: no fixed-point loop here. The obvious
    #   while yy - placed[-1] < gap: yy = placed[-1] + gap
    # does not terminate: (x + gap) - x can round to slightly LESS than gap in
    # floating point, so the condition stays true forever. A single max() does
    # the same job and always terminates.
    gap = dy_min * span
    x_off = 14
    placed = []
    for label, y, c in items:
        yy = max(y, placed[-1] + gap) if placed else y
        placed.append(yy)
        ax.annotate(label, xy=(x, y), xytext=(x_off, yy), textcoords=("offset points", "data"),
                    fontsize=6.5, color=c, va="center", xycoords="data",
                    annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", color=c, lw=.6, alpha=.45,
                                    shrinkA=0, shrinkB=2)
                    if abs(yy - y) > 1e-9 else None)


def critical_plot(summary: dict, out: str) -> None:
    """THE plot: validated extrapolation ability against training compute."""
    runs = summary["runs"]
    by = defaultdict(lambda: defaultdict(list))
    for r in runs:
        for a, v in r["per_age"].items():
            f = v.get("flops")
            p = v["holdout"]["PRIMARY_extrapolative_accuracy"]
            if f:
                by[r["group"]][int(a)].append((f, p))

    fig, axes = _fig(1, 2, figsize=(12.0, 4.5))

    for ax, mode in ((axes[0], "flops"), (axes[1], "age")):
        ends = []
        for g in ORDER:
            if g not in by:
                continue
            label, c, m, ls = GROUP_STYLE[g]
            ages = sorted(by[g])
            ys = [sum(y for _, y in by[g][a]) / len(by[g][a]) for a in ages]
            se = [(sum((y - ys[i]) ** 2 for _, y in by[g][a])
                   / max(1, len(by[g][a]) - 1)) ** .5 / max(1, len(by[g][a])) ** .5
                  for i, a in enumerate(ages)]
            xs = ([sum(x for x, _ in by[g][a]) / len(by[g][a]) for a in ages]
                  if mode == "flops" else list(ages))
            if mode == "flops":
                # nudge each group's error bars apart, otherwise six identical
                # spreads stack on the same x and read as noise rather than
                # as "the spread is larger than the differences"
                k = ORDER.index(g) - len(ORDER) / 2
                jx = [x * (1.0 + 0.035 * k) for x in xs]
                ax.errorbar(jx, ys, yerr=se, fmt="none", ecolor=c,
                            elinewidth=1.0, capsize=1.8, alpha=.45)
            ax.plot(xs, ys, ls=ls, color=c, marker=m, ms=4.5, lw=2.0, label=label,
                    markeredgecolor=SURFACE, markeredgewidth=.7)
            ends.append((label, ys[-1], c, xs[-1]))
        if mode == "flops":
            ax.set_xscale("log")
            ax.set_xlim(right=ax.get_xlim()[1] * 3.4)
            lo, hi = ax.get_xlim()
            ticks = [v for v in (2.5e11, 5e11, 1e12, 2e12) if lo <= v <= hi]
            if ticks:
                ax.set_xticks(ticks)
                ax.set_xticklabels([f"{v/1e12:g}" for v in ticks])
                ax.set_xticks([], minor=True)
            _style(ax, "Validated extrapolation vs training compute",
                   "cumulative training FLOPs (trillions)",
                   "extrapolative accuracy (mean of D1, D2, D3)",
                   sub="error bars = 1 SE across seeds; every group on the same FLOPs budget")
        else:
            ax.set_xlim(right=max(e[3] for e in ends) + 3.0)
            _style(ax, "Developmental age vs accuracy on still-unseen facts",
                   "developmental age (revelation epoch)",
                   "extrapolative accuracy (mean of D1, D2, D3)",
                   sub="does the model become a better guesser as it develops?")
        _label_ends(ax, [(l, y, c) for l, y, c, _ in ends], max(e[3] for e in ends))
        _legend(ax, loc="lower right")

    fig.tight_layout(); fig.savefig(out, dpi=170, facecolor=SURFACE); plt.close(fig)


def bucket_plot(summary: dict, out: str) -> None:
    t = summary["aggregate"]["groups"]
    fig, axes = _fig(1, 2, figsize=(11.5, 4.4))
    ax = axes[0]
    names = [DIST_NAMES[d] for d in range(5)]
    w = 0.8 / max(1, len([g for g in ORDER if g in t]))
    for i, g in enumerate([g for g in ORDER if g in t]):
        label, c, _, _ = GROUP_STYLE[g]
        vals = [t[g]["buckets"].get(n, float("nan")) for n in names]
        ax.bar([x + i * w for x in range(5)], vals, w * 0.88, color=c, label=label,
               edgecolor=SURFACE, linewidth=1.0)
    # Per-bucket majority-class baseline. A single 0.5 line would be wrong for
    # D0 and D1, whose answer spaces are not binary.
    base = [0.340, 0.287, 0.501, 0.497, 0.545]
    nb = len([g for g in ORDER if g in t])
    for i, bv in enumerate(base):
        ax.plot([i - 0.06, i + nb * w - w * 0.06], [bv, bv], color=INK_2,
                ls=(0, (3, 2)), lw=1.1,
                label="majority baseline" if i == 0 else None, zorder=4)
    ax.set_xticks([x + 0.4 for x in range(5)])
    ax.set_xticklabels(["D0\ninterp", "D1\nnear", "D2\ncompos", "D3\ndistant",
                        "D4\nunsupported"], fontsize=7)
    _style(ax, "Accuracy by extrapolation distance", "", "holdout accuracy",
           sub="D4 is unpredictable by construction: it is the leakage alarm, not a score")
    ax.set_ylim(0, 1.14)
    _legend(ax, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.0))

    ax = axes[1]
    for g in [g for g in ORDER if g in t]:
        label, c, m, _ = GROUP_STYLE[g]
        ax.scatter(t[g]["ece"], t[g]["info_gain_bits"], s=70, color=c,
                   marker=m, label=label)
        ax.annotate(label, (t[g]["ece"], t[g]["info_gain_bits"]),
                    fontsize=6, xytext=(4, 4), textcoords="offset points")
    _style(ax, "Pareto: novelty vs calibration", "expected calibration error",
           "validated info gain (bits/prediction)", sub="up and to the left is better")
    fig.tight_layout(); fig.savefig(out, dpi=170, facecolor=SURFACE); plt.close(fig)


def lineage_plot(run_json: str, out: str) -> None:
    """Growth lineage for one developmental run: what was created at each
    event, what survived the competition, and what each candidate was worth."""
    r = json.load(open(run_json))
    events = r["events"]
    live, dead = GROUP_STYLE["D"][1], "#b9b8b4"
    fig, ax = _fig(figsize=(10.5, 4.4))
    end_step = r["compute"]["steps"]
    ymax = max(1, len(events))
    all_scores = []

    for i, e in enumerate(events):
        x = e["step"]
        sc = e["prune_scores"].get("marginal_val_loss_increase_when_masked", {})
        all_scores += list(sc.values())
        ax.axvline(x, color="#e4e3df", lw=1.0, zorder=0)
        ax.annotate(e["kind"].replace("GROW_", "").replace("ADD_", "") + f"\n@{e['site']}",
                    (x, ymax + 0.62), fontsize=6.5, ha="center", color=INK_2)
        for j, uid in enumerate(e["added_uids"]):
            alive = uid in e["survivors"]
            y = i + 1 + (j - (len(e["added_uids"]) - 1) / 2) * 0.17
            ax.scatter([x], [y], s=52 if alive else 28, zorder=3,
                       color=live if alive else dead,
                       edgecolor=SURFACE, linewidth=.8)
            tag = uid.split("@")[0]
            if uid in sc:
                tag += f"   {sc[uid]:+.4f}"
            ax.annotate(tag, (x, y), xytext=(8, 5 if alive else 0),
                        textcoords="offset points", fontsize=5.8,
                        color=live if alive else "#96958f", va="center")
            if alive:
                ax.plot([x, end_step], [y, y], color=live, lw=1.4, alpha=.5, zorder=1)

    neg = sum(1 for v in all_scores if v < 0)
    ax.set_position(ax.get_position())
    _style(ax, "Growth lineage", "training step", "growth event",
           sub=(f"filled = survived, grey = pruned; number = val-loss increase when "
                f"that unit alone is masked off.\n{neg}/{len(all_scores)} candidates "
                f"scored NEGATIVE: masking them off IMPROVED validation loss."))
    ax.set_ylim(0.3, ymax + 1.05)
    ax.set_yticks(range(1, ymax + 1))
    ax.set_xlim(right=end_step * 1.02)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out, dpi=170, facecolor=SURFACE); plt.close(fig)


def trajectory_plot(summary: dict, out: str) -> None:
    """Compute accounting: did the developmental groups simply spend more?"""
    runs = summary["runs"]
    t = summary["aggregate"]["groups"]
    fig, axes = _fig(1, 3, figsize=(13.5, 4.0))

    # --- parameter trajectory against spent compute -----------------------
    ax = axes[0]
    seen = set()
    for g in ORDER:
        for r in runs:
            if r["group"] != g or g in seen:
                continue
            seen.add(g)
            label, c, m, ls = GROUP_STYLE[g]
            ages = sorted(r["per_age"], key=lambda k: int(k))
            xs = [r["per_age"][a]["flops"] for a in ages]
            ys = [r["per_age"][a]["params"] / 1e3 for a in ages]
            ax.plot(xs, ys, ls=ls, color=c, marker=m, ms=4, lw=1.9, label=label,
                    markeredgecolor=SURFACE, markeredgewidth=.7)
    _style(ax, "Parameter trajectory", "cumulative training FLOPs (trillions)",
           "total parameters (thousands)",
           sub="seed 0; the sawtooth is overgrow-then-prune")
    ax.set_xscale("log")
    lo, hi = ax.get_xlim()
    ticks = [v for v in (2.5e11, 5e11, 1e12, 2e12) if lo <= v <= hi]
    if ticks:
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{v/1e12:g}" for v in ticks])
        ax.set_xticks([], minor=True)
    _legend(ax, loc="upper left")

    # --- steps actually taken, the price of being bigger -------------------
    ax = axes[1]
    gs = [g for g in ORDER if g in t]
    ax.barh(range(len(gs)), [t[g]["steps"] for g in gs],
            color=[GROUP_STYLE[g][1] for g in gs], height=.72,
            edgecolor=SURFACE, linewidth=1.0)
    for i, g in enumerate(gs):
        ax.annotate(f"{t[g]['steps']:,.0f}", (t[g]["steps"], i), fontsize=6.5,
                    color=INK_2, xytext=(4, 0), textcoords="offset points",
                    va="center")
    ax.set_yticks(range(len(gs)))
    ax.set_yticklabels([GROUP_STYLE[g][0] for g in gs], fontsize=6.5, color=INK_2)
    ax.invert_yaxis()
    ax.set_xlim(right=max(t[g]["steps"] for g in gs) * 1.22)
    _style(ax, "Optimizer steps bought by the same FLOPs", "steps", "",
           sub="a bigger model costs more per step, so it takes fewer of them")

    # --- is it just parameters? -------------------------------------------
    ax = axes[2]
    pts = []
    for g in gs:
        label, c, m, _ = GROUP_STYLE[g]
        x, y = t[g]["params_final"] / 1e3, t[g]["primary_mean"]
        ax.errorbar(x, y, yerr=t[g]["primary_sd"], fmt="none", ecolor=c,
                    elinewidth=1.2, capsize=3, alpha=.8)
        ax.scatter([x], [y], s=80, color=c, marker=m, zorder=3,
                   edgecolor=SURFACE, linewidth=.8)
        pts.append((label, x, y, c))
    ax.set_xlim(right=ax.get_xlim()[1] * 1.30)
    span = ax.get_ylim()[1] - ax.get_ylim()[0]
    placed = []
    for label, x, y, c in sorted(pts, key=lambda p: p[2]):
        yy = max(y, placed[-1] + 0.075 * span) if placed else y
        placed.append(yy)
        ax.annotate(label, (x, y), xytext=(8, yy), fontsize=6.5, color=c,
                    textcoords=("offset points", "data"), va="center",
                    arrowprops=dict(arrowstyle="-", color=c, lw=.6, alpha=.45,
                                    shrinkA=0, shrinkB=3) if abs(yy - y) > 1e-9 else None)
    _style(ax, "Is it just more parameters?", "final parameters (thousands)",
           "extrapolative accuracy", sub="error bars = 1 SD across seeds")
    fig.tight_layout(); fig.savefig(out, dpi=170, facecolor=SURFACE); plt.close(fig)


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
