"""Progress-over-generations figure for Exp 1 (steered vs blind).

Reads results/main_{steered,blind}_s*.json (whatever seeds exist) and writes
figures/generations.png. Standalone: python plot_generations.py
"""
import glob
import json
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES, FIG = os.path.join(HERE, "results"), os.path.join(HERE, "figures")
SOLVED = 0.95
CHANCE = 0.125
GENS_PER_CYCLE = 10
LOOKUP = (0, 1, 2)            # A, B, Ainv
N_NICHES = 4
BG, INK, MUTED, GRID = "#fcfcfb", "#1f1f1f", "#6b6b6b", "#e6e6e3"
COLORS = {"steered": "#2a78d6", "blind": "#eb6834"}


def load(mode):
    runs = {}
    for p in sorted(glob.glob(os.path.join(RES, f"main_{mode}_s*.json"))):
        s = int(re.search(r"_s(\d+)\.json$", p).group(1))
        with open(p) as f:
            runs[s] = json.load(f)
    return runs


def per_gen(run, field, niches):
    """(n_gens,) array: mean of `field` over the given niches at each generation."""
    h = run["niche_hist"]
    G = max(r["gen"] for r in h) + 1
    M = np.full((G, N_NICHES), np.nan)
    for r in h:
        M[r["gen"], r["niche"]] = r[field]
    return M[:, list(niches)]


def series(run):
    acc = per_gen(run, "mean_acc", range(N_NICHES))
    mastered = np.maximum.accumulate(acc >= SOLVED, axis=0).sum(1)       # cumulative skills
    lookup = np.nanmean(per_gen(run, "mean_acc", LOOKUP), 1)
    lr = np.nanmedian(per_gen(run, "lr", range(N_NICHES)), 1)
    cyc = sorted({p["cycle"] for p in run["polymer"]})
    poly = np.array([np.mean([p["polymer_acc"] for p in run["polymer"] if p["cycle"] == c]) for c in cyc])
    mono = np.array([np.mean([p["monomer_acc"] for p in run["polymer"] if p["cycle"] == c]) for c in cyc])
    return dict(mastered=mastered, lookup=lookup, lr=lr,
                pgen=(np.array(cyc) + 1) * GENS_PER_CYCLE, poly=poly, mono=mono)


def stack(seq):
    n = min(len(a) for a in seq)
    return np.stack([a[:n] for a in seq])


def band(ax, x, Y, color, label, ls="-", marker=None, band_alpha=0.15, log=False):
    mean = np.exp(np.log(Y).mean(0)) if log else Y.mean(0)
    if len(Y) > 1:
        ax.fill_between(x, Y.min(0), Y.max(0), color=color, alpha=band_alpha, lw=0)
    ax.plot(x, mean, color=color, lw=2, ls=ls, marker=marker, ms=4, label=label)


def style(ax, title, xmax):
    ax.set_facecolor(BG)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#b9b9b4")
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=8.5, length=3, width=0.6)
    ax.grid(True, axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for c in range(GENS_PER_CYCLE, xmax + 1, GENS_PER_CYCLE):
        ax.axvline(c, color="#d9d9d4", lw=0.7, zorder=0)
    ax.set_xlim(0, xmax)
    ax.set_xticks(range(0, xmax + 1, GENS_PER_CYCLE))
    ax.set_xlabel("generation", color=MUTED, fontsize=9)
    ax.set_title(title, loc="left", fontsize=10.5, color=INK, fontweight="bold", pad=8)


def legend(ax, loc):
    lg = ax.legend(loc=loc, fontsize=8, frameon=True, framealpha=0.9, edgecolor="none",
                   facecolor=BG, handlelength=2.2)
    for t in lg.get_texts():
        t.set_color(INK)


def main():
    data = {m: load(m) for m in ("steered", "blind")}
    S = {m: {s: series(r) for s, r in runs.items()} for m, runs in data.items()}
    seeds = {m: sorted(S[m]) for m in S}
    xmax = max(len(v["mastered"]) for m in S for v in S[m].values())
    xmax = int(np.ceil(xmax / GENS_PER_CYCLE) * GENS_PER_CYCLE)

    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig, axs = plt.subplots(2, 2, figsize=(11, 7.6), facecolor=BG)
    (a, b), (c, d) = axs

    for m, col in COLORS.items():
        if not S[m]:
            continue
        lab = f"{m} (n={len(S[m])})"
        runs = list(S[m].values())
        # (a) capability
        Y = stack([r["mastered"] for r in runs]).astype(float)
        band(a, np.arange(1, Y.shape[1] + 1), Y, col, lab)
        # (b) lookup-niche accuracy
        Y = stack([r["lookup"] for r in runs])
        band(b, np.arange(1, Y.shape[1] + 1), Y, col, lab)
        # (c) polymers
        P_ = stack([r["poly"] for r in runs]); M_ = stack([r["mono"] for r in runs])
        x = runs[0]["pgen"][: P_.shape[1]]
        band(c, x, P_, col, f"{m} polymer", marker="o")
        band(c, x, M_, col, f"{m} best single monomer", ls="--", band_alpha=0.08)
        # (d) improver gene
        Y = stack([r["lr"] for r in runs])
        band(d, np.arange(1, Y.shape[1] + 1), Y, col, lab, log=True)

    style(a, "(a) skills mastered (niche mean acc ≥ 0.95)", xmax)
    a.set_ylim(-0.15, N_NICHES + 0.3); a.set_yticks(range(N_NICHES + 1))
    a.set_ylabel("skills mastered (of 4)", color=MUTED, fontsize=9)
    legend(a, "lower right")

    style(b, "(b) elite accuracy on lookup skills A, B, Ainv", xmax)
    b.axhline(CHANCE, color=MUTED, lw=1, ls=":")
    b.text(xmax * 0.99, CHANCE + 0.02, "chance 1/8", ha="right", va="bottom", fontsize=8, color=MUTED)
    b.set_ylim(0, 1.03)
    b.set_ylabel("mean elite accuracy", color=MUTED, fontsize=9)
    legend(b, "upper left")

    style(c, "(c) multi-step problems: polymer vs single monomer", xmax)
    c.axhline(CHANCE, color=MUTED, lw=1, ls=":")
    c.text(xmax * 0.99, CHANCE - 0.02, "chance 1/8", ha="right", va="top", fontsize=8, color=MUTED)
    c.set_ylim(0, 1.03)
    c.set_ylabel("accuracy (mean of 4 problems)", color=MUTED, fontsize=9)
    legend(c, "upper left")

    style(d, "(d) evolved improver gene: lifetime learning rate", xmax)
    d.set_yscale("log")
    d.grid(True, axis="y", which="both", color=GRID, lw=0.5)
    d.set_ylabel("median lr across niches (log)", color=MUTED, fontsize=9)
    legend(d, "upper left")

    sd = "; ".join(f"{m}: seeds {', '.join(map(str, seeds[m])) or 'none'}" for m in seeds)
    fig.text(0.01, 0.005, f"Exp 1, lines = seed mean, bands = min–max across seeds ({sd}). "
             "Vertical lines mark cycle boundaries (10 generations).", fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.025, 1, 1), h_pad=2.2, w_pad=2.5)
    os.makedirs(FIG, exist_ok=True)
    out = os.path.join(FIG, "generations.png")
    fig.savefig(out, dpi=150, facecolor=BG)
    print("wrote", out, "seeds:", seeds)


if __name__ == "__main__":
    main()
