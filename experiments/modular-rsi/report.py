"""Figures and summary for the modular RSI experiments (reads results/*.json)."""
import glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figures")
C = {"steered": "#2a78d6", "blind": "#eb6834", "fresh": "#1baf7a", "inherit": "#2a78d6",
     "ink": "#0b0b0b", "ink2": "#52514e", "grid": "#e4e3df", "surface": "#fcfcfb", "violet": "#4a3aa7"}
plt.rcParams.update({"figure.facecolor": C["surface"], "axes.facecolor": C["surface"],
                     "axes.edgecolor": C["ink2"], "axes.labelcolor": C["ink"], "text.color": C["ink"],
                     "xtick.color": C["ink2"], "ytick.color": C["ink2"], "axes.grid": True,
                     "grid.color": C["grid"], "grid.linewidth": 0.8, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 10, "lines.linewidth": 2,
                     "legend.frameon": False})
LOOKUPS = ("A", "B", "Ainv")   # succ is trivial; lookup niches carry the signal


def load(pattern):
    return [json.load(open(f)) for f in sorted(glob.glob(os.path.join(OUT, pattern)))]


def _band(ax, x, ys, color, label):
    ys = np.array(ys)
    ax.plot(x, ys.mean(0), color=color, label=label)
    ax.fill_between(x, ys.min(0), ys.max(0), color=color, alpha=0.15, linewidth=0)


def fig_main(summary):
    runs = {m: load(f"main_{m}_s*.json") for m in ("steered", "blind")}
    if not all(runs.values()):
        return
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for m, rs in runs.items():
        accs, succ = [], []
        for r in rs:
            h = [x for x in r["niche_hist"]]
            names = ("A", "B", "Ainv", "succ")
            per = {i: [x["mean_acc"] for x in h if x["niche"] == i] for i in range(4)}
            accs.append(np.mean([per[names.index(k)] for k in LOOKUPS], 0))
            succ.append(np.mean([r["success_frac"][names.index(k)] for k in LOOKUPS], 0))
        x = np.arange(1, len(accs[0]) + 1)
        _band(axs[0], x, accs, C[m], m)
        _band(axs[1], x, succ, C[m], m)
        solved = [[r["solved_gen"].get(k) for k in LOOKUPS] for r in rs]
        summary.setdefault("main", {})[m] = dict(
            seeds=len(rs),
            gens_to_solve_lookup=solved,
            mean_gens_to_solve=float(np.mean([g if g else 100 for s in solved for g in s])),
            final_success_frac=float(np.mean([s[-1] for s in succ])),
            mean_success_frac_all_gens=float(np.mean(succ)),
            variants=int(np.sum([r["variants"] for r in rs])),
            polymer_evals=int(np.sum([r["polymer_evals"] for r in rs])))
    for ax, t in zip(axs, ("Elite accuracy on lookup niches (A, B, Ainv)",
                           "Fraction of new variants already successful (acc ≥ 0.9)")):
        ax.set_title(t, loc="left", fontsize=11)
        ax.set_xlabel("generation (10 per cycle)")
        ax.set_ylim(0, 1.02)
        for c in range(10, len(x), 10):
            ax.axvline(c + 0.5, color=C["grid"], linewidth=1, zorder=0)
    axs[0].axhline(1 / 8, color=C["ink2"], linestyle=":", linewidth=1)
    axs[0].text(len(x), 1 / 8 + 0.02, "chance", ha="right", color=C["ink2"], fontsize=9)
    axs[0].legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "1_steered_vs_blind.png"), dpi=150)


def fig_polymer(summary):
    rs = load("main_steered_s*.json")
    if not rs:
        return
    last = max(p["cycle"] for p in rs[0]["polymer"])
    tg = [tuple(p["target"]) for p in rs[0]["polymer"] if p["cycle"] == last]
    poly = np.array([[p["polymer_acc"] for p in r["polymer"] if p["cycle"] == last] for r in rs])
    mono = np.array([[p["monomer_acc"] for p in r["polymer"] if p["cycle"] == last] for r in rs])
    chains = [[p["chain_niches"] for p in r["polymer"] if p["cycle"] == last] for r in rs]
    # polymer accuracy over cycles (all seeds, mean over targets)
    cyc = sorted({p["cycle"] for p in rs[0]["polymer"]})
    over = np.array([[np.mean([p["polymer_acc"] for p in r["polymer"] if p["cycle"] == c]) for c in cyc] for r in rs])
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    xi = np.arange(len(tg))
    axs[0].bar(xi - 0.2, mono.mean(0), 0.38, color=C["blind"], label="best single monomer")
    axs[0].bar(xi + 0.2, poly.mean(0), 0.38, color=C["steered"], label="evolved polymer")
    for i in xi:
        axs[0].text(i + 0.2, poly.mean(0)[i] + 0.02, f"{poly.mean(0)[i]:.2f}", ha="center", fontsize=9)
        axs[0].text(i - 0.2, mono.mean(0)[i] + 0.02, f"{mono.mean(0)[i]:.2f}", ha="center", fontsize=9)
    axs[0].set_xticks(xi, [f"L={len(t)}\n" + "→".join(t) for t in tg], fontsize=8)
    axs[0].set_ylim(0, 1.1)
    axs[0].set_title("Hidden multi-step problems, final cycle", loc="left", fontsize=11)
    axs[0].legend(loc="upper right", fontsize=9)
    _band(axs[1], np.array(cyc) + 1, over, C["steered"], "polymer")
    axs[1].set_ylim(0, 1.02)
    axs[1].set_xlabel("cycle")
    axs[1].set_title("Mean polymer accuracy across cycles (4 problems)", loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "2_polymers.png"), dpi=150)
    summary["polymer"] = dict(targets=["→".join(t) for t in tg], polymer_acc=poly.mean(0).tolist(),
                              monomer_acc=mono.mean(0).tolist(), chains_seed0=chains[0],
                              literal_match=[[list(c) == list(t) for c, t in zip(ch, tg)] for ch in chains],
                              polymer_acc_by_cycle=over.mean(0).tolist())


def fig_curriculum(summary):
    res = {m: load(f"curriculum_{m}_s*.json") for m in ("inherit", "fresh", "blind")}
    if not all(res.values()):
        return
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    skills = [s["skill"] for s in res["inherit"][0]["skills"]]
    x = np.arange(len(skills))
    out = {}
    for m, rs in res.items():
        g = np.array([[s["gens_to_solve"] or 90 for s in r["skills"]] for r in rs], float)
        out[m] = dict(per_skill_mean=g.mean(0).tolist(), per_seed=g.tolist(),
                      total_mean=float(g.sum(1).mean()), lookups_after_first=float(g[:, 1:].mean()))
        lab = {"inherit": "steered, inherits evolved improver", "fresh": "steered, default improver",
               "blind": "blind mutation"}[m]
        axs[0].plot(x, g.mean(0), marker="o", markersize=7, color=C[m] if m != "inherit" else C["violet"], label=lab)
    axs[0].set_xticks(x, skills)
    axs[0].set_ylabel("generations to solve (mean of seeds)")
    axs[0].set_xlabel("skill, learned in this order (weights always from scratch)")
    axs[0].set_title("Does the improver improve?", loc="left", fontsize=11)
    axs[0].legend(fontsize=9)
    for k, col in (("lr", C["violet"]), ("sigma", C["fresh"])):
        v = np.array([[s["genes"][k] for s in r["skills"]] for r in res["inherit"]])
        axs[1].plot(x, v.mean(0), marker="o", markersize=7, color=col, label=f"evolved {k}")
    axs[1].set_yscale("log")
    axs[1].set_xticks(x, skills)
    axs[1].set_title("Improver genes carried forward (inherit runs)", loc="left", fontsize=11)
    axs[1].legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "3_recursive_improver.png"), dpi=150)
    summary["curriculum"] = dict(skills=skills, **out)


def fig_scale(summary):
    p = os.path.join(OUT, "scale.json")
    if not os.path.exists(p):
        return
    r = json.load(open(p))
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    labs = [k for k in r if k.endswith("skills)") or k.endswith("skills")]
    cols = [C["steered"], C["violet"]]
    for lab, col in zip(labs, cols):
        n = np.arange(1, len(r[lab]["community_acc"]) + 1)
        ax.plot(n, r[lab]["community_acc"], marker="o", markersize=6, color=col, label=f"community, {lab}")
    ax.plot(n, r[labs[-1]]["monomer_acc"], marker="s", markersize=6, color=C["blind"], label="best single monomer")
    ax.axhline(1 / 8, color=C["ink2"], linestyle=":", linewidth=1)
    ax.text(n[-1], 1 / 8 + 0.02, "chance", ha="right", color=C["ink2"], fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("program length n  (= modules recruited)")
    ax.set_ylabel("accuracy")
    ax.set_title(f"Scaling by recruitment ({r['params_per_monomer']:,} params per module)", loc="left", fontsize=11)
    ax.legend(fontsize=9, loc="center right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "4_scaling.png"), dpi=150)
    summary["scale"] = r


def fig_lineage(summary):
    rs = load("main_steered_s0.json")
    if not rs:
        return
    fig, ax = plt.subplots(figsize=(7.5, 4))
    cols = [C["steered"], C["blind"], C["fresh"], C["violet"]]
    for (k, path), col in zip(rs[0]["champion_lineages"].items(), cols):
        if path:
            g, a = zip(*path)
            ax.step(np.array(g) + 1, a, where="post", color=col, label=k)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("generation of ancestor")
    ax.set_ylabel("ancestor accuracy")
    ax.set_title("Remembered paths: ancestry of each niche's champion (seed 0)", loc="left", fontsize=11)
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "5_champion_lineages.png"), dpi=150)
    summary["lineage_lengths"] = {k: len(v) for k, v in rs[0]["champion_lineages"].items()}


def fig_grow(summary):
    runs = {m: load(f"grow_{m}_s*.json") for m in ("hier", "flat")}
    if not runs["hier"]:
        return
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    out = {}
    for m, rs in runs.items():
        if not rs:
            continue
        col = C["steered"] if m == "hier" else C["blind"]
        lab = "hierarchical (solved polymers become units)" if m == "hier" else "flat control (monomers only, ≤5 per polymer)"
        cyc = sorted({r["cycle"] for r in rs[0]["log"]} | set(range(1, 11)))
        maxlen = []
        for r in rs:
            best, cur = [], 1
            for c in range(1, 11):
                L = [x["length"] for x in r["log"] if x["cycle"] == c and x["val_acc"] >= 0.95]
                cur = max([cur] + L)
                best.append(cur)
            maxlen.append(best)
        maxlen = np.array(maxlen, float)
        x = np.arange(1, 11)
        for row in maxlen:
            axs[0].plot(x, row, color=col, alpha=0.25, linewidth=1)
        gm = np.exp(np.log(maxlen).mean(0))
        axs[0].plot(x, gm, color=col, marker="o", markersize=6, label=lab)
        ev = [[x_["evals"] for x_ in r["log"] if x_["cycle"] == c] for r in rs for c in range(1, 11)]
        solved = [x_ for r in rs for x_ in r["log"] if x_["val_acc"] >= 0.95]
        if m == "hier":
            axs[1].scatter([x_["length"] for x_ in solved], [x_["evals"] for x_ in solved], s=22,
                           color=col, label="solved problems", zorder=3)
            fails = [x_ for r in rs for x_ in r["log"] if x_["val_acc"] < 0.95]
            axs[1].scatter([x_["length"] for x_ in fails], [x_["evals"] for x_ in fails], s=22,
                           facecolors="none", edgecolors=C["ink2"], label="not found within budget", zorder=3)
        # growth rate: slope of log2(max length) vs cycle
        slope = np.polyfit(x, np.log2(gm), 1)[0]
        execs = [x_.get("exec_acc") for x_ in solved if x_.get("exec_acc") is not None]
        out[m] = dict(max_len_by_cycle_geomean=gm.tolist(), max_len_per_seed=maxlen.tolist(),
                      doublings_per_cycle=float(slope), growth_factor_per_cycle=float(2 ** slope),
                      solved=len(solved), attempted=sum(len(r["log"]) for r in rs),
                      mean_evals_per_problem=float(np.mean([x_["evals"] for r in rs for x_ in r["log"]])),
                      exec_verified=len(execs), exec_acc_min=float(min(execs)) if execs else None,
                      longest_exec_verified=max([x_.get("exec_len") or 0 for x_ in solved if x_.get("exec_acc") is not None] or [0]))
    axs[0].set_yscale("log", base=2)
    axs[0].set_xlabel("cycle")
    axs[0].set_ylabel("longest problem solved (primitive steps, log₂)")
    axs[0].set_title("Capability: longest multi-step problem solved", loc="left", fontsize=11)
    axs[0].legend(fontsize=9, loc="upper left")
    axs[1].set_xscale("log"); axs[1].set_yscale("log")
    axs[1].axhline(6000, color=C["ink2"], linestyle=":", linewidth=1)
    axs[1].text(1.1, 6600, "search budget", color=C["ink2"], fontsize=9)
    axs[1].set_xlabel("problem length (primitive steps)")
    axs[1].set_ylabel("polymer evaluations to solve")
    axs[1].set_title("Cost stays flat while length explodes (hierarchical)", loc="left", fontsize=11)
    axs[1].legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "6_exponential_growth.png"), dpi=150)
    summary["grow"] = out


def fig_acquire(summary):
    runs = {m: load(f"acquire_{m}_s*.json") for m in ("scratch", "dup", "smart")}
    if not all(runs.values()):
        return
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    cols = {"scratch": C["blind"], "dup": C["fresh"], "smart": C["steered"]}
    labs = {"scratch": "from scratch (control)", "dup": "gene duplication",
            "smart": "duplication + path credit + edit memory"}
    out = {}
    for m, rs in runs.items():
        skills = [s["skill"] for s in rs[0]["skills"]]
        g = np.array([[s["gens"] or 80 for s in r["skills"]] for r in rs], float)
        ev = np.array([np.cumsum([s["evals"] for s in r["skills"]]) for r in rs], float)
        x = np.arange(len(skills))
        axs[0].plot(x, g.mean(0), marker="o", markersize=6, color=cols[m], label=labs[m])
        axs[0].fill_between(x, g.min(0), g.max(0), color=cols[m], alpha=0.12, linewidth=0)
        axs[1].plot(ev.mean(0), np.arange(1, len(skills) + 1), marker="o", markersize=5,
                    color=cols[m], label=labs[m])
        half = len(skills) // 2
        out[m] = dict(gens_per_skill_mean=g.mean(0).tolist(), per_seed=g.tolist(),
                      first_half_mean=float(g[:, 1:half].mean()), second_half_mean=float(g[:, half:].mean()),
                      total_evals_mean=float(ev[:, -1].mean()),
                      unsolved=int(sum(s["gens"] is None for r in rs for s in r["skills"])))
    axs[0].set_xticks(x, skills, fontsize=8)
    axs[0].set_ylabel("generations to master (mean, band = min–max)")
    axs[0].set_xlabel("skill, acquired in this order (library grows →)")
    axs[0].set_title("Cost of each new skill as the library grows", loc="left", fontsize=11)
    axs[0].legend(fontsize=9)
    axs[1].set_xlabel("cumulative evaluations (compute)")
    axs[1].set_ylabel("skills mastered")
    axs[1].set_title("Capability per unit of compute", loc="left", fontsize=11)
    axs[1].legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "7_smarter_as_it_grows.png"), dpi=150)
    summary["acquire"] = dict(skills=skills, **out)


def report():
    os.makedirs(FIG, exist_ok=True)
    summary = {}
    for f in (fig_main, fig_polymer, fig_curriculum, fig_scale, fig_lineage, fig_grow, fig_acquire):
        f(summary)
    with open(os.path.join(OUT, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "scale"}, indent=1)[:4000])


if __name__ == "__main__":
    report()
