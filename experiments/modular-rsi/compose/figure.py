"""Accuracy vs program length for the composition test (Part A and Part B)."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
C = {"modular": "#2a78d6", "looped": "#1baf7a", "transformer": "#eb6834", "evolved": "#4a3aa7"}
plt.rcParams.update({"figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "axes.grid": True,
                     "grid.color": "#e4e3df", "axes.spines.top": False, "axes.spines.right": False,
                     "font.size": 10, "lines.linewidth": 2, "legend.frameon": False})
fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
for ax, (fn, title) in zip(axs, (("analysis.json", "Part A: 6,000 steps (undertrained)"),
                                  ("analysis_B.json", "Part B: 30,000 steps (to convergence)"))):
    r = json.load(open(os.path.join(HERE, "results", fn)))
    for k, acc in r["hard_acc"].items():
        L = sorted(int(x) for x in acc)
        ax.plot(L, [acc[str(l)] for l in L], marker="o", markersize=6, color=C[k], label=k)
    ax.axvspan(0.5, 3.5, color="#e4e3df", alpha=0.6, linewidth=0)
    ax.text(2, 0.03, "trained\nlengths", ha="center", fontsize=8, color="#52514e")
    ax.axhline(1 / 8, color="#52514e", linestyle=":", linewidth=1)
    ax.set_xlabel("program length (steps)"); ax.set_title(title, loc="left", fontsize=11)
    ax.set_xticks([1, 2, 3, 4, 6, 8, 12, 16])
axs[0].set_ylabel("accuracy (hard interface)"); axs[0].set_ylim(0, 1.03)
axs[1].legend(loc="center right", fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "..", "figures", "9_composition.png"), dpi=150)
