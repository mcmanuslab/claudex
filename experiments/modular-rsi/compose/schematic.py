"""Schematic: step-by-step composition through a discrete symbol interface."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
INK, INK2, GRID, BG = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, GREEN, ORANGE = "#2a78d6", "#1baf7a", "#eb6834"
plt.rcParams.update({"font.size": 10, "figure.facecolor": BG})

fig = plt.figure(figsize=(13, 7.6))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 130); ax.set_ylim(0, 76); ax.axis("off")


def box(x, y, w, h, text, fc, ec, tc=INK, fs=9.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.2",
                                fc=fc, ec=ec, lw=1.6))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc,
            fontweight="bold" if bold else "normal")


def arrow(x0, y0, x1, y1, c=INK2, lw=1.6):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=12, color=c, lw=lw))


def symbol(x, y, s, c=BLUE):
    ax.add_patch(plt.Circle((x, y), 1.9, fc="white", ec=c, lw=1.6))
    ax.text(x, y, s, ha="center", va="center", fontsize=9, color=c, fontweight="bold")


def chain(y, ops, syms, color, label_left):
    ax.text(3, y + 2.2, label_left, ha="left", va="center", fontsize=9.5, color=INK2)
    x = 22
    symbol(x, y + 2.2, syms[0], color)
    for op, s in zip(ops, syms[1:]):
        arrow(x + 2, y + 2.2, x + 4.2, y + 2.2)
        box(x + 4.5, y, 6.6, 4.4, op, "#eef4fc" if color == BLUE else "#e8f7f0", color, fs=8.8, bold=True)
        arrow(x + 11.3, y + 2.2, x + 13.2, y + 2.2)
        x += 15.2
        symbol(x, y + 2.2, s, color)
    return x


# ---- title
ax.text(3, 72.5, "Step-by-step composition through a discrete symbol interface", fontsize=15, fontweight="bold", color=INK)
ax.text(3, 69.3, "Train on short programs only (1–3 steps, final answer only)  →  run long programs never seen (up to 16 steps)",
        fontsize=10.5, color=INK2)

# ---- panel 1: training
ax.text(3, 63.5, "1  TRAINING  (programs of 1–3 steps)", fontsize=11, fontweight="bold", color=BLUE)
chain(57.5, ["look up C", "+1"], ["x=3", "s=6", "y=7"], BLUE, "2-step program")
ax.text(76, 59.7, "Each step = one small block reading the context\n(4 random lookup tables, new every example).\n"
        "Between steps only a SYMBOL is passed\n(one of 8, argmax: the discrete interface).", fontsize=9, color=INK2, va="center")

# ---- panel 2: testing long
ax.text(3, 50, "2  TEST  (programs of 16 steps, never seen in training)", fontsize=11, fontweight="bold", color=GREEN)
ops = ["look up C", "+1", "inv A", "×3", "look up B", "−1"]
xe = chain(43.5, ops, ["x=3", "6", "7", "2", "6", "1", "0"], GREEN, "same blocks,\nreused in sequence")
ax.text(xe + 2.6, 45.7, "… 10 more\n    steps", fontsize=9, color=GREEN, va="center")
ax.text(22, 38.8, "Because each step only has to map symbol → symbol, any number of steps chains cleanly: "
        "errors do not blur across steps.", fontsize=9, color=INK2)

# ---- panel 3: transformer
ax.text(3, 32.5, "3  STANDARD TRANSFORMER  (same size, same training)", fontsize=11, fontweight="bold", color=ORANGE)
box(22, 22.5, 40, 7, "reads the whole program at once:\n[context]  [op₁ op₂ … op₁₆]  [x]", "#fdf0ea", ORANGE, fs=9.5)
arrow(62.5, 26, 67.5, 26, ORANGE)
box(68, 23.5, 12, 5, "answer?", "white", ORANGE, tc=ORANGE, bold=True)
ax.text(22, 19.3, "No step structure: it learns the length-1–3 patterns it saw, and is at chance (0.12) beyond 3 steps.",
        fontsize=9, color=INK2)

# ---- results inset
r = json.load(open(os.path.join(HERE, "results", "analysis_B.json")))["hard_acc"]
ia = fig.add_axes([0.66, 0.05, 0.31, 0.27]); ia.set_facecolor(BG)
for k, c, lab in (("looped", GREEN, "chained, one shared block"), ("modular", BLUE, "chained, 12 modules"),
                  ("transformer", ORANGE, "standard transformer")):
    L = sorted(int(x) for x in r[k]); ia.plot(L, [r[k][str(l)] for l in L], marker="o", ms=4, lw=2, color=c, label=lab)
ia.axvspan(0.5, 3.5, color=GRID, alpha=0.8, lw=0); ia.text(2, 0.03, "trained", ha="center", fontsize=8, color=INK2)
ia.axhline(1 / 8, ls=":", lw=1, color=INK2)
ia.set_ylim(0, 1.05); ia.set_xticks([1, 3, 4, 8, 12, 16]); ia.set_xlabel("program length (steps)", fontsize=8.5)
ia.set_ylabel("accuracy", fontsize=8.5); ia.tick_params(labelsize=8)
for s in ("top", "right"): ia.spines[s].set_visible(False)
ia.legend(fontsize=7.5, frameon=False, loc="center right")
ia.set_title("Result (5 seeds, ~47k params each)", fontsize=9.5, loc="left")

ax.text(3, 12.5, "Result: 0.97–1.00 accuracy at 16 steps for the chained systems,\nvs 0.12 (chance) for the transformer.",
        fontsize=10.5, fontweight="bold", color=INK, va="center")
ax.text(3, 7.2, "What matters is the interface + iteration, not separate modules: one shared block reused per step\n"
        "does as well as 12 specialist modules. The evolutionary outer loop added nothing over gradient training.",
        fontsize=9, color=INK2, va="top")

out = os.path.join(HERE, "..", "figures", "10_composition_schematic.png")
fig.savefig(out, dpi=170); print(out)
