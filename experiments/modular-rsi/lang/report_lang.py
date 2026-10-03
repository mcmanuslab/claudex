"""Figure + summary for Level 1 (spelling community vs monoliths)."""
import glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT, FIG = os.path.join(HERE, "results"), os.path.join(HERE, "..", "figures")
C = {"rsi": "#2a78d6", "random": "#1baf7a", "mono": "#eb6834", "mono8k": "#4a3aa7",
     "ink2": "#52514e", "grid": "#e4e3df", "surface": "#fcfcfb"}
plt.rcParams.update({"figure.facecolor": C["surface"], "axes.facecolor": C["surface"], "axes.grid": True,
                     "grid.color": C["grid"], "axes.spines.top": False, "axes.spines.right": False,
                     "font.size": 10, "lines.linewidth": 2, "legend.frameon": False,
                     "axes.edgecolor": C["ink2"], "xtick.color": C["ink2"], "ytick.color": C["ink2"]})


def main():
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    summ = {}
    for mode, lab in (("rsi", "community, RSI growth (duplicate + mutate + evolved lr)"),
                      ("random", "community, fresh modules only (ablation)")):
        rs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(OUT, f"grow_{mode}_s*.json")))]
        if not rs:
            continue
        n = min(len(r["log"]) for r in rs)
        b = np.array([x["bytes"] for x in rs[0]["log"][:n]])
        bpc = np.array([[x["bpc"] for x in r["log"][:n]] for r in rs])
        val = np.array([[x["valid"] for x in r["log"][:n]] for r in rs])
        for ax, ys in ((axs[0], bpc), (axs[1], val)):
            ax.plot(b, ys.mean(0), color=C[mode], label=lab)
            ax.fill_between(b, ys.min(0), ys.max(0), color=C[mode], alpha=0.15, linewidth=0)
        dup = [np.mean([not x["winner_fresh"] for x in r["log"][1:n]]) for r in rs]
        summ[mode] = dict(seeds=len(rs), K=int(n), bytes=b.tolist(), bpc=bpc.mean(0).tolist(), valid=val.mean(0).tolist(),
                          final_bpc=float(bpc[:, -1].mean()), final_valid=float(val[:, -1].mean()),
                          dup_winner_frac=float(np.mean(dup)),
                          final_lr=[r["log"][n - 1]["genes"]["lr"] for r in rs],
                          samples={str(k): rs[0]["log"][k - 1]["sample"] for k in (1, 4, 8, 16, 32) if k <= n})
    mono = sorted([r for f in glob.glob(os.path.join(OUT, "monolith_K*.json")) for r in json.load(open(f))],
                  key=lambda r: (r["steps"] == 8000, r["params"]))
    for key, lab, sel in (("mono", "single model, same training steps", lambda r: r["steps"] != 8000),
                          ("mono8k", "single model, 8,000 steps", lambda r: r["steps"] == 8000)):
        rr = [r for r in mono if sel(r)]
        if rr:
            axs[0].plot([r["params"] for r in rr], [r["bpc"] for r in rr], marker="o", markersize=6, color=C[key], label=lab)
            axs[1].plot([r["params"] for r in rr], [r["valid"] for r in rr], marker="o", markersize=6, color=C[key], label=lab)
            summ[key] = [{k: r[k] for k in ("params", "steps", "bpc", "valid", "sample")} for r in rr]
    for ax in axs:
        ax.set_xscale("log"); ax.set_xlabel("total size (bytes at int8 = parameters)")
    axs[0].set_ylabel("bits per character (lower is better)")
    axs[0].set_title("Language modelling as the community grows", loc="left", fontsize=11)
    axs[1].set_ylabel("generated words that are real English")
    axs[1].set_title("Spelling", loc="left", fontsize=11)
    axs[0].legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "8_language_community.png"), dpi=150)
    json.dump(summ, open(os.path.join(OUT, "summary_lang.json"), "w"), indent=1)
    for k, v in summ.items():
        if isinstance(v, dict):
            print(k, {kk: vv for kk, vv in v.items() if kk not in ("bytes", "bpc", "valid", "samples")})
        else:
            print(k, [(r["params"], r["steps"], round(r["bpc"], 3), round(r["valid"], 2)) for r in v])


if __name__ == "__main__":
    main()
