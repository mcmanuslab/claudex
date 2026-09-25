"""Turn a run database into an analysis report (Markdown + optional figures).

Organised around the decision rules in `PREREGISTRATION.md` rather than around whatever
happens to look interesting. Every complexity number is printed against the neutral arm
when one is supplied, because a raw growth figure on its own is not reportable -- see
`DESIGN.md` section 3.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:                                    # pragma: no cover
    HAVE_MPL = False


def load_generations(db: sqlite3.Connection) -> list[dict]:
    return [json.loads(r[0]) for r in
            db.execute("SELECT payload FROM generation ORDER BY generation")]


def load_probes(db: sqlite3.Connection) -> dict:
    out: dict = {}
    for gen, cls, cond, metric, val in db.execute(
            "SELECT generation, cls, condition, metric, value FROM probe"):
        out.setdefault((cls, cond, metric), []).append((gen, val))
    return {k: sorted(v) for k, v in out.items()}


def slope(xs, ys) -> float:
    if len(xs) < 2:
        return float("nan")
    x, y = np.asarray(xs, float), np.asarray(ys, float)
    return float(np.polyfit(x, y, 1)[0])


def spark(vals, width: int = 40) -> str:
    """Compact inline trace, so the tables stay readable without opening a figure."""
    if not len(vals):
        return ""
    chars = " .:-=+*#%@"
    v = np.asarray(vals, float)
    if len(v) > width:
        v = np.interp(np.linspace(0, len(v) - 1, width), np.arange(len(v)), v)
    lo, hi = float(np.min(v)), float(np.max(v))
    if hi - lo < 1e-12:
        return chars[len(chars) // 2] * len(v)
    idx = ((v - lo) / (hi - lo) * (len(chars) - 1)).round().astype(int)
    return "".join(chars[i] for i in idx)


def drift_per_gen(gens: list[dict]) -> float:
    if len(gens) < 2:
        return float("nan")
    g = [x["generation"] for x in gens]
    lp = [x["params_log_mean"] for x in gens]
    return slope(g, lp)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("db")
    ap.add_argument("--neutral", default=None,
                    help="Neutral-arm database. Complexity claims require it.")
    ap.add_argument("--out", default=None)
    ap.add_argument("--figures", action="store_true")
    args = ap.parse_args()

    db = sqlite3.connect(args.db)
    gens = load_generations(db)
    probes = load_probes(db)
    manifest = {k: json.loads(v) for k, v in db.execute("SELECT key, value FROM run")}
    out_dir = Path(args.out or Path(args.db).parent)
    out_dir.mkdir(parents=True, exist_ok=True)

    L: list[str] = []
    A = L.append
    A(f"# Run report: {manifest.get('config', {}).get('name', '?')}")
    A("")
    A(f"- config hash `{manifest.get('config_hash')}`  git `{str(manifest.get('git_commit'))[:12]}`")
    A(f"- world-family seal `{manifest.get('suite_seal')}`  split {manifest.get('suite_summary')}")
    A(f"- generations {len(gens)}, seed {manifest.get('config', {}).get('seed')}")
    A(f"- Class C-test opened: **{manifest.get('config', {}).get('open_test_set')}**")
    A("")

    # ---------------------------------------------------------------- population
    A("## Population trajectory")
    A("")
    A("| metric | first | last | trace |")
    A("|---|---|---|---|")
    for key, label in [("n_alive", "alive"), ("params_median", "median params"),
                       ("params_min", "min params"), ("params_max", "max params"),
                       ("score_mean", "mean score"), ("score_max", "max score"),
                       ("gain_mean", "in-context gain"), ("n_species", "species"),
                       ("archive_coverage", "archive cells"),
                       ("gene_p_growth_bias", "growth-bias gene"),
                       ("gene_p_structural", "structural-mutation gene"),
                       ("gene_temperature", "temperature gene")]:
        v = [g[key] for g in gens if key in g]
        if not v:
            continue
        A(f"| {label} | {v[0]:.4g} | {v[-1]:.4g} | `{spark(v)}` |")
    A("")

    # -------------------------------------------------------------- complexity
    A("## Complexity: is the trend driven or passive?")
    A("")
    sel = drift_per_gen(gens)
    A(f"- selected arm: **{sel:+.5f}** nats/generation in mean ln(params)")
    if args.neutral and Path(args.neutral).exists():
        ndb = sqlite3.connect(args.neutral)
        ngens = load_generations(ndb)
        neu = drift_per_gen(ngens)
        A(f"- neutral arm:  **{neu:+.5f}** nats/generation")
        A(f"- **selected minus neutral: {sel - neu:+.5f} nats/generation**")
        A("")
        A("Only the difference is interpretable. A positive selected-arm number on its own "
          "is consistent with passive diffusion off the lower size bound.")
    else:
        A("- neutral arm: **not supplied** -- this number is NOT interpretable on its own. "
          "Run `configs/neutral.json` and pass `--neutral`.")
    A("")
    pmin = [g["params_min"] for g in gens]
    pmed = [g["params_median"] for g in gens]
    A("McShea's driven-trend test asks whether the *minimum* of the distribution moves, "
      "not just the mean:")
    A("")
    A(f"- min params {pmin[0]:.0f} -> {pmin[-1]:.0f} (slope {slope(range(len(pmin)), pmin):+.2f}/gen)")
    A(f"- median params {pmed[0]:.0f} -> {pmed[-1]:.0f} (slope {slope(range(len(pmed)), pmed):+.2f}/gen)")
    A("")
    A("A rising median with a stationary minimum is the signature of **passive diffusion**, "
      "not a driven trend.")
    A("")
    eff = probes.get(("-", "effective_params", "effective"), [])
    frac = probes.get(("-", "effective_params", "fraction"), [])
    if eff:
        A("### Effective vs raw parameters (bloat check)")
        A("")
        A(f"- effective parameters {eff[0][1]:.0f} -> {eff[-1][1]:.0f} "
          f"(slope {slope([g for g, _ in eff], [v for _, v in eff]):+.1f}/gen)")
        if frac:
            A(f"- effective fraction {frac[0][1] * 100:.1f}% -> {frac[-1][1] * 100:.1f}% "
              f"`{spark([v for _, v in frac])}`")
        A("")
        A("Raw parameters growing while the effective fraction falls is **bloat**, not "
          "complexity: capacity accumulating with no behavioural effect.")
        A("")

    # ------------------------------------------------------- displaced founders
    rows = db.execute(
        "SELECT lineage_root, MIN(birth_gen), AVG(params) FROM organism "
        "WHERE alive=1 GROUP BY lineage_root").fetchall()
    founders = db.execute(
        "SELECT lineage_root, params FROM organism WHERE birth_gen=0 "
        "GROUP BY lineage_root").fetchall()
    if founders:
        fmap = dict(founders)
        big = [(r, p) for r, _, p in rows if fmap.get(r, 0) > 20000]
        small = [(r, p) for r, _, p in rows if 0 < fmap.get(r, 0) <= 20000]
        if big and small:
            A("### Subclade test (displaced founders)")
            A("")
            A(f"- lineages founded large (>20K): now mean {np.mean([p for _, p in big]):.0f} params")
            A(f"- lineages founded small: now mean {np.mean([p for _, p in small]):.0f} params")
            A("")
            A("Under a **driven** trend, displaced lineages keep growing. Under **passive "
              "diffusion** they regress toward the bulk.")
            A("")

    # -------------------------------------------------------------- adaptation
    A("## Adaptation: the evolvability decomposition")
    A("")
    classes = sorted({c for c, _, m in probes if m == "score"})
    conds = sorted({k for _, k, m in probes if m == "score"})
    if classes:
        A("Slope of adaptation AUC against generation, per class and transplant condition. "
          "The hypothesis requires `full` to beat `no_feedback` and `capacity_matched`.")
        A("")
        A("| class | " + " | ".join(conds) + " |")
        A("|---" * (len(conds) + 1) + "|")
        for cls in classes:
            cells = []
            for cond in conds:
                pts = probes.get((cls, cond, "score"), [])
                cells.append(f"{slope([g for g, _ in pts], [v for _, v in pts]):+.5f}"
                             if len(pts) > 1 else "-")
            A(f"| {cls} | " + " | ".join(cells) + " |")
        A("")
        A("Final-generation AUC by class and condition:")
        A("")
        A("| class | " + " | ".join(conds) + " |")
        A("|---" * (len(conds) + 1) + "|")
        for cls in classes:
            cells = []
            for cond in conds:
                pts = probes.get((cls, cond, "score"), [])
                cells.append(f"{pts[-1][1]:+.4f}" if pts else "-")
            A(f"| {cls} | " + " | ".join(cells) + " |")
        A("")
        for cls in classes:
            f = probes.get((cls, "full", "score"), [])
            nf = probes.get((cls, "no_feedback", "score"), [])
            if f and nf:
                d = f[-1][1] - nf[-1][1]
                A(f"- **{cls}**: feedback-attributable adaptation = "
                  f"`full - no_feedback` = {d:+.4f}"
                  + ("  (positive: the organism is using its own action/reward history)"
                     if d > 0 else "  (non-positive: improvement is a reactive prior, "
                                   "not in-context learning)"))
        A("")

    # ------------------------------------------------------------- architecture
    A("## Architecture and mutation")
    A("")
    ops = db.execute(
        "SELECT mutations, COUNT(*), AVG(score) FROM organism "
        "WHERE mutations != '' GROUP BY mutations ORDER BY COUNT(*) DESC").fetchall()
    if ops:
        A("| operator | births | mean score | survived to adulthood |")
        A("|---|---|---|---|")
        for op, n, sc in ops[:14]:
            surv = db.execute(
                "SELECT COUNT(DISTINCT oid) FROM organism WHERE mutations=? AND age>0",
                (op,)).fetchone()[0]
            born = db.execute(
                "SELECT COUNT(DISTINCT oid) FROM organism WHERE mutations=?",
                (op,)).fetchone()[0]
            A(f"| `{op}` | {born} | {sc:+.4f} | {surv}/{born} "
              f"({100 * surv / max(1, born):.0f}%) |")
        A("")
    sigs = db.execute(
        "SELECT COUNT(DISTINCT sig) FROM organism").fetchone()[0]
    A(f"- distinct architectures that ever existed: **{sigs}**")
    A(f"- organisms recorded: {db.execute('SELECT COUNT(*) FROM organism').fetchone()[0]}")
    A(f"- extinctions: {db.execute('SELECT COUNT(*) FROM organism WHERE alive=0').fetchone()[0]}")
    A("")

    if args.figures and HAVE_MPL:
        _figures(db, gens, probes, out_dir)
        A(f"Figures written to `{out_dir}`.")
        A("")

    text = "\n".join(L)
    (out_dir / "report.md").write_text(text)
    print(text)
    print(f"\n[report written to {out_dir / 'report.md'}]")


def _figures(db, gens, probes, out_dir: Path) -> None:
    g = [x["generation"] for x in gens]
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))

    ax[0, 0].fill_between(g, [x["params_min"] for x in gens],
                          [x["params_max"] for x in gens], alpha=0.2, label="min-max")
    ax[0, 0].plot(g, [x["params_median"] for x in gens], lw=2, label="median")
    ax[0, 0].plot(g, [x["params_min"] for x in gens], lw=1, ls="--", label="minimum")
    ax[0, 0].set_yscale("log"); ax[0, 0].set_xlabel("generation")
    ax[0, 0].set_ylabel("parameters")
    ax[0, 0].set_title("Size distribution (minimum is the driven-trend test)")
    ax[0, 0].legend(fontsize=8)

    for (cls, cond, metric), pts in sorted(probes.items()):
        if metric == "score" and cond in ("full", "no_feedback"):
            ax[0, 1].plot([p[0] for p in pts], [p[1] for p in pts],
                          marker="o", ms=3,
                          ls="-" if cond == "full" else ":", label=f"{cls}/{cond}")
    ax[0, 1].axhline(0, color="k", lw=0.5)
    ax[0, 1].set_xlabel("generation"); ax[0, 1].set_ylabel("adaptation AUC")
    ax[0, 1].set_title("Adaptation by class; dotted = feedback ablated")
    ax[0, 1].legend(fontsize=7)

    ax[1, 0].plot(g, [x["gene_p_growth_bias"] for x in gens], label="p_growth_bias")
    ax[1, 0].axhline(0.5, color="k", ls="--", lw=0.8, label="neutral expectation")
    ax[1, 0].plot(g, [x["gene_p_structural"] for x in gens], label="p_structural")
    ax[1, 0].set_xlabel("generation"); ax[1, 0].set_ylabel("median gene value")
    ax[1, 0].set_title("Evolvability genes"); ax[1, 0].legend(fontsize=8)

    # Phylogeny: one line per organism from birth to death, y = parameter count,
    # with architectural innovations marked.
    rows = db.execute(
        "SELECT oid, MIN(gen), MAX(gen), AVG(params), MAX(mutations) "
        "FROM organism GROUP BY oid").fetchall()
    for oid, g0, g1, p, mut in rows:
        ax[1, 1].plot([g0, g1], [p, p], lw=0.4, alpha=0.5, color="0.4")
        if mut:
            ax[1, 1].plot([g0], [p], marker=".", ms=3,
                          color="tab:red" if "add" in mut or "widen" in mut
                          or "expand" in mut else "tab:blue")
    ax[1, 1].set_yscale("log"); ax[1, 1].set_xlabel("generation")
    ax[1, 1].set_ylabel("parameters")
    ax[1, 1].set_title("Lineages; red = growth event, blue = shrink event")

    fig.tight_layout()
    fig.savefig(out_dir / "report.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
