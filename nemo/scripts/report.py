#!/usr/bin/env python3
"""Generate the research dashboard for a NEMO run.

    python3 scripts/report.py results/pilot

Self-contained HTML with inline SVG -- no plotting dependency, no CDN, opens
offline.  The panel the whole project exists for is the last one: the module
phylogeny with duplication events marked, so a
duplication -> divergence -> specialisation chain can be read off by eye.
"""

from __future__ import annotations

import argparse
import html
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

W, H, PAD = 640, 200, 44
COLOURS = {"MVG": "#3b7dd8", "RVG": "#d98c3b", "FIX": "#5aa469",
           "DRIFT": "#9b9b9b", "other": "#8e6bbf"}


def cond(name: str) -> str:
    return str(name).split("_r")[0]


def colour(c: str) -> str:
    return COLOURS.get(c, COLOURS["other"])


def load(out: Path):
    con = sqlite3.connect(str(out / "nemo.sqlite"))
    gens = con.execute(
        "SELECT generation, run_name, mean_reward, mean_genome_len, mean_active,"
        " mean_flops, n_duplications, n_deletions, mean_q_str FROM generations"
    ).fetchall()
    pairs = con.execute(
        "SELECT run, lane, generation, gene, src_gene, innov, parent_innov "
        "FROM duplicate_pairs").fetchall()
    lineage = con.execute(
        "SELECT innov, parent, born_generation FROM module_lineage").fetchall()
    events = con.execute(
        "SELECT kind, COUNT(*) FROM events GROUP BY kind").fetchall()
    con.close()
    return gens, pairs, lineage, events


def series(gens, field_idx: int):
    """condition -> (generations, mean over replicates)."""
    by: dict[str, dict[int, list[float]]] = {}
    for row in gens:
        g, name = int(row[0]), row[1]
        v = row[field_idx]
        if v is None or not np.isfinite(v):
            continue
        by.setdefault(cond(name), {}).setdefault(g, []).append(float(v))
    return {c: (np.array(sorted(d)), np.array([np.mean(d[g]) for g in sorted(d)]))
            for c, d in by.items()}


def line_chart(data: dict, title: str, ylabel: str, zero_line: bool = False) -> str:
    if not data:
        return ""
    allx = np.concatenate([x for x, _ in data.values()])
    ally = np.concatenate([y for _, y in data.values()])
    if len(allx) == 0:
        return ""
    x0, x1 = float(allx.min()), float(max(allx.max(), 1))
    y0, y1 = float(ally.min()), float(ally.max())
    if y1 - y0 < 1e-9:
        y0, y1 = y0 - 0.5, y1 + 0.5
    pad = (y1 - y0) * 0.08
    y0, y1 = y0 - pad, y1 + pad

    def sx(v):
        return PAD + (v - x0) / max(x1 - x0, 1e-9) * (W - PAD - 14)

    def sy(v):
        return H - PAD + 6 - (v - y0) / max(y1 - y0, 1e-9) * (H - PAD - 18)

    parts = [f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" '
             f'aria-label="{html.escape(title)}">']
    parts.append(f'<rect x="0" y="0" width="{W}" height="{H}" fill="none"/>')
    for frac in (0, 0.5, 1.0):
        yv = y0 + frac * (y1 - y0)
        parts.append(f'<line x1="{PAD}" y1="{sy(yv):.1f}" x2="{W-14}" y2="{sy(yv):.1f}" '
                     f'class="grid"/>')
        parts.append(f'<text x="{PAD-6}" y="{sy(yv)+4:.1f}" class="tick" '
                     f'text-anchor="end">{yv:.3g}</text>')
    if zero_line and y0 < 0 < y1:
        parts.append(f'<line x1="{PAD}" y1="{sy(0):.1f}" x2="{W-14}" y2="{sy(0):.1f}" '
                     f'class="zero"/>')
    for c in sorted(data):
        x, y = data[c]
        if len(x) < 2:
            continue
        pts = " ".join(f"{sx(a):.1f},{sy(b):.1f}" for a, b in zip(x, y))
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{colour(c)}" '
                     f'stroke-width="2" stroke-linejoin="round"/>')
    parts.append(f'<text x="{PAD}" y="{H-12}" class="tick">{x0:.0f}</text>')
    parts.append(f'<text x="{W-14}" y="{H-12}" class="tick" text-anchor="end">'
                 f'{x1:.0f} generations</text>')
    parts.append(f'<text x="{PAD-34}" y="14" class="ylab">{html.escape(ylabel)}</text>')
    parts.append("</svg>")
    legend = " ".join(
        f'<span class="key"><i style="background:{colour(c)}"></i>{html.escape(c)}</span>'
        for c in sorted(data))
    return (f'<figure><figcaption>{html.escape(title)}</figcaption>'
            f'{"".join(parts)}<div class="legend">{legend}</div></figure>')


def phylogeny_svg(lineage, pairs, max_nodes: int = 260) -> str:
    """Module phylogeny: x = generation born, y = lineage, duplications marked.

    This is the panel the project exists for.  A duplication -> divergence
    chain reads as a fork: one node splitting into two that then acquire
    different descendants.
    """
    if not lineage:
        return "<p class='muted'>No module lineage recorded.</p>"
    born = {int(i): int(g) for i, _, g in lineage}
    parent = {int(i): int(p) for i, p, _ in lineage}
    dup_innov = {int(r[5]) for r in pairs}
    nodes = sorted(born, key=lambda i: born[i])[:max_nodes]
    if not nodes:
        return "<p class='muted'>No module lineage recorded.</p>"
    gmax = max(max(born[i] for i in nodes), 1)

    depth: dict[int, int] = {}
    for i in nodes:
        d, p, guard = 0, parent.get(i, -1), 0
        while p is not None and p >= 0 and guard < 64:
            d += 1
            p = parent.get(p, -1)
            guard += 1
        depth[i] = d
    dmax = max(max(depth.values()), 1)

    ph = 420
    pos = {}
    lanes: dict[int, int] = {}
    for i in nodes:
        d = depth[i]
        k = lanes.get(d, 0)
        lanes[d] = k + 1
        x = PAD + born[i] / gmax * (W - PAD - 24)
        y = 26 + d / dmax * (ph - 60) + (k % 5) * 4
        pos[i] = (x, y)

    out = [f'<svg viewBox="0 0 {W} {ph}" class="chart" role="img" '
           f'aria-label="module phylogeny">']
    for i in nodes:
        p = parent.get(i, -1)
        if p in pos:
            x0, y0 = pos[p]
            x1, y1 = pos[i]
            out.append(f'<path d="M{x0:.1f},{y0:.1f} C{(x0+x1)/2:.1f},{y0:.1f} '
                       f'{(x0+x1)/2:.1f},{y1:.1f} {x1:.1f},{y1:.1f}" class="edge"/>')
    for i in nodes:
        x, y = pos[i]
        if i in dup_innov:
            out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.4" '
                       f'fill="#d94f4f"><title>module {i}: duplicate of '
                       f'{parent.get(i,-1)}, born gen {born[i]}</title></circle>')
        else:
            out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.2" fill="#6b8cc7">'
                       f'<title>module {i}, born gen {born[i]}</title></circle>')
    out.append(f'<text x="{PAD}" y="{ph-6}" class="tick">generation 0</text>')
    out.append(f'<text x="{W-14}" y="{ph-6}" class="tick" text-anchor="end">'
               f'{gmax}</text>')
    out.append("</svg>")
    return (f'<figure><figcaption>Module phylogeny &mdash; red nodes are '
            f'duplication events; a fork that persists is a '
            f'duplication&rarr;divergence chain</figcaption>{"".join(out)}</figure>')


CSS = """
:root{--bg:#ffffff;--fg:#15171a;--muted:#5d6570;--card:#f6f7f9;--line:#dfe3e8;
      --grid:#e6e9ee;--zero:#b9bfc8}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){
  --bg:#14161a;--fg:#e9ecf1;--muted:#98a1ad;--card:#1c1f25;--line:#2b3038;
  --grid:#272c34;--zero:#4a515b}}
:root[data-theme=dark]{--bg:#14161a;--fg:#e9ecf1;--muted:#98a1ad;--card:#1c1f25;
  --line:#2b3038;--grid:#272c34;--zero:#4a515b}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--fg);margin:0;padding:32px 16px 64px;
  font:15px/1.6 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}
.wrap{max-width:1100px;margin:0 auto}
h1{font-size:1.6rem;margin:0 0 4px} h2{font-size:1.05rem;margin:32px 0 10px}
.sub{color:var(--muted);margin:0 0 24px}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:18px}
figure{background:var(--card);border:1px solid var(--line);border-radius:10px;
  margin:0;padding:12px 12px 8px;overflow:hidden}
figcaption{font-size:.82rem;color:var(--muted);margin-bottom:6px}
.chart{width:100%;height:auto;display:block}
.grid{stroke:var(--grid);stroke-width:1}
.zero{stroke:var(--zero);stroke-width:1;stroke-dasharray:3 3}
.edge{stroke:var(--line);stroke-width:1;fill:none}
.tick,.ylab{fill:var(--muted);font-size:10px}
.legend{display:flex;flex-wrap:wrap;gap:12px;margin-top:6px;font-size:.78rem;
  color:var(--muted)}
.key i{display:inline-block;width:10px;height:10px;border-radius:2px;
  margin-right:5px;vertical-align:-1px}
table{border-collapse:collapse;width:100%;font-size:.86rem}
th,td{text-align:right;padding:6px 10px;border-bottom:1px solid var(--line)}
th:first-child,td:first-child{text-align:left}
th{color:var(--muted);font-weight:600}
.muted{color:var(--muted)} code{font-size:.85em}
.note{background:var(--card);border:1px solid var(--line);border-left:3px solid #d98c3b;
  border-radius:8px;padding:12px 14px;margin:18px 0;font-size:.9rem}
@media (max-width:600px){body{padding:20px 16px 48px}}
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    args = ap.parse_args()
    gens, pairs, lineage, events = load(args.out)
    meta = json.loads((args.out / "meta.json").read_text())
    summ = {}
    sp = args.out / "summary.json"
    if sp.exists():
        summ = json.loads(sp.read_text())

    charts = [
        line_chart(series(gens, 2), "Mean reward", "reward", zero_line=True),
        line_chart(series(gens, 3), "Genome length", "genes"),
        line_chart(series(gens, 4), "Active modules", "modules"),
        line_chart(series(gens, 5), "Metabolic cost", "FLOPs"),
        line_chart(series(gens, 6), "Duplication events per generation", "events"),
        line_chart(series(gens, 8), "Structural modularity Q_str", "Q"),
    ]
    ev_rows = "".join(f"<tr><td>{html.escape(k)}</td><td>{v:,}</td></tr>"
                      for k, v in sorted(events, key=lambda x: -x[1]))
    doc = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>NEMO Run Report</title><style>{CSS}</style></head><body><div class="wrap">
<h1>NEMO run report</h1>
<p class="sub">{html.escape(str(args.out))} &middot; preset
<code>{html.escape(str(meta.get('preset')))}</code> &middot;
{meta.get('lanes', 0):,} lanes &middot;
{len(meta.get('runs', []))} runs &middot;
{meta.get('generations', 0):,} generations</p>

<div class="note"><strong>Read the drift control first.</strong> Every curve
below is only interpretable against the grey <code>DRIFT</code> line, which runs
identical machinery with fitness permuted within island. A rising genome-length
curve that tracks DRIFT is bloat, not complexity. A Q_str above zero that does
not exceed its degree-preserving rewired null is sparsity, not modularity.</div>

<h2>Population dynamics</h2>
<div class="grid2">{''.join(c for c in charts if c)}</div>

<h2>Module phylogeny</h2>
{phylogeny_svg(lineage, pairs)}

<h2>Totals</h2>
<div class="grid2">
<figure><figcaption>Mutation events</figcaption>
<table><thead><tr><th>kind</th><th>count</th></tr></thead>
<tbody>{ev_rows}</tbody></table></figure>
<figure><figcaption>Run summary</figcaption><table><tbody>
<tr><td>duplicate pairs</td><td>{summ.get('n_duplicate_pairs', 0):,}</td></tr>
<tr><td>modules ever created</td><td>{summ.get('n_modules_ever', 0):,}</td></tr>
<tr><td>seconds / generation</td><td>{summ.get('s_per_generation', 0):.3f}</td></tr>
<tr><td>distinct module weight sets</td><td>
{summ.get('module_dedup', {}).get('distinct_fraction', 0):.1%}</td></tr>
</tbody></table></figure>
</div>
<p class="sub" style="margin-top:28px">Statistical verdicts are in
<code>analysis.json</code> (<code>scripts/analyse.py</code>); this page shows
trajectories, not tests.</p>
</div></body></html>"""
    dest = args.out / "report.html"
    dest.write_text(doc)
    print(f"written -> {dest}  ({len(doc)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
