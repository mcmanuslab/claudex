"""Compounding continual-acquisition test (pre-registered in compound_prereg.md).

A library of tiny modules learns 8 text domains one after another. For each new domain
the system adds ONE new module (1,520 params) that may read the library: its logits are
added to a gated vote of the top-4 library modules (by zero-shot bpc on the new domain's
selection split; gates are learned scalars). Library modules are frozen.

Question 1 (compounding): does the compute needed to reach a fixed target bpc on a new
domain fall as the library grows?
Question 2 (RSI loop): does the evolutionary outer loop (mutated duplicates, lineage
credit, evolved lr/sigma, 4 candidates) beat a cheap heuristic (probe the library, copy
the best module, fine-tune) at equal compute?

Arms (equal newcomer-training budget: compute counted in candidate-steps; cap 12,000/domain):
  full        4 candidates: 3 mutated duplicates (parent by lineage credit) + 1 fresh;
              evolved lr and sigma; winner = best on selection split
  nocredit    as full, parents uniform
  probe       copy the library module with best zero-shot bpc, fine-tune (1 candidate)
  latest      copy the most recently added module, fine-tune (1 candidate)
  fresh       new random module, no library (1 candidate)
  monolith    one model (~library size at the end) fine-tuned continually, 10% replay

A domain ends when a candidate reaches the target on the selection split (checked every
100 candidate-steps) or at the cap. Targets T1/T2 come from calibration.json.

  python compound.py --arm full --order 0
"""
import argparse, json, os
import numpy as np
import torch
from tinylm import TinyLM
from domains import build
from evaluate import windows

torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "compound"); os.makedirs(OUT, exist_ok=True)
V, C, B = 96, 16, 64
MOD = dict(d=8, mlp=16)            # 1,520 params
CAP, EVAL_EVERY, TOPK = 12_000, 100, 4
NAMES = ["shakespeare", "alice", "python", "javascript", "latex", "markdown", "legal", "french"]


def order_for(o):
    return [str(n) for n in np.random.default_rng(10_000 + o).permutation(NAMES)]


@torch.no_grad()
def logits_all(lm, G, x, chunk=512):
    return torch.cat([lm.logits(G, x[i:i + chunk]) for i in range(0, len(x), chunk)], 1)   # (P,N,C,V)


def ce_bits(lg, y):
    """lg (P,N,C,V), y (N,C) -> per-candidate bpc."""
    P = lg.shape[0]
    return torch.nn.functional.cross_entropy(lg.reshape(-1, V), y.repeat(P, 1, 1).reshape(-1),
                                             reduction="none").reshape(P, -1).mean(1) / np.log(2)


class Domain:
    """Cached data + library logits for one domain."""

    def __init__(self, splits, lm, lib, rng):
        tr, se, te = splits
        self.xs, self.ys = windows(tr[int(rng.integers(0, C)):], C)
        self.xse, self.yse = windows(se, C)
        self.xte, self.yte = windows(te, C)
        self.top, self.lib_tr, self.lib_se, self.lib_te, self.probe = [], None, None, None, []
        if len(lib):
            Gl = torch.stack(lib)
            self.probe = ce_bits(logits_all(lm, Gl, self.xse), self.yse).numpy()       # zero-shot bpc
            self.top = list(np.argsort(self.probe)[:TOPK])
            Gt = Gl[self.top]
            self.lib_tr, self.lib_se, self.lib_te = (logits_all(lm, Gt, x) for x in (self.xs, self.xse, self.xte))

    def mix(self, lg_new, gates, part, idx=None):
        """new-module logits + gated vote of the top library modules."""
        lib = {"tr": self.lib_tr, "se": self.lib_se, "te": self.lib_te}[part]
        if lib is None:
            return lg_new
        lib = lib if idx is None else lib[:, idx]
        return lg_new + torch.einsum("pk,kncv->pncv", gates, lib)


def train_domain(lm, dom, cands, lrs, target, rng):
    """Train candidates in parallel until one reaches `target` bpc on the selection split
    (checked every EVAL_EVERY candidate-steps) or the cap. Returns stats + winner."""
    nC = len(cands)
    P = torch.stack(cands).clone().requires_grad_(True)
    gates = torch.full((nC, len(dom.top)), 0.5, requires_grad=True)
    params = [P, gates] if len(dom.top) else [P]
    lr = torch.tensor(lrs)[:, None]
    st = [dict(m=torch.zeros_like(t), v=torch.zeros_like(t)) for t in params]
    every = max(1, EVAL_EVERY // nC)
    hit, steps, curve = None, 0, []
    while steps * nC < CAP:
        idx = torch.from_numpy(rng.integers(0, len(dom.xs), B))
        lg = dom.mix(lm.logits(P, dom.xs[idx]), gates, "tr", idx)
        l = torch.nn.functional.cross_entropy(lg.reshape(-1, V), dom.ys[idx].repeat(nC, 1, 1).reshape(-1),
                                              reduction="none").reshape(nC, -1).mean(1)
        grads = torch.autograd.grad(l.sum(), params)
        steps += 1
        with torch.no_grad():                       # Adam with bias correction (matches torch.optim.Adam)
            for t, g, s in zip(params, grads, st):
                s["m"].mul_(0.9).add_(0.1 * g); s["v"].mul_(0.999).add_(0.001 * g * g)
                mh, vh = s["m"] / (1 - 0.9 ** steps), s["v"] / (1 - 0.999 ** steps)
                t -= (lr if t is P else lr[:, :1]) * mh / (vh.sqrt() + 1e-8)
        if steps % every == 0:
            with torch.no_grad():
                sel = ce_bits(dom.mix(logits_all(lm, P, dom.xse), gates, "se"), dom.yse).numpy()
            curve.append((steps * nC, float(sel.min())))
            if sel.min() <= target:
                hit = steps * nC
                break
    with torch.no_grad():
        sel = ce_bits(dom.mix(logits_all(lm, P, dom.xse), gates, "se"), dom.yse).numpy()
        w = int(np.argmin(sel))
        test = float(ce_bits(dom.mix(logits_all(lm, P[w:w + 1], dom.xte), gates[w:w + 1], "te"), dom.yte)[0])
    return dict(cost=hit, cost_censored=hit is None, cand_steps=steps * nC, sel_bpc=float(sel[w]),
                test_bpc=test, curve=curve, winner=w), P[w].detach(), gates[w].detach()


def run(arm, o, target_key="T2"):
    cal = json.load(open(os.path.join(OUT, "calibration.json")))
    rng = np.random.default_rng(o); torch.manual_seed(o)
    data = build()
    order = order_for(o)
    lm = TinyLM(MOD["d"], C, MOD["mlp"], V=V)
    lib, credit, parents_of, genes = [], [], [], dict(lr=0.01, sigma=0.3)
    log = []
    if arm == "monolith":
        return run_monolith(o, order, data, cal, target_key)
    for k, name in enumerate(order):
        dom = Domain(data[name], lm, lib, rng)
        target = cal[name][target_key]
        cands, lrs, pars, cg = [], [], [], []
        if arm in ("full", "nocredit") and lib:
            for c in range(4):
                gg = dict(lr=float(np.clip(genes["lr"] * np.exp(0.3 * rng.standard_normal()), 1e-3, 0.1)),
                          sigma=float(np.clip(genes["sigma"] * np.exp(0.3 * rng.standard_normal()), 0.01, 2.0)))
                if c == 0:
                    par, g0 = -1, lm.init(1, rng)[0]
                else:
                    if arm == "full":
                        w = np.array(credit) + 1.0; par = int(rng.choice(len(lib), p=w / w.sum()))
                    else:
                        par = int(rng.integers(0, len(lib)))
                    g0 = lib[par] + gg["sigma"] * torch.randn(lm.n)
                cands.append(g0); lrs.append(gg["lr"]); pars.append(par); cg.append(gg)
        elif arm in ("full", "nocredit"):          # first domain: 4 fresh candidates
            for c in range(4):
                cands.append(lm.init(1, rng)[0]); lrs.append(0.01); pars.append(-1); cg.append(genes)
        else:
            if arm == "probe" and lib:
                par = int(np.argmin(dom.probe))
            elif arm == "latest" and lib:
                par = len(lib) - 1
            else:
                par = -1
            cands = [lib[par].clone() if par >= 0 else lm.init(1, rng)[0]]
            lrs, pars, cg = [0.01], [par], [dict(lr=0.01, sigma=0.0)]
        if arm.startswith("freshlr"):          # exploratory control: fresh module, fixed higher lr
            lrs = [float(arm[len("freshlr"):])]
        if arm == "fresh" or arm.startswith("freshlr"):
            dom.top, dom.lib_tr = [], None
            dom.lib_se = dom.lib_te = None
        res, G, gates = train_domain(lm, dom, cands, lrs, target, rng)
        w = res["winner"]
        lib.append(G); credit.append(0.0)
        if pars[w] >= 0:
            credit[pars[w]] += 1.0
        if arm in ("full", "nocredit"):
            genes = cg[w]
        res.update(k=k, domain=name, target=target, parent=pars[w], lr=lrs[w],
                   top=[int(t) for t in dom.top], probe_best=float(np.min(dom.probe)) if len(dom.probe) else None)
        res.pop("curve_full", None)
        log.append(res)
        print(f"[{arm} o{o}] k={k} {name:11s} cost={res['cost']} test={res['test_bpc']:.3f} "
              f"target={target:.3f} parent={pars[w]} lr={lrs[w]:.4f}", flush=True)
    json.dump(dict(arm=arm, order=o, domains=order, target_key=target_key, log=log),
              open(os.path.join(OUT, f"{arm}_o{o}.json"), "w"), indent=1)


def run_monolith(o, order, data, cal, target_key):
    """One model, size ~ final library (8 x 1,520 params), fine-tuned domain after domain,
    10% replay from earlier domains. Cost counted in param-matched candidate-steps."""
    rng = np.random.default_rng(o); torch.manual_seed(o)
    lm = TinyLM(32, C, 64, V=V)                     # 11,936 params ~ 8 x 1,520
    ratio = lm.n / TinyLM(MOD["d"], C, MOD["mlp"], V=V).n
    G = lm.init(1, rng).requires_grad_(True)
    opt = torch.optim.Adam([G], lr=0.003)
    seen, log = [], []
    for k, name in enumerate(order):
        tr, se, te = data[name]
        xs, ys = windows(tr, C); xse, yse = windows(se, C)
        target, steps, hit = cal[name][target_key], 0, None
        budget = int(CAP / ratio)                   # same FLOP-proxy budget as a module arm
        every = max(1, int(EVAL_EVERY / ratio))
        while steps < budget:
            n_rep = int(0.1 * B) if seen else 0
            idx = rng.integers(0, len(xs), B - n_rep)
            x, y = xs[idx], ys[idx]
            if n_rep:
                px, py = seen[int(rng.integers(0, len(seen)))]
                j = rng.integers(0, len(px), n_rep)
                x, y = torch.cat([x, px[j]]), torch.cat([y, py[j]])
            l = lm.loss(G, x, y).sum(); opt.zero_grad(); l.backward(); opt.step()
            steps += 1
            if steps % every == 0:
                sel = float(ce_bits(logits_all(lm, G.detach(), xse), yse)[0])
                if sel <= target:
                    hit = int(steps * ratio)
                    break
        seen.append((xs, ys))
        log.append(dict(k=k, domain=name, target=target, cost=hit, cost_censored=hit is None,
                        cand_steps=int(steps * ratio)))
        print(f"[monolith o{o}] k={k} {name:11s} cost={hit}", flush=True)
    final = {}
    for name in order:                               # forgetting: final model on every domain
        xte, yte = windows(data[name][2], C)
        final[name] = float(ce_bits(logits_all(lm, G.detach(), xte), yte)[0])
    json.dump(dict(arm="monolith", order=o, domains=order, target_key=target_key, log=log, final_test=final,
                   params=lm.n), open(os.path.join(OUT, f"monolith_o{o}.json"), "w"), indent=1)


def calibrate(steps=6000, seed=12345):
    """Targets per domain: T1 = selection-split bpc a FRESH 1,520-param module reaches after
    6,000 steps; T2 = the same for a fresh 3,968-param module (the harder, primary target)."""
    data = build(); out = {}
    for name in NAMES:
        tr, se, _ = data[name]
        row = {}
        for key, (d, mlp) in (("T1", (8, 16)), ("T2", (16, 32))):
            rng = np.random.default_rng(seed); torch.manual_seed(seed)
            lm = TinyLM(d, C, mlp, V=V)
            G = lm.init(1, rng).requires_grad_(True)
            opt = torch.optim.Adam([G], lr=0.01)
            xs, ys = windows(tr, C)
            for s in range(steps):
                idx = rng.integers(0, len(xs), B)
                l = lm.loss(G, xs[idx], ys[idx]).sum(); opt.zero_grad(); l.backward(); opt.step()
            xse, yse = windows(se, C)
            row[key] = float(ce_bits(logits_all(lm, G.detach(), xse), yse)[0])
        out[name] = row
        print(name, row, flush=True)
    json.dump(out, open(os.path.join(OUT, "calibration.json"), "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="full"); ap.add_argument("--order", type=int, default=0)
    ap.add_argument("--calibrate", action="store_true")
    a = ap.parse_args()
    calibrate() if a.calibrate else run(a.arm, a.order)
