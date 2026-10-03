"""Phase 0: does the RSI outer loop add anything beyond "copy a module and train it"?

Re-runs Level 1 growth (voting community of 340-byte modules, old modules frozen)
with the judge's fixes: disjoint train / selection / test splits (select on the
selection split, report only on test), int8-quantised test bpc, word metrics over
several samples, and FLOP-matched ablations of the outer loop. Every arm spends the
same newcomer-training budget per cycle: 12 x 400 candidate-steps.

  full       12 candidates: 1 fresh + 11 mutated duplicates, parent by lineage credit,
             evolved lr and mutation size (the RSI loop)
  fixedgenes as full, but lr and mutation size fixed (no improver evolution)
  nocredit   as full, but parent chosen uniformly (no lineage credit)
  inherit    1 candidate = copy of the latest module, no mutation, fixed lr, 12x steps
             (a strong, cheap heuristic: warm start + more training)
  fresh      12 fresh candidates, fixed lr (no reuse)

Pre-registered (judge): the RSI-loop hypothesis survives Phase 0 only if `full` beats
`inherit` on final test bpc by more than 2 paired SDs across 4 seeds.

  python phase0.py --arm full --seed 0
"""
import argparse, json, os
import numpy as np
import torch
from data import splits, dictionary, decode
from tinylm import TinyLM, batches, V
from evaluate import bpc_vote, quantize, word_metrics, windows

torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "phase0"); os.makedirs(OUT, exist_ok=True)


@torch.no_grad()
def per_candidate_bpc(lm, G, P, ids, chunk=256):
    """Selection-split bpc of (frozen community G + each candidate in P)."""
    x, y = windows(ids, lm.C)
    tot = torch.zeros(len(P)); per_window = []
    for s in range(0, len(x), chunk):
        xb, yb = x[s:s + chunk], y[s:s + chunk]
        base = lm.logits(G, xb).sum(0) if len(G) else 0.0
        lg = lm.logits(P, xb) + base
        ce = torch.nn.functional.cross_entropy(lg.reshape(-1, V), yb.repeat(len(P), 1, 1).reshape(-1),
                                               reduction="none").reshape(len(P), len(xb), -1).mean(2)
        per_window.append(ce)
    pw = torch.cat(per_window, 1) / np.log(2)          # (P, n_windows)
    return pw.mean(1).numpy(), pw.numpy()


def run(arm, seed, cycles=32, M=12, S=400, B=64, d=4, mlp=8, C=16):
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    tr, se, te = splits(); D = dictionary()
    lm = TinyLM(d, C, mlp)
    G = torch.zeros(0, lm.n); credit = np.zeros(0)
    genes = dict(lr=0.03, sigma=0.3)
    nC, steps = (1, S * M) if arm == "inherit" else (M, S)
    log, cand_steps = [], 0
    for cyc in range(cycles):
        K = len(G)
        cand, parents, cg = [], [], []
        for c in range(nC):
            if arm == "inherit":
                par = K - 1 if K else -1
            elif arm == "fresh" or K == 0 or c == 0:
                par = -1
            elif arm == "nocredit":
                par = int(rng.integers(0, K))
            else:
                w = credit + 1.0; par = int(rng.choice(K, p=w / w.sum()))
            if arm in ("full", "nocredit"):
                gg = dict(lr=float(np.clip(genes["lr"] * np.exp(0.3 * rng.standard_normal()), 0.003, 0.2)),
                          sigma=float(np.clip(genes["sigma"] * np.exp(0.3 * rng.standard_normal()), 0.02, 2.0)))
            else:
                gg = dict(lr=0.03, sigma=0.0 if arm == "inherit" else 0.3)
            g0 = lm.init(1, rng)[0] if par < 0 else G[par].clone() + gg["sigma"] * torch.randn(lm.n)
            cand.append(g0); parents.append(par); cg.append(gg)
        P = torch.stack(cand).requires_grad_(True)
        lr = torch.tensor([g["lr"] for g in cg])[:, None]
        m = torch.zeros_like(P); v = torch.zeros_like(P)
        # frozen-community logits cached once per cycle on an aligned window grid
        # (random offset each cycle); every arm samples training data this same way
        xs, ys = windows(tr[int(rng.integers(0, C)):], C)
        with torch.no_grad():
            base_all = (torch.cat([lm.logits(G, xs[i:i + 512]).sum(0) for i in range(0, len(xs), 512)])
                        if K else None)
        for s in range(steps):
            idx = torch.from_numpy(rng.integers(0, len(xs), B))
            x, y = xs[idx], ys[idx]
            base = base_all[idx] if K else 0.0
            lg = lm.logits(P, x) + base
            l = torch.nn.functional.cross_entropy(lg.reshape(-1, V), y.repeat(nC, 1, 1).reshape(-1),
                                                  reduction="none").reshape(nC, -1).mean(1)
            (g,) = torch.autograd.grad(l.sum(), P)
            with torch.no_grad():
                m.mul_(0.9).add_(0.1 * g); v.mul_(0.99).add_(0.01 * g * g)
                P -= lr * m / (v.sqrt() + 1e-6)
        cand_steps += nC * steps
        sel, pw = per_candidate_bpc(lm, G, P.detach(), se)
        w = int(np.argmin(sel))
        # selection noise: SE of the winner-vs-runner-up difference over windows
        if nC > 1:
            r2 = int(np.argsort(sel)[1]); diff = pw[w] - pw[r2]
            spread, noise = float(sel.max() - sel.min()), float(diff.std() / np.sqrt(len(diff)))
        else:
            spread, noise = 0.0, 0.0
        G = torch.cat([G, P[w:w + 1].detach()])
        credit = np.append(credit, 0.0)
        if parents[w] >= 0:
            credit[parents[w]] += 1.0
        if arm in ("full", "nocredit"):
            genes = cg[w]
        test = bpc_vote(lm, G, te)
        rec = dict(cycle=cyc, K=len(G), params=int(len(G) * lm.n), sel_bpc=float(sel[w]), test_bpc=test,
                   spread=spread, sel_noise_se=noise, winner_parent=parents[w], genes=cg[w], cand_steps=cand_steps)
        log.append(rec)
        print(f"[{arm} s{seed}] K={len(G):2d} test={test:.3f} sel={sel[w]:.3f} spread={spread:.3f} "
              f"noiseSE={noise:.4f} parent={parents[w]} lr={cg[w]['lr']:.3f}", flush=True)
    final = dict(test_bpc=log[-1]["test_bpc"], test_bpc_int8=bpc_vote(lm, quantize(lm, G), te),
                 **word_metrics(lm, G, D, decode, seed=1000 + seed))
    print(f"[{arm} s{seed}] FINAL {json.dumps({k: v for k, v in final.items() if k != 'sample'})}", flush=True)
    torch.save(G, os.path.join(OUT, f"community_{arm}_s{seed}.pt"))
    json.dump(dict(arm=arm, seed=seed, module_params=lm.n, log=log, final=final),
              open(os.path.join(OUT, f"{arm}_s{seed}.json"), "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="full"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cycles", type=int, default=32)
    a = ap.parse_args()
    run(a.arm, a.seed, a.cycles)
