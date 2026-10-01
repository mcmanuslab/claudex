"""Level 1 - spelling by a growing community of tiny modules (RSI growth loop).

The community starts with one ~340-byte module and grows one module per cycle. Old
modules are FROZEN: nothing is ever retrained, so growth is incremental and old skills
cannot be forgotten. Each cycle:

  1. VARY     M candidate newcomers. In 'rsi' mode most are mutated duplicates of
              existing modules; the parent is chosen by lineage credit (modules whose
              offspring won before are favoured) and each candidate carries improver
              genes (lr, sigma) mutated from the last winner's. One candidate is fresh.
              In 'random' mode every candidate is a fresh module with fixed lr.
  2. DEVELOP  every candidate trains S steps against the frozen community (votes are
              summed logits, so the frozen part is computed once per batch).
  3. SELECT   the candidate giving the lowest held-out bits/char joins the community.
  4. REMEMBER lineage (parent, genes, gain) is logged; the winner's parent gains credit.

  python grow_spell.py --mode rsi|random --seed 0 --cycles 32
"""
import argparse, json, os, time
import numpy as np
import torch
from data import corpus, dictionary, decode
from tinylm import TinyLM, batches, V

torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results"); os.makedirs(OUT, exist_ok=True)


def valid_rate(s, D):
    ws = [w.strip(".") for w in s.split() if len(w.strip(".")) >= 2]
    return sum(w in D for w in ws) / max(1, len(ws))


@torch.no_grad()
def sample(lm, G, n, rng, temp=0.8):
    ids = [0]
    for _ in range(n):
        x = torch.tensor([ids[-lm.C:]])
        lg = lm.logits(G, x)[:, 0, -1].sum(0)
        p = torch.softmax(lg / temp, -1).numpy()
        ids.append(int(rng.choice(V, p=p / p.sum())))
    return ids


def run(mode, seed, cycles=32, M=12, S=400, B=64, d=4, mlp=8, C=16):
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    _, tr, va = corpus(); D = dictionary()
    lm = TinyLM(d, C, mlp)
    xv, yv = batches(va, 512, C, np.random.default_rng(99))
    G = torch.zeros(0, lm.n)                       # frozen community
    credit = np.zeros(0)                           # lineage credit per module
    genes = dict(lr=0.03, sigma=0.3)               # last winner's improver genes
    log, steps_total = [], 0
    for cyc in range(cycles):
        K = len(G)
        # ---- 1. vary
        cand, parents, cg = [], [], []
        for c in range(M):
            fresh = mode == "random" or K == 0 or c == 0
            if fresh:
                g0 = lm.init(1, rng)[0]; par = -1
            else:
                w = credit + 1.0; par = int(rng.choice(K, p=w / w.sum()))
                g0 = G[par].clone()
            if mode == "rsi":
                gg = dict(lr=float(np.clip(genes["lr"] * np.exp(0.3 * rng.standard_normal()), 0.003, 0.2)),
                          sigma=float(np.clip(genes["sigma"] * np.exp(0.3 * rng.standard_normal()), 0.02, 2.0)))
            else:
                gg = dict(lr=0.03, sigma=0.0)
            if not fresh:
                g0 = g0 + gg["sigma"] * torch.randn(lm.n)
            cand.append(g0); parents.append(par); cg.append(gg)
        P = torch.stack(cand).requires_grad_(True)
        lr = torch.tensor([g["lr"] for g in cg])[:, None]
        m = torch.zeros_like(P); v = torch.zeros_like(P)
        # ---- 2. develop: Adam per candidate, frozen community logits computed once per batch
        for s in range(S):
            x, y = batches(tr, B, C, rng)
            with torch.no_grad():
                base = lm.logits(G, x).sum(0) if K else 0.0
            lg = lm.logits(P, x) + base                                    # (M,B,T,V)
            l = torch.nn.functional.cross_entropy(lg.reshape(-1, V), y.repeat(M, 1, 1).reshape(-1),
                                                  reduction="none").reshape(M, -1).mean(1)
            (g,) = torch.autograd.grad(l.sum(), P)
            with torch.no_grad():
                m.mul_(0.9).add_(0.1 * g); v.mul_(0.99).add_(0.01 * g * g)
                P -= lr * m / (v.sqrt() + 1e-6)
        steps_total += S
        # ---- 3. select on held-out text
        with torch.no_grad():
            base = lm.logits(G, xv).sum(0) if K else 0.0
            lg = lm.logits(P, xv) + base
            bpc = torch.nn.functional.cross_entropy(lg.reshape(-1, V), yv.repeat(M, 1, 1).reshape(-1),
                                                    reduction="none").reshape(M, -1).mean(1).numpy() / np.log(2)
        w = int(np.argmin(bpc))
        G = torch.cat([G, P[w:w + 1].detach()])
        credit = np.append(credit, 0.0)
        if parents[w] >= 0:
            credit[parents[w]] += 1.0                                       # path credit
        genes = cg[w]
        smp = decode(sample(lm, G, 1500, np.random.default_rng(cyc)))
        rec = dict(cycle=cyc, K=len(G), bytes=int(len(G) * lm.n), bpc=float(bpc[w]),
                   bpc_cands=bpc.round(3).tolist(), winner_parent=parents[w], winner_fresh=parents[w] < 0,
                   genes=genes, valid=valid_rate(smp, D), sample=smp[:200], steps_total=steps_total)
        log.append(rec)
        print(f"[{mode} s{seed}] K={len(G):2d} bytes={rec['bytes']:5d} bpc={rec['bpc']:.3f} valid={rec['valid']:.2f} "
              f"parent={parents[w]:2d} lr={genes['lr']:.3f} sig={genes['sigma']:.2f} | {smp[:70]}", flush=True)
    torch.save(G, os.path.join(OUT, f"community_{mode}_s{seed}.pt"))
    json.dump(dict(mode=mode, seed=seed, module_params=lm.n, log=log), open(os.path.join(OUT, f"grow_{mode}_s{seed}.json"), "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="rsi"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cycles", type=int, default=32)
    a = ap.parse_args()
    run(a.mode, a.seed, a.cycles)
