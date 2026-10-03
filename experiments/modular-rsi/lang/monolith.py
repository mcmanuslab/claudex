"""Baseline: one monolithic tiny transformer trained from scratch at each size the
community passes through (K modules x 340 bytes). Two budgets: the same number of
training steps the community's winners received (K x 400), and a generous 8,000."""
import json, os, sys
import numpy as np
import torch
from data import corpus, dictionary, decode
from tinylm import TinyLM, batches, V
from grow_spell import sample, valid_rate, OUT

torch.set_num_threads(1)


def size_for(target, C=16):
    best = None
    for d in range(2, 64):
        n = TinyLM(d, C, 2 * d).n
        if best is None or abs(n - target) < abs(best[1] - target):
            best = (d, n)
    return best[0]


def train(d, steps, seed=0, C=16, B=64):
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    _, tr, va = corpus(); D = dictionary()
    lm = TinyLM(d, C, 2 * d)
    G = lm.init(1, rng).requires_grad_(True)
    opt = torch.optim.Adam([G], lr=0.03 if d < 12 else 0.01)
    for s in range(steps):
        x, y = batches(tr, B, C, rng)
        l = lm.loss(G, x, y).sum(); opt.zero_grad(); l.backward(); opt.step()
    xv, yv = batches(va, 512, C, np.random.default_rng(99))
    with torch.no_grad():
        bpc = float(lm.loss(G, xv, yv)) / np.log(2)
    smp = decode(sample(lm, G.detach(), 1500, np.random.default_rng(0)))
    return dict(d=d, params=lm.n, steps=steps, bpc=bpc, valid=valid_rate(smp, D), sample=smp[:200])


if __name__ == "__main__":
    K = int(sys.argv[1])
    d = size_for(K * 340)
    res = [train(d, K * 400), train(d, 8000)]
    for r in res:
        r["K_equiv"] = K
        print(json.dumps({k: v for k, v in r.items() if k != "sample"}), flush=True)
    json.dump(res, open(os.path.join(OUT, f"monolith_K{K}.json"), "w"), indent=1)
