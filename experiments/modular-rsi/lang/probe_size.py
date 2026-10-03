"""How small can a transformer be and still learn to spell? Gradient-trained baseline
across sizes (the evolutionary experiments then start from this map)."""
import json, os, sys, numpy as np, torch
from data import corpus, dictionary, decode
from tinylm import TinyLM, batches, V
torch.set_num_threads(1)
rng = np.random.default_rng(0)
text, tr, va = corpus()
D = dictionary()


def word_stats(s):
    ws = [w.strip(".") for w in s.split() if len(w.strip(".")) >= 2]
    return (sum(w in D for w in ws) / max(1, len(ws))), ws[:12]


def bpc(lm, G, ids, B=256):
    r = np.random.default_rng(1)
    x, y = batches(ids, B, lm.C, r)
    with torch.no_grad():
        return float(lm.loss(G, x, y).mean()) / np.log(2)


res = []
# reference: unigram and bigram tables (not transformers)
cnt = np.bincount(tr, minlength=V) + 1; p = cnt / cnt.sum()
res.append(dict(name="unigram table", params=V - 1, bpc=float(-(np.log2(p[va])).mean())))
big = np.ones((V, V)); np.add.at(big, (tr[:-1], tr[1:]), 1); big /= big.sum(1, keepdims=True)
res.append(dict(name="bigram table", params=V * (V - 1), bpc=float(-(np.log2(big[va[:-1], va[1:]])).mean())))
for d, mlp, C in [(2, 0, 8), (3, 0, 8), (4, 0, 8), (4, 8, 16), (8, 0, 16), (8, 16, 16), (16, 32, 16), (24, 48, 24)]:
    lm = TinyLM(d, C, mlp)
    G = lm.init(1, rng).requires_grad_(True)
    opt = torch.optim.Adam([G], lr=0.03)
    for s in range(3000):
        x, y = batches(tr, 64, C, rng)
        l = lm.loss(G, x, y).sum(); opt.zero_grad(); l.backward(); opt.step()
        if s == 2000:
            for g in opt.param_groups: g["lr"] = 0.01
    Gd = G.detach()
    smp = decode(lm.sample(Gd[0], 400, np.random.default_rng(2), prompt=[0]))
    vw, ex = word_stats(smp)
    r = dict(name=f"transformer d={d} mlp={mlp} ctx={C}", params=lm.n, bytes_int8=lm.n,
             bpc=bpc(lm, Gd, va), valid_word_rate=vw, sample=smp[:160])
    res.append(r); print(json.dumps(r), flush=True)
json.dump(res, open(os.path.join(os.path.dirname(__file__), "probe_size.json"), "w"), indent=1)
