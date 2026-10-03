"""Tuned single-model baseline for Phase 0: same train/test splits and metrics as
phase0.py, sizes matched to K = 4, 8, 16, 32 modules (x 340 params), 12,800 steps
(= 32 cycles x 400), learning rate swept over {0.003, 0.01, 0.03}.

  python monolith_tuned.py K LR
"""
import json, os, sys
import numpy as np
import torch
from data import splits, dictionary, decode
from tinylm import TinyLM, batches
from evaluate import bpc_vote, quantize, word_metrics
from monolith import size_for

torch.set_num_threads(1)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "phase0")
os.makedirs(OUT, exist_ok=True)

K, lr = int(sys.argv[1]), float(sys.argv[2])
rng = np.random.default_rng(0); torch.manual_seed(0)
tr, se, te = splits(); D = dictionary()
d = size_for(K * 340)
lm = TinyLM(d, 16, 2 * d)
G = lm.init(1, rng).requires_grad_(True)
opt = torch.optim.Adam([G], lr=lr)
for s in range(12800):
    x, y = batches(tr, 64, 16, rng)
    l = lm.loss(G, x, y).sum(); opt.zero_grad(); l.backward(); opt.step()
G = G.detach()
r = dict(K_equiv=K, d=d, params=lm.n, lr=lr, steps=12800, sel_bpc=bpc_vote(lm, G, se), test_bpc=bpc_vote(lm, G, te),
         test_bpc_int8=bpc_vote(lm, quantize(lm, G), te), **word_metrics(lm, G, D, decode, seed=1000))
print(json.dumps({k: v for k, v in r.items() if k != "sample"}), flush=True)
json.dump(r, open(os.path.join(OUT, f"monolith_K{K}_lr{lr}.json"), "w"), indent=1)
