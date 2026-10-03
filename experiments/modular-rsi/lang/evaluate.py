"""Honest evaluation for the language track.

* bpc over ALL non-overlapping C-char windows of a split (no random sub-batch).
* int8: weights are quantised per tensor (symmetric, 127 levels) and the model is
  re-evaluated, so "bytes" = int8 parameters is an actual storage size.
* real-word rate over several independent samples, with mean and std, both for
  words of >=2 letters and >=3 letters (2-letter words are easy by chance).
"""
import numpy as np
import torch
from tinylm import V


def windows(ids, C):
    n = (len(ids) - 1) // C
    x = torch.from_numpy(ids[: n * C].reshape(n, C))
    y = torch.from_numpy(ids[1: n * C + 1].reshape(n, C))
    return x, y


@torch.no_grad()
def bpc_vote(lm, G, ids, chunk=512):
    """bits/char of a voting community (sum of logits) or a single model (G (1,n))."""
    x, y = windows(ids, lm.C)
    tot, cnt = 0.0, 0
    for s in range(0, len(x), chunk):
        lg = lm.logits(G, x[s:s + chunk]).sum(0)
        tot += float(torch.nn.functional.cross_entropy(lg.reshape(-1, V), y[s:s + chunk].reshape(-1), reduction="sum"))
        cnt += y[s:s + chunk].numel()
    return tot / cnt / np.log(2)


def quantize(lm, G):
    """Per-tensor symmetric int8 quantisation of every segment of every genome."""
    W = (G * lm.scale).clone()
    for k, (a, b, s) in lm.L.items():
        seg = W[:, a:b]
        m = seg.abs().amax(1, keepdim=True).clamp_min(1e-8)
        W[:, a:b] = torch.round(seg / m * 127) / 127 * m
    return W / lm.scale


@torch.no_grad()
def sample_vote(lm, G, n, rng, temp=0.8):
    ids = [0]
    for _ in range(n):
        x = torch.tensor([ids[-lm.C:]])
        p = torch.softmax(lm.logits(G, x)[:, 0, -1].sum(0) / temp, -1).numpy()
        ids.append(int(rng.choice(V, p=p / p.sum())))
    return ids


def word_metrics(lm, G, D, decode, n_samples=5, n_chars=1200, seed=0):
    r2, r3, ex = [], [], ""
    for i in range(n_samples):
        s = decode(sample_vote(lm, G, n_chars, np.random.default_rng(seed + i)))
        ws = [w.strip(".") for w in s.split()]
        w2 = [w for w in ws if len(w) >= 2]; w3 = [w for w in ws if len(w) >= 3]
        r2.append(sum(w in D for w in w2) / max(1, len(w2)))
        r3.append(sum(w in D for w in w3) / max(1, len(w3)))
        ex = ex or s[:200]
    return dict(real2_mean=float(np.mean(r2)), real2_std=float(np.std(r2)),
                real3_mean=float(np.mean(r3)), real3_std=float(np.std(r3)), sample=ex)
