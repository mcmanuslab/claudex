"""A tiny 1-layer transformer "monomer", evaluated for P genomes at once (torch).

Architecture (one residual block, read out at the query position):
    x_q  = soft query embedding (distribution over symbols @ code.qry + qtag)
    o    = MultiHeadAttention(query=x_q, keys/values = context pair tokens)
    r    = x_q + o
    r    = r + MLP(rms(r))
    out  = softmax(rms(r) @ W_r + b_r)        -> distribution over V symbols

Context tokens are sums of (key, value, tag) embeddings and carry no position
(the context is a set), so attention scores and values decompose over the
three embedding tables; we exploit that to evaluate thousands of genomes cheaply.

Genomes are flat float32 vectors in a normalised space (every gene ~N(0,1) at
init); SCALE maps them to actual weights. ~3.9k parameters per monomer.
"""
import numpy as np
import torch
from .world import V, D, N_TAGS

torch.set_num_threads(4)
H, DK, M = 2, 8, 24  # heads, head dim, MLP hidden
HD = H * DK

_SEGMENTS = [
    ("Wq", (D, HD)), ("Wk", (D, HD)), ("Wv", (D, HD)), ("Wo", (HD, D)),
    ("W1", (D, M)), ("b1", (M,)), ("W2", (M, D)),
    ("Wr", (D, V)), ("br", (V,)),
]
LAYOUT = {}
_off = 0
for _name, _shape in _SEGMENTS:
    _n = int(np.prod(_shape))
    LAYOUT[_name] = (_off, _off + _n, _shape)
    _off += _n
N_PARAMS = _off

_scale = np.empty(N_PARAMS, np.float32)
for _name, (_a, _b, _shape) in LAYOUT.items():
    _scale[_a:_b] = (1.0 / np.sqrt(_shape[0])) if _name[0] == "W" else 0.1
SCALE = torch.from_numpy(_scale)


def unpack(G):
    """G: (P, N_PARAMS) normalised genomes -> dict of weight tensors (P, ...)."""
    W = G * SCALE
    return {k: W[:, a:b].reshape((len(G),) + s) for k, (a, b, s) in LAYOUT.items()}


def _rms(x):
    return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + 1e-5)


class CodeT:
    """Torch copy of the genetic code + context tensors."""

    def __init__(self, code):
        f = lambda a: torch.from_numpy(np.asarray(a, np.float32))
        self.key, self.val, self.tag = f(code.key), f(code.val), f(code.tag)
        self.qry, self.qtag = f(code.qry), f(code.qtag)


def ctx_tensors(ctx):
    eye_v, eye_t = torch.eye(V), torch.eye(N_TAGS)
    k, v, t = (torch.from_numpy(ctx[n]).long() for n in ("key", "val", "tag"))
    return [(k, eye_v[k]), (v, eye_v[v]), (t, eye_t[t])]


def forward_t(G, codeT, cparts, qdist, log=False):
    """Differentiable core. G (p,N) tensor; qdist (B,V) or (p,B,V) tensor -> (p,B,V) probs/logprobs."""
    w = unpack(G)
    p_ = G.shape[0]
    xq = qdist @ codeT.qry + codeT.qtag                      # (B,D) or (p,B,D)
    q = xq @ w["Wq"]                                         # (p,B,HD)
    B = q.shape[1]
    xq = xq.expand(p_, B, D)
    q = q.reshape(p_, B, H, DK).transpose(1, 2)              # (p,H,B,DK)
    tables = (codeT.key, codeT.val, codeT.tag)
    score, Vs = 0.0, []
    for table, (idx, _) in zip(tables, cparts):
        K = (table @ w["Wk"]).reshape(p_, -1, H, DK).permute(0, 2, 3, 1)   # (p,H,DK,n)
        Vs.append((table @ w["Wv"]).reshape(p_, -1, H, DK).transpose(1, 2))  # (p,H,n,DK)
        s_sym = q @ K                                        # (p,H,B,n) score per symbol
        score = score + torch.gather(s_sym, -1, idx[None, None].expand(p_, H, -1, -1))
    att = torch.softmax(score / DK ** 0.5, -1)               # (p,H,B,T)
    o = 0.0
    for (_, oh), Vv in zip(cparts, Vs):
        mass = (att.unsqueeze(-2) @ oh).squeeze(-2)          # (p,H,B,n) attention mass per symbol
        o = o + mass @ Vv
    o = o.transpose(1, 2).reshape(p_, B, HD) @ w["Wo"]
    r = xq + o
    h = torch.relu(_rms(r) @ w["W1"] + w["b1"][:, None])
    r = r + h @ w["W2"]
    logits = _rms(r) @ w["Wr"] + w["br"][:, None]
    return torch.log_softmax(logits, -1) if log else torch.softmax(logits, -1)


@torch.no_grad()
def forward(G, code, ctx, qdist, chunk=1024):
    """numpy in/out convenience wrapper: (P,N) genomes -> (P,B,V) probabilities."""
    codeT = code if isinstance(code, CodeT) else CodeT(code)
    cparts = ctx_tensors(ctx)
    G = torch.as_tensor(G)
    qd = torch.as_tensor(np.asarray(qdist, np.float32))
    outs = [forward_t(G[s:s + chunk], codeT, cparts, qd if qd.dim() == 2 else qd[s:s + chunk])
            for s in range(0, len(G), chunk)]
    return torch.cat(outs).numpy()
