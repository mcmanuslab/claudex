"""Hybrid model: shared transformer core (frozen after base training) + per-skill adapters.
Everything is batched over C adapters (candidates or skills), each with its own examples."""
import math
import numpy as np
import torch
import torch.nn as nn
from world import V, R

D, H, FF, RMAX = 48, 4, 96, 16
N_CTX = 2 * V


def _ln(x):
    return (x - x.mean(-1, keepdim=True)) * torch.rsqrt(x.var(-1, keepdim=True, unbiased=False) + 1e-5)


class Core(nn.Module):
    def __init__(self, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        f = lambda *s: nn.Parameter(torch.randn(*s, generator=g) / math.sqrt(s[0]))
        self.slot = nn.Parameter(torch.randn(R, D, generator=g) * 0.3)
        self.val = nn.Parameter(torch.randn(V, D, generator=g) * 0.3)
        self.ckey = nn.Parameter(torch.randn(V, D, generator=g) * 0.3)
        self.cval = nn.Parameter(torch.randn(V, D, generator=g) * 0.3)
        self.ctag = nn.Parameter(torch.randn(2, D, generator=g) * 0.3)
        self.role = nn.Parameter(torch.randn(2, D, generator=g) * 0.3)       # register is operand i / j
        self.argi = nn.Parameter(torch.randn(R, D, generator=g) * 0.3)       # operands on the instruction token
        self.argj = nn.Parameter(torch.randn(R, D, generator=g) * 0.3)
        self.layers = nn.ModuleList()
        for _ in range(2):
            m = nn.Module()
            m.Wqkv, m.Wo, m.W1, m.W2 = f(D, 3 * D), f(D, D), f(D, FF), f(FF, D)
            self.layers.append(m)
        self.out = f(D, V)

    def _layer(self, m, x, A=None, Bm=None, mask=None):
        C, B, T, _ = x.shape
        xn = _ln(x)
        qkv = xn @ m.Wqkv
        if A is not None:                                  # low-rank adapter on the attention projection
            qkv = qkv + torch.einsum("cbtk,cke->cbte", torch.einsum("cbtd,cdk->cbtk", xn, A) * mask[:, None, None], Bm)
        q, k, v = qkv.split(D, -1)
        sh = lambda t: t.reshape(C, B, T, H, D // H).transpose(2, 3)
        a = torch.softmax(sh(q) @ sh(k).transpose(-1, -2) / math.sqrt(D // H), -1) @ sh(v)
        x = x + a.transpose(2, 3).reshape(C, B, T, D) @ m.Wo
        return x + torch.relu(_ln(x) @ m.W1) @ m.W2

    def forward(self, ad, regs, tabs, i, j):
        """ad: adapter dict (leading dim C); regs (C,B,R), tabs (C,B,2,V), i/j (C,B) long.
        Returns logits (C,B,R,V)."""
        C, B = regs.shape[:2]
        sl = torch.arange(R)
        reg_tok = (self.slot + self.val[regs] + (sl == i[..., None]).float()[..., None] * self.role[0]
                   + (sl == j[..., None]).float()[..., None] * self.role[1])            # (C,B,R,D)
        keys = torch.arange(V).expand(2, V)
        ctx = self.ckey[keys] + self.cval[tabs] + self.ctag[:, None]          # (C,B,2,V,D)
        ctx = ctx.reshape(C, B, N_CTX, D)
        op = (ad["emb"][:, None] + self.argi[i] + self.argj[j])[:, :, None]
        x = torch.cat([reg_tok, ctx, op], 2)
        x = self._layer(self.layers[0], x, ad["A0"], ad["B0"], ad["mask"])
        r = x[:, :, :R]
        h = torch.relu(torch.einsum("cbrd,cdk->cbrk", _ln(r), ad["Wd"]) * ad["mask"][:, None, None])
        x = torch.cat([r + torch.einsum("cbrk,ckd->cbrd", h, ad["Wu"]), x[:, :, R:]], 2)
        x = self._layer(self.layers[1], x, ad["A1"], ad["B1"], ad["mask"])
        return _ln(x[:, :, :R]) @ self.out


def new_adapters(C, rank, gen=None):
    """Fresh adapters: random embedding, small down-projection, zero up-projection."""
    g = gen or torch.Generator().manual_seed(0)
    mask = torch.zeros(C, RMAX)
    for c in range(C):
        mask[c, : (rank[c] if hasattr(rank, "__len__") else rank)] = 1
    return dict(emb=torch.randn(C, D, generator=g) * 0.3,
                Wd=torch.randn(C, D, RMAX, generator=g) / math.sqrt(D), Wu=torch.zeros(C, RMAX, D),
                A0=torch.randn(C, D, RMAX, generator=g) / math.sqrt(D), B0=torch.zeros(C, RMAX, 3 * D),
                A1=torch.randn(C, D, RMAX, generator=g) / math.sqrt(D), B1=torch.zeros(C, RMAX, 3 * D),
                mask=mask)


TRAINABLE = ("emb", "Wd", "Wu", "A0", "B0", "A1", "B1")


def stack(ads):
    return {k: torch.stack([a[k] for a in ads]) for k in ads[0]}


def unstack(ad, c):
    return {k: v[c].detach().clone() for k, v in ad.items()}


def to_t(data, C=None):
    """data = (regs, tabs, i, j, y) numpy -> long tensors, optionally expanded to C copies."""
    out = [torch.from_numpy(np.asarray(a)).long() for a in data]
    if C is not None:
        out = [z.unsqueeze(0).expand(C, *z.shape) for z in out]
    return out


def loss_acc(core, ad, regs, tabs, i, j, y):
    lg = core(ad, regs, tabs, i, j)
    C = lg.shape[0]
    l = torch.nn.functional.cross_entropy(lg.reshape(-1, V), y.reshape(-1), reduction="none").reshape(C, -1).mean(1)
    acc = (lg.argmax(-1) == y).all(-1).float().mean(1)               # whole state correct
    slot_acc = (lg.argmax(-1) == y).float().mean(1)                  # (C,R)
    return l, acc, slot_acc
