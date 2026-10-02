"""Models for the composition test. All chained systems use the same Block, so modular vs
looped differ only in how parameters are organised (12 small blocks vs 1 big shared one)."""
import math
import numpy as np
import torch
import torch.nn as nn
from modrsi.world import V, D, N_TAGS

N_OPS = 12


def ctx_parts(ctx):
    eye_v, eye_t = torch.eye(V), torch.eye(N_TAGS)
    k, v, t = (torch.from_numpy(ctx[n]).long() for n in ("key", "val", "tag"))
    return [(k, eye_v[k]), (v, eye_v[v]), (t, eye_t[t])]


def _rms(x):
    return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + 1e-5)


class Block(nn.Module):
    """n_mod one-layer attention blocks with stacked parameters; each example uses the block
    its index selects (modular: index = instruction; looped: n_mod=1, shared block plus an
    instruction embedding). Same math as separate modules, one batched pass."""

    def __init__(self, code, n_mod=1, H=2, DK=8, M=24, n_ops=0):
        super().__init__()
        self.H, self.DK, self.n_mod = H, DK, n_mod
        f = lambda *s: nn.Parameter(torch.randn(n_mod, *s) / math.sqrt(s[0]))
        self.Wq, self.Wk, self.Wv, self.Wo = f(D, H * DK), f(D, H * DK), f(D, H * DK), f(H * DK, D)
        self.W1, self.b1, self.W2 = f(D, M), nn.Parameter(torch.zeros(n_mod, M)), f(M, D)
        self.Wr, self.br = f(D, V), nn.Parameter(torch.zeros(n_mod, V))
        self.op = nn.Parameter(torch.randn(n_ops, D) * 0.3) if n_ops else None
        for n in ("key", "val", "tag", "qry", "qtag"):
            self.register_buffer(n, torch.from_numpy(np.asarray(getattr(code, n), np.float32)))

    def forward(self, parts, q, op):
        """parts: ctx_parts (B contexts); q (B,V) symbol distribution; op (B,) long -> (B,V) logits."""
        if self.n_mod == 1:
            return self._shared(parts, q, op)
        H, DK = self.H, self.DK
        mi = op if self.n_mod > 1 else torch.zeros_like(op)
        xq = q @ self.qry + self.qtag
        if self.op is not None:
            xq = xq + self.op[op]
        B = xq.shape[0]
        qh = torch.einsum("bd,bde->be", xq, self.Wq[mi]).reshape(B, H, DK)
        Wk, Wv = self.Wk[mi], self.Wv[mi]                                  # (B,D,HD)
        score, vals = 0.0, []
        for table, (idx, oh) in zip((self.key, self.val, self.tag), parts):
            K = torch.einsum("nd,bde->bne", table, Wk).reshape(B, -1, H, DK)
            vals.append(torch.einsum("nd,bde->bne", table, Wv).reshape(B, -1, H, DK))
            s_sym = torch.einsum("bhk,bnhk->bhn", qh, K)
            score = score + torch.gather(s_sym, 2, idx[:, None, :].expand(B, H, -1))
        att = torch.softmax(score / DK ** 0.5, -1)                           # (B,H,T)
        o = 0.0
        for (idx, oh), Vv in zip(parts, vals):
            mass = torch.einsum("bht,btn->bhn", att, oh)
            o = o + torch.einsum("bhn,bnhk->bhk", mass, Vv)
        r = xq + torch.einsum("be,bed->bd", o.reshape(B, H * DK), self.Wo[mi])
        h = torch.relu(torch.einsum("bd,bdm->bm", _rms(r), self.W1[mi]) + self.b1[mi])
        r = r + torch.einsum("bm,bmd->bd", h, self.W2[mi])
        return torch.einsum("bd,bdv->bv", _rms(r), self.Wr[mi]) + self.br[mi]


    def _shared(self, parts, q, op):
        """Same computation for a single shared block, without per-example weight copies."""
        H, DK = self.H, self.DK
        w = {n: getattr(self, n)[0] for n in ("Wq", "Wk", "Wv", "Wo", "W1", "b1", "W2", "Wr", "br")}
        xq = q @ self.qry + self.qtag
        if self.op is not None:
            xq = xq + self.op[op]
        B = xq.shape[0]
        qh = (xq @ w["Wq"]).reshape(B, H, DK)
        score, vals = 0.0, []
        for table, (idx, oh) in zip((self.key, self.val, self.tag), parts):
            K = (table @ w["Wk"]).reshape(-1, H, DK)
            vals.append((table @ w["Wv"]).reshape(-1, H, DK))
            s_sym = torch.einsum("bhk,nhk->bhn", qh, K)
            score = score + torch.gather(s_sym, 2, idx[:, None, :].expand(B, H, -1))
        att = torch.softmax(score / DK ** 0.5, -1)
        o = 0.0
        for (idx, oh), Vv in zip(parts, vals):
            o = o + torch.einsum("bhn,nhk->bhk", torch.einsum("bht,btn->bhn", att, oh), Vv)
        r = xq + o.reshape(B, H * DK) @ w["Wo"]
        r = r + torch.relu(_rms(r) @ w["W1"] + w["b1"]) @ w["W2"]
        return _rms(r) @ w["Wr"] + w["br"]


class Chain(nn.Module):
    """Runs a program step by step through the shared symbol interface. kind='modular':
    12 blocks, step j uses block[op_j]; 'looped': one big shared block conditioned on op_j.
    prog (B,L) with -1 = no-op padding."""

    def __init__(self, code, kind):
        super().__init__()
        self.kind = kind
        self.block = (Block(code, n_mod=N_OPS) if kind == "modular"
                      else Block(code, n_mod=1, H=8, DK=16, M=470, n_ops=N_OPS))

    def forward(self, parts, x0, prog, hard=False):
        q = torch.eye(V)[torch.from_numpy(x0)]
        logits = None
        for j in range(prog.shape[1]):
            ops = prog[:, j]
            live = np.nonzero(ops >= 0)[0]
            if not len(live):
                break
            li = torch.from_numpy(live)
            sub = [(i[li], oh[li]) for i, oh in parts]
            lg_live = self.block(sub, q[li], torch.from_numpy(ops[live]).long())
            logits = (torch.zeros(len(x0), V) if logits is None else logits).index_put((li,), lg_live)
            p = torch.softmax(lg_live, -1)
            if hard:
                p = torch.eye(V)[p.argmax(-1)]
            q = q.index_put((li,), p)
        return logits


class ProgramTransformer(nn.Module):
    """Standard pre-LN transformer over [context tokens | instruction tokens | query]."""

    def __init__(self, code, d=32, L=4, heads=4, ff=128, max_len=64):
        super().__init__()
        for n in ("key", "val", "tag", "qry", "qtag"):
            self.register_buffer(n, torch.from_numpy(np.asarray(getattr(code, n), np.float32)))
        self.op = nn.Embedding(N_OPS + 1, d, padding_idx=N_OPS)
        pe = torch.zeros(max_len, d)
        pos = torch.arange(max_len)[:, None]
        div = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer("pe", pe)
        layer = nn.TransformerEncoderLayer(d, heads, ff, dropout=0.0, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, L, enable_nested_tensor=False)
        self.out = nn.Linear(d, V)

    def forward(self, ctx, x0, prog, hard=False):
        k, v, t = (torch.from_numpy(ctx[n]).long() for n in ("key", "val", "tag"))
        ctx_tok = self.key[k] + self.val[v] + self.tag[t]                       # (B,T,32)
        p = torch.from_numpy(np.where(prog < 0, N_OPS, prog)).long()
        op_tok = self.op(p) + self.pe[: p.shape[1]]
        q_tok = (self.qry[torch.from_numpy(x0)] + self.qtag)[:, None]
        X = torch.cat([ctx_tok, op_tok, q_tok], 1)
        pad = torch.cat([torch.zeros(k.shape, dtype=torch.bool), p == N_OPS,
                         torch.zeros(len(x0), 1, dtype=torch.bool)], 1)
        return self.out(self.enc(X, src_key_padding_mask=pad)[:, -1])


def n_params(m):
    return sum(p.numel() for p in m.parameters())
