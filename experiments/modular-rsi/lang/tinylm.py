"""Tiny character-level transformer, batched over P independent models (same trick as
the monomers: one genome per model, so populations can be trained/evolved together).

Embedding is tied with the read-out. One layer, single head, causal attention over a
context of C characters, optional MLP. Parameter count is tiny by design; `bytes` below
assumes int8 weights (1 byte per parameter)."""
import numpy as np
import torch

V = 28


def layout(d, C, mlp):
    segs = [("emb", (V, d)), ("pos", (C, d)), ("Wq", (d, d)), ("Wk", (d, d)), ("Wv", (d, d)), ("Wo", (d, d))]
    if mlp:
        segs += [("W1", (d, mlp)), ("b1", (mlp,)), ("W2", (mlp, d))]
    segs += [("bout", (V,))]
    out, o = {}, 0
    for n, s in segs:
        k = int(np.prod(s)); out[n] = (o, o + k, s); o += k
    return out, o


class TinyLM:
    def __init__(self, d=8, C=16, mlp=0):
        self.d, self.C, self.mlp = d, C, mlp
        self.L, self.n = layout(d, C, mlp)
        sc = np.empty(self.n, np.float32)
        for k, (a, b, s) in self.L.items():
            sc[a:b] = (1 / np.sqrt(s[0])) if k[0] == "W" else (0.3 if k in ("emb", "pos") else 0.1)
        self.scale = torch.from_numpy(sc)

    def init(self, P, rng):
        return torch.from_numpy(rng.standard_normal((P, self.n)).astype(np.float32))

    def unpack(self, G):
        W = G * self.scale
        return {k: W[:, a:b].reshape((len(G),) + s) for k, (a, b, s) in self.L.items()}

    def logits(self, G, x):
        """G (P,n); x (B,T) int with T<=C -> (P,B,T,V) next-char logits."""
        w = self.unpack(G)
        P, (B, T) = len(G), x.shape
        h = w["emb"][:, x] + w["pos"][:, None, :T]                          # (P,B,T,d)
        hn = h / (h.pow(2).mean(-1, keepdim=True) + 1e-5).sqrt()
        q, k, v = (hn @ w[n][:, None] for n in ("Wq", "Wk", "Wv"))
        att = (q @ k.transpose(-1, -2)) / self.d ** 0.5
        att = att.masked_fill(torch.triu(torch.ones(T, T, dtype=torch.bool), 1), float("-inf"))
        h = h + (torch.softmax(att, -1) @ v) @ w["Wo"][:, None]
        if self.mlp:
            hn = h / (h.pow(2).mean(-1, keepdim=True) + 1e-5).sqrt()
            h = h + torch.relu(hn @ w["W1"][:, None] + w["b1"][:, None, None]) @ w["W2"][:, None]
        hn = h / (h.pow(2).mean(-1, keepdim=True) + 1e-5).sqrt()
        return hn @ w["emb"].transpose(-1, -2)[:, None] + w["bout"][:, None, None]

    def loss(self, G, x, y):
        lg = self.logits(G, x)
        return torch.nn.functional.cross_entropy(lg.reshape(-1, V), y.repeat(len(G), 1, 1).reshape(-1),
                                                 reduction="none").reshape(len(G), -1).mean(1)

    @torch.no_grad()
    def sample(self, G1, n, rng, prompt=None, temp=0.8):
        ids = list(prompt if prompt is not None else [0])
        for _ in range(n):
            x = torch.tensor([ids[-self.C:]])
            p = torch.softmax(self.logits(G1[None], x)[0, 0, -1] / temp, -1).numpy()
            ids.append(int(rng.choice(V, p=p / p.sum())))
        return ids


def batches(ids, B, C, rng):
    i = rng.integers(0, len(ids) - C - 1, B)
    x = np.stack([ids[j:j + C] for j in i]); y = np.stack([ids[j + 1:j + C + 1] for j in i])
    return torch.from_numpy(x), torch.from_numpy(y)
