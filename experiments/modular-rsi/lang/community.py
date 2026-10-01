"""A community of tiny char-level transformer modules that write text together.

Each module is a TinyLM (a few hundred bytes). Two ways to combine K modules:

  mix  - a router picks who speaks: p(next) = sum_k g_k(context) p_k(next), with the
         gate g computed from the last character (a 28 x K table, i.e. a few bytes per module).
  vote - every module speaks: logits(next) = sum_k logits_k(next)  (product of experts).

Both are plain tensors over a stacked genome (K, n), so whole communities - and
populations of candidate communities - train in one batched call.
"""
import numpy as np
import torch
from tinylm import TinyLM, V


class Community:
    def __init__(self, lm: TinyLM, mode="mix"):
        self.lm, self.mode = lm, mode

    def logprobs(self, G, R, x):
        """G (K,n) module genomes; R (V,K) router table; x (B,T) -> (B,T,V) log-probs."""
        lg = self.lm.logits(G, x)                                   # (K,B,T,V)
        if self.mode == "vote":
            return torch.log_softmax(lg.sum(0), -1)
        lp = torch.log_softmax(lg, -1)
        gate = torch.log_softmax(R[x], -1).permute(2, 0, 1)[..., None]   # (K,B,T,1)
        return torch.logsumexp(lp + gate, 0)

    def loss(self, G, R, x, y):
        lp = self.logprobs(G, R, x)
        return -lp.gather(-1, y[..., None]).mean()

    @torch.no_grad()
    def sample(self, G, R, n, rng, temp=0.8):
        ids = [0]
        for _ in range(n):
            x = torch.tensor([ids[-self.lm.C:]])
            p = torch.exp(self.logprobs(G, R, x)[0, -1] / temp).numpy()
            ids.append(int(rng.choice(V, p=p / p.sum())))
        return ids

    def bytes(self, K):
        return K * self.lm.n + (V * K if self.mode == "mix" else 0)
