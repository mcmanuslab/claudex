"""Mutation, lifetime learning, selection and lineage memory for monomer populations.

Every individual is a genome = (weights, improver genes, lineage path):

  weights        ~3.9k transformer parameters (normalised space)
  improver genes  sigma : mutation step size
                  m     : drift along the lineage path      (exploit what worked)
                  a     : extra spread along the lineage path (search *around* it)
                  b     : spread in the elite subspace       (what worked for other lineages)
                  lr    : lifetime-learning rate
  path            exponentially-weighted memory of the steps that led to this
                  individual (like CMA-ES's evolution path), inherited and extended.

One generation (per niche):
  1. VARY     each child copies a parent and mutates:
                 delta/sigma = z + (m + a*g)*kappa*p_hat + b*U^T h
              (z~N(0,I), g~N(0,1), h~N(0,I_k); p_hat = lineage path direction,
               kappa = how consistent that path has been, U = top principal
               directions of recently accepted steps across all lineages)
              Improver genes themselves are mutated (log-normal self-adaptation).
  2. DEVELOP  K steps of lifetime learning (Adam) at the child's own lr. Learned weights
              and optimizer moments are inherited (Lamarckian).
  3. SELECT   (mu + lambda) truncation on held-out fitness (+ credit earned in polymers).
  4. REMEMBER every variant goes into the lineage log; selected children extend their
              parent's path; the elite subspace U is refreshed.

Stress response: when the niche's own progress log shows a plateau (best fitness
up by < STALL_EPS over STALL generations) the improver genes mutate at TAU_STRESS
instead of TAU. The lifetime lr is floored at its default: on a plateau, myopic
selection otherwise drives lr down until the lineage stops learning (observed; see README).

Because the improver genes ride along with the weights they produced, selection acts
on the *process of improvement* as well as on the solution: the RSI loop.
mode="blind" is the control: fixed sigma and lr, isotropic mutation, no path memory.
"""
import numpy as np
import torch
from .monomer import N_PARAMS, CodeT, ctx_tensors, forward_t
from .world import V, make_batch

N = N_PARAMS
C_PATH = 0.25
TAU = 0.2
TAU_STRESS = 0.8      # improver-gene mutation rate while the niche is stagnating
STALL, STALL_EPS = 6, 0.05
LR_FLOOR = 0.02       # = the default lr: the improver can speed learning up but not switch it off
SUB_K, SUB_BUF = 6, 96
GENES = ("sigma", "m", "a", "b", "lr")


class Lineage:
    """Append-only record of every variant ever produced. Weights are only kept
    for elites (checkpoints), so millions of records fit in a few MB."""

    COLS = ("id", "parent", "gen", "niche", "fitness", "acc") + GENES + ("selected",)

    def __init__(self):
        self.rows, self.next_id = [], 0

    def new_ids(self, n):
        ids = np.arange(self.next_id, self.next_id + n)
        self.next_id += n
        return ids

    def add(self, **cols):
        self.rows.append({k: np.asarray(cols[k]) for k in self.COLS})

    def arrays(self):
        return {k: np.concatenate([np.atleast_1d(r[k]) for r in self.rows]) for k in self.COLS}

    def save(self, path):
        np.savez_compressed(path, **self.arrays())


def nll(G, codeT, ctx, x0, y):
    """Per-genome mean negative log-likelihood and accuracy (G: tensor (P,N))."""
    lp = forward_t(G, codeT, ctx_tensors(ctx), torch.eye(V)[torch.from_numpy(x0)], log=True)
    yy = torch.from_numpy(y)
    l = -lp[:, torch.arange(len(y)), yy].mean(1)
    return l, (lp.argmax(-1) == yy).float().mean(1)


def lifetime_learning(G, lr, opt_m, opt_v, codeT, rng, program, K, B, b1=0.9, b2=0.99):
    """K steps of per-individual Adam. Optimizer moments (opt_m, opt_v) are inherited
    from the parent, so a child keeps moving along the direction its lineage was
    already learning in. G, opt_m, opt_v: (P,N) tensors; lr: (P,) array."""
    G, opt_m, opt_v = G.clone(), opt_m.clone(), opt_v.clone()
    lr = torch.from_numpy(lr)[:, None]
    for _ in range(K):
        ctx, x0, y = make_batch(rng, program, B)
        G.requires_grad_(True)
        l, _ = nll(G, codeT, ctx, x0, y)
        (g,) = torch.autograd.grad(l.sum(), G)
        with torch.no_grad():
            opt_m.mul_(b1).add_((1 - b1) * g)
            opt_v.mul_(b2).add_((1 - b2) * g * g)
            G = (G - lr * opt_m / (opt_v.sqrt() + 1e-6)).detach()
    return G, opt_m, opt_v


class Niche:
    def __init__(self, idx, program, rng, lineage, codeT, mu=16, lam=112, mode="steered",
                 K=16, B_learn=64, B_eval=128, init=None):
        self.idx, self.program, self.rng, self.lin, self.codeT = idx, tuple(program), rng, lineage, codeT
        self.mu, self.lam, self.mode, self.K, self.B_learn, self.B_eval = mu, lam, mode, K, B_learn, B_eval
        self.G = torch.from_numpy(rng.standard_normal((mu, N)).astype(np.float32))
        self.path = torch.zeros(mu, N)
        self.opt_m, self.opt_v = torch.zeros(mu, N), torch.zeros(mu, N)
        init = init or dict(sigma=0.02, m=0.0, a=1.0, b=1.0, lr=0.02)
        for k in GENES:
            v = np.full(mu, init[k], np.float32)
            if isinstance(init[k], np.ndarray):            # a seeded improver distribution
                v = rng.choice(init[k], mu).astype(np.float32)
            setattr(self, k, v)
        if mode == "blind":
            self.m[:] = 0; self.a[:] = 0; self.b[:] = 0
        self.ids = lineage.new_ids(mu)
        self.fit = np.zeros(mu, np.float32)
        self.acc = np.zeros(mu, np.float32)
        self.bonus = np.zeros(mu, np.float32)
        self.buf = torch.zeros(0, N)
        self.U = torch.zeros(0, N)
        self.gen = 0
        self.history = []

    # ------------------------------------------------------------------ vary
    def stagnating(self):
        """Lineage memory says: no real progress over the last STALL generations."""
        h = self.history
        if len(h) < STALL:
            return False
        return h[-1]["best_fit"] - h[-STALL]["best_fit"] < STALL_EPS

    def _mutate_genes(self, pidx):
        k, rng = len(pidx), self.rng
        g = {n: getattr(self, n)[pidx].copy() for n in GENES}
        if self.mode == "blind":
            return g
        # stress-induced hypermutation: a stalled niche explores *how it improves*
        tau = TAU_STRESS if self.stagnating() else TAU
        ln = lambda: np.exp(tau * rng.standard_normal(k)).astype(np.float32)
        g["sigma"] = np.clip(g["sigma"] * ln(), 1e-3, 0.5)
        g["lr"] = np.clip(g["lr"] * ln(), LR_FLOOR, 0.5)   # may learn faster, never stop learning
        g["a"] = np.clip(g["a"] * ln(), 0, 20)
        g["b"] = np.clip(g["b"] * ln(), 0, 20)
        g["m"] = np.clip(g["m"] + tau * rng.standard_normal(k).astype(np.float32), -5, 5)
        return g

    def _mutate(self, pidx, g):
        k = len(pidx)
        z = torch.randn(k, N)
        y = z
        if self.mode == "steered":
            p = self.path[pidx]
            prms = p.pow(2).mean(1, keepdim=True).sqrt() + 1e-8          # = kappa
            phat = p / (prms * np.sqrt(N))                               # unit vector
            gg = torch.randn(k, 1)
            coef = torch.from_numpy(g["m"])[:, None] + torch.from_numpy(g["a"])[:, None] * gg
            y = y + coef * prms * np.sqrt(N) * phat                     # kappa*sqrt(N) along path
            if len(self.U):
                h = torch.randn(k, len(self.U))
                y = y + torch.from_numpy(g["b"])[:, None] * (h @ self.U)
        return y * torch.from_numpy(g["sigma"])[:, None]

    # ------------------------------------------------------------------ one generation
    def step(self, bonus_w=0.0):
        rng = self.rng
        pidx = rng.integers(0, self.mu, self.lam)
        g = self._mutate_genes(pidx)
        Gc = self.G[pidx] + self._mutate(pidx, g)
        Gc, om, ov = lifetime_learning(Gc, g["lr"], self.opt_m[pidx], self.opt_v[pidx], self.codeT,
                                       rng, self.program, self.K, self.B_learn)
        ctx, x0, y = make_batch(rng, self.program, self.B_eval)
        with torch.no_grad():
            l, acc = nll(torch.cat([self.G, Gc]), self.codeT, ctx, x0, y)
        f, acc = (-l).numpy(), acc.numpy()
        fp, fc = f[: self.mu], f[self.mu:]
        enrich = float((fc > fp[pidx]).mean())
        cids = self.lin.new_ids(self.lam)
        score = f + bonus_w * np.concatenate([self.bonus, self.bonus[pidx]])
        keep = np.argsort(-score)[: self.mu]
        self.lin.add(id=cids, parent=self.ids[pidx], gen=np.full(self.lam, self.gen),
                     niche=np.full(self.lam, self.idx), fitness=fc, acc=acc[self.mu:],
                     selected=np.isin(np.arange(self.lam) + self.mu, keep), **g)
        # assemble next generation
        newG, newP, newM, newV, acc_steps = [], [], [], [], []
        new = {k: [] for k in GENES + ("ids", "bonus")}
        for j in keep:
            if j < self.mu:
                newG.append(self.G[j]); newP.append(self.path[j])
                newM.append(self.opt_m[j]); newV.append(self.opt_v[j])
                for k in new: new[k].append(getattr(self, k)[j])
            else:
                c, par = j - self.mu, pidx[j - self.mu]
                step = Gc[c] - self.G[par]                               # mutation + learning
                u = step / (step.pow(2).mean().sqrt() + 1e-8)            # unit-RMS direction
                newG.append(Gc[c]); newM.append(om[c]); newV.append(ov[c])
                newP.append((1 - C_PATH) * self.path[par] + np.sqrt(C_PATH * (2 - C_PATH)) * u)
                for k in GENES: new[k].append(g[k][c])
                new["ids"].append(cids[c]); new["bonus"].append(0.0)   # credit must be re-earned
                acc_steps.append(u / np.sqrt(N))
        self.G, self.path = torch.stack(newG), torch.stack(newP)
        self.opt_m, self.opt_v = torch.stack(newM), torch.stack(newV)
        for k, v in new.items():
            setattr(self, k, np.array(v))
        self.fit, self.acc = f[keep], acc[keep]
        if self.mode == "steered" and acc_steps:
            self.buf = torch.cat([torch.stack(acc_steps), self.buf])[:SUB_BUF]
            if len(self.buf) >= 2 * SUB_K:
                self.U = torch.linalg.svd(self.buf, full_matrices=False)[2][:SUB_K]
        rec = dict(gen=self.gen, niche=self.idx, stressed=bool(self.stagnating()), best_acc=float(acc[keep].max()),
                   mean_acc=float(acc[keep].mean()), best_fit=float(f[keep[0]]), enrich=enrich,
                   **{k: float(np.median(getattr(self, k))) for k in GENES})
        self.history.append(rec)
        self.gen += 1
        return rec

    def elites(self, k):
        o = np.argsort(-self.fit)[:k]
        return self.G[o], self.ids[o], self.acc[o]
