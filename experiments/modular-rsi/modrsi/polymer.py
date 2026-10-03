"""Polymers and communities: how monomers bind to solve bigger problems.

Binding interface. A monomer maps (context, query distribution) -> distribution over
symbols. Because input and output live in the same space, the output of one monomer
can be the query of the next. A *polymer* is an ordered chain of up to LMAX monomers:

    q_0 = onehot(x)      q_{j+1} = monomer_j(context, q_j)      answer = q_L

Two ways to put polymers to work:

  PolymerSearch  - "hidden program" problems. The task is known only through examples.
                   Chains are evolved (point / insert / delete / swap mutations) and,
                   like monomers, their mutation is *steered* by memory: a per-slot
                   profile of which modules the elite polymers used.

  Community      - "instructed" problems. Each example carries its own program
                   (a sequence of instruction tokens). A tiny routing table maps each
                   instruction to a module. Once the routing is learned, the same
                   community solves programs of ANY length n by recruiting n modules:
                   problem size scales with polymer size and nothing is retrained.
"""
import numpy as np
import torch
from .monomer import ctx_tensors, forward_t
from .world import V, PRIMITIVES, make_batch, sample_contexts, apply_primitive

LMAX = 5


@torch.no_grad()
def chain_forward(pool, chains, codeT, ctx, x0):
    """pool (U,N) tensor; chains (C,L) int array, -1 = empty slot -> (C,B,V) probs."""
    cparts = ctx_tensors(ctx)
    C, L = chains.shape
    q = torch.eye(V)[torch.from_numpy(x0)].expand(C, -1, -1).clone()
    for j in range(L):
        act = np.nonzero(chains[:, j] >= 0)[0]
        if len(act):
            q[act] = forward_t(pool[chains[act, j]], codeT, cparts, q[act])
    return q


def chain_fitness(pool, chains, codeT, ctx, x0, y):
    q = chain_forward(pool, chains, codeT, ctx, x0)
    p = q[:, torch.arange(len(y)), torch.from_numpy(y)]
    return torch.log(p + 1e-6).mean(1).numpy(), (q.argmax(-1).numpy() == y).mean(1)


def canon(ch):
    """Drop empty slots (they are identities) and left-pack; keeps chains comparable."""
    out = np.full(LMAX, -1)
    live = ch[ch >= 0]
    out[: len(live)] = live
    return out


class PolymerSearch:
    """Evolve chains of monomers for one hidden-program problem."""

    def __init__(self, pool, codeT, rng, mu=24, lam=200, steered=True):
        self.pool, self.codeT, self.rng = pool, codeT, rng
        self.U = len(pool)
        self.mu, self.lam, self.steered = mu, lam, steered
        self.pop = np.array([canon(self._random_chain()) for _ in range(mu)])
        self.fit = np.full(mu, -np.inf)
        self.evals = 0
        self.history = []

    def _random_chain(self):
        L = self.rng.integers(1, LMAX + 1)
        ch = np.full(LMAX, -1)
        ch[:L] = self.rng.integers(0, self.U, L)
        return ch

    def _profile(self):
        """Per-slot module frequencies among elites (+ smoothing): the polymer path memory."""
        prof = np.ones((LMAX, self.U + 1))
        top = self.pop[np.argsort(-self.fit)[: max(4, self.mu // 3)]]
        for ch in top:
            for j, u in enumerate(ch):
                prof[j, u] += 3.0           # index -1 lands in the last column (= empty)
        return prof / prof.sum(1, keepdims=True)

    def _mutate(self, ch, prof):
        rng, ch = self.rng, ch.copy()
        L = int((ch >= 0).sum())
        op = rng.choice(["point", "insert", "delete", "swap"], p=[0.55, 0.2, 0.15, 0.1])
        draw = lambda j: (rng.choice(self.U + 1, p=prof[j]) if self.steered else rng.integers(0, self.U + 1))
        if op == "point" and L:
            j = rng.integers(0, L)
            u = draw(j)
            ch[j] = -1 if u == self.U else u
        elif op == "insert" and L < LMAX:
            j = rng.integers(0, L + 1)
            u = draw(j)
            u = rng.integers(0, self.U) if u == self.U else u
            ch = np.insert(ch[:L], j, u)
        elif op == "delete" and L > 1:
            ch = np.delete(ch[:L], rng.integers(0, L))
        elif op == "swap" and L > 1:
            i, j = rng.choice(L, 2, replace=False)
            ch[i], ch[j] = ch[j], ch[i]
        out = np.full(LMAX, -1)
        out[: min(len(ch), LMAX)] = ch[:LMAX]
        return canon(out)

    def step(self, program, B=96):
        rng = self.rng
        prof = self._profile()
        par = self.pop[rng.integers(0, self.mu, self.lam)]
        kids = np.array([self._mutate(c, prof) for c in par])
        allc = np.concatenate([self.pop, kids])
        ctx, x0, y = make_batch(rng, program, B)
        f, acc = chain_fitness(self.pool, allc, self.codeT, ctx, x0, y)
        self.evals += len(allc)
        # unique chains only, then truncation
        _, first = np.unique(allc, axis=0, return_index=True)
        order = first[np.argsort(-f[first])][: self.mu]
        self.pop, self.fit = allc[order], f[order]
        self.acc = acc[order]
        rec = dict(best_acc=float(acc[order[0]]), best_fit=float(f[order[0]]),
                   best_chain=self.pop[0].tolist(), evals=self.evals)
        self.history.append(rec)
        return rec

    def run(self, program, gens=40, B=96):
        for _ in range(gens):
            self.step(program, B)
        return self.history[-1]


# --------------------------------------------------------------------------- community
def instructed_batch(rng, n, B, ops=PRIMITIVES):
    """B problems, each with its own random program of length n over `ops`."""
    ctx = sample_contexts(rng, B)
    x0 = rng.integers(0, V, size=B)
    prog = rng.integers(0, len(ops), size=(B, n))
    y = x0.copy()
    for j in range(n):
        for k, op in enumerate(ops):
            m = prog[:, j] == k
            if m.any():
                sub = {key: v[m] for key, v in ctx.items()}
                y[m] = apply_primitive(sub, op, y[m])
    return ctx, x0, prog, y


@torch.no_grad()
def community_forward(pool, router, codeT, ctx, x0, prog):
    """router[k] = pool index for instruction k. Each step routes every example to the
    module its instruction names. -> (B,V) probabilities."""
    q = torch.eye(V)[torch.from_numpy(x0)]
    for j in range(prog.shape[1]):
        mods = router[prog[:, j]]
        nq = q.clone()
        for u in np.unique(mods):
            m = np.nonzero(mods == u)[0]
            sub = {key: v[m] for key, v in ctx.items()}
            nq[m] = forward_t(pool[u:u + 1], codeT, ctx_tensors(sub), q[m])[0]
        q = nq
    return q


def community_accuracy(pool, router, codeT, rng, n, B=256, ops=PRIMITIVES):
    ctx, x0, prog, y = instructed_batch(rng, n, B, ops)
    q = community_forward(pool, router, codeT, ctx, x0, prog)
    return float((q.argmax(-1).numpy() == y).mean())


def evolve_router(pool, codeT, rng, ops=PRIMITIVES, gens=30, popsize=48, B=128, lengths=(1, 2, 3)):
    """Evolve the instruction->module routing table on short instructed programs."""
    U, K = len(pool), len(ops)
    pop = rng.integers(0, U, size=(popsize, K))
    hist = []
    for g in range(gens):
        kids = pop.copy()
        for c in kids:
            c[rng.integers(0, K)] = rng.integers(0, U)
        allr = np.unique(np.concatenate([pop, kids]), axis=0)
        n = int(rng.choice(lengths))
        ctx, x0, prog, y = instructed_batch(rng, n, B, ops)
        acc = np.array([(community_forward(pool, r, codeT, ctx, x0, prog).argmax(-1).numpy() == y).mean()
                        for r in allr])
        pop = allr[np.argsort(-acc)[:popsize]]
        hist.append(float(acc.max()))
    return pop[0], hist
