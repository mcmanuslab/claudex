"""Hierarchical polymerization: polymers that work are frozen into new units ("macros")
and become building blocks for the next cycle. This is the compounding step.

Hard interface. Between units the symbol is passed as an argmax (one-hot), i.e. the
interface is digital: small per-step errors do not accumulate as blur. On a given
context every unit (monomer or macro) is then a map  V -> V , which we call its table.
  - a monomer's table is computed by actually running the transformer on all V queries;
  - a macro's table is the composition of its children's tables (exactly what running
    the chain would compute, at the cost of <=5 table look-ups instead of re-running
    every monomer inside it). This is the compute-reuse that makes depth cheap.
`verify_by_execution` re-runs a macro monomer-by-monomer to check the claim.

Problems grow by a hierarchical curriculum: each new problem is a concatenation of 2-3
problems from the previous level (the system is not told the decomposition; it sees
only (context, query) -> answer examples). Problem length therefore grows roughly
geometrically, and the question is whether a fixed per-cycle search budget keeps up.
"""
import numpy as np
import torch
from .monomer import forward_t, ctx_tensors
from .world import V, apply_primitive, sample_contexts

SLOTS = 5


class Unit:
    def __init__(self, kind, genome=None, children=None, name="", length=1, cycle=0):
        self.kind, self.genome, self.children = kind, genome, children
        self.name, self.length, self.cycle = name, length, cycle


def rep_ctx(ctx):
    return {k: np.repeat(v, V, axis=0) for k, v in ctx.items()}


@torch.no_grad()
def unit_tables(units, codeT, ctx):
    """(U, B, V) int tables for every unit on B contexts."""
    B = len(ctx["key"])
    rc = ctx_tensors(rep_ctx(ctx))
    q = torch.eye(V)[torch.arange(V).repeat(B)]
    T = np.zeros((len(units), B, V), np.int64)
    mono = [i for i, u in enumerate(units) if u.kind == "monomer"]
    if mono:
        G = torch.stack([units[i].genome for i in mono])
        out = forward_t(G, codeT, rc, q).argmax(-1).numpy().reshape(len(mono), B, V)
        T[mono] = out
    for i, u in enumerate(units):
        if u.kind == "macro":
            T[i] = compose(T, u.children, B)
    return T


def compose(T, chain, B):
    t = np.tile(np.arange(V), (B, 1))
    for c in chain:
        if c >= 0:
            t = np.take_along_axis(T[c], t, axis=1)
    return t


def chains_tables(T, chains):
    """chains (C, SLOTS) -> (C, B, V) tables."""
    C, B = len(chains), T.shape[1]
    t = np.broadcast_to(np.arange(V), (C, B, V)).copy()
    for j in range(chains.shape[1]):
        c = chains[:, j]
        live = c >= 0
        if live.any():
            t[live] = np.take_along_axis(T[c[live]], t[live], axis=2)
    return t


def target_table(ctx, program):
    B = len(ctx["key"])
    y = np.tile(np.arange(V), (B, 1)).reshape(-1)
    rc = rep_ctx(ctx)
    for op in program:
        y = apply_primitive(rc, op, y)
    return y.reshape(B, V)


def search(units, codeT, program, rng, prior, budget=6000, lam=160, mu=32, B=48):
    """Evolve chains of <=SLOTS units whose composed table matches the target.
    Mutation draws units from `prior` (lineage memory: recently formed units are the
    likeliest building blocks of new problems) blended with the elite slot profile."""
    U = len(units)
    pop = np.full((mu, SLOTS), -1)
    for r in pop:
        L = rng.integers(1, SLOTS + 1); r[:L] = rng.choice(U, L, p=prior)
    fit, evals = np.zeros(mu), 0
    while evals < budget:
        ctx = sample_contexts(rng, B)
        T = unit_tables(units, codeT, ctx)
        y = target_table(ctx, program)
        prof = np.ones((SLOTS, U)) * 0.5 + prior[None] * U
        for ch in pop[np.argsort(-fit)[: mu // 4]]:
            for j, u in enumerate(ch):
                if u >= 0: prof[j, u] += 2
        prof /= prof.sum(1, keepdims=True)
        kids = pop[rng.integers(0, mu, lam)].copy()
        for k in kids:
            L = int((k >= 0).sum()); op = rng.random()
            if op < 0.5 and L:
                j = rng.integers(0, L); k[j] = rng.choice(U, p=prof[j])
            elif op < 0.7 and L < SLOTS:
                j = rng.integers(0, L + 1); k[:] = np.insert(k[:L], j, rng.choice(U, p=prof[j]))[:SLOTS].tolist() + [-1] * (SLOTS - L - 1)
            elif op < 0.85 and L > 1:
                j = rng.integers(0, L); k[:] = np.concatenate([np.delete(k[:L], j), [-1] * (SLOTS - L + 1)])
            else:
                L2 = rng.integers(1, SLOTS + 1); k[:] = -1; k[:L2] = rng.choice(U, L2, p=prior)
        allc = np.unique(np.concatenate([pop, kids]), axis=0)
        f = (chains_tables(T, allc) == y[None]).mean((1, 2))
        evals += len(allc)
        o = np.argsort(-f)[:mu]
        pop, fit = allc[o], f[o]
        if fit[0] >= 0.999:
            break
    return pop[0], float(fit[0]), evals


@torch.no_grad()
def verify_by_execution(units, unit_idx, codeT, ctx, program, max_len=2000):
    """Actually run the macro monomer-by-monomer (hard interface) on B contexts."""
    seq = []

    def flat(i):
        u = units[i]
        if u.kind == "monomer":
            seq.append(i)
        else:
            for c in u.children:
                if c >= 0: flat(c)
    flat(unit_idx)
    if len(seq) > max_len:
        return None, len(seq)
    B = len(ctx["key"])
    rng = np.random.default_rng(0)
    x0 = rng.integers(0, V, B)
    cp = ctx_tensors(ctx)
    x = x0.copy()
    for i in seq:
        q = torch.eye(V)[torch.from_numpy(x)]
        x = forward_t(units[i].genome[None], codeT, cp, q)[0].argmax(-1).numpy()
    y = x0.copy()
    for op in program:
        y = apply_primitive(ctx, op, y)
    return float((x == y).mean()), len(seq)
