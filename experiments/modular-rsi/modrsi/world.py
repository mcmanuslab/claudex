"""The shared world: a fixed "genetic code" (token embeddings) and the task family.

Every problem instance is a small *relational context* plus a *query symbol*:

  context  = two random permutation tables over V symbols, written as tagged
             (key -> value) pair tokens, shuffled:  {A:a->f_A(a)} U {B:b->f_B(b)}
  query    = a symbol x (or, inside a polymer, a probability distribution over symbols)

Primitive operations (the monomer "niches"):
  A     : f_A(x)          (look x up in table A)
  B     : f_B(x)          (look x up in table B)
  Ainv  : f_A^{-1}(x)     (reverse lookup in table A)
  Binv  : f_B^{-1}(x)     (reverse lookup in table B)
  succ  : (x + 1) mod V   (context-free arithmetic)

A *program* is a sequence of primitives, e.g. ("A", "succ", "Binv").
The answer to a program is the composition applied to the query. A program of
length L is an L-step reasoning problem over the same context; a single
1-layer monomer can, at best, do one step. Problem size scales with L.
"""
import numpy as np

V = 8           # symbols
D = 32          # residual width (the shared "interface" every module speaks)
N_TAGS = 2      # table tags A, B
PRIMITIVES = ("A", "B", "Ainv", "Binv", "succ")


class GeneticCode:
    """Fixed, shared token embeddings. Every monomer reads/writes this space,
    which is what makes monomers interchangeable building blocks.

    Layout of the D=32 residual stream:
      dims 0..V-1   : symbol identity, "key" role (also used for the query symbol)
      dims V..2V-1  : symbol identity, "value" role
      dims 2V, 2V+1 : table tag A / B
      dim  2V+2     : "this is the query" marker
      rest          : free scratch space for modules
    Optionally a small random rotation-free jitter is added so the code is not
    perfectly axis-aligned (noise > 0).
    """

    def __init__(self, seed=0, noise=0.0):
        rng = np.random.default_rng(seed)
        I = np.eye(D, dtype=np.float32)
        self.key = I[0:V].copy()
        self.val = I[V:2 * V].copy()
        self.tag = I[2 * V:2 * V + N_TAGS].copy()
        self.qry = self.key.copy()
        self.qtag = I[2 * V + N_TAGS].copy()
        if noise:
            for t in (self.key, self.val, self.tag, self.qtag):
                t += (noise * rng.standard_normal(t.shape)).astype(np.float32)
            self.qry = self.key.copy()


def sample_contexts(rng, B):
    """B random contexts. Returns dict of int arrays (B, T) with T = 2V."""
    fA = np.argsort(rng.random((B, V)), axis=1)
    fB = np.argsort(rng.random((B, V)), axis=1)
    keys = np.concatenate([np.tile(np.arange(V), (B, 1))] * 2, axis=1)
    vals = np.concatenate([fA, fB], axis=1)
    tags = np.concatenate([np.zeros((B, V), int), np.ones((B, V), int)], axis=1)
    perm = np.argsort(rng.random((B, 2 * V)), axis=1)
    take = lambda a: np.take_along_axis(a, perm, axis=1)
    return dict(key=take(keys), val=take(vals), tag=take(tags), fA=fA, fB=fB)


def apply_primitive(ctx, op, x):
    """Ground-truth primitive on integer symbols x (B,)."""
    b = np.arange(len(x))
    if op == "A":
        return ctx["fA"][b, x]
    if op == "B":
        return ctx["fB"][b, x]
    if op == "Ainv":
        return np.argsort(ctx["fA"], axis=1)[b, x]
    if op == "Binv":
        return np.argsort(ctx["fB"], axis=1)[b, x]
    if op == "succ":
        return (x + 1) % V
    raise ValueError(op)


def make_batch(rng, program, B):
    """A batch of problems for a program. Returns (ctx, x0, y)."""
    ctx = sample_contexts(rng, B)
    x0 = rng.integers(0, V, size=B)
    y = x0
    for op in program:
        y = apply_primitive(ctx, op, y)
    return ctx, x0, y
