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
N_TAGS = 4      # up to four relation tables A, B, C, D (Exp 1-4 use two)
TABLES = "ABCD"
PRIMITIVES = ("A", "B", "Ainv", "Binv", "succ")
# Exp 5 (open-ended growth) draws on the larger skill universe below.
ALL_SKILLS = ("A", "B", "Ainv", "succ", "Binv", "C", "Cinv", "pred", "D", "Dinv", "neg", "triple")


class GeneticCode:
    """Fixed, shared token embeddings. Every monomer reads/writes this space,
    which is what makes monomers interchangeable building blocks.

    Layout of the D=32 residual stream:
      dims 0..V-1   : symbol identity, "key" role (also used for the query symbol)
      dims V..2V-1  : symbol identity, "value" role
      dims 2V, 2V+1 : table tag A / B
      dim  2V+2     : "this is the query" marker
      dims 2V+3,2V+4: table tag C / D (added for Exp 5; unused by 2-table contexts)
      rest          : free scratch space for modules
    Optionally a small random rotation-free jitter is added so the code is not
    perfectly axis-aligned (noise > 0).
    """

    def __init__(self, seed=0, noise=0.0):
        rng = np.random.default_rng(seed)
        I = np.eye(D, dtype=np.float32)
        self.key = I[0:V].copy()
        self.val = I[V:2 * V].copy()
        self.tag = I[[2 * V, 2 * V + 1, 2 * V + 3, 2 * V + 4]].copy()   # tags A,B,C,D
        self.qry = self.key.copy()
        self.qtag = I[2 * V + 2].copy()
        if noise:
            for t in (self.key, self.val, self.tag, self.qtag):
                t += (noise * rng.standard_normal(t.shape)).astype(np.float32)
            self.qry = self.key.copy()


def sample_contexts(rng, B, n_tables=2):
    """B random contexts with n_tables permutation tables. Returns dict of int arrays;
    key/val/tag are (B, T) with T = n_tables*V, f is (B, n_tables, V)."""
    fs = [np.argsort(rng.random((B, V)), axis=1) for _ in range(n_tables)]
    keys = np.tile(np.arange(V), (B, n_tables))
    vals = np.concatenate(fs, axis=1)
    tags = np.repeat(np.arange(n_tables), V)[None].repeat(B, 0)
    perm = np.argsort(rng.random((B, n_tables * V)), axis=1)
    take = lambda a: np.take_along_axis(a, perm, axis=1)
    ctx = dict(key=take(keys), val=take(vals), tag=take(tags), f=np.stack(fs, 1))
    ctx["fA"], ctx["fB"] = fs[0], fs[1]
    return ctx


def apply_primitive(ctx, op, x):
    """Ground-truth primitive on integer symbols x (B,)."""
    b = np.arange(len(x))
    if op in TABLES:
        return ctx["f"][b, TABLES.index(op), x]
    if op.endswith("inv") and op[0] in TABLES:
        return np.argsort(ctx["f"][:, TABLES.index(op[0])], axis=1)[b, x]
    if op == "succ":
        return (x + 1) % V
    if op == "pred":
        return (x - 1) % V
    if op == "neg":
        return (-x) % V
    if op == "triple":
        return (3 * x) % V
    raise ValueError(op)


def make_batch(rng, program, B, n_tables=2):
    """A batch of problems for a program. Returns (ctx, x0, y)."""
    ctx = sample_contexts(rng, B, n_tables)
    x0 = rng.integers(0, V, size=B)
    y = x0
    for op in program:
        y = apply_primitive(ctx, op, y)
    return ctx, x0, y
