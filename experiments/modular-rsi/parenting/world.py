"""Register machine with opcodes that differ in kind (see PREREG.md).

A skill is an OPCODE; each instruction also carries operands (i, j) sampled per example,
so the shared core learns routing for every operand combination and a new skill only
has to supply new opcode semantics."""
import numpy as np

V, R = 8, 3
FAMILIES = ("lookA", "lookB", "invA", "invB", "lookAA", "affine", "add", "sub", "mul", "max", "min", "swap")
TWO_REG = {"add", "sub", "mul", "max", "min", "swap"}


def contexts(rng, B):
    return np.argsort(rng.random((B, 2, V)), axis=2)            # (B, 2 tables, V)


def operands(rng, op, B):
    i = rng.integers(0, R, B)
    j = rng.integers(0, R, B)
    if op[0] in TWO_REG:
        j = np.where(j == i, (i + 1 + rng.integers(0, R - 1, B)) % R, j)
    return i, j


def apply(op, regs, tabs, i, j):
    fam, p = op
    b = np.arange(len(regs))
    out = regs.copy()
    x, xi = regs[b, j], regs[b, i]
    if fam == "lookA":
        v = tabs[b, 0, x]
    elif fam == "lookB":
        v = tabs[b, 1, x]
    elif fam == "invA":
        v = np.argsort(tabs[:, 0], axis=1)[b, x]
    elif fam == "invB":
        v = np.argsort(tabs[:, 1], axis=1)[b, x]
    elif fam == "lookAA":
        v = tabs[b, 0, tabs[b, 0, x]]
    elif fam == "affine":
        v = (p[0] * x + p[1]) % V
    elif fam == "add":
        v = (xi + x) % V
    elif fam == "sub":
        v = (xi - x) % V
    elif fam == "mul":
        v = (xi * x) % V
    elif fam == "max":
        v = np.maximum(xi, x)
    elif fam == "min":
        v = np.minimum(xi, x)
    elif fam == "swap":
        out[b, j] = xi
        v = x
    out[b, i] = v
    return out


def batch(rng, op, B):
    tabs = contexts(rng, B)
    regs = rng.integers(0, V, (B, R))
    i, j = operands(rng, op, B)
    return regs, tabs, i, j, apply(op, regs, tabs, i, j)


def random_op(rng):
    fam = FAMILIES[rng.integers(len(FAMILIES))]
    p = (int(rng.choice([1, 3, 5, 7])), int(rng.integers(0, V))) if fam == "affine" else ()
    return (fam, p)


BASE_OPS = [("lookA", ()), ("invA", ()), ("affine", (3, 1)), ("add", ()), ("swap", ()), ("max", ())]


def change_profile(op, rng, n=512):
    """Cheap behavioural signature: P(target changes), P(other operand changes), and
    the distribution of (new value - old value) mod V at the target register (8 bins)."""
    regs, tabs, i, j, y = batch(rng, op, n)
    b = np.arange(n)
    d = (y[b, i] - regs[b, i]) % V
    return np.concatenate([[(y[b, i] != regs[b, i]).mean(), (y[b, j] != regs[b, j]).mean()],
                           np.bincount(d, minlength=V) / n])
