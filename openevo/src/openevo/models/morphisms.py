"""Architecture mutation operators: function-preserving growth, least-damage shrinkage.

Two properties matter here, and both are enforced by construction and checked by tests.

**1. Growth is exactly function-preserving at ``noise=0``.**
Each growth operator adds capacity whose contribution to the output is initially zero,
but whose *gradient* is non-zero, so the new capacity is trainable rather than dead.
The trick differs per operator: new attention heads get zeroed rows of ``Wo``; new FFN
units get zeroed rows of ``W2``; new blocks get both zeroed (an exact residual
identity); new residual channels get zeroed RMSNorm gains, with the surviving gains
rescaled by ``sqrt(d/d')`` so that RMS normalisation over the wider vector reproduces
the narrower one exactly. A small ``noise`` is used in practice to break symmetry, which
makes growth *approximately* function-preserving in a controlled, measurable way.

**2. Grow and shrink are exact inverses on the dimension ladder.**
This is the part that protects the central scientific claim. If ``widen`` multiplies a
dimension by 1.25 and ``narrow`` divides by 1.25, integer rounding makes the two steps
unequal in log space (22 -> 28 is +0.241 nats, 22 -> 18 is only -0.201), so a *neutral*
population with ``p_grow == p_shrink`` would drift towards larger models and manufacture
exactly the "spontaneous complexity growth" this study is trying to detect. Instead each
dimension lives on a fixed geometric **ladder** of integers and mutations move +-1 rung.
The walk on rung indices is then provably unbiased, and ``shrink(grow(x)) == x``
identically. Any residual bias in *parameter count* (a nonlinear function of the
dimensions) is measured, not assumed -- see the neutral-drift control.
"""

from __future__ import annotations

import re
from dataclasses import replace

import numpy as np

from .genome import BOUNDS, ArchGenome
from .transformer import DTYPE, Params

GROWTH_OPS = ("widen", "add_head", "widen_head", "expand_ffn", "add_block")
SHRINK_OPS = ("narrow", "remove_head", "narrow_head", "contract_ffn", "remove_block")
STRUCTURAL_OPS = GROWTH_OPS + SHRINK_OPS
# Which ladder each op moves, and in which direction.
OP_FIELD: dict[str, tuple[str, int]] = {
    "widen": ("d_model", +1), "narrow": ("d_model", -1),
    "add_head": ("n_heads", +1), "remove_head": ("n_heads", -1),
    "widen_head": ("head_dim", +1), "narrow_head": ("head_dim", -1),
    "expand_ffn": ("d_ff", +1), "contract_ffn": ("d_ff", -1),
    "add_block": ("n_blocks", +1), "remove_block": ("n_blocks", -1),
}


def _geometric_ladder(lo: int, hi: int, ratio: float, *, even: bool = False) -> list[int]:
    vals, v = [], float(lo)
    while True:
        x = int(round(v))
        if even and x % 2:
            x += 1
        x = max(lo, min(hi, x))
        if not vals or x > vals[-1]:
            vals.append(x)
        if x >= hi:
            break
        v = max(v * ratio, v + 1.0)
    return vals


LADDERS: dict[str, list[int]] = {
    "d_model": _geometric_ladder(*BOUNDS["d_model"], 1.25),
    "n_heads": _geometric_ladder(*BOUNDS["n_heads"], 1.5),
    "head_dim": _geometric_ladder(*BOUNDS["head_dim"], 1.5, even=True),
    "d_ff": _geometric_ladder(*BOUNDS["d_ff"], 1.25),
    "n_blocks": _geometric_ladder(*BOUNDS["n_blocks"], 1.5),
}


def snap(field: str, value: int) -> int:
    """Nearest rung, so seeded/hand-written genomes live on the ladder too."""
    ladder = LADDERS[field]
    return min(ladder, key=lambda v: (abs(v - value), v))


def rung(field: str, value: int) -> int:
    return LADDERS[field].index(snap(field, value))


def snap_genome(g: ArchGenome) -> ArchGenome:
    return replace(g, **{f: snap(f, getattr(g, f)) for f in LADDERS})


def step_field(g: ArchGenome, field: str, direction: int) -> ArchGenome | None:
    """Move `field` one rung. Returns None if the move leaves the ladder."""
    ladder = LADDERS[field]
    i = rung(field, getattr(g, field)) + direction
    if not 0 <= i < len(ladder):
        return None
    cand = replace(g, **{field: ladder[i]})
    return cand if cand.valid() else None


# --------------------------------------------------------------- tensor surgery

def _pad_axis(a: np.ndarray, axis: int, new: int, fill: np.ndarray | float) -> np.ndarray:
    """Grow `axis` to `new`, filling with `fill` (scalar or correctly-shaped array)."""
    old = a.shape[axis]
    if new <= old:
        idx = [slice(None)] * a.ndim
        idx[axis] = slice(0, new)
        return np.ascontiguousarray(a[tuple(idx)])
    shape = list(a.shape)
    shape[axis] = new - old
    block = (np.broadcast_to(fill, shape) if np.isscalar(fill) or fill.shape != tuple(shape)
             else fill)
    return np.concatenate([a, np.asarray(block, dtype=DTYPE)], axis=axis).astype(DTYPE)


def _take(a: np.ndarray, axis: int, keep: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.take(a, keep, axis=axis))


_BLOCK_KEY = re.compile(r"^b\d+\.")


def _is_block_key(k: str) -> bool:
    """True only for per-block tensors. Guards against matching ``b_head``."""
    return _BLOCK_KEY.match(k) is not None


def _break_scale(noise: float) -> float:
    """Symmetry-breaking magnitude for newly created capacity.

    Zero means zero: ``noise=0`` must give *exactly* function-preserving growth so the
    property is testable. Above zero we impose a small floor, because capacity that is
    perfectly symmetric with its neighbours can stay degenerate for a long time.
    """
    return 0.0 if noise <= 0 else max(noise, 1e-3)


def _noise(rng: np.random.Generator, shape: tuple[int, ...], scale: float) -> np.ndarray:
    if scale <= 0:
        return np.zeros(shape, dtype=DTYPE)
    return rng.normal(0.0, scale, size=shape).astype(DTYPE)


def _head_slices(g: ArchGenome, keep_heads: np.ndarray) -> np.ndarray:
    """Column indices into a d_attn axis that survive keeping `keep_heads`."""
    return np.concatenate(
        [np.arange(h * g.head_dim, (h + 1) * g.head_dim) for h in keep_heads]
    )


def apply_morphism(
    p: Params, g: ArchGenome, op: str, rng: np.random.Generator, *, noise: float = 0.0
) -> tuple[Params, ArchGenome] | None:
    """Apply a structural mutation. Returns (new_params, new_genome), or None if invalid.

    `p` carries a leading organism axis (size 1 for a single organism); all surgery
    happens on the trailing axes so the same code serves both cases.
    """
    field, direction = OP_FIELD[op]
    g2 = step_field(g, field, direction)
    if g2 is None:
        return None
    q: Params = {k: v.copy() for k, v in p.items()}
    L = g.n_blocks

    # ---------------------------------------------------------------- d_model
    if field == "d_model":
        d, d2 = g.d_model, g2.d_model
        if d2 > d:  # ---- widen: new residual channels are gated off by zero gains
            s = float(np.sqrt(d / d2))
            for nm in ("E_obs", "E_act", "E_rew"):
                q[nm] = _pad_axis(q[nm], -1, d2, _noise(rng, (1,), noise))
            for nm in [f"b{b}.g1" for b in range(L)] + [f"b{b}.g2" for b in range(L)] + ["g_final"]:
                q[nm] = _pad_axis(q[nm] * s, -1, d2, 0.0)
            for b in range(L):
                for nm in ("Wq", "Wk", "Wv"):
                    q[f"b{b}.{nm}"] = _pad_axis(q[f"b{b}.{nm}"], -2, d2, _noise(rng, (1,), noise))
                q[f"b{b}.Wo"] = _pad_axis(q[f"b{b}.Wo"], -1, d2, 0.0)
                q[f"b{b}.W1"] = _pad_axis(q[f"b{b}.W1"], -2, d2, _noise(rng, (1,), noise))
                q[f"b{b}.W2"] = _pad_axis(q[f"b{b}.W2"], -1, d2, 0.0)
            q["W_head"] = _pad_axis(q["W_head"], -2, d2, 0.0)
        else:  # ---- narrow: drop the channels with the least influence
            score = np.zeros(d)
            for b in range(L):
                for nm in ("Wq", "Wk", "Wv", "W1"):
                    score += np.linalg.norm(q[f"b{b}.{nm}"][0], axis=-1)  # read-out rows
                score += np.linalg.norm(q[f"b{b}.Wo"][0], axis=0)         # write-in cols
                score += np.linalg.norm(q[f"b{b}.W2"][0], axis=0)
            score += np.linalg.norm(q["W_head"][0], axis=-1)
            keep = np.sort(np.argsort(-score)[:d2])
            s = float(np.sqrt(d / d2))
            for nm in ("E_obs", "E_act", "E_rew"):
                q[nm] = _take(q[nm], -1, keep)
            for nm in [f"b{b}.g1" for b in range(L)] + [f"b{b}.g2" for b in range(L)] + ["g_final"]:
                q[nm] = _take(q[nm] * s, -1, keep)
            for b in range(L):
                for nm in ("Wq", "Wk", "Wv", "W1"):
                    q[f"b{b}.{nm}"] = _take(q[f"b{b}.{nm}"], -2, keep)
                q[f"b{b}.Wo"] = _take(q[f"b{b}.Wo"], -1, keep)
                q[f"b{b}.W2"] = _take(q[f"b{b}.W2"], -1, keep)
            q["W_head"] = _take(q["W_head"], -2, keep)

    # ------------------------------------------------- n_heads / head_dim (d_attn)
    elif field in ("n_heads", "head_dim"):
        da2 = g2.d_attn
        if da2 > g.d_attn:
            if field == "n_heads":  # new heads appended: zero Wo rows -> zero output
                for nm in ("Wq", "Wk", "Wv"):
                    for b in range(L):
                        q[f"b{b}.{nm}"] = _pad_axis(
                            q[f"b{b}.{nm}"], -1, da2, _noise(rng, (1,), _break_scale(noise))
                        )
                for b in range(L):
                    q[f"b{b}.Wo"] = _pad_axis(q[f"b{b}.Wo"], -2, da2, 0.0)
            else:
                # head_dim grows: interleave new dims inside every head. Zero the new
                # Wo rows; also rescale q by sqrt(hd2/hd) because attention divides by
                # sqrt(head_dim) and the added dims contribute nothing to the scores.
                hd, hd2, H = g.head_dim, g2.head_dim, g.n_heads
                scale = float(np.sqrt(hd2 / hd))
                for b in range(L):
                    for nm in ("Wq", "Wk", "Wv"):
                        w = q[f"b{b}.{nm}"].reshape(*q[f"b{b}.{nm}"].shape[:-1], H, hd)
                        w = _pad_axis(w, -1, hd2, _noise(rng, (1,), _break_scale(noise)))
                        if nm == "Wq":
                            w = w.copy()
                            w[..., :hd] *= scale
                            w[..., hd:] = 0.0  # keep scores exact at noise=0
                        q[f"b{b}.{nm}"] = w.reshape(*w.shape[:-2], H * hd2)
                    wo = q[f"b{b}.Wo"].reshape(q[f"b{b}.Wo"].shape[0], H, hd, -1)
                    wo = _pad_axis(wo, -2, hd2, 0.0)
                    q[f"b{b}.Wo"] = wo.reshape(wo.shape[0], H * hd2, -1)
        else:
            if field == "n_heads":  # drop the heads contributing least through Wo
                contrib = np.zeros(g.n_heads)
                for b in range(L):
                    wo = q[f"b{b}.Wo"][0].reshape(g.n_heads, g.head_dim, -1)
                    contrib += np.linalg.norm(wo, axis=(1, 2))
                keep = np.sort(np.argsort(-contrib)[: g2.n_heads])
                cols = _head_slices(g, keep)
                for b in range(L):
                    for nm in ("Wq", "Wk", "Wv"):
                        q[f"b{b}.{nm}"] = _take(q[f"b{b}.{nm}"], -1, cols)
                    q[f"b{b}.Wo"] = _take(q[f"b{b}.Wo"], -2, cols)
            else:
                hd, hd2, H = g.head_dim, g2.head_dim, g.n_heads
                scale = float(np.sqrt(hd2 / hd))
                for b in range(L):
                    keepd = None
                    for nm in ("Wq", "Wk", "Wv"):
                        w = q[f"b{b}.{nm}"].reshape(*q[f"b{b}.{nm}"].shape[:-1], H, hd)
                        if keepd is None:
                            wo = q[f"b{b}.Wo"][0].reshape(H, hd, -1)
                            keepd = np.sort(np.argsort(-np.linalg.norm(wo, axis=(0, 2)))[:hd2])
                        w = _take(w, -1, keepd)
                        if nm == "Wq":
                            w = w * scale
                        q[f"b{b}.{nm}"] = w.reshape(*w.shape[:-2], H * hd2)
                    wo = q[f"b{b}.Wo"].reshape(q[f"b{b}.Wo"].shape[0], H, hd, -1)
                    wo = _take(wo, -2, keepd)
                    q[f"b{b}.Wo"] = wo.reshape(wo.shape[0], H * hd2, -1)

    # ----------------------------------------------------------------- d_ff
    elif field == "d_ff":
        f, f2 = g.d_ff, g2.d_ff
        if f2 > f:  # new hidden units: zero W2 rows -> zero output, non-zero gradient
            for b in range(L):
                q[f"b{b}.W1"] = _pad_axis(q[f"b{b}.W1"], -1, f2,
                                          _noise(rng, (1,), _break_scale(noise)))
                q[f"b{b}.W2"] = _pad_axis(q[f"b{b}.W2"], -2, f2, 0.0)
        else:
            for b in range(L):
                keep = np.sort(np.argsort(-np.linalg.norm(q[f"b{b}.W2"][0], axis=-1))[:f2])
                q[f"b{b}.W1"] = _take(q[f"b{b}.W1"], -1, keep)
                q[f"b{b}.W2"] = _take(q[f"b{b}.W2"], -2, keep)

    # -------------------------------------------------------------- n_blocks
    elif field == "n_blocks":
        if g2.n_blocks > g.n_blocks:
            from .transformer import init_params  # local import: avoids a cycle
            n = next(iter(p.values())).shape[0]
            add = g2.n_blocks - g.n_blocks
            fresh = init_params(g2, rng, n=n, scale=max(_break_scale(noise), 1e-3) / 0.5)
            at = int(rng.integers(0, g.n_blocks + 1))  # insertion point
            new: Params = {k: v for k, v in q.items() if not _is_block_key(k)}
            order = list(range(at)) + [None] * add + list(range(at, g.n_blocks))
            for dst, src in enumerate(order):
                for nm in ("g1", "Wq", "Wk", "Wv", "Wo", "g2", "W1", "W2"):
                    if src is None:
                        val = fresh[f"b{dst}.{nm}"]
                        if nm in ("Wo", "W2"):
                            val = np.zeros_like(val)       # exact residual identity
                        elif nm in ("g1", "g2"):
                            val = np.ones_like(val)
                    else:
                        val = q[f"b{src}.{nm}"]
                    new[f"b{dst}.{nm}"] = np.array(val, dtype=DTYPE)
            q = new
        else:
            # Drop the block closest to the identity (smallest output-path norm).
            resid = [float(np.linalg.norm(q[f"b{b}.Wo"][0]) + np.linalg.norm(q[f"b{b}.W2"][0]))
                     for b in range(L)]
            drop = int(np.argmin(resid))
            keep = [b for b in range(L) if b != drop]
            new = {k: v for k, v in q.items() if not _is_block_key(k)}
            for dst, src in enumerate(keep):
                for nm in ("g1", "Wq", "Wk", "Wv", "Wo", "g2", "W1", "W2"):
                    new[f"b{dst}.{nm}"] = q[f"b{src}.{nm}"]
            q = new
    else:  # pragma: no cover - OP_FIELD is exhaustive
        raise ValueError(op)

    return {k: np.ascontiguousarray(v, dtype=DTYPE) for k, v in q.items()}, g2
