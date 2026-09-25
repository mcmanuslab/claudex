"""Effective parameter count: how much of an architecture is actually doing anything.

Raw parameter count cannot distinguish complexity from bloat. The genetic-programming
literature is unambiguous that variable-size representations accumulate material with no
behavioural effect -- introns -- and that this happens for reasons unrelated to fitness
(removal bias, hitchhiking, and simply that far more long representations than short ones
encode any given behaviour). A growth curve in raw parameters is therefore compatible with
no increase in complexity at all, which is why `PREREGISTRATION.md` H3 requires effective
parameters to grow too.

"Effective" is defined by ablation, at the granularity evolution actually operates on:
attention heads, FFN units and whole blocks -- the units the morphisms create and destroy.
A unit counts as effective if zeroing its output path changes the organism's action
distribution by more than `threshold` in total variation, on a fixed probe batch.

The ablations are evaluated as one batched forward pass: each candidate ablation becomes
one organism in a batch that shares the architecture, so measuring several hundred units
costs about one extra evaluation rather than several hundred.
"""

from __future__ import annotations

import numpy as np

from ..models.genome import ArchGenome
from ..models.transformer import DTYPE, Params, forward_full, stack, unstack
from ..storage.scheduler import max_group


def _probs(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def _units(arch: ArchGenome) -> list[tuple[str, int, int, int]]:
    """(kind, block, index, parameters attributable to the unit)."""
    d, hd, f = arch.d_model, arch.head_dim, arch.d_ff
    out: list[tuple[str, int, int, int]] = []
    for b in range(arch.n_blocks):
        for h in range(arch.n_heads):
            out.append(("head", b, h, 3 * d * hd + hd * d))
        for j in range(f):
            out.append(("ffn", b, j, d + d))
    return out


def _ablate(w: Params, arch: ArchGenome, kind: str, b: int, i: int) -> Params:
    q = {k: v.copy() for k, v in w.items()}
    if kind == "head":
        lo, hi = i * arch.head_dim, (i + 1) * arch.head_dim
        q[f"b{b}.Wo"][lo:hi, :] = 0.0
    elif kind == "ffn":
        q[f"b{b}.W2"][i, :] = 0.0
    return q


def effective_params(weights: Params, arch: ArchGenome, obs: np.ndarray,
                     prev_act: np.ndarray, prev_rew: np.ndarray,
                     *, threshold: float = 0.01,
                     mem_budget: int = 1_000_000_000) -> dict[str, float]:
    """Ablation-based effective parameter count for a single organism.

    `obs`/`prev_act`/`prev_rew` are a fixed probe batch of shape (1, B, T); using the
    same probe batch across generations is what makes the numbers comparable.
    """
    base_logits, _ = forward_full(weights, arch, obs, prev_act, prev_rew)
    base = _probs(base_logits)[0]

    single = unstack(weights, 0)
    units = _units(arch)
    if not units:
        return {"effective": float(arch.n_params), "raw": float(arch.n_params),
                "fraction": 1.0, "n_units": 0, "n_effective": 0}

    deltas = np.zeros(len(units))
    cap = max(1, max_group(arch, obs.shape[1], obs.shape[2], mem_budget))
    for start in range(0, len(units), cap):
        chunk = units[start : start + cap]
        batch = stack([_ablate(single, arch, k, b, i) for k, b, i, _ in chunk])
        n = len(chunk)
        logits, _ = forward_full(batch, arch,
                                 np.repeat(obs, n, axis=0), np.repeat(prev_act, n, axis=0),
                                 np.repeat(prev_rew, n, axis=0))
        p = _probs(logits)
        # Total variation between the ablated and intact action distributions.
        deltas[start : start + n] = 0.5 * np.abs(p - base[None]).sum(axis=-1).mean(axis=(1, 2))

    live = deltas > threshold
    unit_params = np.array([p for _, _, _, p in units], dtype=float)
    # Parameters not attributable to any ablatable unit (embeddings, norms, output head)
    # are counted as effective: they are on every path through the network.
    structural = float(unit_params.sum())
    shared = max(0.0, arch.n_params - structural)
    eff = shared + float(unit_params[live].sum())
    return {
        "effective": eff,
        "raw": float(arch.n_params),
        "fraction": eff / max(1.0, arch.n_params),
        "n_units": float(len(units)),
        "n_effective": float(live.sum()),
        "mean_delta": float(deltas.mean()),
    }


def probe_batch(arch: ArchGenome, rng: np.random.Generator, batch: int = 8,
                seq: int = 64) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A fixed token batch for ablation probing. Same seed => same batch across runs."""
    return (rng.integers(0, arch.n_obs, (1, batch, seq)),
            rng.integers(0, arch.n_act, (1, batch, seq)),
            rng.integers(0, arch.n_rew, (1, batch, seq)))
