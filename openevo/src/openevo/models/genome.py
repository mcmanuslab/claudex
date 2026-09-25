"""Architecture genome for the tiny transformer, with exact parameter and FLOP accounting.

Design notes
------------
`d_model`, the attention width (`n_heads * head_dim`) and `d_ff` are deliberately
*decoupled*. In a stock transformer `d_model == n_heads * head_dim`, which couples
"widen the residual stream" to "add a head" and makes the two mutations impossible to
attribute separately. Here attention projects ``d_model -> n_heads*head_dim`` and back,
so `widen`, `add_head`, `expand_ffn` and `add_block` are four independent, individually
exactly-function-preserving operators (see ``models.morphisms``).

Positions use RoPE, which costs zero parameters. With learned positional embeddings a
5K-parameter model would spend ~60% of its budget on the position table, which would
make the parameter count a poor proxy for computational capacity.

The FLOP model here is *analytic and exact* for the ops we execute. Fitness is charged
against this number, never against wall-clock time, so that backend inefficiency
(padding, bucketing, dispatch overhead) can never leak into the science.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace, asdict
from typing import Any

# Hard structural bounds. These are safety rails, not targets: evolution may sit
# anywhere inside them. `max_params` is enforced separately by the resource scheduler.
BOUNDS: dict[str, tuple[int, int]] = {
    "d_model": (4, 4096),
    "n_heads": (1, 64),
    "head_dim": (2, 256),
    "d_ff": (4, 16384),
    "n_blocks": (1, 48),
}


@dataclass(frozen=True)
class ArchGenome:
    """Heritable architecture. Frozen: mutation returns a new instance."""

    d_model: int = 22
    n_heads: int = 2
    head_dim: int = 8
    d_ff: int = 44
    n_blocks: int = 1
    # Interface dimensions. Fixed across every world in the study so that adaptation
    # speed on held-out worlds is never confounded by an interface change.
    n_obs: int = 32
    n_act: int = 5
    n_rew: int = 4

    # ---- validity -------------------------------------------------------------
    def valid(self) -> bool:
        for field, (lo, hi) in BOUNDS.items():
            v = getattr(self, field)
            if not (lo <= v <= hi):
                return False
        # RoPE pairs dimensions, so head_dim must be even.
        return self.head_dim % 2 == 0

    def clipped(self) -> "ArchGenome":
        kw: dict[str, int] = {}
        for field, (lo, hi) in BOUNDS.items():
            kw[field] = int(min(hi, max(lo, getattr(self, field))))
        if kw["head_dim"] % 2:
            kw["head_dim"] += 1
        return replace(self, **kw)

    # ---- derived sizes --------------------------------------------------------
    @property
    def d_attn(self) -> int:
        return self.n_heads * self.head_dim

    # ---- exact parameter count -----------------------------------------------
    def param_breakdown(self) -> dict[str, int]:
        d, da, f = self.d_model, self.d_attn, self.d_ff
        # Embeddings: observation, previous action, previous (quantised) reward.
        emb = (self.n_obs + self.n_act + self.n_rew) * d
        # Per block: q,k,v projections (d x da each), output projection (da x d),
        # FFN up (d x f) + down (f x d), two RMSNorm gains (d each).
        per_block = 3 * d * da + da * d + d * f + f * d + 2 * d
        blocks = self.n_blocks * per_block
        final_norm = d
        head = d * self.n_act + self.n_act  # weight + bias
        return {
            "embeddings": emb,
            "blocks": blocks,
            "final_norm": final_norm,
            "head": head,
            "total": emb + blocks + final_norm + head,
        }

    @property
    def n_params(self) -> int:
        return self.param_breakdown()["total"]

    # ---- exact FLOP model ----------------------------------------------------
    def flops_per_token(self, context: int) -> int:
        """Multiply-accumulates counted as 2 FLOPs, for one token of a forward pass.

        `context` is the number of keys attended over, which for a full causal
        sequence of length T averages (T+1)/2 per token; callers pass the value they
        actually mean. Softmax/normalisation/elementwise terms are included because at
        this scale they are not negligible relative to the matmuls.
        """
        d, da, f, h, hd = self.d_model, self.d_attn, self.d_ff, self.n_heads, self.head_dim
        per_block = 0
        per_block += 2 * d * da * 3          # q, k, v projections
        per_block += 2 * h * context * hd    # q @ k^T
        per_block += 2 * h * context * hd    # attn @ v
        per_block += 5 * h * context         # softmax (max, exp, sum, div)
        per_block += 2 * da * d              # output projection
        per_block += 2 * d * f + 2 * f * d   # FFN
        per_block += 4 * f                   # GELU-ish activation
        per_block += 8 * d                   # two RMSNorms
        total = self.n_blocks * per_block
        total += 4 * d                       # final norm
        total += 2 * d * self.n_act          # output head
        return int(total)

    def flops_forward(self, seq_len: int) -> int:
        """Forward FLOPs for one full causal sequence of `seq_len` tokens."""
        # Sum over positions t=1..T of per-token cost with context t.
        # Terms linear in `context` sum to T(T+1)/2; constant terms scale by T.
        const_part = self.flops_per_token(context=0)
        slope = self.flops_per_token(context=1) - const_part
        return int(seq_len * const_part + slope * seq_len * (seq_len + 1) // 2)

    def flops_train_step(self, seq_len: int) -> int:
        """Forward + backward. Backward is ~2x forward for these dense matmuls."""
        return 3 * self.flops_forward(seq_len)

    # ---- identity / serialisation -------------------------------------------
    def signature(self) -> tuple[int, ...]:
        """Shape bucket key. Organisms sharing a signature can be batched together."""
        return (self.d_model, self.n_heads, self.head_dim, self.d_ff, self.n_blocks)

    def arch_hash(self) -> str:
        return hashlib.sha256(
            json.dumps(asdict(self), sort_keys=True).encode()
        ).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "ArchGenome":
        fields = ArchGenome.__dataclass_fields__.keys()
        return ArchGenome(**{k: int(v) for k, v in d.items() if k in fields})


def scale_to_params(
    target: int,
    *,
    base: ArchGenome | None = None,
    n_blocks: int = 2,
    ff_mult: float = 2.0,
    head_dim: int = 8,
) -> ArchGenome:
    """Smallest genome with >= `target` parameters, grown along d_model.

    Used to build the ancestral scale ladder (A/B/C) and, importantly, the
    capacity-matched controls: "a generation-0 shaped architecture with the same
    parameter count as this generation-t organism".

    ``n_blocks`` defaults to 2 rather than 1 on the basis of a measurement, not taste.
    With single-block founders the depth dimension sits on the floor of its ladder, so
    `remove_block` can never fire while `add_block` always can, and a population under
    *no selection at all* drifts at +0.0067 nats/generation -- a 28x increase in
    parameter count over 500 generations, purely passively. Starting one rung clear of
    the floor drops that to +0.0024 +- 0.0047 nats/generation, statistically
    indistinguishable from zero (3.2x over 500 generations, CI spanning 1). Founders are
    therefore placed off the wall, and the residual is still measured by the neutral arm
    of every experiment.
    """
    base = base or ArchGenome()
    best = None
    for d in range(BOUNDS["d_model"][0], 1200):
        heads = max(1, round(d / head_dim))
        g = replace(
            base,
            d_model=d,
            n_heads=heads,
            head_dim=head_dim,
            d_ff=max(4, int(round(d * ff_mult))),
            n_blocks=n_blocks,
        ).clipped()
        if g.n_params >= target:
            best = g
            break
    if best is None:
        raise ValueError(f"cannot reach {target} params within bounds")
    # Snap onto the mutation ladder. Organisms live on it (mutations move +-1 rung), so a
    # genome off the ladder would be silently resized the first time it is used and the
    # parameter count reported for a founder, or for a capacity-matched control, would
    # not be the one actually evaluated.
    from .morphisms import snap_genome, step_field  # deferred: morphisms imports this
    snapped = snap_genome(best)
    # Snapping rounds to the nearest rung, which can land below the target. Step d_model
    # back up until the contract ("at least `target` parameters") holds again.
    while snapped.n_params < target:
        up = step_field(snapped, "d_model", +1)
        if up is None:
            break
        snapped = up
    return snapped
