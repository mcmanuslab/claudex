"""Resource scheduler: bound peak memory before running anything, never after.

Checkpoint size is a poor guide to what an evaluation costs. The attention probabilities
alone are ``organisms x instances x heads x T^2`` floats, which for a large group can
dwarf the weights by orders of magnitude. The scheduler estimates the peak footprint of a
batched group analytically and splits the group until it fits, so an oversized bucket is
chunked rather than discovered by the machine starting to swap.
"""

from __future__ import annotations

from ..models.genome import ArchGenome

BYTES = 4  # float32


def rollout_bytes(arch: ArchGenome, n_org: int, n_inst: int, seq: int) -> int:
    """Peak bytes for a KV-cached incremental rollout of one group."""
    kv = 2 * n_org * n_inst * arch.n_heads * seq * arch.head_dim * arch.n_blocks
    step = n_org * n_inst * (arch.n_heads * seq + 12 * arch.d_model + 4 * arch.d_ff)
    weights = n_org * arch.n_params
    return int((kv + step + weights) * BYTES)


def train_bytes(arch: ArchGenome, n_org: int, n_inst: int, seq: int) -> int:
    """Peak bytes for a cached full-sequence forward plus backward."""
    probs = n_org * n_inst * arch.n_heads * seq * seq * arch.n_blocks
    acts = n_org * n_inst * seq * (10 * arch.d_model + 2 * arch.d_ff
                                   + 6 * arch.d_attn) * arch.n_blocks
    weights = 3 * n_org * arch.n_params  # params + grads + optimiser moments
    return int((probs + acts + weights) * BYTES)


def max_group(arch: ArchGenome, n_inst: int, seq: int, budget_bytes: int,
              *, training: bool = False, floor: int = 1) -> int:
    """Largest number of organisms that fits in `budget_bytes`."""
    fn = train_bytes if training else rollout_bytes
    per = max(1, fn(arch, 1, n_inst, seq))
    return max(floor, int(budget_bytes // per))


def chunks(n: int, size: int):
    for i in range(0, n, size):
        yield list(range(i, min(n, i + size)))
