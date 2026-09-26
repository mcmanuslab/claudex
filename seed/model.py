"""
model.py -- a tiny pre-norm decoder-only transformer whose structure can change
during training.

The whole architecture exists to make one thing true: every growth operator we
use is EXACTLY function preserving, to floating-point, with no masking tricks
and no LayerNorm surgery.

We get that by a deliberate scope restriction: we never grow `d_model`.
MSG (ICLR 2024) had to fold masks into the LayerNorm statistics precisely
because widening `d_model` changes what LN normalises over. Our two operators
leave `d_model` alone:

  GROW_DEPTH        insert a Block whose attention out-projection and whose
                    every MLP branch down-projection are zero. A pre-norm block
                    computes `x + attn(ln(x)) + mlp(ln(x))`, so a zeroed output
                    contributes exactly 0 and the block is the identity.

  ADD_MLP_CAPACITY  add a parallel MLP branch to an existing block with a
                    zero down-projection. Same argument, and `d_ff` is not
                    normalised over, so LN never sees it.

Both are identity at the instant of growth and both have non-zero gradient on
the down-projection immediately, so they start learning on the first step.
(The up-projection has zero gradient for exactly one step, which is the
standard zero-init behaviour and is harmless.)

Every growable thing is a "unit" with a stable id, a gate we can set to 0 for
masking ablation, and a birth record. That is what makes the growth lineage and
the competitive pruning in `growth.py` possible.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 71
    d_model: int = 128
    n_head: int = 4
    d_ff: int = 256          # width of ONE mlp branch
    n_layer: int = 4
    seq_len: int = 5
    dropout: float = 0.0


class MLPBranch(nn.Module):
    """One parallel MLP branch. The unit of width growth and width pruning."""

    def __init__(self, d_model: int, d_ff: int, unit_id: str, zero_init: bool = False):
        super().__init__()
        self.unit_id = unit_id
        self.up = nn.Linear(d_model, d_ff)
        self.down = nn.Linear(d_ff, d_model)
        if zero_init:
            nn.init.zeros_(self.down.weight)
            nn.init.zeros_(self.down.bias)
        # gate is a buffer, not a parameter: it is an ablation switch, never learned
        self.register_buffer("gate", torch.ones(()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if float(self.gate) == 0.0:
            return torch.zeros_like(x)
        return self.down(F.gelu(self.up(x))) * self.gate


class Block(nn.Module):
    """Pre-norm block. The unit of depth growth and depth pruning."""

    def __init__(self, cfg: ModelConfig, unit_id: str, zero_init: bool = False,
                 n_branch: int = 1):
        super().__init__()
        self.unit_id = unit_id
        self.cfg = cfg
        d = cfg.d_model
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        if zero_init:
            nn.init.zeros_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)
        self.branches = nn.ModuleList([
            MLPBranch(d, cfg.d_ff, f"{unit_id}.b{i}", zero_init=zero_init)
            for i in range(n_branch)
        ])
        self.register_buffer("gate", torch.ones(()))

    def attn(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        H = self.cfg.n_head
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, H, C // H).transpose(1, 2)
        k = k.view(B, T, H, C // H).transpose(1, 2)
        v = v.view(B, T, H, C // H).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if float(self.gate) == 0.0:
            return x
        g = self.gate
        x = x + g * self.attn(self.ln1(x))
        h = self.ln2(x)
        mlp = None
        for br in self.branches:
            out = br(h)
            mlp = out if mlp is None else mlp + out
        if mlp is not None:
            x = x + g * mlp
        return x

    def live_branches(self) -> list[MLPBranch]:
        return [b for b in self.branches if float(b.gate) != 0.0]


class SeedTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Embedding(cfg.seq_len, cfg.d_model)
        self.blocks = nn.ModuleList([
            Block(cfg, unit_id=f"L{i}") for i in range(cfg.n_layer)
        ])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self._next_uid = cfg.n_layer
        self.apply(self._init)
        # re-zero anything that was meant to be zero (apply() overwrites it)
        for b in self.blocks:
            pass

    def _init(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def new_uid(self, prefix: str) -> str:
        self._next_uid += 1
        return f"{prefix}{self._next_uid}"

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))[None]
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.ln_f(x))

    def answer_logits(self, idx: torch.Tensor) -> torch.Tensor:
        """Logits predicting the ANSWER token, i.e. at the position just before
        it. Every metric in this study is computed from exactly this tensor."""
        return self.forward(idx[:, :-1])[:, -1, :]

    # -------------------------------------------------------------- structure
    def live_blocks(self) -> list[Block]:
        return [b for b in self.blocks if float(b.gate) != 0.0]

    def units(self) -> list[tuple[str, nn.Module]]:
        """Every ablatable/prunable unit, as (kind, module)."""
        out: list[tuple[str, nn.Module]] = []
        for b in self.blocks:
            out.append(("block", b))
            for br in b.branches:
                out.append(("branch", br))
        return out

    def find_unit(self, uid: str) -> nn.Module | None:
        for _, m in self.units():
            if m.unit_id == uid:
                return m
        return None

    def n_params(self, active_only: bool = False) -> int:
        if not active_only:
            return sum(p.numel() for p in self.parameters())
        n = self.tok.weight.numel() + self.pos.weight.numel()
        n += sum(p.numel() for p in self.ln_f.parameters())
        n += self.head.weight.numel()
        for b in self.live_blocks():
            n += sum(p.numel() for p in b.ln1.parameters())
            n += sum(p.numel() for p in b.ln2.parameters())
            n += sum(p.numel() for p in b.qkv.parameters())
            n += sum(p.numel() for p in b.proj.parameters())
            for br in b.live_branches():
                n += sum(p.numel() for p in br.parameters())
        return n

    def flops_per_token(self, training: bool = True) -> float:
        """Forward FLOPs per token from the ACTIVE structure, plus the explicit
        attention term. Training is charged 3x forward (fwd + bwd).

        Charging from the active structure is what makes the compute-matched
        comparison honest: a group that temporarily runs four candidate
        branches pays for four candidate branches."""
        d, T = self.cfg.d_model, self.cfg.seq_len
        f = 0.0
        for b in self.live_blocks():
            f += 2 * (3 * d * d)          # qkv
            f += 2 * (d * d)              # out proj
            f += 4 * T * d                # attention scores + weighted sum
            for br in b.live_branches():
                f += 2 * (d * br.up.out_features) * 2   # up and down
        f += 2 * d * self.cfg.vocab_size  # lm head
        return f * (3.0 if training else 1.0)

    def arch_signature(self) -> dict:
        return {
            "n_blocks": len(self.live_blocks()),
            "blocks": [
                {"uid": b.unit_id, "branches": [br.unit_id for br in b.live_branches()],
                 "d_ff_total": sum(br.up.out_features for br in b.live_branches())}
                for b in self.live_blocks()
            ],
            "params_total": self.n_params(False),
            "params_active": self.n_params(True),
            "flops_per_token_train": self.flops_per_token(True),
        }
