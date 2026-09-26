"""
growth.py -- growth operators, function-preservation checks, competitive
overgrowth and pruning.

Three things in here are load-bearing and easy to get silently wrong:

1. FUNCTION PRESERVATION IS VERIFIED, NOT ASSUMED. Every growth call returns
   the measured max |logit difference| on a probe batch. Gate 1 asserts it is
   at floating-point zero. If a future edit breaks preservation, the assert
   fires rather than the experiment quietly producing a worse number.

2. OPTIMIZER STATE IS THE PART EVERYONE FORGETS. Staged Training's central
   point is that preserving the *loss* is not enough; you must preserve the
   *training dynamics*. A newly created parameter enters AdamW with zero first
   and second moments, and AdamW's first update on such a parameter has
   magnitude ~lr regardless of gradient size. Dropping a zero-initialised block
   into a converged model and then hitting it with a full-size step destroys
   the function we just took care to preserve. So new parameters get their own
   optimizer group with an LR warmup, and pre-existing parameters keep their
   moments exactly.

3. THE OVERGROWTH WINDOW IS CHARGED FOR. `flops_per_token` reads the live
   structure, so while K candidates are active the method pays for K
   candidates. Competitive overgrowth is only allowed to look good if it is
   still ahead after paying for the candidates it threw away.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import Block, MLPBranch, ModelConfig, SeedTransformer


# ---------------------------------------------------------------------------
# growth event record (the lineage)
# ---------------------------------------------------------------------------

@dataclass
class GrowthEvent:
    step: int
    age: int
    kind: str                       # GROW_DEPTH | ADD_MLP_CAPACITY
    strategy: str                   # simple | competitive
    reason: dict                    # exactly why the controller fired
    added_uids: list[str] = field(default_factory=list)
    site: str = ""
    params_before: int = 0
    params_after: int = 0
    logit_delta: float = 0.0        # measured functional perturbation
    val_loss_before: float = 0.0
    val_loss_after: float = 0.0
    # filled in later
    survivors: list[str] = field(default_factory=list)
    pruned: list[str] = field(default_factory=list)
    prune_scores: dict = field(default_factory=dict)
    final_ablation: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# function-preservation probe
# ---------------------------------------------------------------------------

@torch.no_grad()
def logit_delta(model: SeedTransformer, before: torch.Tensor,
                probe: torch.Tensor) -> float:
    after = model.answer_logits(probe)
    return (after - before).abs().max().item()


@torch.no_grad()
def snapshot_logits(model: SeedTransformer, probe: torch.Tensor) -> torch.Tensor:
    return model.answer_logits(probe).clone()


# ---------------------------------------------------------------------------
# operators
# ---------------------------------------------------------------------------

def grow_depth(model: SeedTransformer, position: int, n_branch: int = 1,
               init_std: float = 0.02) -> Block:
    """Insert an identity block at `position`. Exactly function preserving."""
    blk = Block(model.cfg, unit_id=model.new_uid("G"), zero_init=True,
                n_branch=n_branch)
    for m in blk.modules():
        if isinstance(m, nn.Linear):
            pass
    # non-output weights get ordinary init; outputs stay zero
    nn.init.normal_(blk.qkv.weight, std=init_std)
    nn.init.zeros_(blk.qkv.bias)
    for br in blk.branches:
        nn.init.normal_(br.up.weight, std=init_std)
        nn.init.zeros_(br.up.bias)
        nn.init.zeros_(br.down.weight)
        nn.init.zeros_(br.down.bias)
    nn.init.zeros_(blk.proj.weight)
    nn.init.zeros_(blk.proj.bias)
    blocks = list(model.blocks)
    blocks.insert(position, blk)
    model.blocks = nn.ModuleList(blocks)
    return blk


def add_mlp_branch(model: SeedTransformer, block_idx: int, d_ff: int,
                   init_std: float = 0.02) -> MLPBranch:
    """Add a parallel MLP branch with a zero down-projection to an existing
    block. Exactly function preserving; `d_model` and therefore LayerNorm are
    untouched."""
    blk = model.blocks[block_idx]
    br = MLPBranch(model.cfg.d_model, d_ff,
                   unit_id=model.new_uid("B") + f"@{blk.unit_id}", zero_init=True)
    nn.init.normal_(br.up.weight, std=init_std)
    nn.init.zeros_(br.up.bias)
    blk.branches.append(br)
    return br


def remove_unit(model: SeedTransformer, uid: str) -> bool:
    """Structurally delete a unit. Parameters and FLOPs actually go away --
    gating alone would let a 'pruned' model keep paying for what it pruned."""
    for bi, blk in enumerate(model.blocks):
        if blk.unit_id == uid:
            blocks = [b for b in model.blocks if b.unit_id != uid]
            model.blocks = nn.ModuleList(blocks)
            return True
        for br in list(blk.branches):
            if br.unit_id == uid:
                keep = [b for b in blk.branches if b.unit_id != uid]
                blk.branches = nn.ModuleList(keep)
                return True
    return False


# ---------------------------------------------------------------------------
# optimizer surgery
# ---------------------------------------------------------------------------

class GrowthAwareOptimizer:
    """AdamW that survives architectural change.

    Pre-existing parameters keep their exp_avg / exp_avg_sq / step exactly.
    Newly born parameters go into their own cohort group whose LR ramps from 0
    over `warmup`, because a zero-moment AdamW parameter otherwise takes a
    ~full-size step immediately and blows away the function we just preserved.
    """

    def __init__(self, model: nn.Module, lr: float, weight_decay: float = 0.01,
                 warmup: int = 100, betas=(0.9, 0.95)):
        self.lr = lr
        self.wd = weight_decay
        self.warmup = warmup
        self.betas = betas
        self.cohorts: list[dict] = [{"birth": None, "params": list(model.parameters())}]
        self._build()

    def _build(self) -> None:
        groups = []
        for c in self.cohorts:
            ps = [p for p in c["params"] if p.requires_grad]
            if ps:
                groups.append({"params": ps, "lr": self.lr, "weight_decay": self.wd})
        self.opt = torch.optim.AdamW(groups, lr=self.lr, betas=self.betas,
                                     weight_decay=self.wd)

    def rebuild(self, model: nn.Module, new_params: list[nn.Parameter],
                step: int) -> None:
        """Called after any structural change. Carries Adam moments across by
        tensor identity, which is stable because growth adds and removes
        modules but never re-allocates surviving parameter tensors."""
        saved = {id(p): self.opt.state[p] for g in self.opt.param_groups
                 for p in g["params"] if p in self.opt.state}
        live = {id(p) for p in model.parameters()}
        new_ids = {id(p) for p in new_params}
        for c in self.cohorts:
            c["params"] = [p for p in c["params"] if id(p) in live and id(p) not in new_ids]
        self.cohorts = [c for c in self.cohorts if c["params"]]
        if new_params:
            self.cohorts.append({"birth": step, "params": list(new_params)})
        self._build()
        for g in self.opt.param_groups:
            for p in g["params"]:
                if id(p) in saved:
                    self.opt.state[p] = saved[id(p)]

    def set_lr(self, base_lr: float, step: int) -> None:
        self.lr = base_lr
        for g, c in zip(self.opt.param_groups, [c for c in self.cohorts if c["params"]]):
            if c["birth"] is None:
                g["lr"] = base_lr
            else:
                age = step - c["birth"]
                g["lr"] = base_lr * min(1.0, max(0.0, age / max(1, self.warmup)))

    def step(self) -> None:
        self.opt.step()

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.opt.zero_grad(set_to_none=set_to_none)


# ---------------------------------------------------------------------------
# competition: masking ablation + redundancy
# ---------------------------------------------------------------------------

@torch.no_grad()
def _loss_on(model: SeedTransformer, x: torch.Tensor, y: torch.Tensor) -> float:
    return F.cross_entropy(model.answer_logits(x), y).item()


@torch.no_grad()
def ablation_scores(model: SeedTransformer, uids: list[str], x: torch.Tensor,
                    y: torch.Tensor) -> dict[str, float]:
    """Marginal contribution of each unit = increase in validation loss when it
    alone is masked off. Positive means the unit is earning its place."""
    base = _loss_on(model, x, y)
    out: dict[str, float] = {}
    for uid in uids:
        u = model.find_unit(uid)
        if u is None:
            continue
        old = u.gate.clone()
        u.gate.zero_()
        out[uid] = _loss_on(model, x, y) - base
        u.gate.copy_(old)
    return out


@torch.no_grad()
def ablation_by_relation(model: SeedTransformer, uids: list[str], x: torch.Tensor,
                         y: torch.Tensor, rels: list[str]) -> dict[str, dict[str, float]]:
    """Per-relation marginal contribution of each unit.

    This is what answers "did the surviving branch actually specialise?".
    A unit whose ablation cost is spread evenly across all nine relations has
    only added generic capacity; a unit whose cost concentrates on, say,
    HEAVIER has taken on a specific job.
    """
    import collections

    def per_rel() -> dict[str, float]:
        ls = F.cross_entropy(model.answer_logits(x), y, reduction="none")
        acc: dict[str, list[float]] = collections.defaultdict(list)
        for r, v in zip(rels, ls.tolist()):
            acc[r].append(v)
        return {k: sum(v) / len(v) for k, v in acc.items()}

    base = per_rel()
    out: dict[str, dict[str, float]] = {}
    for uid in uids:
        u = model.find_unit(uid)
        if u is None:
            continue
        old = u.gate.clone()
        u.gate.zero_()
        masked = per_rel()
        u.gate.copy_(old)
        out[uid] = {r: round(masked[r] - base[r], 6) for r in base}
    return out


@torch.no_grad()
def branch_outputs(model: SeedTransformer, uids: list[str],
                   x: torch.Tensor) -> dict[str, torch.Tensor]:
    """Capture each candidate's contribution to the residual stream, so we can
    measure whether two candidates are doing the same job."""
    store: dict[str, torch.Tensor] = {}
    hooks = []

    def mk(uid):
        def hook(_m, _i, o):
            store[uid] = o.detach().reshape(-1).float()
        return hook

    for uid in uids:
        u = model.find_unit(uid)
        if u is not None:
            hooks.append(u.register_forward_hook(mk(uid)))
    model.answer_logits(x)
    for h in hooks:
        h.remove()
    return store


def redundancy_matrix(outs: dict[str, torch.Tensor]) -> dict[tuple[str, str], float]:
    uids = list(outs)
    m: dict[tuple[str, str], float] = {}
    for i, a in enumerate(uids):
        for b in uids[i + 1:]:
            va, vb = outs[a], outs[b]
            if va.norm() < 1e-9 or vb.norm() < 1e-9:
                m[(a, b)] = 0.0
            else:
                m[(a, b)] = float(F.cosine_similarity(va, vb, dim=0).abs())
    return m


def select_survivors(scores: dict[str, float], redund: dict[tuple[str, str], float],
                     keep: int, min_gain: float | None, redundancy_thresh: float,
                     rng: torch.Generator | None = None,
                     random_mode: bool = False) -> tuple[list[str], list[str], dict]:
    """Pick which candidates live.

    `random_mode` is the honest control: same number of survivors, chosen by
    coin flip. "When BERT Plays the Lottery" found random structured pruning
    competitive with importance-based pruning, so competitive selection has to
    beat random selection before we may claim the competition did anything.
    """
    uids = list(scores)
    if random_mode:
        order = torch.randperm(len(uids), generator=rng).tolist()
        ranked = [uids[i] for i in order]
        why = {"mode": "random"}
    else:
        ranked = sorted(uids, key=lambda u: -scores[u])
        why = {"mode": "competitive"}

    # What a "only keep units that help" rule WOULD have done. Recorded even
    # when the rule is off, because "the competition found no candidate that
    # improved validation loss" is a first-class finding, not a detail.
    why["would_fail_min_gain"] = [u for u in uids if scores[u] < 0.0]
    why["all_candidates_unhelpful"] = len(why["would_fail_min_gain"]) == len(uids)

    survivors: list[str] = []
    pruned: list[str] = []
    reasons: dict[str, str] = {}
    for u in ranked:
        if len(survivors) >= keep:
            pruned.append(u); reasons[u] = "over budget"; continue
        if min_gain is not None and not random_mode and scores[u] < min_gain:
            pruned.append(u); reasons[u] = f"marginal gain {scores[u]:.5f} < {min_gain}"
            continue
        dup = None
        if not random_mode:
            for s in survivors:
                r = redund.get((s, u), redund.get((u, s), 0.0))
                if r > redundancy_thresh:
                    dup = (s, r); break
        if dup is not None:
            pruned.append(u); reasons[u] = f"redundant with {dup[0]} (cos={dup[1]:.3f})"
            continue
        survivors.append(u); reasons[u] = "kept"
    why["per_unit"] = reasons
    return survivors, pruned, why
