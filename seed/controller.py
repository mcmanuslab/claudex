"""
controller.py -- the developmental controller.

The scientific claim this file exists to test is narrow and falsifiable:

    growth triggered by persistent learning difficulty beats growth
    triggered by the clock.

So the controller is deliberately simple and completely logged. It is
heuristic, not learned. Every decision records the full evidence that produced
it, so a growth event can be audited after the fact rather than taken on faith.

The matched control, `RandomController`, is the whole point of the design: it
emits the SAME number of growth events, of the SAME kinds, in the SAME order,
producing the SAME parameter trajectory -- but at uniformly random steps. If
the developmental model beats FIXED SMALL but not RANDOM, then what helps is
"getting bigger", not "growing under pressure", and Question 2 answers NO.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import torch


@dataclass
class ControllerConfig:
    eval_every: int = 100          # steps between controller observations
    window: int = 5                # observations in the plateau window
    plateau_rel: float = 0.01      # < this relative improvement over window = stalled
    patience: int = 2              # consecutive stalled observations before growing
    cooldown: int = 400            # steps after a growth event before another may fire
    max_events: int = 4
    max_blocks: int = 8
    depth_first: int = 2           # first N events grow depth, then MLP width
    warmup_steps: int = 300        # never grow before this (initial loss drop is not a plateau)


class DevelopmentalController:
    def __init__(self, cfg: ControllerConfig):
        self.cfg = cfg
        self.hist: deque[float] = deque(maxlen=2 * cfg.window)
        self.stalled = 0
        self.events = 0
        self.last_growth = -10**9
        self.log: list[dict] = []

    def observe(self, step: int, val_loss: float, per_cat: dict[str, float],
                strain: dict[str, float], n_blocks: int) -> dict:
        """Return a decision dict: {'action': 'CONTINUE'|'GROW', ...} with the
        complete reason attached."""
        c = self.cfg
        self.hist.append(val_loss)
        rec = {
            "step": step, "val_loss": val_loss, "window": list(self.hist),
            "stalled_count": self.stalled, "events_so_far": self.events,
        }

        if len(self.hist) < 2 * c.window:
            rec.update(action="CONTINUE", why="window not full")
            self.log.append(rec); return rec

        # Plateau = the BEST loss stopped improving, comparing the best of the
        # older half of the window against the best of the recent half.
        # Comparing endpoints was the obvious thing to do and it is wrong here:
        # the real validation curve bounces by ~15% step to step, so an
        # endpoint comparison is dominated by noise and the controller
        # degenerates into "fire whenever the cooldown expires" -- which is
        # exactly the random schedule it is supposed to be distinguished from.
        h = list(self.hist)
        old = min(h[:c.window])
        recent = min(h[c.window:])
        rel = (old - recent) / max(abs(old), 1e-8)
        rec["best_old_half"] = old
        rec["best_recent_half"] = recent
        rec["rel_improvement_over_window"] = rel
        if rel < c.plateau_rel:
            self.stalled += 1
        else:
            self.stalled = 0
        rec["stalled_count"] = self.stalled

        blocked = None
        if step < c.warmup_steps:
            blocked = "warmup"
        elif self.events >= c.max_events:
            blocked = "growth budget exhausted"
        elif step - self.last_growth < c.cooldown:
            blocked = f"cooldown ({step - self.last_growth} < {c.cooldown})"
        elif self.stalled < c.patience:
            blocked = f"not yet persistent (stalled {self.stalled} < {c.patience})"

        if blocked:
            rec.update(action="CONTINUE", why=blocked)
            self.log.append(rec); return rec

        # --- decide WHAT and WHERE -----------------------------------------
        kind = "GROW_DEPTH" if (self.events < c.depth_first and n_blocks < c.max_blocks) \
            else "ADD_MLP_CAPACITY"
        # WHERE: the block carrying the most gradient per parameter is the one
        # under the most optimisation strain, so that is where we add capacity.
        site = max(strain, key=strain.get) if strain else "L0"
        worst_cat = max(per_cat, key=per_cat.get) if per_cat else None

        rec.update(
            action="GROW", kind=kind, site=site,
            why=(f"best val loss improved only {rel*100:.2f}% "
                 f"({old:.4f} -> {recent:.4f}) across the last "
                 f"{2*c.window} observations, for {self.stalled} consecutive "
                 f"observations (threshold {c.plateau_rel*100:.1f}%)"),
            per_category_loss=dict(sorted(per_cat.items(), key=lambda kv: -kv[1])),
            worst_category=worst_cat,
            strain=dict(sorted(strain.items(), key=lambda kv: -kv[1])),
        )
        self.events += 1
        self.last_growth = step
        self.stalled = 0
        self.hist.clear()
        self.log.append(rec)
        return rec


class RandomController:
    """Matched control: identical event count, kinds and order; random timing.

    Timings are drawn ONCE at construction from the same legal window the
    developmental controller operates in, so the two differ only in *why* the
    event fires."""

    def __init__(self, cfg: ControllerConfig, total_steps: int, seed: int,
                 kinds: list[str] | None = None):
        self.cfg = cfg
        g = torch.Generator().manual_seed(seed)
        lo, hi = cfg.warmup_steps, max(cfg.warmup_steps + 1, total_steps - cfg.cooldown)
        # sample event steps with the same cooldown constraint, so the parameter
        # trajectory is comparable rather than all growth landing at once
        steps: list[int] = []
        for _ in range(400):
            cand = sorted(int(torch.randint(lo, hi, (1,), generator=g)) for _ in range(cfg.max_events))
            if all(b - a >= cfg.cooldown for a, b in zip(cand, cand[1:])):
                steps = cand; break
        if not steps:
            span = max(1, (hi - lo) // max(1, cfg.max_events))
            steps = [lo + i * span for i in range(cfg.max_events)]
        self.steps = steps
        self.kinds = kinds or (["GROW_DEPTH"] * cfg.depth_first
                               + ["ADD_MLP_CAPACITY"] * (cfg.max_events - cfg.depth_first))
        self.events = 0
        self.log: list[dict] = []

    def observe(self, step: int, val_loss: float, per_cat: dict, strain: dict,
                n_blocks: int) -> dict:
        rec = {"step": step, "val_loss": val_loss, "events_so_far": self.events,
               "scheduled_steps": self.steps}
        if self.events < len(self.steps) and step >= self.steps[self.events]:
            kind = self.kinds[min(self.events, len(self.kinds) - 1)]
            site = max(strain, key=strain.get) if strain else "L0"
            rec.update(action="GROW", kind=kind, site=site,
                       why=f"RANDOM SCHEDULE: pre-drawn step {self.steps[self.events]}")
            self.events += 1
        else:
            rec.update(action="CONTINUE", why="not a scheduled random step")
        self.log.append(rec)
        return rec


class NullController:
    """Fixed architectures never grow."""

    def __init__(self) -> None:
        self.log: list[dict] = []

    def observe(self, step, val_loss, per_cat, strain, n_blocks) -> dict:
        return {"action": "CONTINUE", "why": "fixed architecture", "step": step}


def block_strain(model) -> dict[str, float]:
    """Gradient norm per parameter, per block. A block whose parameters are
    being pushed hard relative to their number is the one the optimiser is
    struggling with, which is where new capacity is most likely to pay."""
    out: dict[str, float] = {}
    for b in model.live_blocks():
        tot, n = 0.0, 0
        for p in b.parameters():
            if p.grad is not None:
                tot += float(p.grad.detach().pow(2).sum())
                n += p.numel()
        out[b.unit_id] = math.sqrt(tot) / max(1, n) * 1e4 if n else 0.0
    return out


class ForcedController:
    """Fires growth at pre-specified steps, ignoring all evidence.

    This is the instrument for the growth-time sweep. Experiment 001 could not
    separate "the unit had more time to integrate" from "the unit was born
    earlier", because with a fixed budget `steps_alive = total - birth_step`
    makes them the same variable. Forcing the birth step and then always
    training a FIXED number of further steps breaks that: birth time varies,
    time-to-integrate is held constant.
    """

    def __init__(self, steps: list[int], kinds: list[str] | None = None):
        self.steps = sorted(steps)
        self.kinds = kinds or ["GROW_DEPTH"] * len(self.steps)
        self.events = 0
        self.log: list[dict] = []

    def observe(self, step: int, val_loss: float, per_cat: dict, strain: dict,
                n_blocks: int) -> dict:
        rec = {"step": step, "val_loss": val_loss, "scheduled": self.steps,
               "events_so_far": self.events}
        if self.events < len(self.steps) and step >= self.steps[self.events]:
            rec.update(action="GROW",
                       kind=self.kinds[min(self.events, len(self.kinds) - 1)],
                       site=max(strain, key=strain.get) if strain else "L0",
                       why=f"FORCED: scheduled growth at step {self.steps[self.events]}")
            self.events += 1
        else:
            rec.update(action="CONTINUE", why="not a forced growth step")
        self.log.append(rec)
        return rec
