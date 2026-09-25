"""
train.py -- one developmental run.

A run is: a world, a model, a controller, and a FLOPs budget divided into
developmental ages. At the end of every age the model writes predictions for
every fact it has not been shown into the hash-chained ledger, and only then is
the next tranche revealed.

Compute is budgeted in FLOPs, not steps. This is deliberate and it is the
single most important fairness decision in the study: if we budgeted steps, a
model that grew would simply have been handed more compute, and every result
would be uninterpretable. Because the budget is FLOPs, growing means taking
fewer steps, and competitive overgrowth means taking fewer steps still --
the method pays for the candidates it discards.
"""

from __future__ import annotations

import argparse, json, math, os, resource, time
from dataclasses import dataclass, asdict, field

import torch
import torch.nn.functional as F

import growth as G
from controller import (ControllerConfig, DevelopmentalController,
                        NullController, RandomController, block_strain)
from ledger import Ledger
from model import ModelConfig, SeedTransformer
from world import World, WorldSpec, DIST_NAMES, RELATIONS


@dataclass
class RunConfig:
    group: str = "D"                 # A | B | C | D | R | D_RANDPRUNE
    seed: int = 0
    world_seed: int = 0
    out: str = "results/run"

    # model
    d_model: int = 128
    n_head: int = 4
    d_ff: int = 256
    n_layer: int = 4

    # optimisation
    lr: float = 3e-3
    batch_size: int = 128
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    lr_warmup: int = 200
    new_param_warmup: int = 150
    sampling_alpha: float = 0.5      # relation-balancing temperature

    # budget
    flops_budget: float = 3.0e13
    n_ages: int = 8
    max_steps: int = 200_000

    # growth
    n_candidates: int = 4            # K in competitive overgrowth
    keep: int = 1                    # survivors per event
    dev_window: int = 300            # steps candidates get to differentiate
    min_gain: float | None = None    # None = keep exactly `keep` (C/D param-matched)
    redundancy_thresh: float = 0.95
    branch_d_ff: int = 256

    controller: ControllerConfig = field(default_factory=ControllerConfig)
    match_arch: str = ""             # path to a run.json whose final arch to copy
    match_events: str = ""           # path to a run.json whose growth EVENTS to copy


def facts_to_tensor(world: World, facts: list) -> tuple[torch.Tensor, torch.Tensor]:
    seqs = [world.encode(f)[0] for f in facts]
    t = torch.tensor(seqs, dtype=torch.long)
    return t, t[:, -1].clone()


def relation_sampling_weights(facts: list, alpha: float) -> torch.Tensor:
    """Per-example sampling weights under temperature-`alpha` relation
    balancing: P(relation) proportional to n_relation ** alpha.

    This is not a free knob, it is a fix for a flaw the pilot exposed. The
    binary relations have ~2256 facts each and the unary ones have 48, so under
    uniform sampling the model sees a SHELL fact 0.5% of the time and simply
    never learns the latent period. But period is exactly what the D3
    extrapolation bucket requires the model to have grounded. Uniform sampling
    would therefore have made the flagship measurement untestable for a reason
    that has nothing to do with growth. alpha=0.5 (square-root temperature) is
    the standard multi-task choice and is applied IDENTICALLY to every group.
    """
    from collections import Counter
    c = Counter(f.rel for f in facts)
    w_rel = {r: (n ** alpha) / n for r, n in c.items()}   # per-example weight
    return torch.tensor([w_rel[f.rel] for f in facts], dtype=torch.float)


class Trainer:
    def __init__(self, cfg: RunConfig):
        self.cfg = cfg
        torch.manual_seed(cfg.seed)
        torch.set_num_threads(int(os.environ.get("SEED_THREADS", "4")))

        self.world = World(WorldSpec(seed=cfg.world_seed, n_tranches=cfg.n_ages))
        mc = ModelConfig(vocab_size=len(self.world.vocab), d_model=cfg.d_model,
                         n_head=cfg.n_head, d_ff=cfg.d_ff, n_layer=cfg.n_layer,
                         seq_len=5)
        self.model = SeedTransformer(mc)

        if cfg.match_arch:
            self._match_architecture(cfg.match_arch)

        self.opt = G.GrowthAwareOptimizer(self.model, lr=cfg.lr,
                                          weight_decay=cfg.weight_decay,
                                          warmup=cfg.new_param_warmup)
        self.ctrl = self._make_controller()

        vx, vy = facts_to_tensor(self.world, self.world.val_facts())
        self.vx, self.vy = vx, vy
        self.val_rel = [f.rel for f in self.world.val_facts()]
        self.probe = vx[:64]

        self.flops = 0.0
        self.tokens = 0
        self.step = 0
        self.age = 0
        self.events: list[G.GrowthEvent] = []
        self.competition: dict | None = None
        self.trace: list[dict] = []
        self.t0 = time.time()
        os.makedirs(cfg.out, exist_ok=True)
        self.ledger = Ledger(os.path.join(cfg.out, "ledger.jsonl"))
        self.rng = torch.Generator().manual_seed(cfg.seed + 9999)

    # ------------------------------------------------------------------ setup
    def _make_controller(self):
        c = self.cfg
        if c.group in ("A", "B"):
            return NullController()
        if c.group == "R":
            est = self._estimate_total_steps()
            kinds, n = None, c.controller.max_events
            if c.match_events:
                evs = json.load(open(c.match_events))["events"]
                kinds = [e["kind"] for e in evs]
                n = len(evs)
            # The random control must match the DEVELOPMENTAL run's ACTUAL
            # growth sequence, not the configured maximum. The controller is
            # evidence-driven, so it fires 3 events on one seed and 4 on
            # another; a control pinned to max_events would end at a different
            # parameter count and the comparison would silently become a
            # parameter comparison again.
            cc = ControllerConfig(**{**asdict(c.controller), "max_events": n})
            return RandomController(cc, est, seed=c.seed + 777, kinds=kinds)
        return DevelopmentalController(c.controller)

    def _estimate_total_steps(self) -> int:
        f = self.model.flops_per_token(True) * self.cfg.batch_size * 4
        return int(min(self.cfg.max_steps, self.cfg.flops_budget / max(f, 1.0)))

    def _match_architecture(self, path: str) -> None:
        """Build FIXED LARGE to exactly the architecture the developmental run
        ended with, trained from scratch with ordinary init."""
        spec = json.load(open(path))["final_arch"]
        cur = len(self.model.blocks)
        for _ in range(spec["n_blocks"] - cur):
            G.grow_depth(self.model, len(self.model.blocks))
        for i, b in enumerate(spec["blocks"]):
            want = len(b["branches"])
            while len(self.model.blocks[i].branches) < want:
                G.add_mlp_branch(self.model, i, self.cfg.branch_d_ff)
        # trained from scratch, so undo the zero-init that growth uses
        for m in self.model.modules():
            if isinstance(m, torch.nn.Linear):
                if float(m.weight.abs().sum()) == 0.0:
                    torch.nn.init.normal_(m.weight, std=0.02)

    # ------------------------------------------------------------- evaluation
    @torch.no_grad()
    def val_loss(self) -> tuple[float, dict[str, float]]:
        self.model.eval()
        logits = self.model.answer_logits(self.vx)
        ls = F.cross_entropy(logits, self.vy, reduction="none")
        per: dict[str, list[float]] = {}
        for r, v in zip(self.val_rel, ls.tolist()):
            per.setdefault(r, []).append(v)
        self.model.train()
        return float(ls.mean()), {k: sum(v) / len(v) for k, v in per.items()}

    @torch.no_grad()
    def predict(self, facts: list) -> list[dict]:
        """Forward pass over facts, returning prediction + calibrated
        confidence. Ground truth is NOT read here -- see ledger.py."""
        self.model.eval()
        out = []
        for i in range(0, len(facts), 1024):
            chunk = facts[i:i + 1024]
            x, _ = facts_to_tensor(self.world, chunk)
            lg = self.model.answer_logits(x)
            p = lg.softmax(-1)
            conf, pred = p.max(-1)
            ent = -(p * (p + 1e-12).log()).sum(-1)
            # full predictive distribution over the answer symbols only, for Brier
            ab = self.world.vocab.ans_base
            pa = p[:, ab:ab + self.world.vocab.n_ans]
            for j, f in enumerate(chunk):
                out.append({
                    "fid": f.fid, "rel": f.rel, "dist": f.dist, "split": f.split,
                    "tranche": f.tranche,
                    "pred_token": int(pred[j]), "conf": float(conf[j]),
                    "entropy": float(ent[j]),
                    "p_answers": [round(float(v), 5) for v in pa[j]],
                })
        self.model.train()
        return out

    # ---------------------------------------------------------------- growth
    def _do_growth(self, dec: dict) -> None:
        cfg = self.cfg
        competitive = cfg.group in ("D", "D_RANDPRUNE")
        k = cfg.n_candidates if competitive else 1
        before = G.snapshot_logits(self.model, self.probe)
        vl_before, _ = self.val_loss()
        p_before = self.model.n_params()
        added: list[str] = []
        new_params: list[torch.nn.Parameter] = []

        site_idx = next((i for i, b in enumerate(self.model.blocks)
                         if b.unit_id == dec.get("site")), len(self.model.blocks) - 1)

        if dec["kind"] == "GROW_DEPTH":
            # K identity blocks at K distinct depths. Each is exactly the
            # identity, so inserting all K is still exactly the identity --
            # which is what lets them compete on equal terms.
            positions = sorted({min(len(self.model.blocks), site_idx + 1 + i)
                                for i in range(k)})
            while len(positions) < k:
                positions.append(min(len(self.model.blocks), positions[-1] + 1))
            for ci, pos in enumerate(positions[:k]):
                std = 0.02 * (1.0 + 0.25 * ci)     # small init differences
                blk = G.grow_depth(self.model, pos, init_std=std)
                added.append(blk.unit_id)
                new_params += list(blk.parameters())
        else:
            for ci in range(k):
                std = 0.02 * (1.0 + 0.25 * ci)
                br = G.add_mlp_branch(self.model, site_idx, cfg.branch_d_ff,
                                      init_std=std)
                added.append(br.unit_id)
                new_params += list(br.parameters())

        delta = G.logit_delta(self.model, before, self.probe)
        assert delta < 1e-5, f"growth broke function preservation: {delta}"
        self.opt.rebuild(self.model, new_params, self.step)
        vl_after, _ = self.val_loss()

        ev = G.GrowthEvent(
            step=self.step, age=self.age, kind=dec["kind"],
            strategy="competitive" if competitive else "simple",
            reason={k2: v for k2, v in dec.items() if k2 != "window"},
            added_uids=added, site=dec.get("site", ""),
            params_before=p_before, params_after=self.model.n_params(),
            logit_delta=delta, val_loss_before=vl_before, val_loss_after=vl_after,
        )
        self.events.append(ev)
        if competitive:
            self.competition = {"uids": list(added), "due": self.step + cfg.dev_window,
                                "event": ev}
        else:
            ev.survivors = list(added)

    def _resolve_competition(self) -> None:
        cfg = self.cfg
        comp = self.competition
        self.competition = None
        uids = [u for u in comp["uids"] if self.model.find_unit(u) is not None]
        scores = G.ablation_scores(self.model, uids, self.vx, self.vy)
        outs = G.branch_outputs(self.model, uids, self.vx)
        red = G.redundancy_matrix(outs)
        survivors, pruned, why = G.select_survivors(
            scores, red, keep=cfg.keep, min_gain=cfg.min_gain,
            redundancy_thresh=cfg.redundancy_thresh, rng=self.rng,
            random_mode=(cfg.group == "D_RANDPRUNE"))
        for u in pruned:
            G.remove_unit(self.model, u)
        self.opt.rebuild(self.model, [], self.step)
        ev = comp["event"]
        ev.survivors, ev.pruned = survivors, pruned
        ev.prune_scores = {
            "marginal_val_loss_increase_when_masked": {k: round(v, 6) for k, v in scores.items()},
            "pairwise_output_cosine": {f"{a}|{b}": round(v, 4) for (a, b), v in red.items()},
            "selection": why,
        }

    # ------------------------------------------------------------------ loop
    def run(self) -> dict:
        cfg = self.cfg
        per_age_flops = cfg.flops_budget / cfg.n_ages
        for age in range(cfg.n_ages):
            self.age = age
            revealed = self.world.revealed(age)
            # --- the no-leakage invariant, enforced not promised -------------
            assert all(f.split == "pool" and f.tranche <= age for f in revealed)
            tx, ty = facts_to_tensor(self.world, revealed)
            tw = relation_sampling_weights(revealed, cfg.sampling_alpha)
            target = per_age_flops * (age + 1)
            while self.flops < target and self.step < cfg.max_steps:
                self._train_step(tx, ty, tw)
                if (self.competition and self.step >= self.competition["due"]):
                    self._resolve_competition()
                if self.step % cfg.controller.eval_every == 0:
                    self._observe()
            self._write_ledger(age)
        return self._finish()

    def _train_step(self, tx: torch.Tensor, ty: torch.Tensor,
                    tw: torch.Tensor | None = None) -> None:
        cfg = self.cfg
        if tw is None:
            idx = torch.randint(0, tx.shape[0], (cfg.batch_size,), generator=self.rng)
        else:
            idx = torch.multinomial(tw, cfg.batch_size, replacement=True,
                                    generator=self.rng)
        x, y = tx[idx], ty[idx]
        logits = self.model.answer_logits(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
        lr = cfg.lr * min(1.0, (self.step + 1) / max(1, cfg.lr_warmup))
        self.opt.set_lr(lr, self.step)
        self.opt.step()
        self.last_loss = float(loss.detach())
        self.flops += self.model.flops_per_token(True) * cfg.batch_size * 4
        self.tokens += cfg.batch_size * 4
        self.step += 1
        # strain is read from the gradients of the step we just took
        if self.step % cfg.controller.eval_every == 0:
            self._strain = block_strain(self.model)
        self.opt.zero_grad()

    def _observe(self) -> None:
        vl, per = self.val_loss()
        dec = self.ctrl.observe(self.step, vl, per, getattr(self, "_strain", {}),
                                len(self.model.live_blocks()))
        self.trace.append({
            "step": self.step, "age": self.age, "flops": self.flops,
            "tokens": self.tokens, "train_loss": getattr(self, "last_loss", None),
            "val_loss": vl, "per_cat": {k: round(v, 4) for k, v in per.items()},
            "params": self.model.n_params(), "params_active": self.model.n_params(True),
            "n_blocks": len(self.model.live_blocks()),
            "action": dec.get("action"),
        })
        if dec.get("action") == "GROW" and self.competition is None:
            self._do_growth(dec)

    def _write_ledger(self, age: int) -> None:
        """Predict every fact the model has NOT been shown, and commit those
        predictions before the next tranche is revealed."""
        targets = self.world.unrevealed_pool(age) + self.world.holdout()
        preds = self.predict(targets)
        arch = self.model.arch_signature()
        head = {
            "kind": "checkpoint", "age": age, "step": self.step,
            "flops": self.flops, "tokens": self.tokens,
            "params": arch["params_total"], "params_active": arch["params_active"],
            "n_blocks": arch["n_blocks"],
            "arch": json.dumps(arch["blocks"], sort_keys=True),
            "wall_s": round(time.time() - self.t0, 2),
        }
        self.ledger.append(head)
        for p in preds:
            p.update(kind="prediction", age=age, step=self.step, flops=self.flops)
            self.ledger.append(p)

    def _finish(self) -> dict:
        arch = self.model.arch_signature()
        all_uids = [m.unit_id for _, m in self.model.units()]
        final_abl = G.ablation_scores(self.model, all_uids, self.vx, self.vy)
        final_abl_rel = G.ablation_by_relation(self.model, all_uids, self.vx,
                                               self.vy, self.val_rel)
        for ev in self.events:
            ev.final_ablation = {u: round(final_abl.get(u, float("nan")), 6)
                                 for u in ev.survivors}
        out = {
            "config": {**{k: v for k, v in asdict(self.cfg).items()}},
            "world": self.world.summary(),
            "final_arch": arch,
            "events": [asdict(e) for e in self.events],
            "controller_log": getattr(self.ctrl, "log", []),
            "trace": self.trace,
            "final_ablation": {k: round(v, 6) for k, v in final_abl.items()},
            "final_ablation_by_relation": final_abl_rel,
            "compute": {
                "flops": self.flops, "tokens": self.tokens, "steps": self.step,
                "wall_s": round(time.time() - self.t0, 2),
                "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                "inference_flops_per_token": self.model.flops_per_token(False),
            },
        }
        with open(os.path.join(self.cfg.out, "run.json"), "w") as f:
            json.dump(out, f, indent=1)
        torch.save({"arch": arch, "state_dict": self.model.state_dict()},
                   os.path.join(self.cfg.out, "model.pt"))
        return out


def load_config(path: str, **over) -> RunConfig:
    raw = json.load(open(path)) if path else {}
    raw = {k: v for k, v in raw.items() if not k.startswith("_")}   # drop comments
    raw.update(over)
    cc = ControllerConfig(**raw.pop("controller", {}))
    return RunConfig(controller=cc, **raw)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="")
    ap.add_argument("--group", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--match_arch", default=None)
    ap.add_argument("--match_events", default=None)
    ap.add_argument("--flops_budget", type=float, default=None)
    a = ap.parse_args()
    over = {k: v for k, v in vars(a).items() if k != "config" and v is not None}
    cfg = load_config(a.config, **over)
    r = Trainer(cfg).run()
    c = r["compute"]
    print(f"[{cfg.group} s{cfg.seed}] steps={c['steps']} flops={c['flops']:.3e} "
          f"params={r['final_arch']['params_total']/1e6:.3f}M "
          f"blocks={r['final_arch']['n_blocks']} wall={c['wall_s']}s "
          f"events={len(r['events'])}")


if __name__ == "__main__":
    main()
