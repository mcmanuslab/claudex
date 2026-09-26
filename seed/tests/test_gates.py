"""
tests/test_gates.py -- the experiment's own assertions.

These are not unit tests for their own sake. Each one pins a property that, if
it silently broke, would make the headline results wrong rather than merely
absent. Run with: python3 -m pytest tests/ -q   (or python3 tests/test_gates.py)
"""
import os, sys, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

import growth as G
import ledger as L
import metrics as M
from controller import ControllerConfig, DevelopmentalController, RandomController
from model import ModelConfig, SeedTransformer
from world import (ATOMIC_RELATIONS, DISTANT_PERIOD, D_DISTANT, D_UNSUPPORTED,
                   World, WorldSpec)


# ---------------------------------------------------------------- GATE 1
def test_depth_growth_is_exactly_function_preserving():
    torch.manual_seed(0)
    m = SeedTransformer(ModelConfig())
    probe = torch.randint(0, 71, (64, 5))
    before = G.snapshot_logits(m, probe)
    for pos in (0, 2, len(m.blocks)):
        G.grow_depth(m, pos)
        assert G.logit_delta(m, before, probe) == 0.0


def test_mlp_growth_is_exactly_function_preserving():
    torch.manual_seed(0)
    m = SeedTransformer(ModelConfig())
    probe = torch.randint(0, 71, (64, 5))
    before = G.snapshot_logits(m, probe)
    for _ in range(3):
        G.add_mlp_branch(m, 1, 128)
        assert G.logit_delta(m, before, probe) == 0.0


def test_growth_preserves_function_on_a_TRAINED_model():
    """Preservation on a random net is easy. The claim that matters is that a
    model which has actually learned something does not lose it."""
    torch.manual_seed(0)
    w = World(WorldSpec(seed=7))
    m = SeedTransformer(ModelConfig(vocab_size=len(w.vocab)))
    facts = w.revealed(3)
    x = torch.tensor([w.encode(f)[0] for f in facts[:512]])
    y = x[:, -1].clone()
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    for _ in range(150):
        loss = F.cross_entropy(m.answer_logits(x), y)
        opt.zero_grad(); loss.backward(); opt.step()
    # NB: recompute after the final optimizer step -- `loss` inside the loop
    # was measured BEFORE the last update, so comparing against it would
    # attribute one ordinary training step to the growth operator.
    with torch.no_grad():
        trained = float(F.cross_entropy(m.answer_logits(x), y))
    assert trained < 1.0, f"the probe model did not learn ({trained})"
    before = G.snapshot_logits(m, x)
    acc_before = (m.answer_logits(x).argmax(-1) == y).float().mean()
    G.grow_depth(m, 2); G.add_mlp_branch(m, 0, 256)
    assert G.logit_delta(m, before, x) == 0.0
    after = float(F.cross_entropy(m.answer_logits(x), y).detach())
    acc_after = (m.answer_logits(x).argmax(-1) == y).float().mean()
    assert abs(after - trained) < 1e-6, (trained, after)
    assert acc_before == acc_after            # zero catastrophic forgetting at t=0


def test_new_params_get_lr_warmup_not_a_full_adam_step():
    """Staged Training's point: preserving the loss is useless if the first
    optimizer step after growth destroys it. A zero-moment AdamW parameter
    takes a ~lr-sized step regardless of gradient magnitude."""
    torch.manual_seed(0)
    m = SeedTransformer(ModelConfig())
    opt = G.GrowthAwareOptimizer(m, lr=1e-2, warmup=100)
    blk = G.grow_depth(m, 1)
    opt.rebuild(m, list(blk.parameters()), step=500)
    opt.set_lr(1e-2, 500)
    lrs = [g["lr"] for g in opt.opt.param_groups]
    assert min(lrs) == 0.0, lrs           # the newborn cohort starts at zero LR
    opt.set_lr(1e-2, 550)
    assert 0 < min(g["lr"] for g in opt.opt.param_groups) < 1e-2
    opt.set_lr(1e-2, 700)
    assert min(g["lr"] for g in opt.opt.param_groups) == 1e-2


def test_optimizer_moments_survive_growth():
    torch.manual_seed(0)
    m = SeedTransformer(ModelConfig())
    opt = G.GrowthAwareOptimizer(m, lr=1e-3)
    x = torch.randint(0, 71, (16, 5))
    for _ in range(5):
        F.cross_entropy(m.answer_logits(x), x[:, -1]).backward()
        opt.step(); opt.zero_grad()
    p = m.tok.weight
    m1 = opt.opt.state[p]["exp_avg"].clone()
    blk = G.grow_depth(m, 1)
    opt.rebuild(m, list(blk.parameters()), step=5)
    assert torch.equal(opt.opt.state[p]["exp_avg"], m1)


# ---------------------------------------------------------------- pruning
def test_pruning_actually_removes_parameters_and_flops():
    """Gating a unit off is not pruning. If 'pruned' capacity kept costing
    FLOPs, every compute-matched comparison in the study would be wrong."""
    torch.manual_seed(0)
    m = SeedTransformer(ModelConfig())
    br = G.add_mlp_branch(m, 0, 256)
    p1, f1 = m.n_params(), m.flops_per_token()
    assert G.remove_unit(m, br.unit_id)
    assert m.n_params() < p1 and m.flops_per_token() < f1


def test_flops_accounting_charges_for_live_candidates():
    torch.manual_seed(0)
    m = SeedTransformer(ModelConfig())
    base = m.flops_per_token()
    uids = [G.add_mlp_branch(m, 0, 256).unit_id for _ in range(4)]
    assert m.flops_per_token() > base * 1.1     # overgrowth is paid for
    for u in uids[1:]:
        G.remove_unit(m, u)
    assert m.flops_per_token() < base * 1.4


def test_random_selection_mode_ignores_scores():
    scores = {"a": 1.0, "b": 0.0, "c": -1.0, "d": -2.0}
    red = {}
    picks = set()
    for s in range(30):
        g = torch.Generator().manual_seed(s)
        surv, _, why = G.select_survivors(scores, red, keep=1, min_gain=0.0,
                                          redundancy_thresh=.95, rng=g,
                                          random_mode=True)
        picks.add(surv[0]); assert why["mode"] == "random"
    assert len(picks) > 1, "random pruning must not always pick the same unit"
    surv, pruned, why = G.select_survivors(scores, red, keep=1, min_gain=0.0,
                                           redundancy_thresh=.95)
    assert surv == ["a"] and why["mode"] == "competitive"


def test_redundant_candidate_is_pruned_even_when_useful():
    scores = {"a": 1.0, "b": 0.99}
    red = {("a", "b"): 0.999}
    surv, pruned, _ = G.select_survivors(scores, red, keep=2, min_gain=0.0,
                                         redundancy_thresh=0.95)
    assert surv == ["a"] and pruned == ["b"]


# ---------------------------------------------------------------- controller
def test_controller_fires_on_plateau_not_on_the_clock():
    cfg = ControllerConfig(eval_every=100, window=5, plateau_rel=0.02,
                           patience=2, cooldown=400, warmup_steps=200, max_events=4)
    flat = DevelopmentalController(cfg)
    steep = DevelopmentalController(cfg)
    for i, s in enumerate(range(0, 4000, 100)):
        flat.observe(s, 0.5 + 1e-4 * (40 - i), {}, {"L0": 1.0}, 4)
        steep.observe(s, 3.0 * (0.90 ** i) + 0.05, {}, {"L0": 1.0}, 4)
    assert flat.events > 0, "a flat loss curve must trigger growth"
    assert steep.events == 0, "a still-improving curve must NOT trigger growth"


def test_controller_records_why():
    cfg = ControllerConfig(warmup_steps=100, cooldown=200, plateau_rel=.02)
    c = DevelopmentalController(cfg)
    for s in range(0, 3000, 100):
        c.observe(s, 1.0, {"HEAVIER": 2.0, "VAL": .1}, {"L1": 9.0, "L0": 1.0}, 4)
    g = [r for r in c.log if r["action"] == "GROW"]
    assert g and "rel_improvement_over_window" in g[0]
    assert g[0]["site"] == "L1" and g[0]["worst_category"] == "HEAVIER"


def test_random_controller_matches_event_count_and_kinds():
    cfg = ControllerConfig(max_events=4, depth_first=2, warmup_steps=200, cooldown=300)
    r = RandomController(cfg, total_steps=5000, seed=3)
    assert len(r.steps) == cfg.max_events
    assert r.kinds.count("GROW_DEPTH") == cfg.depth_first
    fired = sum(r.observe(s, 1.0, {}, {"L0": 1.0}, 4)["action"] == "GROW"
                for s in range(0, 5000, 50))
    assert fired == cfg.max_events


# ---------------------------------------------------------------- world
def test_token_ids_do_not_leak_latent_coordinates():
    for s in range(6):
        w = World(WorldSpec(seed=s))
        for k, v in w.token_id_leak.items():
            assert abs(v) < 0.05, (s, k, v)


def test_unsupported_bucket_is_genuinely_unpredictable():
    """D4 is our leakage alarm, so it must be unpredictable from anything the
    model can see. Two entities with identical latent coordinates must be able
    to disagree on OMEN."""
    w = World(WorldSpec(seed=0))
    omen = [f for f in w.facts if f.rel == "OMEN"]
    assert 0.35 < sum(f.answer for f in omen) / len(omen) < 0.65
    # OMEN is of course a deterministic function of the entity PAIR -- every
    # pair has distinct latent coordinates, so it could not be otherwise, and
    # a large enough model can memorise the OMEN facts it is shown. The
    # property that makes it a valid leakage alarm is different and stronger:
    # its answer must be uncorrelated with every latent feature a model could
    # generalise FROM, so that held-out OMEN stays unpredictable.
    from world import _pearson
    feats = {
        "group_gap": lambda f: abs(w.group[f.a1] - w.group[f.a2]),
        "period_gap": lambda f: abs(w.period[f.a1] - w.period[f.a2]),
        "group_sum": lambda f: w.group[f.a1] + w.group[f.a2],
        "mass_order": lambda f: int(w.mass[f.a1] > w.mass[f.a2]),
        "a1_group": lambda f: w.group[f.a1],
        "a1_period": lambda f: w.period[f.a1],
    }
    ys = [f.answer for f in omen]
    for name, fn in feats.items():
        r = _pearson([fn(f) for f in omen], ys)
        assert abs(r) < 0.12, f"OMEN correlates with {name}: r={r:.3f}"


def test_distant_bucket_is_grounded_but_withheld():
    """D3 must be extrapolation, not ignorance: the period of a distant entity
    must be stated at age 0, while the rule's application at that period is
    withheld entirely."""
    w = World(WorldSpec(seed=0))
    d5 = [e for e in range(48) if w.period[e] == DISTANT_PERIOD]
    assert d5
    for e in d5:
        shell = [f for f in w.facts if f.rel == "SHELL" and f.a1 == e]
        assert shell and shell[0].split == "pool" and shell[0].tranche == 0
        heav = [f for f in w.facts if f.rel == "HEAVIER" and e in (f.a1, f.a2)]
        assert heav and all(f.split == "holdout" for f in heav)


def test_atomic_facts_are_all_available_at_age_zero():
    w = World(WorldSpec(seed=1))
    for f in w.facts:
        if f.rel in ATOMIC_RELATIONS and f.split == "pool":
            assert f.tranche == 0


def test_no_fact_appears_in_two_splits_and_revelation_is_monotone():
    w = World(WorldSpec(seed=2))
    assert len({f.fid for f in w.facts}) == len(w.facts)
    prev = set()
    for age in range(8):
        cur = {f.fid for f in w.revealed(age)}
        assert prev <= cur, "revelation must never un-reveal a fact"
        assert not (cur & {f.fid for f in w.holdout()})
        assert not (cur & {f.fid for f in w.val_facts()})
        prev = cur


def test_holdout_is_never_trainable_at_any_age():
    w = World(WorldSpec(seed=3))
    hold = {f.fid for f in w.holdout()}
    for age in range(12):
        assert not ({f.fid for f in w.revealed(age)} & hold)


# ---------------------------------------------------------------- ledger
def test_ledger_detects_tampering(tmp_path="/tmp/_seedtest"):
    import json as J
    os.makedirs(tmp_path, exist_ok=True)
    p = os.path.join(tmp_path, "l.jsonl")
    lg = L.Ledger(p)
    lg.append_many([{"age": 0, "fid": i, "pred_token": i, "conf": .5} for i in range(5)])
    assert L.verify(p)[0]
    rows = open(p).read().splitlines()
    d = J.loads(rows[2]); d["pred_token"] = 999
    rows[2] = J.dumps(d, sort_keys=True, separators=(",", ":"))
    open(p, "w").write("\n".join(rows) + "\n")
    assert not L.verify(p)[0]


def test_ledger_refuses_to_store_ground_truth():
    lg = L.Ledger("/tmp/_seedtest/l2.jsonl")
    for bad in ({"correct": 1}, {"truth": 3}):
        try:
            lg.append(bad); raise AssertionError("should have refused " + str(bad))
        except AssertionError as e:
            assert "ledger must never" in str(e) or "should have refused" not in str(e)


# ---------------------------------------------------------------- metrics
def test_majority_guesser_scores_near_zero_information_gain():
    """The direct test of 'a model can game this by making safe predictions'."""
    w = World(WorldSpec(seed=0))
    prior = w.answer_prior(0)
    recs = []
    for f in w.holdout():
        maj = max(prior[f.rel], key=prior[f.rel].get)
        recs.append({"kind": "prediction", "fid": f.fid, "rel": f.rel,
                     "dist": f.dist, "split": "holdout", "tranche": -1,
                     "pred_token": w.vocab.ans(maj), "conf": 0.9, "entropy": 0.1,
                     "p_answers": [1.0 if a == maj else 0.0
                                   for a in range(w.vocab.n_ans)]})
    s = M.summarize(M.score_records(w, recs, 0), "holdout")
    assert s["overall_info_gain_bits"] < 0.85, s["overall_info_gain_bits"]
    oracle = []
    for f in w.holdout():
        oracle.append({**recs[0], "fid": f.fid, "rel": f.rel, "dist": f.dist,
                       "pred_token": w.vocab.ans(f.answer),
                       "p_answers": [1.0 if a == f.answer else 0.0
                                     for a in range(w.vocab.n_ans)]})
    so = M.summarize(M.score_records(w, oracle, 0), "holdout")
    assert so["overall_info_gain_bits"] > s["overall_info_gain_bits"] * 1.5


def test_leakage_alarm_fires_on_an_oracle():
    w = World(WorldSpec(seed=0))
    recs = [{"kind": "prediction", "fid": f.fid, "rel": f.rel, "dist": f.dist,
             "split": "holdout", "tranche": -1,
             "pred_token": w.vocab.ans(f.answer), "conf": .9, "entropy": .1,
             "p_answers": [1.0 if a == f.answer else 0.0 for a in range(w.vocab.n_ans)]}
            for f in w.holdout()]
    s = M.summarize(M.score_records(w, recs, 0), "holdout")
    assert s["leakage_flag"], "a cheating model must trip the D4 alarm"


def test_primary_endpoint_weights_buckets_equally():
    w = World(WorldSpec(seed=0))
    recs = []
    for f in w.holdout():
        # right on D3 only; if buckets were weighted by count, D3's ~700 facts
        # would move the primary very differently than an equal weighting does
        ok = (f.dist == 3)
        recs.append({"kind": "prediction", "fid": f.fid, "rel": f.rel,
                     "dist": f.dist, "split": "holdout", "tranche": -1,
                     "pred_token": w.vocab.ans(f.answer if ok else (f.answer + 1) % 5),
                     "conf": .5, "entropy": 1.0, "p_answers": [0.2] * w.vocab.n_ans})
    s = M.summarize(M.score_records(w, recs, 0), "holdout")
    assert abs(s["PRIMARY_extrapolative_accuracy"] - 1 / 3) < 0.02


if __name__ == "__main__":
    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_")]
    bad = 0
    for name, fn in fns:
        try:
            fn(); print(f"  PASS  {name}")
        except Exception as e:
            bad += 1; print(f"  FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{len(fns)-bad}/{len(fns)} passed")
    sys.exit(1 if bad else 0)
