"""pilot.py -- calibration only. Runs on world_seed 99, which is NEVER used in
the frozen experiment. Its job is to answer three questions before the protocol
is frozen: does the task learn at all, does FIXED SMALL actually plateau at a
non-trivial error (if it solved the task, growth would have nothing to do), and
what plateau threshold corresponds to a real stall."""
import json, sys, time
import torch, torch.nn.functional as F
from train import RunConfig, Trainer, facts_to_tensor
from world import DIST_NAMES

def holdout_acc(tr):
    facts = tr.world.holdout()
    preds = tr.predict(facts)
    by = {}
    for f, p in zip(facts, preds):
        ok = (p["pred_token"] == tr.world.vocab.ans(f.answer))
        by.setdefault(f.dist, []).append(ok)
    return {DIST_NAMES[d]: sum(v)/len(v) for d, v in sorted(by.items())}

cfg = RunConfig(group="A", seed=0, world_seed=99, out="/tmp/pilot",
                flops_budget=float(sys.argv[1]) if len(sys.argv)>1 else 1.5e13, n_ages=8)
tr = Trainer(cfg)
per_age = cfg.flops_budget / cfg.n_ages
print(f"{'step':>6} {'age':>3} {'train':>7} {'val':>7} {'D0':>6} {'D1':>6} {'D2':>6} {'D3':>6} {'D4':>6}  worst-cat")
t0=time.time()
for age in range(cfg.n_ages):
    tr.age = age
    rev = tr.world.revealed(age)
    tx, ty = facts_to_tensor(tr.world, rev)
    target = per_age*(age+1)
    while tr.flops < target and tr.step < cfg.max_steps:
        tr._train_step(tx, ty)
        if tr.step % 500 == 0:
            vl, per = tr.val_loss()
            h = holdout_acc(tr)
            worst = max(per, key=per.get)
            print(f"{tr.step:6d} {age:3d} {tr.last_loss:7.4f} {vl:7.4f} "
                  + " ".join(f"{h.get(k,0):6.3f}" for k in
                    ["D0-interpolation","D1-near","D2-compositional","D3-distant","D4-unsupported"])
                  + f"  {worst}={per[worst]:.3f}")
print(f"\nwall {time.time()-t0:.0f}s  steps {tr.step}  flops {tr.flops:.2e}")
vl, per = tr.val_loss()
print("final per-category val loss:", json.dumps({k: round(v,3) for k,v in sorted(per.items(), key=lambda kv:-kv[1])}))
