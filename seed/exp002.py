"""exp002.py -- the three follow-ups Experiment 001 pointed at.

A  GROWTH-TIME SWEEP.  Exp 001 could not separate "the unit had more time to
   integrate" from "the unit was born earlier", because with a fixed budget
   steps_alive = total - birth_step makes them one variable. Here the birth step
   is forced and the run then continues for a FIXED number of further steps, so
   every unit gets exactly the same time to integrate regardless of when it was
   born. Flat effect vs birth step => time is what matters, and recursion just
   needs a longer budget. Declining effect => earliness is what matters, and
   recursion is self-defeating, because each generation is later than the last.

B  LONG-HORIZON RECURSION.  Exp 001 gave growth 3 cycles crammed into the back
   half of a short run. Here: double the budget, up to 10 cycles, so later
   generations actually have room to build on earlier ones.

C  STAGED EXPRESSION.  A seed's genome is fixed; growth expresses a blueprint it
   already holds. The analogue is allocating the whole architecture at step 0 --
   never creating anything mid-run -- and switching units on over time. Tests
   staged EXPRESSION of pre-allocated capacity against staged CREATION of new
   capacity, at the same architecture and the same timing.
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time

SWEEP_BIRTHS = [0, 400, 800, 1600, 2400, 3200]
SWEEP_POST = 2400          # every swept unit gets exactly this long to integrate


def launch(name, root, threads=1, **over):
    out = os.path.join(root, name)
    if os.path.exists(os.path.join(out, "run.json")):
        return None
    cmd = [sys.executable, "-u", "-c",
           "import json,sys;from train import RunConfig,Trainer;"
           "from controller import ControllerConfig;"
           "o=json.loads(sys.argv[1]);cc=ControllerConfig(**o.pop('controller',{}));"
           "Trainer(RunConfig(controller=cc,**o)).run()",
           json.dumps({**over, "out": out})]
    env = dict(os.environ, SEED_THREADS=str(threads), OMP_NUM_THREADS=str(threads))
    log = open(os.path.join(root, name + ".log"), "w")
    return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)


def wait(procs, label):
    procs = [p for p in procs if p]
    t0 = time.time()
    for p in procs:
        p.wait()
    bad = [p.returncode for p in procs if p.returncode]
    print(f"  {label}: {len(procs)} runs in {time.time()-t0:.0f}s"
          + (f"  FAILURES {bad}" if bad else ""), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/exp002")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--arms", default="A,B,C")
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.split(",")]
    os.makedirs(a.root, exist_ok=True)
    arms = a.arms.split(",")
    base = dict(d_model=64, n_head=4, d_ff=128, n_layer=2, lr=3e-3, batch_size=128,
                sampling_alpha=0.5, n_ages=8, branch_d_ff=128, flops_budget=0.0)

    if "A" in arms:
        print(f"arm A: growth-time sweep, births {SWEEP_BIRTHS}, "
              f"each unit gets {SWEEP_POST} steps to integrate", flush=True)
        jobs = [(s, b) for s in seeds for b in SWEEP_BIRTHS]
        for i in range(0, len(jobs), a.parallel):
            procs = [launch(f"sweep_b{b}_s{s}", a.root, group="SWEEP", seed=s,
                            step_budget=b + SWEEP_POST, forced_growth_steps=[b],
                            n_candidates=1, **base)
                     for s, b in jobs[i:i + a.parallel]]
            wait(procs, f"sweep batch {i//a.parallel+1}")

    if "B" in arms:
        print("arm B: long-horizon recursion, 2x budget, up to 10 cycles", flush=True)
        ctrl = dict(eval_every=100, window=3, plateau_rel=0.02, patience=2,
                    cooldown=900, max_events=10, max_blocks=14, depth_first=5,
                    warmup_steps=400)
        jobs = [(g, s) for s in seeds for g in ("A", "D")]
        for i in range(0, len(jobs), a.parallel):
            procs = [launch(f"long_{g}_s{s}", a.root, group=g, seed=s,
                            n_candidates=4 if g == "D" else 1, keep=1,
                            dev_window=400, min_gain=None, controller=ctrl,
                            **{**base, "flops_budget": 4.0e12})
                     for g, s in jobs[i:i + a.parallel]]
            wait(procs, f"long batch {i//a.parallel+1}")

    if "C" in arms:
        print("arm C: staged expression of a pre-allocated architecture", flush=True)
        procs = []
        for s in seeds:
            d = f"results/exp001/D_s{s}/run.json"
            if not os.path.exists(d):
                print(f"  skip S seed {s}: reference D run missing"); continue
            procs.append(launch(f"staged_s{s}", a.root, group="S", seed=s,
                                match_arch=d, expression_ramp=400,
                                **{**base, "flops_budget": 2.0e12}))
        wait(procs, "staged batch")
    print("done", flush=True)


if __name__ == "__main__":
    main()
