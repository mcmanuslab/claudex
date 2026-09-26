"""run_experiment.py -- the frozen protocol, executed.

Two phases, because FIXED LARGE must match the architecture the developmental
model actually ends with rather than an architecture we guessed in advance:

  phase 1   A (fixed small), C (growth only), D (developmental),
            D_RANDPRUNE (competition with random selection), R (random schedule)
  phase 2   B (fixed large), built to copy D's final architecture for the
            same seed, trained from scratch under the same FLOPs budget
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time

PHASE1 = ["A", "C", "D", "D_RANDPRUNE"]


def launch(group, seed, cfg, root, match=None, match_events=None, threads=1):
    out = os.path.join(root, f"{group}_s{seed}")
    if os.path.exists(os.path.join(out, "run.json")):
        return None
    cmd = [sys.executable, "-u", "train.py", "--config", cfg, "--group", group,
           "--seed", str(seed), "--out", out]
    if match:
        cmd += ["--match_arch", match]
    if match_events:
        cmd += ["--match_events", match_events]
    env = dict(os.environ, SEED_THREADS=str(threads), OMP_NUM_THREADS=str(threads))
    log = open(os.path.join(root, f"{group}_s{seed}.log"), "w")
    return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)


def wait_all(procs, label):
    procs = [p for p in procs if p]
    t0 = time.time()
    for p in procs:
        p.wait()
    bad = [p.returncode for p in procs if p.returncode != 0]
    print(f"  {label}: {len(procs)} runs in {time.time()-t0:.0f}s"
          + (f"  FAILURES: {bad}" if bad else ""))
    return not bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/exp001_cpu.json")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--root", default="results/exp001")
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    os.makedirs(a.root, exist_ok=True)

    jobs = [(g, s) for s in seeds for g in PHASE1]
    print(f"phase 1: {len(jobs)} runs, {a.parallel} at a time, {a.threads} thread(s) each")
    for i in range(0, len(jobs), a.parallel):
        chunk = jobs[i:i + a.parallel]
        procs = [launch(g, s, a.config, a.root, threads=a.threads) for g, s in chunk]
        wait_all(procs, f"batch {i//a.parallel+1} {[f'{g}{s}' for g,s in chunk]}")

    print("phase 2: FIXED LARGE (arch-matched to D) and RANDOM DEV "
          "(event-matched to D), per seed")
    procs = []
    for s in seeds:
        d = os.path.join(a.root, f"D_s{s}", "run.json")
        if not os.path.exists(d):
            print(f"  skip seed {s}: D run missing"); continue
        procs.append(launch("B", s, a.config, a.root, match=d, threads=a.threads))
        procs.append(launch("R", s, a.config, a.root, match_events=d, threads=a.threads))
        if len(procs) >= a.parallel:
            wait_all(procs, "phase2 batch"); procs = []
    wait_all(procs, "phase2 batch")
    print("done. now: python3 evaluate.py && python3 visualize.py")


if __name__ == "__main__":
    main()
