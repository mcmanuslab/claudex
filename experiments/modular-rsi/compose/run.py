"""Composition test (pre-registered in PREREG.md).

  python run.py --system modular|looped|transformer --lr 3e-3 --seed 0
  python run.py --system evolved --seed 0
"""
import argparse, copy, json, os, sys, time
import numpy as np
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from modrsi.world import GeneticCode, ALL_SKILLS, sample_contexts, apply_primitive, V
from models import Chain, ProgramTransformer, ctx_parts, n_params

torch.set_num_threads(1)
OUT = os.path.join(HERE, "results"); os.makedirs(OUT, exist_ok=True)
STEPS, B = 6000, 128
TEST_LENS = (1, 2, 3, 4, 6, 8, 12, 16)


def batch(rng, B, lens):
    ctx = sample_contexts(rng, B, n_tables=4)
    x0 = rng.integers(0, V, B)
    L = rng.choice(lens, B)
    Lmax = int(max(lens))
    prog = np.full((B, Lmax), -1)
    for i in range(B):
        prog[i, : L[i]] = rng.integers(0, len(ALL_SKILLS), L[i])
    y = x0.copy()
    for j in range(Lmax):
        for k, op in enumerate(ALL_SKILLS):
            m = prog[:, j] == k
            if m.any():
                y[m] = apply_primitive({n: a[m] for n, a in ctx.items()}, op, y[m])
    return ctx, x0, prog, y


def logits_of(model, ctx, x0, prog, hard=False):
    if isinstance(model, ProgramTransformer):
        return model(ctx, x0, prog)
    return model(ctx_parts(ctx), x0, prog, hard=hard)


@torch.no_grad()
def accuracy(model, data, hard):
    ctx, x0, prog, y = data
    accs = []
    for s in range(0, len(x0), 256):
        sl = slice(s, s + 256)
        lg = logits_of(model, {k: v[sl] for k, v in ctx.items()}, x0[sl], prog[sl], hard)
        accs.append((lg.argmax(-1).numpy() == y[sl]))
    return float(np.concatenate(accs).mean())


def make(system, code):
    return ProgramTransformer(code) if system == "transformer" else Chain(code, system)


def train(model, opt, rng, steps):
    for _ in range(steps):
        ctx, x0, prog, y = batch(rng, B, (1, 2, 3))
        lg = logits_of(model, ctx, x0, prog)
        loss = torch.nn.functional.cross_entropy(lg, torch.from_numpy(y))
        opt.zero_grad(); loss.backward(); opt.step()


def evaluate(model, val, tests):
    out = dict(val=accuracy(model, val, hard=False))
    for L, d in tests.items():
        out[f"hard_{L}"] = accuracy(model, d, hard=True)
        if not isinstance(model, ProgramTransformer):
            out[f"soft_{L}"] = accuracy(model, d, hard=False)
    return out


def fixed_sets():
    val = batch(np.random.default_rng(777), 1024, (1, 2, 3))
    tests = {L: batch(np.random.default_rng(1000 + L), 1024, (L,)) for L in TEST_LENS}
    return val, tests


def run(system, lr, seed, steps=STEPS):
    code = GeneticCode(0); val, tests = fixed_sets()
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    model = make(system, code)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    t0 = time.time()
    train(model, opt, rng, steps)
    res = dict(system=system, lr=lr, seed=seed, params=n_params(model), steps=steps,
               seconds=time.time() - t0, **evaluate(model, val, tests))
    print(json.dumps(res), flush=True)
    json.dump(res, open(os.path.join(OUT, f"{system}_lr{lr}_s{seed}.json"), "w"), indent=1)


def run_evolved(seed, pop=6, gen_steps=250, total=3 * STEPS):
    """PBT-style evolution of the modular architecture: same total gradient steps as one
    system's lr grid (3 x STEPS), split across the population."""
    code = GeneticCode(0); val, tests = fixed_sets()
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    members = []
    for i in range(pop):
        m = make("modular", code)
        lr = float(np.exp(rng.uniform(np.log(1e-3), np.log(1e-2))))
        members.append(dict(model=m, opt=torch.optim.Adam(m.parameters(), lr=lr), lr=lr, sigma=0.02))
    used, gens, t0 = 0, 0, time.time()
    while used + pop * gen_steps <= total:
        for mb in members:
            train(mb["model"], mb["opt"], rng, gen_steps)
        used += pop * gen_steps; gens += 1
        scores = [accuracy(mb["model"], val, hard=False) for mb in members]
        order = np.argsort(scores)[::-1]
        top, bottom = order[: pop // 2], order[pop // 2:]
        for b, t in zip(bottom, top):
            src = members[t]
            child = copy.deepcopy(src["model"])
            sigma = float(np.clip(src["sigma"] * np.exp(0.3 * rng.standard_normal()), 1e-3, 0.2))
            lr = float(np.clip(src["lr"] * np.exp(0.3 * rng.standard_normal()), 3e-4, 3e-2))
            with torch.no_grad():
                for p in child.parameters():
                    p.add_(sigma * p.abs().mean() * torch.randn_like(p))
            opt = torch.optim.Adam(child.parameters(), lr=lr)
            opt.load_state_dict(src["opt"].state_dict())
            for g in opt.param_groups:
                g["lr"] = lr
            members[b] = dict(model=child, opt=opt, lr=lr, sigma=sigma)
    best = members[int(np.argmax([accuracy(mb["model"], val, hard=False) for mb in members]))]
    res = dict(system="evolved", seed=seed, params=n_params(best["model"]), steps_total=used, generations=gens,
               final_lr=best["lr"], final_sigma=best["sigma"], seconds=time.time() - t0,
               **evaluate(best["model"], val, tests))
    print(json.dumps(res), flush=True)
    json.dump(res, open(os.path.join(OUT, f"evolved_s{seed}.json"), "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", default="modular"); ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--steps", type=int, default=STEPS)
    a = ap.parse_args()
    run_evolved(a.seed) if a.system == "evolved" else run(a.system, a.lr, a.seed, a.steps)
