"""The parent (expensive search) and the decision-situation generator.

For a new skill, candidates = source ∈ {fresh, copy of each library adapter} × rank ∈ {4, 16}.
The parent trains every candidate for S adapter steps (core frozen) and measures validation
loss; its choice is the argmin. Alongside, cheap features for the student are recorded."""
import os, sys, json, time
import numpy as np
import torch
from world import batch, random_op, change_profile, R
from model import Core, new_adapters, stack, unstack, loss_acc, RMAX, TRAINABLE, to_t

torch.set_num_threads(int(os.environ.get("THREADS", "1")))
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results"); os.makedirs(OUT, exist_ok=True)
RANKS = (4, 16)
S_DEFAULT, B, LR = 150, 32, 3e-3


def load_core():
    ck = torch.load(os.path.join(OUT, "core.pt"))
    core = Core(); core.load_state_dict(ck["core"])
    for p in core.parameters():
        p.requires_grad_(False)
    lib = [dict(skill=s, ad=unstack(ck["adapters"], i), rank=16, parent=None, copies=0, wins=0,
                depth=0, age=0, own_acc=float(ck["acc"][i])) for i, s in enumerate(ck["skills"])]
    return core, lib


def candidate_adapter(lib, src, rank, gen):
    a = unstack(new_adapters(1, rank, gen), 0)
    if src >= 0:
        s = lib[src]["ad"]
        k = min(rank, lib[src]["rank"])
        a["emb"] = s["emb"].clone()
        for down, up in (("Wd", "Wu"), ("A0", "B0"), ("A1", "B1")):
            a[down][:, :k] = s[down][:, :k]          # copy the source's first k ranks;
            a[up][:k] = s[up][:k]                    # extra ranks stay fresh (zero up-projection)
    return a


def candidates(lib, gen):
    srcs = [-1] + list(range(len(lib)))
    return [(s, r) for s in srcs for r in RANKS], [candidate_adapter(lib, s, r, gen) for s in srcs for r in RANKS]


def evaluate(core, ad, val):
    with torch.no_grad():
        C = ad["emb"].shape[0]
        return loss_acc(core, ad, *(v.unsqueeze(0).expand(C, *v.shape) for v in val))


def train(core, ad, skill, rng, steps):
    ad = {k: v.clone() for k, v in ad.items()}
    for k in TRAINABLE:
        ad[k].requires_grad_(True)
    opt = torch.optim.Adam([ad[k] for k in TRAINABLE], lr=LR)
    C = ad["emb"].shape[0]
    for _ in range(steps):
        l, _, _ = loss_acc(core, ad, *to_t(batch(rng, skill, B), C))
        opt.zero_grad(); l.sum().backward(); opt.step()
    return {k: v.detach() for k, v in ad.items()}


def val_set(skill, seed, n=512):
    return tuple(to_t(batch(np.random.default_rng(seed), skill, n)))


def features(core, lib, skill, cands, ads, rng):
    """Cheap features per candidate (what the student sees)."""
    pv = val_set(skill, 4242, 256)
    l0, a0, sl0 = evaluate(core, ads, pv)                     # zero-shot probe of each candidate init
    prof = change_profile(skill, np.random.default_rng(7))
    feats = []
    for c, (src, rank) in enumerate(cands):
        e = lib[src] if src >= 0 else None
        sp = change_profile(e["skill"], np.random.default_rng(7)) if e else np.zeros_like(prof)
        feats.append([1.0 if src < 0 else 0.0, 1.0 if rank == 16 else 0.0,
                      float(l0[c]), float(a0[c]), *[float(v) for v in sl0[c]],
                      e["own_acc"] if e else 0.0, float(e["copies"]) if e else 0.0, float(e["wins"]) if e else 0.0,
                      float(e["depth"]) if e else 0.0, float(e["age"]) if e else 0.0,
                      *[float(v) for v in prof], float(np.abs(prof - sp).sum()) if e else 2.0])
    return np.array(feats, np.float32)


def decide(core, lib, skill, rng, gen, steps=S_DEFAULT, long_steps=0):
    cands, ad_list = candidates(lib, gen)
    ads = stack(ad_list)
    feats = features(core, lib, skill, cands, ads, rng)
    vs = val_set(skill, 999)
    trained = train(core, ads, skill, rng, steps)
    l, acc, _ = evaluate(core, trained, vs)
    out = dict(cands=cands, feats=feats.tolist(), loss=l.tolist(), acc=acc.tolist(), steps=steps)
    if long_steps:
        tl = train(core, ads, skill, np.random.default_rng(rng.integers(1 << 30)), long_steps)
        ll, la, _ = evaluate(core, tl, vs)
        out.update(loss_long=ll.tolist(), acc_long=la.tolist())
    best = int(np.argmin(l.numpy()))
    return out, best, unstack(trained, best)


HELD_OUT = {"invB", "min"}


def universe(uid, arrivals=6, long_steps=0):
    rng = np.random.default_rng(10_000 + uid); gen = torch.Generator().manual_seed(uid)
    core, lib = load_core()
    seen = {tuple(map(str, s["skill"])) for s in lib}
    log = []
    for k in range(arrivals):
        while True:
            sk = random_op(rng)
            if uid < 100 and sk[0] in HELD_OUT:          # training universes never see these arrive
                continue
            if tuple(map(str, sk)) not in seen:
                break
        seen.add(tuple(map(str, sk)))
        t0 = time.time()
        d, best, ad = decide(core, lib, sk, rng, gen, long_steps=long_steps)
        src, rank = d["cands"][best]
        d.update(uid=uid, k=k, skill=[str(x) for x in sk], best=best, seconds=time.time() - t0,
                 lib_skills=[[str(x) for x in e["skill"]] for e in lib])
        log.append(d)
        if src >= 0:
            lib[src]["copies"] += 1; lib[src]["wins"] += 1
        for e in lib:
            e["age"] += 1
        lib.append(dict(skill=sk, ad=ad, rank=rank, parent=src, copies=0, wins=0,
                        depth=(lib[src]["depth"] + 1) if src >= 0 else 0, age=0, own_acc=float(d["acc"][best])))
        print(f"[u{uid}] k={k} {sk} best={d['cands'][best]} acc={d['acc'][best]:.3f} "
              f"fresh16={d['acc'][1]:.3f} {d['seconds']:.0f}s", flush=True)
    json.dump(log, open(os.path.join(OUT, f"universe_{uid}{'_long' if long_steps else ''}.json"), "w"))


if __name__ == "__main__":
    uid = int(sys.argv[1]); long_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    universe(uid, arrivals=6 if not long_steps else 2, long_steps=long_steps)
