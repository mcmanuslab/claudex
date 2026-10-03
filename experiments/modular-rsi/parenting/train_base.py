"""Train the shared core jointly with the 8 base-skill adapters, then freeze it."""
import os, sys, time, json
import numpy as np
import torch
from world import BASE_OPS as BASE_SKILLS, batch
from model import Core, new_adapters, stack, unstack, loss_acc, TRAINABLE

torch.set_num_threads(int(os.environ.get("THREADS", "4")))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"); os.makedirs(OUT, exist_ok=True)


def sample(rng, skills, B):
    data = [batch(rng, s, B) for s in skills]
    return [torch.stack([torch.from_numpy(d[k]).long() for d in data]) for k in range(5)]


def main(steps=int(sys.argv[1]) if len(sys.argv) > 1 else 20000, seed=0):
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    core = Core(seed)
    g = torch.Generator().manual_seed(seed)
    ad = stack([unstack(new_adapters(1, 16, g), 0) for _ in BASE_SKILLS])
    for k in TRAINABLE:
        ad[k].requires_grad_(True)
    params = list(core.parameters()) + [ad[k] for k in TRAINABLE]
    opt = torch.optim.Adam(params, lr=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    t0 = time.time()
    for s in range(steps):
        l, acc, _ = loss_acc(core, ad, *sample(rng, BASE_SKILLS, 32))
        opt.zero_grad(); l.sum().backward(); opt.step(); sched.step()
        if s % 1000 == 0 or s == steps - 1:
            with torch.no_grad():
                l, acc, _ = loss_acc(core, ad, *sample(np.random.default_rng(99), BASE_SKILLS, 256))
            print(s, "acc", np.round(acc.numpy(), 3), f"{time.time() - t0:.0f}s", flush=True)
    torch.save(dict(core=core.state_dict(), adapters={k: v.detach() for k, v in ad.items()},
                    skills=BASE_SKILLS, acc=acc.tolist()), os.path.join(OUT, "core.pt"))
    json.dump(dict(acc=acc.tolist(), skills=[list(map(str, s)) for s in BASE_SKILLS]),
              open(os.path.join(OUT, "core_acc.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
