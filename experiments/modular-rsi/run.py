"""Modular recursive self-improvement experiments.

  python run.py main  --mode steered --seed 0     # Exp 1: mutate -> polymerize -> dissociate cycles
  python run.py curriculum --improver inherit --seed 0   # Exp 2: does the improver improve?
  python run.py scale                              # Exp 3: grow polymer/community n
  python run.py report                             # figures + results/summary.json

Everything is written under results/.
"""
import argparse, json, os, time
import numpy as np
import torch
from modrsi.world import GeneticCode, PRIMITIVES, make_batch, sample_contexts
from modrsi.monomer import CodeT, N_PARAMS
from modrsi.evolve import Niche, IslandNiche, Lineage, GENES, nll
from modrsi import polymer as P

torch.set_num_threads(1)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)
MAIN_NICHES = ("A", "B", "Ainv", "succ")          # Binv is held out (novel skill for Exp 2/3)
TARGETS = [("A", "B"), ("B", "succ", "Ainv"), ("A", "succ", "A", "B"),
           ("Ainv", "B", "succ", "A", "Ainv")]     # "big" hidden-program problems, L = 2..5
SOLVED = 0.95


def jdump(obj, name):
    with open(os.path.join(OUT, name), "w") as f:
        json.dump(obj, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))


def make_niche(i, program, rng, lin, codeT, mode, init=None):
    """steered = island niche with lineage-steered mutation + PBT-selected improvers;
    blind = one population, fixed isotropic mutation and fixed learning rate."""
    if mode == "blind":
        return Niche(i, program, rng, lin, codeT, mode="blind")
    return IslandNiche(i, program, rng, lin, codeT, init=init)


# ============================================================================ Exp 1
def exp_main(mode, seed, cycles=10, gens=10, pool_k=3):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    codeT = CodeT(GeneticCode(0))
    lin = Lineage()
    niches = [make_niche(i, (op,), rng, lin, codeT, mode) for i, op in enumerate(MAIN_NICHES)]
    log = dict(mode=mode, seed=seed, niche_hist=[], polymer=[], solved_gen={})
    t0, polymer_evals = time.time(), 0
    for cyc in range(cycles):
        # ---- 1. monomer phase: mutation + lifetime learning + selection, per niche
        for g in range(gens):
            for n in niches:
                r = n.step(bonus_w=0.5)
                r["cycle"] = cyc
                log["niche_hist"].append(r)
                if r["mean_acc"] >= SOLVED and n.program[0] not in log["solved_gen"]:
                    log["solved_gen"][n.program[0]] = n.gen
        # ---- 2. polymerization: the best monomers of every niche form a shared pool
        pool, pool_ids, pool_niche = [], [], []
        for n in niches:
            G, ids, _ = n.elites(pool_k)
            pool.append(G); pool_ids += list(ids); pool_niche += [n.idx] * len(ids)
        pool = torch.cat(pool)
        credit = {}
        for tgt in TARGETS:
            ps = P.PolymerSearch(pool, codeT, rng, steered=(mode == "steered"))
            hist = [ps.step(tgt) for _ in range(30)]
            polymer_evals += ps.evals
            # best single monomer on the same big problem (the "monomer alone" baseline)
            ctx, x0, y = make_batch(rng, tgt, 256)
            mono = np.full((len(pool), P.LMAX), -1); mono[:, 0] = np.arange(len(pool))
            _, mono_acc = P.chain_fitness(pool, mono, codeT, ctx, x0, y)
            _, best_acc = P.chain_fitness(pool, ps.pop[:1], codeT, ctx, x0, y)
            chain = [int(u) for u in ps.pop[0] if u >= 0]
            first = next((h["evals"] for h in hist if h["best_acc"] >= SOLVED), None)
            log["polymer"].append(dict(cycle=cyc, target=tgt, polymer_acc=float(best_acc[0]),
                                       monomer_acc=float(mono_acc.max()),
                                       chain_niches=[MAIN_NICHES[pool_niche[u]] for u in chain],
                                       evals_to_solve=first, evals=ps.evals))
            for u in chain if best_acc[0] >= 0.5 else []:   # credit flows back to monomers of working polymers
                credit[pool_ids[u]] = max(credit.get(pool_ids[u], 0), float(best_acc[0]))
        # ---- 3. dissociate: monomers return to their niches carrying polymer credit
        for n in niches:
            n.bonus = np.array([credit.get(i, 0.0) for i in n.ids], np.float32)
        print(f"[{mode} s{seed}] cycle {cyc} "
              f"niche acc {[round(n.acc.mean(), 2) for n in niches]} "
              f"polymer {[round(p['polymer_acc'], 2) for p in log['polymer'][-len(TARGETS):]]} "
              f"{time.time() - t0:.0f}s", flush=True)
    A = lin.arrays()
    log["variants"] = int(len(A["id"]))
    log["polymer_evals"] = int(polymer_evals)
    # fraction of children that are already "successful" (acc>=0.9), per generation
    log["success_frac"] = [[float(A["acc"][(A["gen"] == g) & (A["niche"] == i)].__ge__(0.9).mean())
                            for g in range(cycles * gens)] for i in range(len(niches))]
    # champion lineages: walk parent pointers back from each niche's best monomer
    parent = dict(zip(A["id"].tolist(), A["parent"].tolist()))
    acc_of = dict(zip(A["id"].tolist(), A["acc"].tolist()))
    gen_of = dict(zip(A["id"].tolist(), A["gen"].tolist()))
    lineages = {}
    for n in niches:
        cur, path = int(n.ids[np.argmax(n.fit)]), []
        while cur in parent:
            path.append((gen_of[cur], acc_of[cur])); cur = parent[cur]
        lineages[n.program[0]] = path[::-1]
    log["champion_lineages"] = lineages
    log["final_genes"] = {k: np.concatenate([getattr(n, k) for n in niches]).tolist() for k in GENES}
    tag = f"main_{mode}_s{seed}"
    lin.save(os.path.join(OUT, f"lineage_{tag}.npz"))
    torch.save({n.program[0]: n.elites(pool_k)[0] for n in niches}, os.path.join(OUT, f"modules_{tag}.pt"))
    jdump(log, f"{tag}.json")


# ============================================================================ Exp 2
CURRICULUM = ("A", "Ainv", "B", "succ", "Binv")


def exp_curriculum(improver, seed, max_gens=90):
    """Learn skills one after another. 'inherit': each new niche is seeded with the
    improver genes (sigma, m, a, b, lr) that the previous niches evolved - the system's
    accumulated know-how about *how to improve*. Weights always start from scratch.
    'fresh': every niche starts from the default improver. 'blind': no steering at all."""
    rng = np.random.default_rng(1000 + seed)
    torch.manual_seed(1000 + seed)
    codeT = CodeT(GeneticCode(0))
    lin = Lineage()
    genes, out = None, []
    for i, op in enumerate(CURRICULUM):
        mode = "blind" if improver == "blind" else "steered"
        init = None
        if improver == "inherit" and genes is not None:
            init = {k: genes[k] for k in GENES}
        n = make_niche(i, (op,), rng, lin, codeT, mode, init=init)
        solved = None
        for g in range(max_gens):
            r = n.step()
            if r["mean_acc"] >= SOLVED:
                solved = g + 1
                break
        # the improver pool handed to the next skill: genes of the elites at solve time
        genes = {k: getattr(n, k).copy() for k in GENES}
        out.append(dict(skill=op, gens_to_solve=solved, genes={k: float(np.median(v)) for k, v in genes.items()}))
        print(f"[curriculum {improver} s{seed}] {op}: solved at {solved}", flush=True)
    jdump(dict(improver=improver, seed=seed, skills=out), f"curriculum_{improver}_s{seed}.json")


# ============================================================================ Exp 3
def exp_scale(seed=0, max_n=12):
    rng = np.random.default_rng(7)
    codeT = CodeT(GeneticCode(0))
    mods = torch.load(os.path.join(OUT, f"modules_main_steered_s{seed}.pt"))
    # grow the community: train the held-out skill (Binv) as a new monomer species
    lin = Lineage()
    n = IslandNiche(9, ("Binv",), rng, lin, codeT)
    for g in range(120):
        if n.step()["mean_acc"] >= 0.99:
            break
    mods["Binv"] = n.elites(3)[0]
    torch.save(mods, os.path.join(OUT, "modules_scale.pt"))
    names = list(mods)
    pool = torch.cat([mods[k] for k in names])
    owner = sum([[k] * len(mods[k]) for k in names], [])
    res = {}
    for ops, label in ((("A", "B", "Ainv", "succ"), "4 skills"), (PRIMITIVES, "5 skills (+Binv)")):
        router, rhist = P.evolve_router(pool, codeT, rng, ops=ops, gens=40)
        accs = [P.community_accuracy(pool, router, codeT, rng, k, B=512, ops=ops) for k in range(1, max_n + 1)]
        # monomer-alone baseline: best single monomer, applied once, on the same problems
        mono = []
        for k in range(1, max_n + 1):
            ctx, x0, prog, y = P.instructed_batch(rng, k, 512, ops)
            best = 0.0
            for u in range(len(pool)):
                r = np.full(len(ops), u)
                q = P.community_forward(pool, r, codeT, ctx, x0, prog[:, :1])
                best = max(best, float((q.argmax(-1).numpy() == y).mean()))
            mono.append(best)
        res[label] = dict(router=[owner[u] for u in router], router_hist=rhist,
                          community_acc=accs, monomer_acc=mono,
                          n_programs=[len(ops) ** k for k in range(1, max_n + 1)])
        print(label, "router", [owner[u] for u in router], "acc", np.round(accs, 3), "mono", np.round(mono, 3), flush=True)
    res["params_per_monomer"] = N_PARAMS
    jdump(res, "scale.json")


# ============================================================================ Exp 4
def exp_grow(hier=True, seed=0, cycles=10, per_cycle=4, budget=6000, modules=None):
    """Compounding growth. Each cycle the environment poses `per_cycle` new problems, each a
    concatenation of 2-3 problems of the previous level. hier=True: solved polymers are
    frozen into macros and join the pool. hier=False (control): the pool stays the 5
    monomers, polymers stay <= 5 units. Same search budget per problem in both."""
    from modrsi import hierarchy as Hy
    rng = np.random.default_rng(100 + seed)
    codeT = CodeT(GeneticCode(0))
    mods = torch.load(modules or os.path.join(OUT, "modules_scale.pt"))
    units = [Hy.Unit("monomer", mods[k][0], name=k) for k in PRIMITIVES]
    solved = {0: [((k,), i) for i, k in enumerate(PRIMITIVES)]}   # level -> [(program, unit idx)]
    log, total_evals = [], 0
    for cyc in range(1, cycles + 1):
        prev = solved.get(cyc - 1) or []
        if not prev:
            break
        solved[cyc] = []
        for _ in range(per_cycle):
            parts = [prev[rng.integers(0, len(prev))] for _ in range(rng.integers(2, 4))]
            program = tuple(op for p, _ in parts for op in p)
            # lineage-memory prior: units formed in the latest cycle are favoured
            w = np.array([10.0 if (u.cycle == cyc - 1 and u.kind == "macro") or (cyc == 1) else 1.0
                          for u in units])
            chain, fit, ev = Hy.search(units, codeT, program, rng, w / w.sum(), budget=budget)
            total_evals += ev
            vctx = sample_contexts(rng, 256)
            T = Hy.unit_tables(units, codeT, vctx)
            val = float((Hy.compose(T, chain, 256) == Hy.target_table(vctx, program)).mean())
            rec = dict(cycle=cyc, length=len(program), evals=ev, search_fit=fit, val_acc=val,
                       chain=[units[c].name for c in chain if c >= 0])
            if val >= 0.95 and not hier:      # control: solved, but the solution is not reusable
                solved[cyc].append((program, None))
            elif val >= 0.95:
                units.append(Hy.Unit("macro", children=[int(c) for c in chain],
                                     name=f"M{cyc}.{len(solved[cyc])}", length=len(program), cycle=cyc))
                solved[cyc].append((program, len(units) - 1))
                acc_exec, n_exec = Hy.verify_by_execution(units, len(units) - 1, codeT,
                                                          sample_contexts(rng, 64), program)
                rec.update(exec_acc=acc_exec, exec_len=n_exec)
            log.append(rec)
            print(f"[grow hier={hier}] cycle {cyc} L={len(program)} solved={val >= 0.95} val={val:.3f} "
                  f"evals={ev} chain={rec['chain']} exec={rec.get('exec_acc')}", flush=True)
    jdump(dict(hier=hier, seed=seed, log=log, total_evals=total_evals), f"grow_{'hier' if hier else 'flat'}_s{seed}.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("exp", choices=["main", "curriculum", "scale", "grow", "report"])
    ap.add_argument("--flat", action="store_true")
    ap.add_argument("--mode", default="steered")
    ap.add_argument("--improver", default="inherit")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.exp == "main":
        exp_main(a.mode, a.seed)
    elif a.exp == "curriculum":
        exp_curriculum(a.improver, a.seed)
    elif a.exp == "scale":
        exp_scale(a.seed)
    elif a.exp == "grow":
        exp_grow(hier=not a.flat, seed=a.seed)
    else:
        from report import report
        report()
