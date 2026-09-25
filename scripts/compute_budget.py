#!/usr/bin/env python3
"""Analytic roofline + dispatch model for the M3 Ultra, and the design
decisions that fall out of it.

This is a *model*, not a measurement.  Every hardware constant below is a
named, overridable assumption, and scripts/bench_backend.py measures each one
on the real machine.  The purpose of the model is to decide which experiment
to build before spending a week of wall-clock discovering the answer.

Run: python3 scripts/compute_budget.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemo.modules.spec import ModuleSpec, OrganismSpec  # noqa: E402


@dataclass(frozen=True)
class Hardware:
    """M3 Ultra, 80-core GPU, 512 GB unified memory.

    peak_flops_fp32 : 80 cores x 128 ALU x 2 flop x ~1.4 GHz ~= 28.7 TFLOP/s.
                      We use a deliberately conservative 26 TFLOP/s.
    bandwidth       : 800 GB/s (published unified-memory bandwidth).
    dispatch_us     : per-kernel launch + graph overhead.  THE critical unknown.
                      MLX is lower-overhead than CUDA but not free; 25 us is a
                      central guess bracketed by 10 / 50 in the sensitivity run.
    """

    name: str = "M3 Ultra (80-core GPU)"
    peak_flops: float = 26e12
    bandwidth: float = 800e9
    dispatch_us: float = 25.0
    small_gemm_efficiency: float = 0.25   # fraction of peak reachable at these sizes


@dataclass(frozen=True)
class Workload:
    module: ModuleSpec
    n_runs: int                 # independent replicate evolutionary runs, batched together
    n_organisms: int            # organisms per run
    n_episodes: int             # parallel episodes per organism (weight-reuse factor E)
    n_active_modules: float     # mean modules executed per organism-timestep
    lifetime_steps: int         # environment interactions per organism lifetime
    exec_rounds: int            # sequential message-passing rounds per timestep
    kernels_per_round: int = 8  # fused-op count per round (attn qkv, scores, out, ffn x2, norms, gather, scatter)
    learn: bool = False         # lifetime learning (adds fwd+bwd+optimizer traffic)

    @property
    def n_weight_sets(self) -> int:
        """Distinct module weight sets resident per generation (bandwidth driver)."""
        return self.n_runs * self.n_organisms

    @property
    def batch(self) -> int:
        """Total parallel module-instance lanes (FLOP driver)."""
        return self.n_runs * self.n_organisms * self.n_episodes


def per_generation(hw: Hardware, wl: Workload, dedup_factor: float = 1.0) -> dict:
    """Wall-clock model for one generation (one full lifetime for everyone).

    dedup_factor: fraction of module weight sets that are byte-distinct.  1.0 =
    no sharing; 0.3 = 70% of modules in the population are exact copies of some
    other module and can be read once (content-addressed grouping).
    """
    m = wl.module
    fwd = m.flops_forward()
    step_flops_mult = 3.0 if wl.learn else 1.0   # fwd + bwd(2x)
    weight_bytes = m.bytes_bf16()
    # Optimizer state (Adam m,v in fp32) is read+written per learning step.
    opt_bytes = (8 * m.n_params) if wl.learn else 0

    # ---- FLOP roof ----
    flops_per_ts = wl.batch * wl.n_active_modules * fwd * step_flops_mult
    t_flop = flops_per_ts / (hw.peak_flops * hw.small_gemm_efficiency)

    # ---- Bandwidth roof ----
    # Weights are read once per timestep per distinct weight set, then reused
    # across the n_episodes lanes.  Activations scale with the full batch.
    act_bytes = wl.batch * wl.n_active_modules * (
        2 * m.d_model * (m.max_in_degree + 4)
    ) * (3 if wl.learn else 1)
    w_bytes = wl.n_weight_sets * wl.n_active_modules * (weight_bytes + opt_bytes) * dedup_factor
    bytes_per_ts = w_bytes + act_bytes
    t_bw = bytes_per_ts / hw.bandwidth

    # ---- Dispatch floor (independent of batch size) ----
    k = wl.exec_rounds * wl.kernels_per_round * (3 if wl.learn else 1)
    t_disp = k * hw.dispatch_us * 1e-6

    t_ts = max(t_flop, t_bw, t_disp)
    t_gen = t_ts * wl.lifetime_steps

    intensity = (wl.n_episodes * fwd) / (weight_bytes + opt_bytes) if weight_bytes else 0.0
    ridge = hw.peak_flops * hw.small_gemm_efficiency / hw.bandwidth

    bound = "dispatch" if t_disp >= max(t_flop, t_bw) else ("bandwidth" if t_bw >= t_flop else "flops")
    return {
        "t_gen_s": t_gen,
        "t_timestep_ms": t_ts * 1e3,
        "bound": bound,
        "t_flop_ms": t_flop * 1e3,
        "t_bw_ms": t_bw * 1e3,
        "t_disp_ms": t_disp * 1e3,
        "arith_intensity": intensity,
        "ridge_point": ridge,
        "achieved_tflops": flops_per_ts / t_ts / 1e12,
        "gpu_util_pct": 100 * (flops_per_ts / t_ts) / hw.peak_flops,
        "batch": wl.batch,
    }


def fmt(v: float, n: int = 2) -> str:
    return f"{v:,.{n}f}"


def main() -> None:
    hw = Hardware()
    m = ModuleSpec()
    o = OrganismSpec(module=m, n_genes=4)

    print("=" * 78)
    print("1. EXACT PARAMETER COUNTS")
    print("=" * 78)
    pb = m.param_breakdown()
    print(f"  module (d={m.d_model}, d_ff={m.d_ff}, K={m.max_in_degree}):")
    for k, v in pb.items():
        print(f"      {k:<12} {v:>8,}")
    print(f"  forward FLOPs/module-step (K={m.max_in_degree}): {m.flops_forward():,}")
    print(f"  arithmetic intensity, 1 episode: {m.flops_forward()/m.bytes_bf16():.2f} FLOP/byte")
    print(f"  M3 Ultra ridge point: {hw.peak_flops*hw.small_gemm_efficiency/hw.bandwidth:.1f} FLOP/byte")
    print()
    ob = o.breakdown()
    print("  ancestral organism (4 genes):")
    for k, v in ob.items():
        print(f"      {k:<20} {v:>8,}")
    print()

    print("=" * 78)
    print("2. MODULE SIZE SWEEP  (the proposal asks: 500 / 1K / 2K / 3K / 5K / 10K)")
    print("=" * 78)
    print(f"  {'d_model':>8} {'d_ff':>6} {'params':>9} {'KB bf16':>9} {'FLOP/step':>11} {'intensity':>10}")
    for d in (4, 8, 12, 16, 20, 24, 32, 48, 64):
        ms = ModuleSpec(d_model=d, d_ff=2 * d)
        print(f"  {d:>8} {2*d:>6} {ms.n_params:>9,} {ms.bytes_bf16()/1024:>9.2f}"
              f" {ms.flops_forward():>11,} {ms.flops_forward()/ms.bytes_bf16():>10.2f}")
    print()

    print("=" * 78)
    print("3. WHY EPISODE BATCHING, NOT POPULATION SIZE, IS THE LEVER")
    print("=" * 78)
    print("  Fixed: 16 runs x 256 organisms, 8 active modules, 512-step lifetime,")
    print("  4 exec rounds, no lifetime learning.  Only E (episodes/organism) varies.")
    print()
    print(f"  {'E':>4} {'batch':>10} {'intens':>8} {'bound':>10} {'ms/step':>9} {'s/gen':>8} {'GPU%':>7}")
    for e in (1, 2, 4, 8, 16, 32, 64):
        wl = Workload(module=m, n_runs=16, n_organisms=256, n_episodes=e,
                      n_active_modules=8, lifetime_steps=512, exec_rounds=4)
        r = per_generation(hw, wl)
        print(f"  {e:>4} {r['batch']:>10,} {r['arith_intensity']:>8.1f} {r['bound']:>10}"
              f" {r['t_timestep_ms']:>9.3f} {r['t_gen_s']:>8.2f} {r['gpu_util_pct']:>7.2f}")
    print()

    print("  Same sweep over POPULATION at fixed E=16:")
    print(f"  {'orgs':>6} {'batch':>10} {'intens':>8} {'bound':>10} {'ms/step':>9} {'s/gen':>8} {'GPU%':>7}")
    for n in (64, 128, 256, 512, 1024, 2048):
        wl = Workload(module=m, n_runs=16, n_organisms=n, n_episodes=16,
                      n_active_modules=8, lifetime_steps=512, exec_rounds=4)
        r = per_generation(hw, wl)
        print(f"  {n:>6} {r['batch']:>10,} {r['arith_intensity']:>8.1f} {r['bound']:>10}"
              f" {r['t_timestep_ms']:>9.3f} {r['t_gen_s']:>8.2f} {r['gpu_util_pct']:>7.2f}")
    print()

    print("=" * 78)
    print("4. LIFETIME LENGTH IS THE EXPENSIVE AXIS (it is serial)")
    print("=" * 78)
    print(f"  {'lifetime':>9} {'s/gen':>8} {'h / 1000 gens':>15}")
    for t in (64, 128, 256, 512, 1024, 2048):
        wl = Workload(module=m, n_runs=16, n_organisms=256, n_episodes=16,
                      n_active_modules=8, lifetime_steps=t, exec_rounds=4)
        r = per_generation(hw, wl)
        print(f"  {t:>9} {r['t_gen_s']:>8.2f} {r['t_gen_s']*1000/3600:>15.2f}")
    print()

    print("=" * 78)
    print("5. COST OF LIFETIME LEARNING (phase-1 go/no-go)")
    print("=" * 78)
    for learn in (False, True):
        wl = Workload(module=m, n_runs=16, n_organisms=256, n_episodes=16,
                      n_active_modules=8, lifetime_steps=512, exec_rounds=4, learn=learn)
        r = per_generation(hw, wl)
        tag = "with lifetime learning" if learn else "evolution only"
        print(f"  {tag:<24} bound={r['bound']:<10} {r['t_gen_s']:>7.2f} s/gen"
              f"   {r['t_gen_s']*1000/3600:>6.2f} h / 1000 gens")
    print()

    print("=" * 78)
    print("6. PAYOFF OF CONTENT-ADDRESSED WEIGHT DEDUPLICATION")
    print("=" * 78)
    print("  (bandwidth-bound regime: sharing identical modules cuts weight traffic)")
    print(f"  {'distinct %':>11} {'bound':>10} {'s/gen':>8} {'speedup':>9}")
    base = None
    for dedup in (1.0, 0.7, 0.5, 0.3, 0.15):
        wl = Workload(module=m, n_runs=16, n_organisms=256, n_episodes=4,
                      n_active_modules=8, lifetime_steps=512, exec_rounds=4)
        r = per_generation(hw, wl, dedup_factor=dedup)
        base = base or r["t_gen_s"]
        print(f"  {dedup*100:>10.0f}% {r['bound']:>10} {r['t_gen_s']:>8.2f} {base/r['t_gen_s']:>8.2f}x")
    print()

    print("=" * 78)
    print("7. DISPATCH-OVERHEAD SENSITIVITY (the one number to measure first)")
    print("=" * 78)
    print(f"  {'dispatch us':>12} {'bound':>10} {'s/gen':>8} {'h / 1000 gens':>15}")
    for us in (5, 10, 25, 50, 100):
        h2 = Hardware(dispatch_us=us)
        wl = Workload(module=m, n_runs=16, n_organisms=256, n_episodes=16,
                      n_active_modules=8, lifetime_steps=512, exec_rounds=4)
        r = per_generation(h2, wl)
        print(f"  {us:>12} {r['bound']:>10} {r['t_gen_s']:>8.2f} {r['t_gen_s']*1000/3600:>15.2f}")
    print()

    print("=" * 78)
    print("8. NAIVE (per-organism python loop) VS VECTORIZED")
    print("=" * 78)
    naive_kernels = 16 * 256 * 8 * 4 * 8   # runs x orgs x modules x rounds x kernels
    naive_s = naive_kernels * hw.dispatch_us * 1e-6 * 512
    wl = Workload(module=m, n_runs=16, n_organisms=256, n_episodes=16,
                  n_active_modules=8, lifetime_steps=512, exec_rounds=4)
    vec = per_generation(hw, wl)
    print(f"  naive      : {naive_kernels:,} dispatches/timestep -> {naive_s/3600:,.1f} h per GENERATION")
    print(f"  vectorized : {wl.exec_rounds*wl.kernels_per_round:,} dispatches/timestep -> {vec['t_gen_s']:,.2f} s per generation")
    print(f"  ratio      : {naive_s/vec['t_gen_s']:,.0f}x")
    print()

    print("=" * 78)
    print("9. MEMORY FOOTPRINT (the 350 GB budget is ~4 orders of magnitude too big)")
    print("=" * 78)
    for label, runs, orgs, genes, eps, learn in (
        ("pilot   (16 runs x 256 org, 8 genes)", 16, 256, 8, 16, False),
        ("pilot + lifetime learning", 16, 256, 8, 16, True),
        ("large   (64 runs x 512 org, 32 genes)", 64, 512, 32, 16, False),
        ("pathological (256 org x 1M params)", 1, 256, 1, 16, False),
    ):
        mm = ModuleSpec(d_model=352, d_ff=704) if "1M" in label else m
        w = runs * orgs * genes * mm.bytes_bf16()
        opt = w * 4 if learn else 0
        acts = runs * orgs * eps * genes * 2 * mm.d_model * (mm.max_in_degree + 4) * (8 if learn else 1)
        tot = w + opt + acts
        print(f"  {label:<40} weights {w/2**30:>7.3f} GB  opt {opt/2**30:>6.3f} GB"
              f"  act {acts/2**30:>6.3f} GB  total {tot/2**30:>7.3f} GB")
    print()




def addendum() -> None:
    """Sections 10-12: where the model's regime boundaries actually sit."""
    hw = Hardware()
    m = ModuleSpec()

    print("=" * 78)
    print("10. WHERE DOES DEDUPLICATION / BANDWIDTH START TO MATTER?")
    print("=" * 78)
    print("  Dedup pays only once weight traffic dominates dispatch.  Sweep module size")
    print("  at 16 runs x 256 organisms, E=4 (low reuse, the worst case for bandwidth):")
    print(f"  {'d_model':>8} {'params':>9} {'bound @100%':>12} {'s/gen 100%':>11} {'s/gen 30%':>10} {'speedup':>8}")
    for d in (16, 32, 64, 128, 256, 352):
        ms = ModuleSpec(d_model=d, d_ff=2 * d)
        wl = Workload(module=ms, n_runs=16, n_organisms=256, n_episodes=4,
                      n_active_modules=8, lifetime_steps=512, exec_rounds=4)
        a = per_generation(hw, wl, dedup_factor=1.0)
        b = per_generation(hw, wl, dedup_factor=0.3)
        print(f"  {d:>8} {ms.n_params:>9,} {a['bound']:>12} {a['t_gen_s']:>11.2f}"
              f" {b['t_gen_s']:>10.2f} {a['t_gen_s']/b['t_gen_s']:>7.2f}x")
    print()

    print("=" * 78)
    print("11. FULL PHASE-1 CAMPAIGN COST")
    print("=" * 78)
    print("  Factorial: goal-structure {MVG, RVG, FIX} x metabolism {on, off}")
    print("             x duplication {on, off} = 12 cells, 12 replicates = 144 runs.")
    print("  All 144 runs execute as ONE batch (they are independent lanes).")
    print()
    for label, runs, orgs, eps, life, gens, learn in (
        ("smoke      (144 runs, 100 gens)", 144, 256, 8, 256, 100, False),
        ("pilot      (144 runs, 1k gens)", 144, 256, 8, 256, 1000, False),
        ("main       (144 runs, 5k gens)", 144, 256, 16, 512, 5000, False),
        ("phase 2    (+lifetime learning)", 144, 256, 16, 512, 5000, True),
        ("phase 3    (+sex, HGT: 288 runs)", 288, 256, 16, 512, 5000, True),
    ):
        wl = Workload(module=m, n_runs=runs, n_organisms=orgs, n_episodes=eps,
                      n_active_modules=8, lifetime_steps=life, exec_rounds=4, learn=learn)
        r = per_generation(hw, wl)
        hrs = r["t_gen_s"] * gens / 3600
        print(f"  {label:<34} batch {r['batch']:>9,}  bound {r['bound']:<10}"
              f" {r['t_gen_s']:>6.2f} s/gen  ->  {hrs:>6.2f} h")
    print()

    print("=" * 78)
    print("12. THE PROPOSAL'S OWN NUMBERS, PRICED")
    print("=" * 78)
    for label, runs, orgs, eps, life in (
        ("proposal as written (1 run, 256 org, 1 episode)", 1, 256, 1, 1024),
        ("  + episode batching E=16", 1, 256, 16, 1024),
        ("  + 12 replicates batched", 12, 256, 16, 1024),
        ("  + 144-run factorial batched", 144, 256, 16, 1024),
    ):
        wl = Workload(module=m, n_runs=runs, n_organisms=orgs, n_episodes=eps,
                      n_active_modules=8, lifetime_steps=life, exec_rounds=4)
        r = per_generation(hw, wl)
        print(f"  {label:<48} {r['t_gen_s']:>6.2f} s/gen  batch {r['batch']:>9,}"
              f"  bound {r['bound']}")
    print()
    print("  Interpretation: the serial cost is set by lifetime x generations alone.")
    print("  Everything parallel (organisms, episodes, replicates, factorial cells)")
    print("  rides along inside the same dispatches until the FLOP roof at ~1.3e5 lanes.")
    print()


if __name__ == "__main__":
    main()
    addendum()
