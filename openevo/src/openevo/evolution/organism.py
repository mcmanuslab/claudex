"""Organisms: genotype (architecture + weights + strategy genes) and their mutation.

The strategy genes are the secondary scientific question made concrete. In particular
``p_growth_bias`` -- the fraction of an organism's structural mutations that are growth
rather than shrinkage -- is *itself heritable*. That converts "does evolution favour
larger architectures?" from an inference about a noisy size trajectory into a directly
measurable trait: under neutrality the gene must random-walk around 0.5, so any
sustained departure is evidence of selection rather than of diffusion off the lower
bound on size.

Strategy genes mutate multiplicatively (log-normal) with a heritable ``meta_sigma``,
because additive noise on a positive, scale-free quantity such as a learning rate is
biased and cannot shrink a value by the same factor it can grow it.

Which genes are actually heritable is gated by ``PhaseConfig``, so that a run can enable
one mechanism at a time and keep the causal attribution interpretable.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field, replace

import numpy as np

from ..models.genome import ArchGenome
from ..models.morphisms import GROWTH_OPS, SHRINK_OPS, apply_morphism, snap_genome
from ..models.transformer import DTYPE, Params, init_params

_ids = itertools.count(1)

# (lo, hi) safety bounds. Wide enough that evolution, not the bounds, decides.
GENE_BOUNDS: dict[str, tuple[float, float]] = {
    "temperature": (0.05, 5.0),
    "lr": (1e-5, 0.3),
    "p_structural": (0.005, 0.9),
    "p_growth_bias": (0.02, 0.98),
    "weight_sigma": (1e-4, 1.0),
    "meta_sigma": (0.01, 1.0),
    "morph_noise": (0.0, 0.2),
}
DEFAULT_GENES: dict[str, float] = {
    "temperature": 1.0, "lr": 0.02, "p_structural": 0.15, "p_growth_bias": 0.5,
    "weight_sigma": 0.05, "meta_sigma": 0.15, "morph_noise": 0.01,
}


@dataclass(frozen=True)
class PhaseConfig:
    """Which mechanisms are switched on. Phases are cumulative."""

    heritable_learning: bool = False     # phase 2: temperature, lr
    heritable_mutation: bool = False     # phase 3: p_structural, p_growth_bias, sigmas
    recombination: bool = False          # phase 4
    env_coevolution: bool = False        # phase 5
    structural_mutation: bool = True     # phase 1 (off => fixed-architecture control)
    weight_inheritance: str = "lamarckian"  # lamarckian | darwinian | partial
    neutral: bool = False                # neutral-drift null: fitness is randomised

    def heritable_genes(self) -> tuple[str, ...]:
        g: list[str] = []
        if self.heritable_learning:
            g += ["temperature", "lr"]
        if self.heritable_mutation:
            g += ["p_structural", "p_growth_bias", "weight_sigma", "meta_sigma",
                  "morph_noise"]
        return tuple(g)


@dataclass(eq=False)
class Organism:
    """Identity, not value, semantics: two organisms are the same only if they are the
    same object. The default dataclass ``__eq__`` would compare weight tensors
    element-wise, which is both wrong here and raises on any list membership test."""

    arch: ArchGenome
    weights: Params                      # single organism, leading axis of size 1
    genes: dict[str, float]
    oid: int = field(default_factory=lambda: next(_ids))
    parents: tuple[int, ...] = ()
    birth_gen: int = 0
    island: int = 0
    lineage_root: int = 0
    mutations: tuple[str, ...] = ()      # ops applied at this birth
    n_struct_events: int = 0             # cumulative down the lineage
    fitness: float = 0.0
    fitness_components: dict[str, float] = field(default_factory=dict)
    behaviour: tuple[float, ...] = ()    # behavioural descriptor, for QD + novelty
    age: int = 0
    eval_flops: int = 0
    death_reason: str = ""

    @property
    def n_params(self) -> int:
        return self.arch.n_params

    def gene(self, k: str) -> float:
        return self.genes.get(k, DEFAULT_GENES[k])


def founder(arch: ArchGenome, rng: np.random.Generator, island: int = 0,
            genes: dict[str, float] | None = None) -> Organism:
    a = snap_genome(arch)
    o = Organism(arch=a, weights=init_params(a, rng, n=1),
                 genes=dict(genes or DEFAULT_GENES), island=island)
    o.lineage_root = o.oid
    return o


def _mutate_genes(genes: dict[str, float], heritable: tuple[str, ...],
                  rng: np.random.Generator) -> dict[str, float]:
    out = dict(genes)
    if not heritable:
        return out
    ms = float(np.clip(genes.get("meta_sigma", DEFAULT_GENES["meta_sigma"]),
                       *GENE_BOUNDS["meta_sigma"]))
    for k in heritable:
        lo, hi = GENE_BOUNDS[k]
        v = out.get(k, DEFAULT_GENES[k])
        if k == "p_growth_bias":
            # A probability: mutate on the logit scale so the walk is symmetric about
            # 0.5 and cannot be biased by the parameterisation itself.
            z = np.log(v / (1 - v)) + rng.normal(0.0, ms)
            v = 1.0 / (1.0 + np.exp(-z))
        elif k == "morph_noise":
            v = abs(v + rng.normal(0.0, ms * 0.02))
        else:
            v = v * float(np.exp(rng.normal(0.0, ms)))
        out[k] = float(np.clip(v, lo, hi))
    return out


def _reference_scale(v: np.ndarray) -> float:
    """The magnitude a tensor of this shape *would* have at initialisation."""
    if v.ndim >= 3:                      # (organisms, fan_in, fan_out) projection
        return 1.0 / np.sqrt(max(1, v.shape[-2]))
    return 0.1                           # gains and biases


def _perturb_weights(w: Params, sigma: float, rng: np.random.Generator) -> Params:
    """Gaussian perturbation, scaled by the tensor but floored at its natural scale.

    A purely scale-relative step (``sigma * rms(v)``) makes **zero an absorbing state**,
    which is fatal here rather than merely inelegant: function-preserving growth and
    `init_params` both start every block's output path (``Wo``, ``W2``) at exactly zero,
    so those tensors would have an effective mutation size of ~5e-6 and could never
    leave zero. The attention and FFN stacks would stay pinned at the identity for the
    whole run, and evolution would silently optimise nothing but the embeddings and the
    output head -- which is precisely what the first pilot did. It was invisible in
    fitness (which rose steadily) and was caught only by the effective-parameter
    ablation, which found 0 of 99 units doing anything.

    Flooring the step at the scale the tensor would have had at initialisation keeps the
    perturbation scale-invariant where that is meaningful, while leaving zero escapable.
    """
    out: Params = {}
    for k, v in w.items():
        rms = float(np.sqrt(np.mean(np.square(v))))
        out[k] = (v + rng.normal(0.0, sigma * max(rms, _reference_scale(v)),
                                 size=v.shape)).astype(DTYPE)
    return out


def reproduce(parent: Organism, rng: np.random.Generator, phase: PhaseConfig,
              generation: int) -> Organism:
    """Asexual reproduction with architecture, weight and strategy-gene mutation."""
    genes = _mutate_genes(parent.genes, phase.heritable_genes(), rng)
    arch, weights, ops = parent.arch, parent.weights, []

    if phase.structural_mutation and rng.random() < genes["p_structural"]:
        # Choose the direction first, then sample uniformly among the dimensions on
        # which that direction is currently legal.
        #
        # This is deliberate, and two alternatives were measured and rejected. Sampling
        # the dimension first and *aborting* when the move is illegal is much worse
        # (+2.75 rungs of drift per 60 neutral generations, 95% CI [1.6, 3.9]): since
        # n_blocks starts pinned at its floor, `remove_block` can never fire while
        # `add_block` always can, so blocked shrinks simply vanish while the matching
        # growth succeeds. Substituting another dimension keeps the rung walk unbiased
        # (-0.20 +- 1.30 rungs over the same test).
        #
        # What remains is a genuine reflecting boundary at minimum viable size -- the
        # "left wall" of the passive-diffusion account of complexity growth. No operator
        # design removes it, because a population sitting against a lower bound really
        # can only move one way. It is therefore *measured* rather than assumed: every
        # experiment runs a neutral arm (PhaseConfig.neutral) with identical operators
        # and demography but randomised fitness, and a complexity claim has to exceed
        # that arm, not merely exceed zero.
        grow = rng.random() < genes["p_growth_bias"]
        pool = list(GROWTH_OPS if grow else SHRINK_OPS)
        rng.shuffle(pool)
        for op in pool:  # first op whose ladder move is legal
            res = apply_morphism(weights, arch, op, rng, noise=genes["morph_noise"])
            if res is not None:
                weights, arch = res
                ops.append(op)
                break

    if phase.weight_inheritance == "darwinian":
        weights = init_params(arch, rng, n=1)
    elif phase.weight_inheritance == "partial":
        # Keep inherited modules; re-initialise anything created at this birth. With
        # function-preserving growth the new module is already an identity, so this
        # differs from `lamarckian` only in that the inherited part is not perturbed.
        weights = {k: v.copy() for k, v in weights.items()}
    else:
        weights = _perturb_weights(weights, genes["weight_sigma"], rng)

    child = Organism(
        arch=arch, weights=weights, genes=genes, parents=(parent.oid,),
        birth_gen=generation, island=parent.island, lineage_root=parent.lineage_root,
        mutations=tuple(ops), n_struct_events=parent.n_struct_events + len(ops),
    )
    return child


def recombine(a: Organism, b: Organism, rng: np.random.Generator, phase: PhaseConfig,
              generation: int) -> Organism | None:
    """Module-level crossover between architecturally compatible parents (phase 4).

    Naive tensor crossover between differently-shaped networks is meaningless, so mating
    is restricted to parents whose architecture signatures match, and inheritance happens
    at the granularity of whole modules (a block, the embeddings, the head) rather than
    individual weights.
    """
    if a.arch.signature() != b.arch.signature():
        return None
    groups: dict[str, list[str]] = {}
    for k in a.weights:
        groups.setdefault(k.split(".")[0] if "." in k else k, []).append(k)
    w: Params = {}
    for g, keys in groups.items():
        src = a if rng.random() < 0.5 else b
        for k in keys:
            w[k] = src.weights[k].copy()
    genes = {k: (a.genes.get(k, DEFAULT_GENES[k]) if rng.random() < 0.5
                 else b.genes.get(k, DEFAULT_GENES[k])) for k in DEFAULT_GENES}
    genes = _mutate_genes(genes, phase.heritable_genes(), rng)
    child = Organism(arch=replace(a.arch), weights=w, genes=genes,
                     parents=(a.oid, b.oid), birth_gen=generation, island=a.island,
                     lineage_root=a.lineage_root, mutations=("recombine",),
                     n_struct_events=max(a.n_struct_events, b.n_struct_events))
    return child
