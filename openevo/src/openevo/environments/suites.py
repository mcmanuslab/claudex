"""Class A / B / C world suites, and the reference normalisation that makes them comparable.

The three classes differ in *what kind of novelty* they present:

* **Class A (ancestral)** -- families drawn from the mechanism vocabulary and the family
  combinations that evolution is selected on. Novel instances, familiar structure.
* **Class B (recombination)** -- the same mechanism vocabulary, but family combinations
  explicitly withheld from the selection sampler. Tests compositional generalisation.
* **Class C (alien)** -- at least one mechanism that never appears under selection at
  all. Tests whether the machinery of adaptation transfers to unfamiliar causal structure.

Two safeguards that the naive version of this experiment lacks:

**Reference normalisation.** A raw score on a held-out world is uninterpretable, because
a rising "adaptation speed" curve could just mean the sampler happened to draw easier
worlds later. Every world is therefore scored as
``(agent - random) / (reference_high - random)``, where both references are measured on
that exact world instance. Worlds whose reference gap is too small to discriminate
between policies are rejected at sampling time, which also matches difficulty across
the three classes rather than hoping it matches.

**A dev/test split inside Class C.** The held-out set is split into ``C-dev``, which may
be inspected while building and debugging the system, and ``C-test``, which is opened
once per pre-registered experiment. Keeping alien performance out of *selection* is not
enough on its own: a researcher who tunes the system while watching alien scores leaks
information through their own choices just as surely as the fitness function would.
"""

from __future__ import annotations

import hashlib
import itertools
import json

import numpy as np

from .worlds import (
    ALIEN, CONTEXT, LATENT_ALIEN, LATENT_ANCESTRAL, N_ACT, OBS_ALIEN, OBS_ANCESTRAL,
    REW_ALIEN, REW_ANCESTRAL, REW_CORES, WorldBatch, WorldSpec,
)

# A world paired with its measured (random, reference_high) scores.
Scored = tuple[WorldSpec, float, float]

PARAM_GRID: dict[str, tuple[float, ...]] = {
    "mask_p": (0.2, 0.35), "distractor_p": (0.2, 0.35), "mimic_p": (0.25, 0.4),
    "delay": (1, 2, 3), "sparse_n": (3, 4), "period": (6, 10),
    "walk_p": (0.6, 0.8), "act_cost": (0.05, 0.15), "cheat_r": (0.25, 0.4),
    "poison_k": (2, 3), "flip_at": (6, 10), "hist_k": (2, 3),
}
ANCESTRAL_OBS_MODS = tuple(m for m in OBS_ANCESTRAL if m != "direct")
ANCESTRAL_REW_MODS = tuple(m for m in REW_ANCESTRAL if m not in REW_CORES)
ALIEN_REW_MODS = tuple(m for m in REW_ALIEN if m not in REW_CORES)


def _sorted_subsets(items: tuple[str, ...], max_len: int) -> list[tuple[str, ...]]:
    out: list[tuple[str, ...]] = [()]
    for r in range(1, max_len + 1):
        out += [tuple(sorted(c)) for c in itertools.combinations(items, r)]
    return out


def enumerate_families(max_obs: int = 2, max_rew: int = 2) -> list[tuple]:
    """All structurally valid families over the full mechanism vocabulary."""
    fams = []
    for lat in LATENT_ANCESTRAL + LATENT_ALIEN:
        for core in sorted(REW_CORES):
            for om in _sorted_subsets(ANCESTRAL_OBS_MODS + OBS_ALIEN, max_obs):
                for rm in _sorted_subsets(ANCESTRAL_REW_MODS + ALIEN_REW_MODS, max_rew):
                    fams.append((lat, om, core, rm))
    return fams


def family_is_alien(fam: tuple) -> bool:
    lat, om, core, rm = fam
    return bool({lat, core, *om, *rm} & ALIEN)


class SuiteSplit:
    """A frozen, hash-sealed partition of world families into A / B / C-dev / C-test.

    The split is a pure function of `seed`, and :meth:`seal` hashes it so a run's
    manifest can prove which partition it used. Class B is carved out of the *ancestral*
    families, so B shares A's vocabulary and differs only in combination.
    """

    def __init__(self, seed: int = 20260101, b_fraction: float = 0.25,
                 c_test_fraction: float = 0.5) -> None:
        self.seed = seed
        rng = np.random.default_rng(seed)
        fams = enumerate_families()
        anc = [f for f in fams if not family_is_alien(f)]
        ali = [f for f in fams if family_is_alien(f)]
        anc_idx = rng.permutation(len(anc))
        n_b = max(1, int(round(b_fraction * len(anc))))
        self.class_b = [anc[i] for i in sorted(anc_idx[:n_b])]
        self.class_a = [anc[i] for i in sorted(anc_idx[n_b:])]
        ali_idx = rng.permutation(len(ali))
        n_t = max(1, int(round(c_test_fraction * len(ali))))
        self.c_test = [ali[i] for i in sorted(ali_idx[:n_t])]
        self.c_dev = [ali[i] for i in sorted(ali_idx[n_t:])]

    def pool(self, cls: str) -> list[tuple]:
        return {"A": self.class_a, "B": self.class_b,
                "C_dev": self.c_dev, "C_test": self.c_test}[cls]

    def seal(self) -> str:
        blob = json.dumps({k: [list(map(list, f)) for f in self.pool(k)]
                           for k in ("A", "B", "C_dev", "C_test")}, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    def summary(self) -> dict[str, int]:
        return {k: len(self.pool(k)) for k in ("A", "B", "C_dev", "C_test")}


def sample_spec(fam: tuple, rng: np.random.Generator, seed: int) -> WorldSpec:
    lat, om, core, rm = fam
    used = {lat, core, *om, *rm}
    params = tuple(
        (name, float(rng.choice(vals)))
        for name, vals in PARAM_GRID.items()
        if any(name.startswith(m[:4]) or name in _PARAM_OWNER.get(m, ()) for m in used)
    )
    return WorldSpec(lat, om, core, rm, k=int(rng.choice([4, 6, 8])),
                     seed=int(seed), params=params)


_PARAM_OWNER = {
    "masked": ("mask_p",), "distractor": ("distractor_p",), "mimic": ("mimic_p",),
    "delay": ("delay",), "sparse": ("sparse_n",), "switch": ("period",),
    "randwalk": ("walk_p",), "action_cost": ("act_cost",),
    "deceptive": ("cheat_r", "poison_k"), "reversal_after": ("flip_at",),
    "aggregate": ("hist_k",), "parity_history": ("hist_k",),
}


# ------------------------------------------------------- reference measurements
def reference_scores(spec: WorldSpec, n: int = 64, seed: int = 0
                     ) -> tuple[float, float]:
    """(random_policy, reference_high) mean per-step reward on this world instance."""
    rng = np.random.default_rng(seed)
    wb = WorldBatch(spec, n, rng)
    lo = 0.0
    for _ in range(CONTEXT):
        wb.observe()
        lo += float(wb.step(rng.integers(0, N_ACT, n)).sum())
    wb = WorldBatch(spec, n, np.random.default_rng(seed))
    hi = 0.0
    for _ in range(CONTEXT):
        wb.observe()
        hi += float(wb.step(wb.oracle_action()).sum())
    return lo / (n * CONTEXT), hi / (n * CONTEXT)


MIN_GAP = 0.08  # reference gap below this cannot discriminate between policies


def build_suite(split: SuiteSplit, cls: str, size: int, rng: np.random.Generator,
                *, min_gap: float = MIN_GAP, max_tries: int = 40) -> list[Scored]:
    """Draw `size` discriminative worlds of class `cls`, with their reference scores."""
    pool = split.pool(cls)
    out: list[Scored] = []
    tries = 0
    while len(out) < size and tries < size * max_tries:
        tries += 1
        fam = pool[int(rng.integers(0, len(pool)))]
        spec = sample_spec(fam, rng, seed=int(rng.integers(0, 2**31)))
        lo, hi = reference_scores(spec, seed=spec.seed)
        if hi - lo >= min_gap:
            out.append((spec, lo, hi))
    if len(out) < size:
        raise RuntimeError(f"only {len(out)}/{size} discriminative worlds for class {cls}")
    return out


def normalise(raw_per_step: float, lo: float, hi: float) -> float:
    """Map a raw per-step reward onto the random=0, reference_high=1 scale."""
    return float(np.clip((raw_per_step - lo) / max(1e-6, hi - lo), -0.5, 1.5))
