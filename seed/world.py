"""
world.py -- the hidden world.

A fully synthetic "lattice chemistry" universe. Every fact is a deterministic
function of latent per-entity coordinates that the model never observes. The
point is that a model which *induces the latent coordinates and the rules over
them* can predict facts it has never seen, while a model which *memorises
observed facts* cannot.

Design constraints that matter (each one is a defence against a specific way
this experiment could lie to us):

  1. Entity token ids are a RANDOM PERMUTATION of the latent grid positions.
     If element #7 were literally at grid position 7, the token id itself would
     leak the latent coordinate and "extrapolation" would be trivial lookup.
  2. One relation (OMEN) is a salted hash -- ground truth exists but is
     information-theoretically unpredictable from anything observable. Any
     model scoring above chance on it is reading something it should not be.
     This is our leakage alarm, and it is load-bearing.
  3. The "distant extrapolation" bucket withholds facts about entities whose
     latent period is 5, while still grounding period-5 entities via SHELL
     facts. So the period *value* is observable but the *rule* f(period,group)
     must be extrapolated one step beyond the range it was fit on. Without
     that grounding the bucket would be unpredictable-in-principle, i.e. a
     second copy of OMEN rather than a real extrapolation test.

Fact surface form (fixed length 5, loss on the final position only):

    [REL] [ARG1] [ARG2] [=] [ANSWER]

Unary relations use a PAD token for ARG2. Training loss and every metric are
computed at the ANSWER position only, so "accuracy" and "confidence" are
unambiguous and the model gets no free loss from predicting the format.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict

# ----------------------------------------------------------------------------
# latent world parameters (the model never sees any of this)
# ----------------------------------------------------------------------------

N_GROUP = 10         # latent x coordinate
N_PERIOD = 8         # latent y coordinate
N_ELEM = N_GROUP * N_PERIOD

VALENCE = [1, 2, 3, 4, 3, 2, 1, 0, 2, 4]  # group -> valence     (len N_GROUP)
N_PRODUCT = 6                              # REACT product classes
SOLUBILITY = [1, 0, 1, 1, 0, 0]            # product class -> soluble
BOND_OF_GAP = [0, 1, 1, 2, 2, 3, 3, 3, 2, 1]  # |g1-g2| -> bond type (len N_GROUP)
METAL_GROUP_CUTOFF = 3                     # group < cutoff => metal
DISTANT_PERIOD = N_PERIOD - 1              # the withheld-rule period
OMEN_SUBSAMPLE = 8                         # keep 1 in N OMEN pairs (noise, not the task)
ATOMIC_RELATIONS = ("VAL", "SHELL", "METAL", "SOLUBLE")  # the world's axioms

RELATIONS = [
    "VAL",       # unary,  f(group)                      -- 1 latent var
    "SHELL",     # unary,  f(period)                     -- 1 latent var
    "METAL",     # unary,  threshold on group
    "BOND",      # binary, f(|g1 - g2|)                  -- relational on group
    "REACT",     # binary, f(g1 + g2)                    -- relational on group
    "SOLUBLE",   # unary on product class
    "HEAVIER",   # binary, comparison on mass = f(p, g)  -- 2 latent vars
    "REACTSOL",  # binary, SOLUBLE(REACT(e1, e2))        -- 2-hop composition
    "OMEN",      # binary, salted hash -- UNPREDICTABLE BY CONSTRUCTION
]

# extrapolation-distance buckets
D_INTERP, D_NEAR, D_COMPOS, D_DISTANT, D_UNSUPPORTED = 0, 1, 2, 3, 4
DIST_NAMES = {
    D_INTERP: "D0-interpolation",
    D_NEAR: "D1-near",
    D_COMPOS: "D2-compositional",
    D_DISTANT: "D3-distant",
    D_UNSUPPORTED: "D4-unsupported",
}


def _pearson(xs: list, ys: list) -> float:
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    den = (sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys)) ** 0.5
    return 0.0 if den == 0 else num / den


def _mass(period: int, group: int) -> int:
    """Latent mass. Linear and monotone in period so that the rule fitted on
    periods 0..4 has a unique correct extrapolation to period 5."""
    return 12 * period + 2 * group


# ----------------------------------------------------------------------------
# vocabulary
# ----------------------------------------------------------------------------

class Vocab:
    """Flat vocabulary. Answers are NOT restricted per-relation at eval time:
    the model must learn the type constraints itself, otherwise we would be
    handing it free information and inflating every accuracy number."""

    def __init__(self) -> None:
        self.itos: list[str] = []
        self.stoi: dict[str, int] = {}
        for t in ["<pad>", "<eq>"]:
            self._add(t)
        self.rel_base = len(self.itos)
        for r in RELATIONS:
            self._add(f"<rel:{r}>")
        self.elem_base = len(self.itos)
        for i in range(N_ELEM):
            self._add(f"<e:{i:02d}>")
        self.prod_base = len(self.itos)
        for i in range(N_PRODUCT):
            self._add(f"<c:{i}>")
        self.ans_base = len(self.itos)
        # answer symbols: shared integer answer space, wide enough for every
        # relation's codomain (valence 0..4, shell 0..5, bool, bond 0..3, ...)
        self.n_ans = max(N_PRODUCT, N_PERIOD, max(VALENCE) + 1, max(BOND_OF_GAP) + 1, 2)
        for i in range(self.n_ans):
            self._add(f"<a:{i}>")

    def _add(self, tok: str) -> int:
        self.stoi[tok] = len(self.itos)
        self.itos.append(tok)
        return self.stoi[tok]

    def __len__(self) -> int:
        return len(self.itos)

    @property
    def pad(self) -> int:
        return self.stoi["<pad>"]

    @property
    def eq(self) -> int:
        return self.stoi["<eq>"]

    def rel(self, name: str) -> int:
        return self.rel_base + RELATIONS.index(name)

    def ans(self, v: int) -> int:
        return self.ans_base + int(v)


@dataclass
class Fact:
    fid: int
    rel: str
    a1: int               # surface token index (element id or product class)
    a2: int               # -1 for unary
    answer: int           # integer answer value
    a1_is_product: bool = False
    dist: int = D_INTERP  # extrapolation-distance bucket (only meaningful in holdout)
    split: str = "pool"   # "pool" | "val" | "holdout"
    tranche: int = -1     # revelation tranche index, -1 for val/holdout


@dataclass
class WorldSpec:
    seed: int = 0
    holdout_frac: float = 0.30
    val_frac: float = 0.08
    n_tranches: int = 8
    initial_reveal_frac: float = 0.25  # fraction of the pool revealed at age 0
    salt: str = "omen-v1"


class World:
    """Generates the complete universe, then hides most of it."""

    def __init__(self, spec: WorldSpec) -> None:
        self.spec = spec
        self.vocab = Vocab()
        rng = _Rng(spec.seed)

        # --- latent coordinates, deliberately decorrelated from token id -----
        # A single random permutation leaves a Pearson r of ~1/sqrt(N) between
        # token id and latent coordinate. That is a real (if weak) side channel
        # through which a model could "extrapolate" by reading the token id, so
        # we rejection-sample until the leak is negligible.
        for _attempt in range(10000):
            perm = list(range(N_ELEM))
            rng.shuffle(perm)
            g = [p % N_GROUP for p in perm]
            q = [p // N_GROUP for p in perm]
            if max(abs(_pearson(list(range(N_ELEM)), g)),
                   abs(_pearson(list(range(N_ELEM)), q))) < 0.05:
                break
        else:
            raise RuntimeError("could not decorrelate token ids from latent grid")
        self.token_id_leak = {
            "pearson_id_group": _pearson(list(range(N_ELEM)), [p % N_GROUP for p in perm]),
            "pearson_id_period": _pearson(list(range(N_ELEM)), [p // N_GROUP for p in perm]),
        }
        # elem_token i sits at latent grid cell perm[i]
        self.group = [perm[i] % N_GROUP for i in range(N_ELEM)]
        self.period = [perm[i] // N_GROUP for i in range(N_ELEM)]
        self.mass = [_mass(self.period[i], self.group[i]) for i in range(N_ELEM)]

        self.facts: list[Fact] = []
        self._build_facts()
        self._assign_splits(rng)

    # ------------------------------------------------------------------ rules
    def _react(self, e1: int, e2: int) -> int:
        return (self.group[e1] + self.group[e2]) % N_PRODUCT

    def _omen(self, e1: int, e2: int) -> int:
        h = hashlib.sha256(f"{self.spec.salt}|{self.spec.seed}|{e1}|{e2}".encode())
        return h.digest()[0] & 1

    def _answer(self, rel: str, a1: int, a2: int) -> int:
        if rel == "VAL":
            return VALENCE[self.group[a1]]
        if rel == "SHELL":
            return self.period[a1]
        if rel == "METAL":
            return int(self.group[a1] < METAL_GROUP_CUTOFF)
        if rel == "SOLUBLE":
            return SOLUBILITY[a1]
        if rel == "BOND":
            return BOND_OF_GAP[abs(self.group[a1] - self.group[a2])]
        if rel == "REACT":
            return self._react(a1, a2)
        if rel == "HEAVIER":
            return int(self.mass[a1] > self.mass[a2])
        if rel == "REACTSOL":
            return SOLUBILITY[self._react(a1, a2)]
        if rel == "OMEN":
            return self._omen(a1, a2)
        raise ValueError(rel)

    def _build_facts(self) -> None:
        fid = 0
        unary = ["VAL", "SHELL", "METAL"]
        binary = ["BOND", "REACT", "HEAVIER", "REACTSOL", "OMEN"]
        for rel in unary:
            for e in range(N_ELEM):
                self.facts.append(Fact(fid, rel, e, -1, self._answer(rel, e, -1)))
                fid += 1
        for c in range(N_PRODUCT):
            self.facts.append(
                Fact(fid, "SOLUBLE", c, -1, SOLUBILITY[c], a1_is_product=True))
            fid += 1
        for rel in binary:
            for e1 in range(N_ELEM):
                for e2 in range(N_ELEM):
                    if e1 == e2:
                        continue
                    # OMEN is pure noise. It must be present (it is our leakage
                    # alarm and our calibration trap) but it must not dominate
                    # the training mixture, or the controller would spend the
                    # whole budget growing capacity to memorise coin flips.
                    if rel == "OMEN" and (e1 * N_ELEM + e2) % OMEN_SUBSAMPLE != 0:
                        continue
                    self.facts.append(Fact(fid, rel, e1, e2, self._answer(rel, e1, e2)))
                    fid += 1

    # ----------------------------------------------------------------- splits
    def _dist_bucket(self, f: Fact) -> int:
        """Structural assignment of extrapolation distance. Computed from the
        world's own geometry, never from model behaviour."""
        if f.rel == "OMEN":
            return D_UNSUPPORTED
        involves_distant = (
            self.period[f.a1] == DISTANT_PERIOD if not f.a1_is_product else False
        ) or (f.a2 >= 0 and self.period[f.a2] == DISTANT_PERIOD)
        # D3 is ONLY the HEAVIER facts touching a distant-period element.
        # REACTSOL was originally binned here too, which was a mistake the
        # pilot caught: REACTSOL(e1,e2) = SOLUBILITY[(g1+g2) mod 6] depends on
        # GROUP alone, so its period-5 instances are perfectly predictable from
        # information the model already has. Leaving them in D3 inflated the
        # flagship extrapolation number with facts that require no
        # extrapolation at all. They are compositional, so they go to D2.
        if f.rel == "HEAVIER" and involves_distant:
            return D_DISTANT
        if f.rel == "REACTSOL":
            return D_COMPOS
        if involves_distant:
            return D_NEAR
        return D_INTERP

    def _assign_splits(self, rng: "_Rng") -> None:
        sp = self.spec
        for f in self.facts:
            f.dist = self._dist_bucket(f)

        # --- hard structural rule, applied BEFORE any random splitting -------
        # Every HEAVIER fact touching a period-5 element is withheld, so the
        # mass rule is fitted only on periods 0..4 and must be extrapolated.
        # SHELL facts for those same elements stay in the pool: the period is
        # observable, only the rule's application at that period is withheld.
        forced_holdout: set[int] = set()
        for f in self.facts:
            touches_distant = (
                (not f.a1_is_product and self.period[f.a1] == DISTANT_PERIOD)
                or (f.a2 >= 0 and self.period[f.a2] == DISTANT_PERIOD)
            )
            if touches_distant and f.rel == "HEAVIER":
                forced_holdout.add(f.fid)
            if f.rel == "SHELL" and touches_distant:
                f.dist = D_NEAR  # grounding fact: stays available

        # SHELL facts for distant-period elements must never be withheld,
        # otherwise D3 degenerates into "unpredictable" rather than
        # "extrapolate the rule one step".
        grounding: set[int] = {
            f.fid for f in self.facts
            if f.rel == "SHELL" and not f.a1_is_product
            and self.period[f.a1] == DISTANT_PERIOD
        }

        rest = [f for f in self.facts if f.fid not in forced_holdout and f.fid not in grounding]
        rng.shuffle(rest)
        n_hold = int(sp.holdout_frac * len(rest))
        n_val = int(sp.val_frac * len(rest))

        for f in self.facts:
            if f.fid in forced_holdout:
                f.split = "holdout"
        for f in rest[:n_hold]:
            f.split = "holdout"
        for f in rest[n_hold:n_hold + n_val]:
            f.split = "val"
        pool = [f for f in rest[n_hold + n_val:]] + [
            f for f in self.facts if f.fid in grounding]
        for f in pool:
            f.split = "pool"

        # --- revelation tranches over the pool -------------------------------
        # ATOMIC facts (the unary property relations) are the axioms of this
        # world: they are the only direct evidence of an entity's latent
        # coordinates. The pilot showed that drip-feeding them across tranches
        # starves the model of the grounding it needs, so that "failure to
        # extrapolate" is really "the axiom had not been stated yet" -- which
        # would be a test of the revelation schedule, not of the model. They
        # are therefore ALL revealed at age 0, and progressive revelation
        # applies to the relational facts. This mirrors the atomic-vs-inferred
        # split used by Grokked Transformers.
        rng.shuffle(pool)
        atomic = [f for f in pool if f.rel in ATOMIC_RELATIONS]
        relational = [f for f in pool if f.rel not in ATOMIC_RELATIONS]
        for f in atomic:
            f.tranche = 0
        n0 = int(sp.initial_reveal_frac * len(relational))
        for f in relational[:n0]:
            f.tranche = 0
        remaining = relational[n0:]
        per = max(1, len(remaining) // max(1, sp.n_tranches - 1))
        for i, f in enumerate(remaining):
            f.tranche = min(sp.n_tranches - 1, 1 + i // per)

    # ------------------------------------------------------------ encoding
    def encode(self, f: Fact) -> tuple[list[int], int]:
        v = self.vocab
        a1 = (v.prod_base + f.a1) if f.a1_is_product else (v.elem_base + f.a1)
        a2 = v.pad if f.a2 < 0 else (v.elem_base + f.a2)
        seq = [v.rel(f.rel), a1, a2, v.eq, v.ans(f.answer)]
        return seq, v.ans(f.answer)

    # ------------------------------------------------------------- accessors
    def revealed(self, age: int) -> list[Fact]:
        """Facts legitimately available for training at developmental age
        `age`. This is the single source of truth for the no-leakage rule."""
        return [f for f in self.facts if f.split == "pool" and 0 <= f.tranche <= age]

    def unrevealed_pool(self, age: int) -> list[Fact]:
        return [f for f in self.facts if f.split == "pool" and f.tranche > age]

    def val_facts(self) -> list[Fact]:
        return [f for f in self.facts if f.split == "val"]

    def holdout(self) -> list[Fact]:
        return [f for f in self.facts if f.split == "holdout"]

    def answer_prior(self, age: int) -> dict[str, dict[int, float]]:
        """Empirical P(answer | relation) over the REVEALED facts only. Used as
        the baseline for information gain, so that a model which always emits
        the majority answer scores ~0 bits of validated novel information."""
        counts: dict[str, dict[int, int]] = {}
        for f in self.revealed(age):
            counts.setdefault(f.rel, {})
            counts[f.rel][f.answer] = counts[f.rel].get(f.answer, 0) + 1
        out: dict[str, dict[int, float]] = {}
        k = self.vocab.n_ans
        for rel in RELATIONS:
            c = counts.get(rel, {})
            tot = sum(c.values())
            # Laplace smoothing: an unseen answer keeps finite information gain,
            # and a relation with no revealed facts falls back to uniform.
            out[rel] = {a: (c.get(a, 0) + 1) / (tot + k) for a in range(k)}
        return out

    def summary(self) -> dict:
        by = {}
        for f in self.facts:
            key = (f.split, DIST_NAMES[f.dist])
            by[f"{key[0]}/{key[1]}"] = by.get(f"{key[0]}/{key[1]}", 0) + 1
        rel_counts = {}
        for f in self.facts:
            rel_counts[f.rel] = rel_counts.get(f.rel, 0) + 1
        return {
            "n_facts": len(self.facts),
            "n_elem": N_ELEM,
            "vocab": len(self.vocab),
            "by_split_dist": dict(sorted(by.items())),
            "by_relation": rel_counts,
            "spec": asdict(self.spec),
            "token_id_leak": self.token_id_leak,
        }


class _Rng:
    """Tiny deterministic RNG so world generation never depends on torch/numpy
    global state (which the training loop reseeds constantly)."""

    def __init__(self, seed: int) -> None:
        self.s = (seed * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)

    def _next(self) -> int:
        self.s = (self.s * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return (self.s >> 33) & 0xFFFFFFFF

    def randint(self, n: int) -> int:
        return self._next() % n

    def shuffle(self, xs: list) -> None:
        for i in range(len(xs) - 1, 0, -1):
            j = self.randint(i + 1)
            xs[i], xs[j] = xs[j], xs[i]


if __name__ == "__main__":
    w = World(WorldSpec(seed=0))
    print(json.dumps(w.summary(), indent=2))
