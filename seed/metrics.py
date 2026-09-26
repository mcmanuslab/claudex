"""
metrics.py -- scoring the ledger.

The primary endpoint has to survive a hostile reading, so it is defined here
once and not adjusted afterwards.

    PRIMARY: mean holdout accuracy over the three genuinely extrapolative
    buckets D1 (near), D2 (compositional), D3 (distant), each bucket weighted
    equally, at a matched training-FLOPs budget.

Equal bucket weights matter: D3 has ~3x the facts of D1, so an unweighted mean
would silently make the primary endpoint "how well do you do on D3".

Why accuracy and not a composite: a hand-built correctness x novelty x
calibration scalar has free parameters, and free parameters in an endpoint are
how experiments get massaged. We report the components separately and show the
trade-off as a Pareto plot instead.

Supporting measures, each answering a specific way the primary could mislead:

  D0 accuracy        interpolation sanity -- if this is at chance nothing
                     learned and the extrapolation numbers are noise.
  D4 accuracy        THE LEAKAGE ALARM. D4 is a salted hash; it is
                     unpredictable in principle. Significantly above chance
                     means information is reaching the model that should not
                     be, and every other number is void.
  information gain   bits of validated novel information per prediction,
                     against the empirical P(answer | relation) computed on
                     REVEALED facts only. A model that always emits the
                     majority answer scores ~0. This is the direct answer to
                     "a model can game accuracy with safe predictions".
  Brier / ECE        calibration. A developmental model that is merely more
                     confident is not a better guesser.
  precision@coverage accuracy among its most-confident predictions, so
                     selective prediction is measured rather than assumed.
"""

from __future__ import annotations

import math
from collections import defaultdict

from world import DIST_NAMES, World

EXTRAPOLATIVE = [1, 2, 3]          # D1, D2, D3
LEAKAGE_BUCKET = 4                 # D4


def _chance(world: World, rel: str) -> float:
    """Chance accuracy for a relation, from the size of its true codomain."""
    sizes = {"VAL": 5, "SHELL": 6, "METAL": 2, "SOLUBLE": 2, "BOND": 4,
             "REACT": 6, "HEAVIER": 2, "REACTSOL": 2, "OMEN": 2}
    return 1.0 / sizes.get(rel, world.vocab.n_ans)


def score_records(world: World, recs: list[dict], age_for_prior: int) -> list[dict]:
    """Join predictions against ground truth. This is the ONLY place truth
    touches a prediction, and it happens strictly after the ledger is written."""
    truth = {f.fid: f for f in world.facts}
    prior = world.answer_prior(age_for_prior)
    out = []
    for r in recs:
        if r.get("kind") != "prediction":
            continue
        f = truth[r["fid"]]
        gold_tok = world.vocab.ans(f.answer)
        correct = int(r["pred_token"] == gold_tok)
        p_base = prior[f.rel].get(f.answer, 1e-6)
        pa = r.get("p_answers") or []
        # multiclass Brier over the answer symbols
        brier = 0.0
        if pa:
            for a, p in enumerate(pa):
                brier += (p - (1.0 if a == f.answer else 0.0)) ** 2
        out.append({
            **r, "correct": correct, "gold": f.answer,
            "info_gain_bits": correct * math.log2(1.0 / max(p_base, 1e-9)),
            "brier": brier,
            "chance": _chance(world, f.rel),
        })
    return out


def _mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def ece(scored: list[dict], bins: int = 10) -> float:
    """Expected calibration error on the top-1 confidence."""
    if not scored:
        return float("nan")
    buckets = defaultdict(list)
    for s in scored:
        buckets[min(bins - 1, int(s["conf"] * bins))].append(s)
    n = len(scored)
    return sum(len(v) / n * abs(_mean(x["correct"] for x in v) - _mean(x["conf"] for x in v))
               for v in buckets.values())


def precision_at_coverage(scored: list[dict], coverage: float) -> float:
    """Accuracy over the most-confident `coverage` fraction. Prevents gaming
    the headline number by abstaining on everything hard."""
    if not scored:
        return float("nan")
    s = sorted(scored, key=lambda r: -r["conf"])
    k = max(1, int(len(s) * coverage))
    return _mean(x["correct"] for x in s[:k])


def summarize(scored: list[dict], split: str = "holdout") -> dict:
    """The full report card for one model at one developmental age."""
    sel = [s for s in scored if s["split"] == split]
    by_d = defaultdict(list)
    for s in sel:
        by_d[s["dist"]].append(s)

    per_bucket = {}
    for d, rows in sorted(by_d.items()):
        n = len(rows)
        acc = _mean(r["correct"] for r in rows)
        ch = _mean(r["chance"] for r in rows)
        per_bucket[DIST_NAMES[d]] = {
            "n": n,
            "accuracy": acc,
            "chance": ch,
            "lift_over_chance": acc - ch,
            "se": math.sqrt(max(acc * (1 - acc), 1e-9) / n) if n else float("nan"),
            "info_gain_bits": _mean(r["info_gain_bits"] for r in rows),
            "brier": _mean(r["brier"] for r in rows),
            "ece": ece(rows),
            "mean_conf": _mean(r["conf"] for r in rows),
            "prec@50": precision_at_coverage(rows, 0.5),
        }

    extrap = [per_bucket[DIST_NAMES[d]]["accuracy"] for d in EXTRAPOLATIVE
              if DIST_NAMES[d] in per_bucket]

    # --- leakage alarm --------------------------------------------------
    # The baseline is NOT 0.5. The salted hash does not land exactly 50/50 on
    # a finite holdout (it is ~0.54 on these worlds), and a model that simply
    # learns the marginal "OMEN usually answers 1" legitimately achieves that
    # rate without any leakage. Testing against 0.5 therefore produces a false
    # alarm at z ~ 2.8 on a perfectly honest model -- which is exactly what
    # the pilot did. The right null is the majority-class rate of the bucket
    # itself: beating THAT is what would require information the model should
    # not have.
    leak_rows = [s for s in sel if s["dist"] == LEAKAGE_BUCKET]
    leak_n = len(leak_rows)
    if leak_n:
        golds = [s["gold"] for s in leak_rows]
        maj = max(set(golds), key=golds.count)
        leak_base = golds.count(maj) / leak_n
        leak_acc = _mean(s["correct"] for s in leak_rows)
        leak_se = math.sqrt(max(leak_base * (1 - leak_base), 1e-9) / leak_n)
        leak_z = (leak_acc - leak_base) / leak_se
    else:
        leak_base = leak_acc = leak_se = leak_z = float("nan")

    return {
        "n": len(sel),
        "PRIMARY_extrapolative_accuracy": _mean(extrap),
        "per_bucket": per_bucket,
        "overall_accuracy": _mean(r["correct"] for r in sel),
        "overall_info_gain_bits": _mean(r["info_gain_bits"] for r in sel),
        "overall_brier": _mean(r["brier"] for r in sel),
        "overall_ece": ece(sel),
        "leakage_baseline": leak_base,
        "leakage_accuracy": leak_acc,
        "leakage_z": leak_z,
        "leakage_flag": bool(leak_n and leak_z > 2.5),
    }


def by_relation(scored: list[dict], split: str = "holdout") -> dict:
    out = defaultdict(list)
    for s in scored:
        if s["split"] == split:
            out[s["rel"]].append(s)
    return {k: {"n": len(v), "accuracy": _mean(r["correct"] for r in v),
                "chance": _mean(r["chance"] for r in v)}
            for k, v in sorted(out.items())}


def paired_bootstrap(a: list[float], b: list[float], n: int = 20000,
                     seed: int = 0) -> dict:
    """Paired comparison across seeds. With 3-5 seeds a t-test is a fiction;
    a paired bootstrap at least reports the uncertainty honestly, and we quote
    the seed-level win count alongside it because n=3 is n=3."""
    import random
    rnd = random.Random(seed)
    d = [x - y for x, y in zip(a, b)]
    if not d:
        return {}
    boots = []
    for _ in range(n):
        s = [d[rnd.randrange(len(d))] for _ in d]
        boots.append(sum(s) / len(s))
    boots.sort()
    mean = sum(d) / len(d)
    sd = (sum((x - mean) ** 2 for x in d) / max(1, len(d) - 1)) ** 0.5
    return {
        "mean_diff": mean,
        "ci95": [boots[int(0.025 * n)], boots[int(0.975 * n)]],
        "cohens_d": mean / sd if sd > 1e-12 else float("inf") if mean else 0.0,
        "n_pairs": len(d),
        "wins": sum(1 for x in d if x > 0),
        "p_boot_gt0": sum(1 for x in boots if x <= 0) / n,
    }
