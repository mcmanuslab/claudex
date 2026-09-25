"""Frozen-organism probes: the evolvability decomposition.

A rising "adaptation speed on novel worlds" curve is, on its own, uninterpretable. At
least four different mechanisms produce that same curve, and only one of them is the
hypothesis:

1. genuine in-context adaptive machinery;
2. a better *reactive prior* that happens to suit the held-out worlds (transfer, not
   adaptation);
3. capacity -- bigger models simply learn faster per interaction;
4. hyperparameters (exploration temperature, learning rate) tuned to the evaluation
   protocol rather than to adapting.

Each frozen organism is therefore re-evaluated under transplant conditions that switch
these off one at a time:

``full``               as it evolved -- the headline number.
``no_feedback``        previous action and reward tokens zeroed. Observations still
                       arrive, but consequences do not. Improvement that survives this
                       is mechanism 2, not mechanism 1.
``reinit_weights``     the evolved architecture and genes, with random weights. What
                       still improves is carried by the *substrate* rather than by
                       learned content -- the strict sense of evolved evolvability.
``capacity_matched``   an ancestral-shaped architecture scaled to the same parameter
                       count, random weights. Isolates mechanism 3.
``fixed_hparams``      evolved weights and architecture, but exploration temperature
                       reset to the ancestral value. Isolates mechanism 4.

Everything is reported per class (A / B / C-dev / C-test) and per condition, and every
condition is scored on the *same* world instances with common random numbers.
"""

from __future__ import annotations

import numpy as np

from ..environments.suites import Scored
from ..evolution.evaluate import evaluate_group
from ..evolution.organism import DEFAULT_GENES, Organism
from ..models.genome import ArchGenome, scale_to_params
from ..models.transformer import init_params

CONDITIONS = ("full", "no_feedback", "reinit_weights", "capacity_matched", "fixed_hparams")


def _capacity_matched(o: Organism, ancestral: ArchGenome,
                      rng: np.random.Generator) -> tuple[ArchGenome, dict]:
    """Ancestral *shape* re-scaled to this organism's parameter count."""
    try:
        arch = scale_to_params(o.n_params, base=ancestral,
                               n_blocks=ancestral.n_blocks,
                               head_dim=ancestral.head_dim)
    except ValueError:
        arch = o.arch
    return arch, init_params(arch, rng, n=1)


def probe_condition(organisms: list[Organism], worlds: list[Scored], condition: str,
                    ancestral: ArchGenome, rng: np.random.Generator,
                    n_inst: int = 12) -> dict[str, float]:
    """Score `organisms` on `worlds` under one transplant condition."""
    by_arch: dict[tuple, list[int]] = {}
    prepared: list[tuple[ArchGenome, dict, float]] = []
    for o in organisms:
        seed_rng = np.random.default_rng(abs(hash((o.oid, condition))) % (2**31))
        if condition == "reinit_weights":
            arch, w = o.arch, init_params(o.arch, seed_rng, n=1)
        elif condition == "capacity_matched":
            arch, w = _capacity_matched(o, ancestral, seed_rng)
        else:
            arch, w = o.arch, o.weights
        temp = (DEFAULT_GENES["temperature"] if condition == "fixed_hparams"
                else o.gene("temperature"))
        prepared.append((arch, w, temp))
    for i, (arch, _, _) in enumerate(prepared):
        by_arch.setdefault(arch.signature(), []).append(i)

    acc: dict[str, list[float]] = {"score": [], "gain": [], "final": [], "params": []}
    for _, idxs in by_arch.items():
        arch = prepared[idxs[0]][0]
        ws = [prepared[i][1] for i in idxs]
        if not ws or not ws[0]:
            continue
        temps = np.array([prepared[i][2] for i in idxs], dtype=np.float32)
        res = evaluate_group(ws, arch, worlds, temps, np.random.default_rng(12345),
                             n_inst, feedback=(condition != "no_feedback"))
        acc["score"] += list(res["score"])
        acc["gain"] += list(res["gain"])
        acc["final"] += list(res["final"])
        acc["params"] += [float(arch.n_params)] * len(idxs)
    if not acc["score"]:
        return {}
    return {
        "score": float(np.mean(acc["score"])),
        "score_sd": float(np.std(acc["score"])),
        "gain": float(np.mean(acc["gain"])),
        "final": float(np.mean(acc["final"])),
        "params": float(np.mean(acc["params"])),
        "n": float(len(acc["score"])),
    }


def run_probes(organisms: list[Organism], suites: dict[str, list[Scored]],
               ancestral: ArchGenome, rng: np.random.Generator,
               conditions: tuple[str, ...] = CONDITIONS,
               n_inst: int = 12) -> list[tuple[str, str, dict[str, float]]]:
    """Cross-product of world classes and transplant conditions for a frozen cohort."""
    out = []
    for cls, worlds in suites.items():
        for cond in conditions:
            m = probe_condition(organisms, worlds, cond, ancestral, rng, n_inst)
            if m:
                out.append((cls, cond, m))
    return out
