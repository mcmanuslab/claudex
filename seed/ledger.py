"""
ledger.py -- the immutable prediction ledger.

A prediction about a hidden fact is only evidence if it was demonstrably made
BEFORE the fact was revealed. "We promise we didn't peek" is not a protocol, so
the ledger is a hash-chained append-only JSONL: every record carries the SHA256
of the previous record, and the records deliberately contain NO ground truth.

Truth is joined in afterwards by `evaluate.py`, straight from the world
generator. If anyone (including a future edit of this codebase) rewrites a
prediction after seeing the answer, `verify()` fails.
"""

from __future__ import annotations

import hashlib
import json
import os


class Ledger:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.prev = "0" * 64
        self.n = 0
        with open(path, "w") as f:
            f.write("")

    def append(self, rec: dict) -> None:
        assert "correct" not in rec and "truth" not in rec, \
            "the ledger must never contain ground truth at write time"
        rec = dict(rec)
        rec["_i"] = self.n
        rec["_prev"] = self.prev
        body = json.dumps(rec, sort_keys=True, separators=(",", ":"))
        h = hashlib.sha256((self.prev + body).encode()).hexdigest()
        rec["_hash"] = h
        with open(self.path, "a") as f:
            f.write(json.dumps(rec, sort_keys=True, separators=(",", ":")) + "\n")
        self.prev = h
        self.n += 1

    def append_many(self, recs: list[dict]) -> None:
        for r in recs:
            self.append(r)


def read(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def verify(path: str) -> tuple[bool, str]:
    prev = "0" * 64
    for i, rec in enumerate(read(path)):
        h = rec.pop("_hash", None)
        if rec.get("_prev") != prev:
            return False, f"record {i}: broken chain link"
        body = json.dumps(rec, sort_keys=True, separators=(",", ":"))
        calc = hashlib.sha256((prev + body).encode()).hexdigest()
        if calc != h:
            return False, f"record {i}: hash mismatch (record was edited)"
        prev = h
    return True, f"ok: {prev[:16]}"
