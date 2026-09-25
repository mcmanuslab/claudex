"""Event and metric storage.

SQLite for events and lineage, Parquet-friendly CSV for per-generation metric
series.  Deliberately not millions of tiny JSON files, per the original
proposal's instruction.

Module tensors are content-addressed (BLAKE2b of the raw bytes), which is where
the proposal's copy-on-write idea belongs: it makes the module phylogeny exact
and deduplicates the archive, without putting indirection into the execution
path where the model shows it buys nothing (RESEARCH.md 5, finding 4).
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS generations (
    generation INTEGER, run INTEGER, run_name TEXT,
    mean_reward REAL, max_reward REAL, mean_genome_len REAL,
    mean_active REAL, mean_flops REAL,
    n_duplications INTEGER, n_deletions INTEGER, n_encapsulations INTEGER,
    mean_q_str REAL, goal TEXT
);
CREATE TABLE IF NOT EXISTS events (
    generation INTEGER, run INTEGER, lane INTEGER, kind TEXT, gene INTEGER
);
CREATE TABLE IF NOT EXISTS module_lineage (
    innov INTEGER PRIMARY KEY, parent INTEGER, born_generation INTEGER
);
CREATE TABLE IF NOT EXISTS duplicate_pairs (
    run INTEGER, lane INTEGER, generation INTEGER,
    gene INTEGER, src_gene INTEGER, innov INTEGER, parent_innov INTEGER
);
CREATE TABLE IF NOT EXISTS module_blobs (
    digest TEXT PRIMARY KEY, n_params INTEGER, n_refs INTEGER
);
CREATE INDEX IF NOT EXISTS ix_gen_run ON generations(run, generation);
CREATE INDEX IF NOT EXISTS ix_ev_gen  ON events(generation);
CREATE INDEX IF NOT EXISTS ix_dp_run  ON duplicate_pairs(run, generation);
"""


class Store:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.con = sqlite3.connect(str(self.path))
        self.con.executescript(SCHEMA)
        self.con.commit()

    def write_generations(self, recs, run_names: list[str]) -> None:
        self.con.executemany(
            "INSERT INTO generations VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [(r.generation, r.run, run_names[r.run], r.mean_reward, r.max_reward,
              r.mean_genome_len, r.mean_active, r.mean_flops, r.n_duplications,
              r.n_deletions, r.n_encapsulations, r.mean_q_str,
              json.dumps(list(r.goal))) for r in recs])
        self.con.commit()

    def write_events(self, events) -> None:
        self.con.executemany("INSERT INTO events VALUES (?,?,?,?,?)", events)
        self.con.commit()

    def write_module_lineage(self, registry) -> None:
        self.con.executemany(
            "INSERT OR REPLACE INTO module_lineage VALUES (?,?,?)",
            [(i, registry.parent_of.get(i, -1), registry.born_at.get(i, 0))
             for i in registry.parent_of])
        self.con.commit()

    def write_duplicate_pairs(self, pairs) -> None:
        self.con.executemany(
            "INSERT INTO duplicate_pairs VALUES (?,?,?,?,?,?,?)",
            [(p["run"], p["lane"], p["generation"], p["gene"], p["src_gene"],
              p["innov"], p["parent_innov"]) for p in pairs])
        self.con.commit()

    def close(self) -> None:
        self.con.close()


def module_digest(*arrays: np.ndarray) -> str:
    """Content address for a module's weights."""
    h = hashlib.blake2b(digest_size=16)
    for a in arrays:
        h.update(np.ascontiguousarray(a, dtype=np.float32).tobytes())
    return h.hexdigest()


def dedup_stats(pop) -> dict:
    """Fraction of module weight sets in the population that are byte-distinct.

    Worth measuring even though the model says deduplication buys nothing in
    the execution path at pilot module size: it is the empirical check on that
    prediction, and it says how much the archive actually saves.
    """
    seen: dict[str, int] = {}
    total = 0
    for lane in range(pop.L):
        for g in np.flatnonzero(pop.alive[lane] > 0):
            d = module_digest(pop.Wq[lane, g], pop.Wk[lane, g], pop.Wv[lane, g],
                              pop.Wo[lane, g], pop.W1[lane, g], pop.W2[lane, g])
            seen[d] = seen.get(d, 0) + 1
            total += 1
    return {"distinct": len(seen), "total": total,
            "distinct_fraction": len(seen) / max(total, 1)}
