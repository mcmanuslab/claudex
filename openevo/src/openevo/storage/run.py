"""Run provenance and the metrics store.

Every run writes an immutable manifest (config, git commit, seeds, machine, the sealed
hash of the world-family split) alongside a SQLite database. SQLite rather than a tree of
JSON files because a long run produces millions of organism records and the analysis is
almost entirely relational: lineages, per-generation aggregates, survival by architecture.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sqlite3
import subprocess
import sys
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS run (
  key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS generation (
  generation INTEGER PRIMARY KEY, payload TEXT);
CREATE TABLE IF NOT EXISTS organism (
  rowid_ INTEGER PRIMARY KEY AUTOINCREMENT,
  oid INTEGER, gen INTEGER, birth_gen INTEGER, island INTEGER,
  parents TEXT, lineage_root INTEGER, params INTEGER, sig TEXT,
  d_model INTEGER, n_heads INTEGER, d_ff INTEGER, n_blocks INTEGER,
  mutations TEXT, n_struct_events INTEGER, fitness REAL, score REAL, gain REAL,
  eval_flops INTEGER, age INTEGER, alive INTEGER, death_reason TEXT,
  beh0 REAL, beh1 REAL, beh2 REAL,
  gene_p_growth_bias REAL, gene_p_structural REAL, gene_temperature REAL,
  gene_lr REAL, gene_weight_sigma REAL, gene_meta_sigma REAL, gene_morph_noise REAL);
CREATE INDEX IF NOT EXISTS ix_org_gen ON organism(gen);
CREATE INDEX IF NOT EXISTS ix_org_oid ON organism(oid);
CREATE TABLE IF NOT EXISTS probe (
  generation INTEGER, cls TEXT, condition TEXT, metric TEXT, value REAL);
CREATE INDEX IF NOT EXISTS ix_probe ON probe(generation, cls, condition);
"""

ORG_COLS = [
    "oid", "gen", "birth_gen", "island", "parents", "lineage_root", "params", "sig",
    "d_model", "n_heads", "d_ff", "n_blocks", "mutations", "n_struct_events",
    "fitness", "score", "gain", "eval_flops", "age", "alive", "death_reason",
    "beh0", "beh1", "beh2", "gene_p_growth_bias", "gene_p_structural",
    "gene_temperature", "gene_lr", "gene_weight_sigma", "gene_meta_sigma",
    "gene_morph_noise",
]


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, timeout=5).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _jsonable(x: Any) -> Any:
    return asdict(x) if is_dataclass(x) and not isinstance(x, type) else x


class RunStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path)
        self.db.executescript(SCHEMA)
        self.db.commit()

    def write_manifest(self, **parts: Any) -> str:
        payload = {k: _jsonable(v) for k, v in parts.items()}
        payload |= {
            "git_commit": git_commit(),
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "machine": platform.machine(),
        }
        blob = json.dumps(payload, sort_keys=True, default=str)
        payload["config_hash"] = hashlib.sha256(blob.encode()).hexdigest()[:16]
        self.db.executemany("INSERT OR REPLACE INTO run VALUES (?,?)",
                            [(k, json.dumps(v, default=str)) for k, v in payload.items()])
        self.db.commit()
        (self.path.parent / f"{self.path.stem}.manifest.json").write_text(
            json.dumps(payload, indent=2, default=str))
        return payload["config_hash"]

    def add_generation(self, snap: dict) -> None:
        self.db.execute("INSERT OR REPLACE INTO generation VALUES (?,?)",
                        (snap["generation"], json.dumps(snap)))

    def add_organisms(self, records: list[dict]) -> None:
        if not records:
            return
        rows = [tuple(r.get(c) for c in ORG_COLS) for r in records]
        self.db.executemany(
            f"INSERT INTO organism ({','.join(ORG_COLS)}) "
            f"VALUES ({','.join('?' * len(ORG_COLS))})", rows)

    def add_probe(self, generation: int, cls: str, condition: str,
                  metrics: dict[str, float]) -> None:
        self.db.executemany(
            "INSERT INTO probe VALUES (?,?,?,?,?)",
            [(generation, cls, condition, k, float(v)) for k, v in metrics.items()])

    def commit(self) -> None:
        self.db.commit()

    def close(self) -> None:
        self.db.commit()
        self.db.close()
