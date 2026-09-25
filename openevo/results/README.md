# results/

Run outputs land here as `results/<run name>/`:

- `run.db` — SQLite: per-generation aggregates, per-organism records (including every
  organism that ever lived, extinct ones included), and probe results
- `run.manifest.json` — immutable provenance: config, config hash, git commit, seed,
  platform, and the sealed hash of the world-family split
- `report.md` / `report.png` — produced by `scripts/report.py`
- `checkpoint.pkl` — resume point, if `checkpoint_every` is set

Databases and checkpoints are gitignored because they regenerate from the config plus the
seed. Reports and manifests are small and worth committing for a run whose results are
being cited.

`DEVIATIONS.md` records any departure from `PREREGISTRATION.md`, with reason and date.
