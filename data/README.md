# data/ (git-ignored)

Local cache that mirrors the GCS layout (`landing/`, `bronze/`, `silver/`, `gold/`, `reports/`). Nothing in here is committed. Only this README is tracked.

`data/toy/` is built and versioned by DVC (`dvc.yaml`; see the root README, "Data versioning with DVC"): `uv run dvc repro` rebuilds it, `uv run dvc pull` restores it.
