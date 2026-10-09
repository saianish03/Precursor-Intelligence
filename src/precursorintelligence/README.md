# src/precursorintelligence: the shared Python package

One installable package; each stage is a subpackage. Stages talk to storage only through `common/` clients (local disk ↔ GCS, Postgres ↔ BigQuery), never through hard-coded paths.

| Subpackage | Owns |
|---|---|
| `common/` | config loader, logging, GCS / BigQuery / Postgres clients, CCN utils, release-month utils |
| `ingestion/` | discover, download, profile, manifest, drift, load, `cli.py` (Step 1: landing → bronze) |
| `validation/` | data-quality checks on raw → staged (Pandera) |
| `features/` | preprocessing and feature engineering (silver → gold) |
| `training/` | model training, calibration, tuning |
| `evaluation/` | metrics, promotion gates, slice checks |
| `serving/` | batch scoring and prediction helpers used by the backend |
| `monitoring/` | drift, delayed-outcome evaluation, alerts |
| `agents/` | Pydantic AI agents, tools, retrieval (BM25), output validation; used by `backend/` and LLM evals |
