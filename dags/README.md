# dags/: Airflow DAGs

DAG files are **thin**: they schedule containerised jobs (the `precursor-pipeline` image) and hold no business logic. Helpers go in `utils/` and must not import heavy libraries (the DAG processor imports them on every parse).

| File | What |
|---|---|
| `cms_nh_pipeline.py` | **The data pipeline, one DAG** (`docs/DATA_PIPELINE_SPEC.md` §15): TaskGroups `ingestion` → `bronze` → `silver` → `gold` (`silver` and `gold` each end with a `validate_*` task, `docs/VALIDATION.md`), then `mirror_metadata` (the metadata store's copy in BigQuery) and the lock release, both teardowns. Scheduled on the asset `cms_nh_release`, or triggered by hand. |
| `cms_nh_release_check.py` | **The hourly release check** (spec §2.0, decision 24): `check_release` decides READY / RESUME / CHANGED (emit `cms_nh_release`, which starts `cms_nh_pipeline`) or PARTIAL / BUSY / BLOCKED / NO_CHANGE (skip); `alert` emails once when a release stays partial or proceeds start no run. Created paused; unpaused when production starts. |
| `cms_nh_ml.py` | **The ML pipeline, one DAG** (`docs/ML_PIPELINE_SPEC.md` §15): `decide` → `prepare` → one path (RETRAIN: `train` → `evaluate` → `gate`; PROMOTE: `promote`; SCORE; RETUNE: `tune` alone) → `score` → `publish` (emits `cms_nh_scores`) → `monitor` → `notify` (every run's summary email) → `mirror_metadata` (the teardown of a no-op setup `start`). Scheduled on `cms_nh_gold`, or triggered by hand with `mode`, `version`, `reason`, `allow_unverified`. Created paused. |
| `utils/pipeline_operator.py` | `PipelineDockerOperator` (exit 2/3/137 fail without retry; exit 1 retried; the exit code kept as an XCom) and `docker_kwargs()` (laptop vs VM, from the environment) |
| `utils/alerts.py` | The alert callbacks (a failed run, a failed `release_lock`, a failed `mirror_metadata`). The emails themselves, incl. new error anomalies, are built and sent by `precursorintelligence.alerts` (`src/`, baked into the Airflow image; mounted over it on a laptop) |

How it works: `docs/AIRFLOW_PIPELINE.md`. How to run it, locally and on the VM: `docs/runbooks/airflow-local.md`. Tests: `tests/dags/`, run inside the `precursor-airflow` image.
