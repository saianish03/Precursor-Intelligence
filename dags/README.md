# dags/: Airflow DAGs

DAG files are **thin**: they schedule containerised jobs (the `precursor-pipeline` image) and hold no business logic. Helpers go in `utils/` and must not import heavy libraries (the DAG processor imports them on every parse).

| File | What |
|---|---|
| `cms_nh_pipeline.py` | **The data pipeline, one DAG** (`docs/DATA_PIPELINE_SPEC.md` §15): TaskGroups `ingestion` → `bronze` → `silver` → `gold` (`silver` and `gold` each end with a `validate_*` task, `docs/VALIDATION.md`), lock released by a teardown. `check_release` and `mirror_metadata` are TODO slots until those exist. |
| `utils/pipeline_operator.py` | `PipelineDockerOperator` (exit 2/3/137 fail without retry; exit 1 retried; the exit code kept as an XCom) and `docker_kwargs()` (laptop vs VM, from the environment) |
| `utils/alerts.py` | The alert callbacks (a failed run, a failed `release_lock`). The emails themselves, incl. new error anomalies, are built and sent by `precursorintelligence.alerts` (`src/`, mounted read-only into the Airflow containers) |

How it works: `docs/AIRFLOW_PIPELINE.md`. How to run it, locally and on the VM: `docs/runbooks/airflow-local.md`. Tests: `tests/dags/`, run inside the `precursor-airflow` image.
