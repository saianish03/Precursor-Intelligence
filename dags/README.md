# dags/: Airflow DAGs

DAG files are **thin**: they schedule and trigger containerised jobs (the pipeline image) and hold no business logic. Helpers that DAGs need go in `utils/` and must not import heavy libraries. Integrity test: `tests/dags/test_dag_integrity.py`.

**The data pipeline is one DAG, `cms_nh_pipeline`** (5 TaskGroups: check_release → ingestion → bronze → silver → gold; `docs/DATA_PIPELINE_SPEC.md` §15). The ingestion TaskGroup's task contract is in `docs/STEP1_DATA_INGESTION_PIPELINE.md` §12.6; its older standalone `ingest_vintage` DAG is superseded. Add the DAG here once the `precursor-pipeline` image exists, with `tests/dags` (import check + `PipelineDockerOperator` exit codes, §12.6).
