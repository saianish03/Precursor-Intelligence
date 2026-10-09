# dags/: Airflow DAGs

DAG files are **thin**: they schedule and trigger containerised jobs (the pipeline image) and hold no business logic. Helpers that DAGs need go in `utils/` and must not import heavy libraries. Integrity test: `tests/dags/test_dag_integrity.py`.
