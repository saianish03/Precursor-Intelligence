# infra/terraform/

`envs/{dev,prod}/` compose the reusable `modules/`:
`gcs`, `bigquery`, `artifact_registry`, `cloud_run_job` (pipeline jobs), `cloud_run_service` (backend + frontend), `wif` (keyless GitHub Actions auth), `composer` (only if a managed Airflow environment is used).
State and `*.tfvars` are git-ignored.
