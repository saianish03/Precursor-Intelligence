---
name: mlops-deployment
description: Platform conventions for Precursor Intelligence on GCP. Use when writing Airflow 3 DAGs, Dockerfiles, docker-compose, Terraform, GitHub Actions workflows, Cloud Run deployment, MLflow server setup, the precursor_meta metadata store, budgets or cost controls.
---

# MLOps deployment

## Runtime topology
- One Compute Engine VM (e2-standard-2) running Docker Compose: `postgres` (databases `airflow`, `mlflow`,
  `precursor_meta`), Airflow 3 (`api-server`, `scheduler`, `dag-processor`, `triggerer`, LocalExecutor),
  `mlflow` (artifacts in GCS). Instance schedule stops the VM overnight and on weekends.
- Cloud Run: `precursor-api` (FastAPI + agents), min-instances 0, max-instances 2. UI is built in Base44.
- Storage: GCS (bronze/silver/gold/artifacts/reports, lifecycle rules) and BigQuery (gold, scores, monitoring,
  `meta` mirror, `llm_calls`). Artifact Registry with a keep-last-5 cleanup policy.
- Never: Cloud Composer, GKE, Vertex AI online endpoints, Cloud SQL, self-hosted GPU LLMs.

## Airflow DAG conventions
- DAGs are thin: every task is a `DockerOperator` running the `precursor-pipeline` image with a CLI command
  (`uv run python -m precursorintelligence.<module> --vintage {{ ... }}`). No business logic in `dags/`.
- Chain DAGs with Airflow **Assets**: `pdc_release_sensor` → `ingest_vintage` → `build_silver` → `build_gold`
  → `train_evaluate` → `batch_score` → `monitor`; plus `backfill_archive`, `llm_eval`, optional `ingest_pbj`.
- Tasks are idempotent and keyed by vintage; reruns overwrite silver/gold partitions; bronze is immutable.
- A failed Pandera or leakage check stops the DAG before gold is published.
- DAG files must import cleanly (`tests/dags` checks import errors and cycles).

## Docker
- Images: `precursor-pipeline`, `precursor-api`, `precursor-airflow`, `precursor-mlflow`; official `postgres:16`.
- Use uv in images: copy `pyproject.toml` + `uv.lock`, `uv sync --frozen --no-dev`, non-root user, pinned base
  image digest or tag. No secrets in build args or layers.

## Terraform
- All GCP resources in `infra/terraform/` (`envs/{dev,prod}` compose `modules/`), state in GCS. Budgets ($25/$50/$100/$150), VM schedule,
  BigQuery datasets, service accounts (least privilege), Workload Identity Federation for GitHub Actions.
- Local commands: `fmt`, `validate`, `plan` only. `apply` runs in CI after review.

## CI/CD (GitHub Actions)
`ci.yml` (lint, types, tests, Docker build, terraform plan), `llm-eval.yml`, `cd.yml` (push images,
Cloud Run no-traffic deploy → smoke → 10% → 100%, DAG sync), `promote-model.yml` (manual override and rollback),
`e2e-nightly.yml`, `pr-hygiene.yml`, `weekly-report.yml`. Auth through WIF only; no JSON keys.

## Metadata
Write pipeline metadata to Postgres `precursor_meta` (vintage_registry, ingestion_log, dataset_manifest,
dq_results, gate_results, promotion_audit, agent_registry, llm_eval_runs). Mirror to BigQuery `meta` at the
end of each DAG run. Publish `status/latest.json` to GCS at the end of `monitor`. Nightly `pg_dump` to GCS.

## Cost guardrails
Every BigQuery query sets `maximum_bytes_billed`; partition tables by month; LLM daily call cap and
`max_tokens`; check spend weekly against the $300 trial budget (planned ~$65–105).
