# .github/

Planned files (added by the owning teammate; empty workflow files are not committed because GitHub Actions would fail on them):

| File | Purpose |
|---|---|
| `CODEOWNERS` | review ownership per folder |
| `pull_request_template.md` | PR checklist |
| `ISSUE_TEMPLATE/{feature,bug,data_issue}.yml` | issue forms |
| `workflows/ci.yml` | ✅ added: PRs into dev/main run ruff + offline tests; `ci-ok` is the required check |
| `workflows/integration.yml` | nightly / manual: real CMS + GCP sandbox |
| `workflows/deploy-ingestion.yml` | main: build → Artifact Registry → Cloud Run job |
| `workflows/deploy-backend.yml`, `deploy-frontend.yml` | main: build → Cloud Run services |
| `workflows/deploy-dags.yml` | main: sync `dags/` to the Airflow environment |
