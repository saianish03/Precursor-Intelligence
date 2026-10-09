# Precursor Intelligence

An end-to-end MLOps application - Nursing Homes Risk Assessment for the Insurance Industry

## Repository layout

```
.github/            CI/CD workflows, PR + issue templates, CODEOWNERS
configs/            base/dev/prod config, dataset registry, versioned schema contracts, LLM prompts
src/precursorintelligence/
                    shared Python package: common, ingestion, validation, features, training,
                    evaluation, serving, monitoring, agents
backend/            FastAPI service (+ Pydantic AI agents via src/precursorintelligence/agents) → Cloud Run
frontend/           Streamlit dashboard → Cloud Run
dags/               thin Airflow DAGs (trigger jobs only, no business logic)
docker/             Dockerfiles per image + local Airflow compose
infra/terraform/    GCP infrastructure: envs/{dev,prod} + reusable modules
tests/              mirrors src/ + backend/frontend/contracts/dags; tiny fixtures only
notebooks/          exploration only (strip outputs before committing)
data/               git-ignored local cache mirroring the GCS layout
docs/               ADRs, data notes, runbooks
```

Each top-level folder has a README describing what belongs there.

## Branching

| Branch | Purpose | Who pushes |
|---|---|---|
| `main` | stable, production-ready | merges from `dev` only (admins) |
| `dev` | integration branch, most active | merges from feature branches via PR |
| `feature/<name>` | one feature or fix, branched from `dev` | the teammate working on it |

Workflow: branch from `dev` → open a PR into `dev` → review + CI green → merge. `dev` is merged into `main` for releases.
