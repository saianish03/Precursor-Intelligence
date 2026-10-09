# Precursor Intelligence

An end-to-end MLOps application - Nursing Homes Risk Assessment for the Insurance Industry

## Repository layout

```
Precursor-Intelligence/
├── .github/                    # CI/CD workflows, PR and issue templates, CODEOWNERS
├── configs/                    # environment config, dataset registry, schema contracts, prompts
├── src/
│   └── precursorintelligence/  # shared Python package
├── backend/                    # FastAPI service
├── frontend/                   # Streamlit dashboard
├── dags/                       # Airflow DAGs
├── docker/                     # Dockerfiles and local Airflow compose
├── infra/terraform/            # GCP infrastructure as code
├── tests/                      # automated tests
├── notebooks/                  # exploration only
├── data/                       # local data cache (git-ignored)
└── docs/                       # ADRs, data notes, runbooks
```

| Folder | What goes here |
|---|---|
| `.github/` | CI/CD workflows, pull request and issue templates, CODEOWNERS |
| `configs/` | base / dev / prod config, dataset registry, versioned schema contracts, LLM prompts |
| `src/precursorintelligence/` | Shared Python package: `common`, `ingestion`, `validation`, `features`, `training`, `evaluation`, `serving`, `monitoring`, `agents` |
| `backend/` | FastAPI service, using the agents in `src/precursorintelligence/agents`; deployed to Cloud Run |
| `frontend/` | Streamlit dashboard; deployed to Cloud Run |
| `dags/` | Thin Airflow DAGs that only trigger jobs (no business logic) |
| `docker/` | One Dockerfile per image, plus the local Airflow compose file |
| `infra/terraform/` | GCP infrastructure: `envs/{dev,prod}` and reusable modules |
| `tests/` | Mirrors `src/`, plus `backend/`, `frontend/`, `contracts/` and `dags/`; tiny fixtures only |
| `notebooks/` | Exploration only; strip outputs before committing |
| `data/` | Git-ignored local cache that mirrors the GCS layout |
| `docs/` | Architecture decision records, data notes, runbooks |

Each top-level folder has its own README describing what belongs there.

## Branching

| Branch | Purpose | Who pushes |
|---|---|---|
| `main` | Stable, production-ready | Merges from `dev` only (admins) |
| `dev` | Integration branch, most active | Merges from feature branches via pull request |
| `feat/<name>` | One feature or fix, branched from `dev` | The teammate working on it |

**Workflow:**

1. Create `feature/<name>` always cloned from `dev`.
2. Open a pull request into `dev`.
3. Merge after review and green CI.
4. Merge `dev` into `main` for each release.
