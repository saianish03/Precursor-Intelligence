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

1. Create `feat/<name>` from the latest `dev`.
2. Open a pull request into `dev`.
3. Merge after review and green CI.
4. Merge `dev` into `main` for each release.

## Rules before making a PR to `dev`

### 1. Branch
- **Branch from the latest `dev`:** `git checkout dev && git pull && git checkout -b <type>/<short-name>`.
- **Prefixes:** `feat/` (new feature), `fix/` (bug fix), `docs/`, `test/`, `refactor/`, `ci/`, `chore/`.
- **Names:** short, lowercase, hyphenated or underscored, e.g. `feat/silver-column-map`, `fix/penalty-dup-seq`, `docs/data_pipeline_steps.md`, etc.
- **One focused change per branch and PR.** Aim for under ~400 changed lines. Split bigger work into several PRs.

### 2. Before you push, check locally (CI runs the same commands)
```bash
uv venv && uv pip install -e ".[dev]"     # or: pip install -e ".[dev]"
ruff check src tests                      # lint must pass
pytest                                    # offline tests must pass (live CMS tests are skipped)
```
- **Bring in the latest `dev`** (`git pull origin dev`, or rebase) and resolve conflicts locally before opening the PR.

### 3. Code and tests
- **New or changed code comes with tests** in `tests/<stage>/`, mirroring `src/precursorintelligence/<stage>/`.
- **Unit tests must not call the network or cloud services.** Mark live tests with `@pytest.mark.network`.
- **Add only required code for the feature, or only tiny fixes** (a few rows). Never put real downloaded data in the repo.
- **No hard-coded paths, buckets or credentials.** Read them from `configs/` and environment variables.
- **New dependencies go into the right group in `pyproject.toml`** (`gcp`, `training`, `backend`, …). Say why in the PR.

### 4. Never commit
- data files (`data/`, CSV / Parquet / zip downloads), logs, model artefacts;
- secrets: `.env`, service-account keys, passwords, tokens;
- notebook outputs (clear them before committing).

### 5. Commits and PR description
- **Commit messages follow Conventional Commits:** `feat(ingestion): …`, `fix(features): …`, `docs: …`, `test: …`, `ci: …`.
- **The PR description says:**
  - **what** changed and **why**;
  - **how it was tested;**
  - **config or schema changes** (`configs/`, column maps, data contracts) and any **breaking changes**;
  - the linked issue, if there is one.
- **Docs:** update the relevant README or `docs/` page when behaviour, configuration or commands change.

### 6. Review and merge
- **CI must be green.** The `ci-ok` check (ruff + tests on Python 3.11 and 3.12) is required.
- **At least 1 approval from a teammate.** Don't merge your own PR without one.
- **Resolve every review conversation** before merging.
- **Merge with "Squash and merge"** so `dev` keeps one commit per PR, and delete the branch after merging.
- **Only admins push to `dev` directly,** and only for repository setup or emergencies.
