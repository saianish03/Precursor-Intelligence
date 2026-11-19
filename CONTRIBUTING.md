# Contributing to Precursor Intelligence

How the team works on this repository: branches, the checks to run before a pull request, and how a PR is reviewed
and merged. The project itself is described in the [README](README.md).

---

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

---

## Rules before making a PR to `dev`

### 1. Branch
- **Branch from the latest `dev`:** `git checkout dev && git pull && git checkout -b <type>/<short-name>`.
- **Prefixes:** `feat/` (new feature), `fix/` (bug fix), `docs/`, `test/`, `refactor/`, `ci/`, `chore/`.
- **Names:** short, lowercase, hyphenated or underscored, e.g. `feat/silver-column-map`, `fix/penalty-dup-seq`, `docs/data_pipeline_steps.md`, etc.
- **One focused change per branch and PR.** Aim for under ~400 changed lines. Split bigger work into several PRs.

### 2. Before you push, check locally (CI runs the same commands)
```bash
uv sync --all-extras                      # exact versions from uv.lock, plus dev tools (uv only, never pip)
uv run ruff check src tests               # lint must pass
uv run pytest                             # offline tests must pass (live CMS tests are skipped)
```
- **macOS only:** LightGBM needs OpenMP: `brew install libomp` once per laptop.
- **Bring in the latest `dev`** (`git pull origin dev`, or rebase) and resolve conflicts locally before opening the PR.
- **Changed `pyproject.toml` or `uv.lock`?** Re-export `requirements.txt` (CI fails when it is stale):
  `uv export --frozen --no-hashes --no-header --extra pipeline --extra ml --extra bias --extra dvc -o requirements.txt`.
- **Changed anything listed under `deps:` in `dvc.yaml`** (the gold, harmonize, common and ingestion code, the ML code under training, evaluation, bias and serving, the configs they load, or `uv.lock`)? Run `uv run dvc repro` (about a minute for `toy_ml`), commit `dvc.lock`, `metrics/toy_gold.json` and `metrics/toy_ml.json` in the same PR, and run `uv run dvc push` (a new output must be in the remote, or teammates' `dvc pull` fails).

### 3. Code and tests
- **New or changed code comes with tests** in `tests/<stage>/`, mirroring `src/precursorintelligence/<stage>/`.
- **Unit tests must not call the network or cloud services.** Mark live tests with `@pytest.mark.network`.
- **Add only required code for the feature, or only tiny fixes** (a few rows). Never put real downloaded data in the repo.
- **No hard-coded paths, buckets or credentials.** Read them from `configs/` and environment variables.
- **New dependencies go into the right extra with `uv add --optional <extra> <package>`** (`pipeline`, `ml`, `ml-extra`, `agents`, `api`, `ui`, `mlflow-server`, `dvc`, `bias`; tools: `uv add --group dev <package>`). Never type versions by hand; commit the updated `uv.lock`. Say why in the PR.

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
- **CI must be green.** The `ci-ok` check (ruff + tests on Python 3.11 and 3.12) is required. When a PR touches `docker/`, `dags/`, `tests/dags/`, `src/`, `configs/`, `.dockerignore`, `pyproject.toml`, `uv.lock` or `.github/workflows/ci.yml`, it also covers building the three images and the DAG tests inside the Airflow image (job `images`).
- **At least 1 approval from a teammate.** Don't merge your own PR without one.
- **Resolve every review conversation** before merging.
- **Merge with "Squash and merge"** so `dev` keeps one commit per PR, and delete the branch after merging.
- **Only admins push to `dev` directly,** and only for repository setup or emergencies.
