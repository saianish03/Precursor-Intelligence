# docker/

| File | Image / stack |
|---|---|
| `pipeline.Dockerfile` | `precursor-pipeline`: every data-pipeline and ML task (`precursor-ingest`, `precursor-pipeline bronze|silver|gold|ml`); extras `pipeline`, `ml`, `bias` |
| `airflow.Dockerfile` | `precursor-airflow`: Airflow 3.1.8 + the pinned Docker provider (4.6.1), the DAGs, the `precursorintelligence` package they import, and the compose bundle (`/opt/precursor/compose/`) the VM deploys from |
| `mlflow.Dockerfile` | `precursor-mlflow`: the MLflow tracking server and registry (extra `mlflow-server`: mlflow, the GCS client, the Postgres driver); ML spec §16.1 |
| `airflow-local/docker-compose.yaml` | Airflow (LocalExecutor) + Postgres (`airflow`, `precursor_meta`, `mlflow`) + docker-socket-proxy + MLflow (task network only; off on the VM until Phase 2). The same stack on a laptop and on the VM; a laptop sets its values in `.env` (`.env.example`) |
| `airflow-local/docker-compose.vm.yaml` | VM only, with `-f` (never auto-loaded): the production values, both images by sha, Postgres data on the data disk |
| `airflow-local/docker-compose.override.yaml` | Laptop only (auto-loaded in that folder): mounts the working tree's DAGs, `src/precursorintelligence` and `tests/dags` over the image's copies |
| `backend.Dockerfile` (planned) | FastAPI service (`backend/`) |
| `frontend.Dockerfile` (planned) | dashboard (`frontend/`) |

Build and run: `docs/runbooks/airflow-local.md`.
