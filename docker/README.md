# docker/

| File | Image / stack |
|---|---|
| `pipeline.Dockerfile` | `precursor-pipeline`: every data-pipeline task (`precursor-ingest`, `precursor-pipeline bronze|silver|gold`) |
| `airflow.Dockerfile` | `precursor-airflow`: Airflow 3.1.8 + the pinned Docker provider (4.6.1), the DAGs, the `precursorintelligence` package they import, and the compose bundle (`/opt/precursor/compose/`) the VM deploys from |
| `airflow-local/docker-compose.yaml` | Airflow (LocalExecutor) + Postgres (`airflow`, `precursor_meta`) + docker-socket-proxy. The same stack on a laptop and on the VM; a laptop sets its values in `.env` (`.env.example`) |
| `airflow-local/docker-compose.vm.yaml` | VM only, with `-f` (never auto-loaded): the production values, both images by sha, Postgres data on the data disk |
| `airflow-local/docker-compose.override.yaml` | Laptop only (auto-loaded in that folder): mounts the working tree's DAGs, `src/precursorintelligence` and `tests/dags` over the image's copies |
| `training.Dockerfile` (planned) | training / evaluation / scoring jobs |
| `backend.Dockerfile` (planned) | FastAPI service (`backend/`) |
| `frontend.Dockerfile` (planned) | dashboard (`frontend/`) |

Build and run: `docs/runbooks/airflow-local.md`.
