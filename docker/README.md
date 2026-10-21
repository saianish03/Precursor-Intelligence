# docker/

| File | Image / stack |
|---|---|
| `pipeline.Dockerfile` | `precursor-pipeline`: every data-pipeline task (`precursor-ingest`, `precursor-pipeline bronze|silver`) |
| `airflow.Dockerfile` | `precursor-airflow`: Airflow 3.1.8 + the pinned Docker provider (4.6.1) |
| `airflow-local/docker-compose.yaml` | Airflow (LocalExecutor) + Postgres (`airflow`, `precursor_meta`) + docker-socket-proxy. The same stack on a laptop and on the VM; only `.env` differs (`.env.example`) |
| `training.Dockerfile` (planned) | training / evaluation / scoring jobs |
| `backend.Dockerfile` (planned) | FastAPI service (`backend/`) |
| `frontend.Dockerfile` (planned) | dashboard (`frontend/`) |

Build and run: `docs/runbooks/airflow-local.md`.
