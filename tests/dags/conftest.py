"""DAG tests need Airflow, which is deliberately not a repo dependency: run them in the precursor-airflow image.

    cd docker/airflow-local && docker compose run --rm --no-deps --entrypoint "" \
      airflow-scheduler python -m pytest -p no:cacheprovider /opt/airflow/tests/dags

`docker compose` in that folder also loads docker-compose.override.yaml, which mounts tests/dags (and the working
tree's DAGs over the ones baked into the image). CI runs them against the baked DAGs (job `images` in ci.yml).
"""

import os
import sys
from pathlib import Path

import pytest

pytest.importorskip("airflow")

DAGS_DIR = Path(os.environ.get("PI_DAGS_DIR", Path(__file__).resolve().parents[2] / "dags"))
if not DAGS_DIR.is_dir():  # in the airflow image: /opt/airflow/dags (baked; mounted on a laptop)
    DAGS_DIR = Path("/opt/airflow/dags")
sys.path.insert(0, str(DAGS_DIR))  # Airflow puts the DAGs folder on sys.path the same way


@pytest.fixture(scope="session")
def dags_dir() -> Path:
    return DAGS_DIR
