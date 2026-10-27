"""DAG tests need Airflow, which is deliberately not a repo dependency: run them in the precursor-airflow image.

    docker compose -f docker/airflow-local/docker-compose.yaml run --rm --no-deps --entrypoint "" \
      airflow-scheduler python -m pytest -p no:cacheprovider /opt/airflow/tests/dags
"""

import os
import sys
from pathlib import Path

import pytest

pytest.importorskip("airflow")

DAGS_DIR = Path(os.environ.get("PI_DAGS_DIR", Path(__file__).resolve().parents[2] / "dags"))
if not DAGS_DIR.is_dir():  # inside the airflow image the DAGs are mounted at /opt/airflow/dags
    DAGS_DIR = Path("/opt/airflow/dags")
sys.path.insert(0, str(DAGS_DIR))  # Airflow puts the DAGs folder on sys.path the same way


@pytest.fixture(scope="session")
def dags_dir() -> Path:
    return DAGS_DIR
