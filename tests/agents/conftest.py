from pathlib import Path

import pytest

from precursorintelligence.common.io.object_store import LocalObjectStore


@pytest.fixture
def store(tmp_path: Path) -> LocalObjectStore:
    return LocalObjectStore(tmp_path / "bucket", immutable=None)
