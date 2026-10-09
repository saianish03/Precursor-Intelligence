"""Fixtures for the ingestion tests (helpers live in cms_fakes.py)."""

import uuid
from pathlib import Path

import pytest

from cms_fakes import FakeCMS, make_settings
from precursorintelligence.common.context import build_context
from precursorintelligence.ingestion.http import SafeHttpClient


@pytest.fixture
def fake_cms() -> FakeCMS:
    return FakeCMS()


@pytest.fixture
def make_ctx(tmp_path):
    def _make(fake: FakeCMS, backend: str = "local", run_id: str | None = None, root: Path | None = None):
        settings = make_settings(root or tmp_path, backend)
        s = settings.sources
        http = SafeHttpClient(s.source, s.http, s.security, session=fake)
        return build_context(settings, run_id=run_id or f"run-{uuid.uuid4().hex[:6]}", http=http)
    return _make
