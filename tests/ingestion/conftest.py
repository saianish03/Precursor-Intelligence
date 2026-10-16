"""Fixtures for the ingestion tests (helpers live in cms_fakes.py)."""

import uuid
from pathlib import Path

import pytest
import requests
from cms_fakes import FakeCMS, make_settings

from precursorintelligence.ingestion.context import build_context
from precursorintelligence.ingestion.http import SafeHttpClient


@pytest.fixture(autouse=True)
def no_real_http(request, monkeypatch):
    """Unit tests are offline: any real HTTP request fails the test (live tests use -m network)."""
    if request.node.get_closest_marker("network"):
        return

    def blocked(self, req, **kwargs):
        raise AssertionError(f"unit test tried to reach the network: {req.method} {req.url}")

    monkeypatch.setattr(requests.Session, "send", blocked)


@pytest.fixture
def fake_cms() -> FakeCMS:
    return FakeCMS()


@pytest.fixture
def make_ctx(tmp_path):
    def _make(
        fake: FakeCMS, backend: str = "local", run_id: str | None = None, root: Path | None = None
    ):
        settings = make_settings(root or tmp_path, backend)
        s = settings.sources
        http = SafeHttpClient(s.source, s.http, s.security, session=fake)
        return build_context(settings, run_id=run_id or f"run-{uuid.uuid4().hex[:6]}", http=http)

    return _make
