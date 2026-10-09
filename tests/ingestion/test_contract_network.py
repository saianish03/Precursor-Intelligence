"""Live contract tests against data.cms.gov. Run explicitly:  pytest -m network

They guard the undocumented archive endpoints: if CMS changes their shape, these fail before a real run does.
"""

import pytest

from precursorintelligence.common.config import load_settings
from precursorintelligence.ingestion.http import SafeHttpClient
from precursorintelligence.ingestion.sources.cms_pdc import REQUIRED_KEYS

pytestmark = pytest.mark.network


@pytest.fixture(scope="module")
def client():
    s = load_settings("dev").sources
    return SafeHttpClient(s.source, s.http, s.security), s


def test_archive_endpoint_shape(client):
    http, s = client
    data = http.get_json(http.api_url(s.source.endpoints.archive.format(theme=s.source.theme_slug)))["data"]
    monthly = [d for d in data if d["type"] == "theme"]
    assert len(monthly) >= 89
    assert all(REQUIRED_KEYS <= d.keys() for d in data)
    for d in monthly:
        http.download_url(d["url"])  # every advertised URL passes the allow-list


def test_current_endpoint_and_range_support(client):
    http, s = client
    cur = [d for d in http.get_json(http.api_url(s.source.endpoints.current))["data"]
           if d.get("theme") == s.source.theme_slug]
    assert len(cur) == 1
    head = http.head(http.download_url(cur[0]["url"]))
    assert head.status == 200 and head.accept_ranges and head.content_type == "application/zip"
