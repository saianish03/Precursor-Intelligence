import pytest
import requests
from cms_fakes import ARCH, BASE, FakeCMS, FakeResponse, make_settings, make_zip

from precursorintelligence.ingestion.http import DownloadValidationError, SafeHttpClient, UnsafeUrlError


@pytest.fixture
def client_factory(tmp_path):
    def make(session):
        s = make_settings(tmp_path).sources
        return SafeHttpClient(s.source, s.http, s.security, session=session)
    return make


@pytest.mark.parametrize("bad", [
    "https://evil.example.com/provider-data/sites/default/files/dataset-archives/x.zip",
    "//evil.example.com/provider-data/sites/default/files/dataset-archives/x.zip",
    f"{ARCH}/theme/../../../etc/passwd.zip",
    f"{ARCH}/theme/nursing-homes/a.zip?redirect=https://evil",
    f"{ARCH}/theme/nursing-homes/a.zip#frag",
    f"{ARCH}/theme/%2e%2e/a.zip",
    "/provider-data/api/1/datastore/query/abc.zip",
    "/etc/passwd",
    f"{ARCH}/theme/nursing-homes/a.exe",
    "",
])
def test_download_url_allow_list_rejects(client_factory, bad):
    with pytest.raises(UnsafeUrlError):
        client_factory(FakeCMS()).download_url(bad)


def test_download_url_allow_list_accepts(client_factory):
    url = client_factory(FakeCMS()).download_url(f"{ARCH}/theme/nursing-homes/nursing-homes_2021-01-27.zip")
    assert url == f"{BASE}{ARCH}/theme/nursing-homes/nursing-homes_2021-01-27.zip"


def test_cross_host_redirect_is_refused(client_factory):
    fake = FakeCMS()
    rel = fake.add("2021-01-27", make_zip("2021-01", "Jan2021"))
    fake.overrides[BASE + rel] = FakeResponse(302, b"", {"Location": "https://evil.example.com/x.zip"})
    client = client_factory(fake)
    with pytest.raises(UnsafeUrlError):
        client.head(client.download_url(rel))


def _download(client, rel, size):
    return b"".join(client.iter_download(client.download_url(rel), size))


def test_valid_download_roundtrip(client_factory):
    fake = FakeCMS()
    body = make_zip("2021-01", "Jan2021")
    rel = fake.add("2021-01-27", body)
    assert _download(client_factory(fake), rel, len(body)) == body


@pytest.mark.parametrize("headers,msg", [
    ({"Content-Type": "text/html"}, "Content-Type"),
    ({"Content-Type": "application/zip", "Content-Length": "5"}, "Content-Length"),
])
def test_bad_response_headers_are_rejected(client_factory, headers, msg):
    fake = FakeCMS()
    body = make_zip("2021-01", "Jan2021")
    rel = fake.add("2021-01-27", body)
    h = {"Content-Length": str(len(body)), **headers}
    fake.overrides[BASE + rel] = FakeResponse(200, body, h)
    with pytest.raises(DownloadValidationError, match=msg):
        _download(client_factory(fake), rel, len(body))


def test_non_zip_body_is_rejected(client_factory):
    fake = FakeCMS()
    body = b"<html>not a zip</html>" * 10
    rel = fake.add("2021-01-27", body)
    with pytest.raises(DownloadValidationError, match="not a zip"):
        _download(client_factory(fake), rel, len(body))


def test_size_cap_is_enforced(client_factory):
    fake = FakeCMS()
    client = client_factory(fake)
    client.security.max_download_bytes = 100
    with pytest.raises(DownloadValidationError, match="exceeds cap"):
        list(client.iter_download(f"{BASE}{ARCH}/theme/x.zip", 10_000))


class FlakySession(FakeCMS):
    """Drops the connection half-way through the first GET; the client must resume with Range."""

    def __init__(self):
        super().__init__()
        self.failed = False

    def request(self, method, url, headers=None, **kw):
        resp = super().request(method, url, headers=headers, **kw)
        if method == "GET" and not (headers or {}).get("Range") and resp.status_code == 200 and not self.failed \
                and url.endswith(".zip"):
            self.failed = True
            body = resp._body
            half = len(body) // 2

            class Broken(FakeResponse):
                def iter_content(self, chunk_size=1024):
                    yield body[:half]
                    raise requests.ConnectionError("connection reset")
            return Broken(200, body, dict(resp.headers))
        return resp


def test_interrupted_download_resumes_with_range(client_factory):
    fake = FlakySession()
    body = make_zip("2021-01", "Jan2021")
    rel = fake.add("2021-01-27", body)
    assert _download(client_factory(fake), rel, len(body)) == body
    assert any(h.get("Range") for m, u, h in fake.requests if m == "GET"), "expected a Range resume request"


def test_retry_after_on_503_then_success(client_factory):
    fake = FakeCMS()
    body = make_zip("2021-01", "Jan2021")
    rel = fake.add("2021-01-27", body)
    calls = {"n": 0}
    original = fake.request

    def flaky(method, url, **kw):
        if method == "HEAD" and calls["n"] == 0:
            calls["n"] += 1
            return FakeResponse(503, b"", {"Retry-After": "0"})
        return original(method, url, **kw)
    fake.request = flaky
    client = client_factory(fake)
    assert client.head(client.download_url(rel)).size == len(body)
