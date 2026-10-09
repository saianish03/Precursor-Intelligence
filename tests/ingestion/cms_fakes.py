"""Test helpers: settings for any backend, a fake CMS server and synthetic archive zips.

The fake server is plugged in at the ``requests.Session`` level, so the real SafeHttpClient (allow-list,
redirect checks, Range resume, response validation) is exercised in every offline test.
"""

from __future__ import annotations

import io
import json
import uuid
import zipfile
from email.utils import formatdate
from pathlib import Path

from requests.structures import CaseInsensitiveDict

from precursorintelligence.common.config import MODULE_ROOT, EnvConfig, Settings, SourcesConfig, _read_yaml

BASE = "https://data.cms.gov"
API = BASE + "/provider-data/api/1"
ARCH = "/provider-data/sites/default/files/dataset-archives"

TABLE_FILES = {  # base names per table (L2 style, month token filled in)
    "provider_info": "NH_ProviderInfo_{tok}.csv",
    "health_citations": "NH_HealthCitations_{tok}.csv",
    "survey_dates": "NH_SurveyDates_{tok}.csv",
    "penalties": "NH_Penalties_{tok}.csv",
    "mds_qm": "NH_QualityMsr_MDS_{tok}.csv",
    "claims_qm": "NH_QualityMsr_Claims_{tok}.csv",
    "citation_lookup": "NH_CitationDescriptions_{tok}.csv",
}
DATASET_IDS = {"provider_info": "4pq5-n9py", "health_citations": "r5ix-sfxw", "survey_dates": "svdt-c123",
               "penalties": "g6vv-u9sr", "mds_qm": "djen-97ju", "claims_qm": "ijh5-nb2v",
               "citation_lookup": "tagd-9999"}


def csv_bytes(month: str, extra_rows: int = 3, survey_date: str = "2019-05-02") -> bytes:
    lines = ["CMS Certification Number (CCN),Survey Date,Processing Date"]
    for i in range(extra_rows):
        lines.append(f"{i:06d},{survey_date},{month}-01")
    return ("\n".join(lines) + "\n").encode()


def make_zip(month: str, tok: str, *, layout: str = "L2", tables=None, junk: bool = False,
             subfolder: str | None = None, extra: dict[str, bytes] | None = None) -> bytes:
    tables = tables or list(TABLE_FILES)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        manifest = []
        for t in tables:
            name = TABLE_FILES[t].format(tok=tok)
            if layout == "L3":
                name = f"{DATASET_IDS[t]}_{month}-01_{name}"
            path = f"{subfolder}/{name}" if subfolder else name
            zf.writestr(path, csv_bytes(month, survey_date="2018-03-04" if t == "health_citations" else "2019-05-02"))
            manifest.append({"dataset_id": DATASET_IDS[t], "resources": [{"filename": TABLE_FILES[t].format(tok=tok)}]})
        zf.writestr("NH_Data_Dictionary.pdf", b"%PDF-1.4 fake")
        if layout == "L3":
            zf.writestr("manifest.json", json.dumps(manifest))
        if junk:
            zf.writestr("__MACOSX/._NH_ProviderInfo.csv", b"\x00" * 64)
            zf.writestr(".DS_Store", b"\x00" * 32)
        for k, v in (extra or {}).items():
            zf.writestr(k, v)
    return buf.getvalue()


class FakeResponse:
    def __init__(self, status: int, body: bytes = b"", headers: dict | None = None) -> None:
        self.status_code, self._body = status, body
        self.headers = CaseInsensitiveDict(headers or {})

    @property
    def content(self) -> bytes:
        return self._body

    def json(self):
        return json.loads(self._body)

    def iter_content(self, chunk_size: int = 1024):
        for i in range(0, len(self._body), chunk_size):
            yield self._body[i:i + chunk_size]

    def close(self) -> None:
        pass


class FakeCMS:
    """Serves the archive/current/metastore endpoints and zip files, with Range support."""

    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.last_modified: dict[str, str] = {}
        self.entries: list[dict] = []
        self.current: list[dict] = []
        self.requests: list[tuple[str, str, dict]] = []
        self.overrides: dict[str, FakeResponse] = {}

    def add(self, date: str, body: bytes, *, kind: str = "theme", path: str | None = None,
            catalog_size: int | None = None) -> str:
        name = f"nursing-homes_{date}.zip" if kind != "current" else "theme_nursing-homes_current.zip"
        rel = path or (f"{ARCH}/current/theme/{name}" if kind == "current" else f"{ARCH}/theme/nursing-homes/{name}")
        self.files[rel] = body
        self.last_modified[rel] = formatdate(1_700_000_000 + len(self.files) * 86400, usegmt=True)
        item = {"name": f"Theme: nursing-homes ({date})", "nid": str(len(self.files)), "identifier": None,
                "type": kind, "theme": "nursing-homes", "url": rel, "size": str(catalog_size or len(body)),
                "date": date, "access_level": "public"}
        (self.current if kind == "current" else self.entries).append(item)
        return rel

    def replace(self, rel: str, body: bytes) -> None:
        self.files[rel] = body
        self.last_modified[rel] = formatdate(1_800_000_000, usegmt=True)
        for item in self.entries + self.current:
            if item["url"] == rel:
                item["size"] = str(len(body))

    # requests.Session-compatible
    def request(self, method, url, headers=None, stream=False, allow_redirects=True, timeout=None, verify=None):
        headers = headers or {}
        self.requests.append((method, url, dict(headers)))
        if url in self.overrides:
            return self.overrides[url]
        if url.startswith(API + "/archive/aggregate/theme/"):
            return self._json({"data": self.entries})
        if url.startswith(API + "/archive/aggregate/current/"):
            return self._json({"data": self.current})
        if url.startswith(API + "/metastore/schemas/dataset/items/"):
            ds = url.rsplit("/", 1)[1].split("?")[0]
            return self._json({"identifier": ds, "title": ds, "modified": "2026-08-01", "released": "2026-08-26",
                               "nextUpdateDate": "2026-09-30",
                               "distribution": [{"data": {"downloadURL": f"{BASE}/x/{ds}.csv"}}]})
        rel = url[len(BASE):]
        if rel not in self.files:
            return FakeResponse(404, b"not found", {"Content-Type": "text/html"})
        body = self.files[rel]
        base_headers = {"Content-Type": "application/zip", "Accept-Ranges": "bytes",
                        "Last-Modified": self.last_modified[rel]}
        if method == "HEAD":
            return FakeResponse(200, b"", {**base_headers, "Content-Length": str(len(body))})
        rng = headers.get("Range")
        if rng:
            start = int(rng.split("=")[1].rstrip("-"))
            part = body[start:]
            return FakeResponse(206, part, {**base_headers, "Content-Length": str(len(part)),
                                            "Content-Range": f"bytes {start}-{len(body) - 1}/{len(body)}"})
        return FakeResponse(200, body, {**base_headers, "Content-Length": str(len(body))})

    @staticmethod
    def _json(obj) -> FakeResponse:
        return FakeResponse(200, json.dumps(obj).encode(), {"Content-Type": "application/json"})


def make_settings(tmp_path: Path, backend: str = "local") -> Settings:
    sources = SourcesConfig.model_validate(_read_yaml(MODULE_ROOT / "configs" / "ingestion" / "sources.yaml"))
    sources.http.backoff_initial_s = 0.01
    sources.http.backoff_max_s = 0.02
    if backend == "local":
        env = {"env": "test-local", "object_store": {"kind": "local", "root": str(tmp_path / "store")},
               "metadata_store": {"url": f"sqlite:///{tmp_path / 'meta.db'}"}}
    elif backend == "memory":
        env = {"env": "test-memory", "object_store": {"kind": "memory", "root": f"memory://sh-{uuid.uuid4().hex}"},
               "metadata_store": {"url": "sqlite:///:memory:"}}
    else:
        raise ValueError(backend)
    return Settings(sources=sources, env=EnvConfig.model_validate(env), module_root=MODULE_ROOT)


