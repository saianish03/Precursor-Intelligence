"""Task ``download`` (fan-out: one call per catalog entry).

Flow for one zip (docs §6.3):
  1. validate the catalog URL against the allow-list            -> UnsafeUrlError => status 'rejected'
  2. HEAD: size / Last-Modified / Content-Type
  3. skip if the landing object exists and size + Last-Modified match the last successful log row
  4. stream GET (resumable) into  quarantine/cms_nh/<run_id>/<zip_name>, hashing while writing
  5. validate the archive (central directory limits + CRC of every member)  -> fail => 'quarantined'
  6. promote: move to landing/cms_nh/zips/zip_date=<date>/<zip_name> and make it immutable
     * if a different sha256 already lives there, store <stem>__<sha8>.zip instead ('republished')
  7. append one row to download_log (every outcome, including skips and failures)
Workers never write to the MetadataStore; they return a result row that the caller appends, so the
fan-out is safe with any backend and maps directly onto mapped DAG tasks.
"""

from __future__ import annotations

import logging
import posixpath
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from dataclasses import dataclass, field

from ..common.context import Context, utc_now
from ..common.io.object_store import sha256_of
from ..common.obs.logging import log_context
from .http import DownloadValidationError, UnsafeUrlError
from .sources.base import CatalogEntry
from .zipsafe import UnsafeArchiveError, inspect_archive, verify_crc

log = logging.getLogger(__name__)

ZIP_PREFIX = "landing/cms_nh/zips"
QUARANTINE_PREFIX = "quarantine/cms_nh"
SUCCESS = {"downloaded", "skipped", "unchanged", "republished", "adopted"}


def landing_key(entry: CatalogEntry, zip_name: str | None = None) -> str:
    return f"{ZIP_PREFIX}/zip_date={entry.zip_date.isoformat()}/{zip_name or entry.zip_name}"


@dataclass
class DownloadResult:
    entry: CatalogEntry
    status: str
    object_key: str | None = None
    sha256: str | None = None
    actual_size: int | None = None
    expected_size: int | None = None
    http_last_modified: str | None = None
    http_content_type: str | None = None
    detail: str = ""
    url: str | None = None
    extra: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status in SUCCESS

    def log_row(self, ctx: Context) -> dict:
        e = self.entry
        return {
            "run_id": ctx.run_id, "entry_id": e.entry_id, "zip_name": e.zip_name, "zip_date": e.zip_date.isoformat(),
            "type": e.type, "url": self.url, "catalog_size": e.size_bytes, "expected_size": self.expected_size,
            "actual_size": self.actual_size, "sha256": self.sha256, "http_last_modified": self.http_last_modified,
            "http_content_type": self.http_content_type, "object_key": self.object_key,
            "object_uri": ctx.object_store.uri_for(self.object_key) if self.object_key else None,
            "status": self.status, "detail": self.detail, "logged_at": utc_now(),
        }


def last_good_download(ctx: Context, entry_id: str) -> dict | None:
    rows = [r for r in ctx.metadata.read("download_log", {"entry_id": entry_id})
            if r["status"] in SUCCESS and r.get("sha256")]
    return max(rows, key=lambda r: r["id"]) if rows else None


def _validate_archive(ctx: Context, key: str) -> None:
    limits = ctx.cfg.security.zip
    with ctx.object_store.open_read(key) as f:
        inspect_archive(f, limits)
    with ctx.object_store.open_read(key) as f:
        verify_crc(f)


def download_entry(ctx: Context, entry: CatalogEntry, prior: dict | None) -> DownloadResult:
    store = ctx.object_store
    res = DownloadResult(entry=entry, status="failed")
    try:
        res.url = url = ctx.http.download_url(entry.url)
    except UnsafeUrlError as exc:
        log.error("rejected catalog URL for %s: %s", entry.entry_id, exc)
        res.status, res.detail = "rejected", str(exc)
        return res

    final = landing_key(entry)
    try:
        head = ctx.http.head(url)
        res.expected_size, res.http_last_modified, res.http_content_type = head.size, head.last_modified, head.content_type
        if head.status != 200 or head.size is None:
            raise DownloadValidationError(f"HEAD returned HTTP {head.status} / size {head.size}")
        if head.size != entry.size_bytes:
            log.warning("HEAD size %d differs from catalog size %d for %s; trusting the server",
                        head.size, entry.size_bytes, entry.zip_name)

        if store.exists(final) and prior and prior["actual_size"] == head.size \
                and prior["http_last_modified"] == head.last_modified:
            res.status, res.object_key, res.sha256 = "skipped", prior["object_key"], prior["sha256"]
            res.actual_size, res.detail = prior["actual_size"], "unchanged since last download (size + Last-Modified)"
            log.info("skip %s: already stored and unchanged", entry.zip_name)
            return res

        if store.exists(final) and prior is None:
            # A zip placed manually (e.g. copied from a teammate) is adopted only after full validation.
            _validate_archive(ctx, final)
            res.status, res.object_key, res.sha256 = "adopted", final, sha256_of(store, final)
            res.actual_size = store.stat(final).size
            res.detail = "pre-existing object validated and adopted"
            log.info("adopted pre-existing %s (sha256 %s)", entry.zip_name, res.sha256[:12])
            return res

        qkey = f"{QUARANTINE_PREFIX}/{ctx.run_id}/{entry.zip_date.isoformat()}/{entry.zip_name}"
        log.info("downloading %s (%.1f MB)", entry.zip_name, head.size / 1e6)
        put = store.put_stream(qkey, ctx.http.iter_download(url, head.size), overwrite=True)
        res.actual_size, res.sha256 = put.size, put.sha256
        try:
            _validate_archive(ctx, qkey)
        except UnsafeArchiveError as exc:
            res.status, res.detail, res.object_key = "quarantined", f"archive validation failed: {exc}", qkey
            log.error("quarantined %s: %s", entry.zip_name, exc)
            return res

        if store.exists(final):
            old_sha = (prior or {}).get("sha256") or sha256_of(store, final)
            if old_sha == put.sha256:
                store.remove(qkey)
                res.status, res.object_key = "unchanged", final
                res.detail = "re-downloaded after header change; content identical"
                log.info("%s re-checked: identical content (sha256 %s)", entry.zip_name, put.sha256[:12])
                return res
            stem, ext = posixpath.splitext(entry.zip_name)
            alt = landing_key(entry, f"{stem}__{put.sha256[:8]}{ext}")
            if store.exists(alt):
                store.remove(qkey)
                res.status, res.object_key, res.detail = "unchanged", alt, "republished version already stored"
                return res
            store.move(qkey, alt)
            store.make_immutable(alt)
            res.status, res.object_key = "republished", alt
            res.detail = f"CMS re-published this zip: sha256 {old_sha[:12]} -> {put.sha256[:12]}; original kept"
            log.warning("%s was re-published by CMS (sha256 %s -> %s); stored side by side as %s",
                        entry.zip_name, old_sha[:12], put.sha256[:12], posixpath.basename(alt))
            return res

        store.move(qkey, final)
        store.make_immutable(final)
        res.status, res.object_key = "downloaded", final
        log.info("stored %s (%d bytes, sha256 %s)", entry.zip_name, put.size, put.sha256[:12])
        return res
    except (DownloadValidationError, UnsafeUrlError) as exc:
        res.status, res.detail = "quarantined" if res.actual_size else "failed", str(exc)
        log.error("download of %s failed validation: %s", entry.zip_name, exc)
        return res
    except Exception as exc:  # network errors after retries, storage errors
        res.status, res.detail = "failed", f"{exc.__class__.__name__}: {exc}"
        log.exception("download of %s failed", entry.zip_name)
        return res


def run_download(ctx: Context, entries: list[CatalogEntry]) -> list[DownloadResult]:
    priors = {e.entry_id: last_good_download(ctx, e.entry_id) for e in entries}

    def work(e: CatalogEntry) -> DownloadResult:
        with log_context(entry_id=e.entry_id, zip_name=e.zip_name):
            return download_entry(ctx, e, priors[e.entry_id])

    results: list[DownloadResult] = []
    with ThreadPoolExecutor(max_workers=ctx.cfg.http.max_parallel_downloads) as pool:
        futures = [pool.submit(copy_context().run, work, e) for e in entries]
        for fut in futures:
            r = fut.result()
            ctx.metadata.append("download_log", [r.log_row(ctx)])
            results.append(r)
    ctx.object_store.prune_empty(QUARANTINE_PREFIX)  # removes empty folders only; quarantined files stay
    counts: dict[str, int] = {}
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1
    log.info("download: %s", ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    return results
