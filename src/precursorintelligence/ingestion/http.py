"""Hardened HTTP client for the CMS Provider Data Catalog.

Security properties (docs §9):
  * HTTPS only, TLS verification always on (certifi bundle);
  * download URLs are built only from catalog paths that match an anchored allow-list regex and are
    joined to the configured base URL (no absolute URLs, no '..', no query strings, no other hosts);
  * redirects are followed manually and only to the same https host;
  * connect/read timeouts on every request; bounded retries with exponential backoff + jitter that
    honour ``Retry-After`` on 429/503;
  * download responses are validated (status, Content-Type, Content-Length, zip magic bytes) and capped
    at ``max_download_bytes``; interrupted downloads resume with HTTP Range requests.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Iterator
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from typing import Any, Protocol
from urllib.parse import urljoin, urlsplit

import certifi
import requests
from tenacity import RetryCallState, Retrying, retry_if_exception, stop_after_attempt, wait_random_exponential

from ..common.config import HttpCfg, SecurityCfg, SourceCfg

log = logging.getLogger(__name__)

ZIP_MAGIC = b"PK\x03\x04"
RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class UnsafeUrlError(ValueError):
    """A URL failed the allow-list / host checks."""


class DownloadValidationError(RuntimeError):
    """A response failed an integrity check; the bytes must not be trusted."""


class _TruncatedBody(RuntimeError):
    """The server closed the body before Content-Length bytes arrived (resumable)."""


class RetryableHTTPError(RuntimeError):
    def __init__(self, status: int, retry_after: float | None) -> None:
        super().__init__(f"HTTP {status}")
        self.status, self.retry_after = status, retry_after


@dataclass(frozen=True)
class HeadInfo:
    status: int
    size: int | None
    content_type: str | None
    last_modified: str | None
    etag: str | None
    accept_ranges: bool


class SessionLike(Protocol):
    def request(self, method: str, url: str, **kwargs: Any) -> Any: ...


def _retry_after_seconds(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            return max(0.0, parsedate_to_datetime(value).timestamp() - time.time())
        except (TypeError, ValueError):
            return None


def _is_retryable(exc: BaseException) -> bool:
    return isinstance(exc, (RetryableHTTPError, requests.ConnectionError, requests.Timeout,
                            requests.exceptions.ChunkedEncodingError))


class SafeHttpClient:
    def __init__(self, source: SourceCfg, http: HttpCfg, security: SecurityCfg,
                 session: SessionLike | None = None) -> None:
        if not security.verify_tls:  # defence in depth; config validation already forbids this
            raise ValueError("TLS verification cannot be disabled")
        self.source, self.http, self.security = source, http, security
        self.base = source.base_url
        self.base_host = urlsplit(self.base).hostname
        self._allowed_path = re.compile(source.allowed_download_path_regex)
        self.session = session or requests.Session()
        if isinstance(self.session, requests.Session):
            self.session.headers.update({"User-Agent": http.user_agent, "Accept-Encoding": "identity"})
            self.session.max_redirects = http.max_redirects

    # ------------------------------------------------------------------ URL safety
    def api_url(self, path: str) -> str:
        """URL for a configured (trusted) API path."""
        return f"{self.base}{self.source.api_prefix}{path}"

    def download_url(self, rel_path: str) -> str:
        """Validate a catalog-supplied relative path and turn it into an absolute download URL."""
        if not isinstance(rel_path, str) or not rel_path:
            raise UnsafeUrlError("empty download path")
        parts = urlsplit(rel_path)
        if parts.scheme or parts.netloc or rel_path.startswith("//"):
            raise UnsafeUrlError(f"absolute URL not allowed: {rel_path!r}")
        if parts.query or parts.fragment or "?" in rel_path or "#" in rel_path:
            raise UnsafeUrlError(f"query/fragment not allowed: {rel_path!r}")
        if any(seg in ("..", ".") for seg in rel_path.split("/")) or "\\" in rel_path or "%" in rel_path:
            raise UnsafeUrlError(f"path traversal / encoded characters not allowed: {rel_path!r}")
        if not self._allowed_path.fullmatch(rel_path):
            raise UnsafeUrlError(f"path not in allow-list: {rel_path!r}")
        url = urljoin(self.base + "/", rel_path.lstrip("/"))
        self._check_same_host(url)
        return url

    def _check_same_host(self, url: str) -> None:
        p = urlsplit(url)
        if p.scheme != "https" or p.hostname != self.base_host:
            raise UnsafeUrlError(f"refusing non-https or foreign-host URL: {url}")

    # ------------------------------------------------------------------ core request with retries
    def _wait(self, state: RetryCallState) -> float:
        base = wait_random_exponential(multiplier=self.http.backoff_initial_s, max=self.http.backoff_max_s)(state)
        exc = state.outcome.exception() if state.outcome else None
        if isinstance(exc, RetryableHTTPError) and exc.retry_after is not None:
            return max(base, min(exc.retry_after, self.http.backoff_max_s))
        return base

    def _before_sleep(self, state: RetryCallState) -> None:
        exc = state.outcome.exception() if state.outcome else None
        log.warning("retrying HTTP request (attempt %d/%d) after %s; sleeping %.1fs",
                    state.attempt_number, self.http.max_retries, exc, state.next_action.sleep if state.next_action else 0)

    def _send_once(self, method: str, url: str, headers: dict[str, str] | None, stream: bool) -> Any:
        self._check_same_host(url)
        for _ in range(self.http.max_redirects + 1):
            resp = self.session.request(
                method, url, headers=headers or {}, stream=stream, allow_redirects=False,
                timeout=(self.http.connect_timeout_s, self.http.read_timeout_s), verify=certifi.where(),
            )
            if resp.status_code in (301, 302, 303, 307, 308):
                location = resp.headers.get("Location", "")
                resp.close()
                url = urljoin(url, location)
                self._check_same_host(url)  # raises UnsafeUrlError for cross-host redirects
                log.debug("following same-host redirect to %s", url)
                continue
            if resp.status_code in RETRYABLE_STATUS:
                retry_after = _retry_after_seconds(resp.headers.get("Retry-After"))
                resp.close()
                raise RetryableHTTPError(resp.status_code, retry_after)
            return resp
        raise UnsafeUrlError(f"too many redirects for {url}")

    def request(self, method: str, url: str, *, headers: dict[str, str] | None = None, stream: bool = False) -> Any:
        retrying = Retrying(
            stop=stop_after_attempt(self.http.max_retries), wait=self._wait,
            retry=retry_if_exception(_is_retryable), before_sleep=self._before_sleep, reraise=True,
        )
        return retrying(self._send_once, method, url, headers, stream)

    # ------------------------------------------------------------------ public helpers
    def get_json(self, url: str) -> Any:
        resp = self.request("GET", url)
        try:
            if resp.status_code != 200:
                raise DownloadValidationError(f"GET {url} returned HTTP {resp.status_code}")
            ctype = (resp.headers.get("Content-Type") or "").split(";")[0].strip().lower()
            if ctype not in ("application/json", "application/ld+json"):
                raise DownloadValidationError(f"GET {url} returned non-JSON Content-Type {ctype!r}")
            return resp.json()
        finally:
            resp.close()

    def get_bytes(self, url: str) -> bytes:
        resp = self.request("GET", url)
        try:
            if resp.status_code != 200:
                raise DownloadValidationError(f"GET {url} returned HTTP {resp.status_code}")
            return resp.content
        finally:
            resp.close()

    def head(self, url: str) -> HeadInfo:
        resp = self.request("HEAD", url)
        try:
            h = resp.headers
            size = h.get("Content-Length")
            return HeadInfo(
                status=resp.status_code,
                size=int(size) if size and size.isdigit() else None,
                content_type=(h.get("Content-Type") or "").split(";")[0].strip().lower() or None,
                last_modified=h.get("Last-Modified"),
                etag=h.get("ETag"),
                accept_ranges="bytes" in (h.get("Accept-Ranges") or "").lower(),
            )
        finally:
            resp.close()

    def iter_download(self, url: str, expected_size: int) -> Iterator[bytes]:
        """Yield the body of ``url`` in chunks, resuming with Range on connection failures.

        Raises DownloadValidationError if any integrity check fails. The caller is expected to write
        the chunks into quarantine storage and only promote them after the archive checks pass.
        """
        cap = self.security.max_download_bytes
        if expected_size > cap:
            raise DownloadValidationError(f"expected size {expected_size} exceeds cap {cap}")
        received, resumes, head_buf = 0, 0, b""
        while received < expected_size:
            headers = {"Range": f"bytes={received}-"} if received else {}
            resp = self.request("GET", url, headers=headers, stream=True)
            try:
                self._validate_download_response(resp, received, expected_size)
                for chunk in resp.iter_content(chunk_size=self.http.chunk_bytes):
                    if not chunk:
                        continue
                    if len(head_buf) < len(ZIP_MAGIC):
                        head_buf += chunk[: len(ZIP_MAGIC) - len(head_buf)]
                        if len(head_buf) >= len(ZIP_MAGIC) and head_buf[:4] != ZIP_MAGIC:
                            raise DownloadValidationError("response body is not a zip archive (bad leading bytes)")
                    received += len(chunk)
                    if received > expected_size or received > cap:
                        raise DownloadValidationError(f"received {received} bytes, more than expected {expected_size}")
                    yield chunk
                if received < expected_size:
                    raise _TruncatedBody(f"body ended at {received}/{expected_size} bytes")
            except (requests.ConnectionError, requests.Timeout, requests.exceptions.ChunkedEncodingError,
                    _TruncatedBody) as exc:
                resumes += 1
                if resumes > self.http.max_retries:
                    raise DownloadValidationError(
                        f"giving up after {resumes - 1} resumes at {received}/{expected_size} bytes: {exc}") from exc
                wait = min(self.http.backoff_max_s, self.http.backoff_initial_s * 2 ** (resumes - 1))
                log.warning("download interrupted at %d/%d bytes (%s); resuming in %.0fs (resume %d/%d)",
                            received, expected_size, exc.__class__.__name__, wait, resumes, self.http.max_retries)
                time.sleep(wait)
            finally:
                resp.close()
        if received != expected_size:
            raise DownloadValidationError(f"size mismatch: received {received}, expected {expected_size}")

    def _validate_download_response(self, resp: Any, offset: int, expected_size: int) -> None:
        want = 206 if offset else 200
        if resp.status_code != want:
            raise DownloadValidationError(
                f"unexpected HTTP {resp.status_code} (wanted {want}{' for Range resume' if offset else ''})")
        ctype = (resp.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        if ctype not in self.security.allowed_content_types:
            raise DownloadValidationError(f"Content-Type {ctype!r} not in {self.security.allowed_content_types}")
        clen = resp.headers.get("Content-Length")
        if clen is None or not clen.isdigit() or int(clen) != expected_size - offset:
            raise DownloadValidationError(f"Content-Length {clen!r} != expected {expected_size - offset}")
        if offset and not (resp.headers.get("Content-Range") or "").startswith(f"bytes {offset}-"):
            raise DownloadValidationError("server did not honour the Range request")
