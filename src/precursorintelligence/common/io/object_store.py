"""ObjectStore port and adapters.

Keys are logical, POSIX-style, relative paths such as ``landing/cms_nh/zips/zip_date=2024-08-29/x.zip``.
Every adapter maps a key to ``<root>/<key>``, so the folder layout is identical on local disk, S3 and GCS.

Rules enforced for every adapter:
  * keys are validated (no absolute paths, no ``..``, no backslashes, no empty segments);
  * writes never overwrite an existing object unless ``overwrite=True`` is passed explicitly;
  * writes are atomic (temp file + rename locally; object stores only publish an object on close);
  * deletes are only allowed under configured scratch prefixes (``quarantine/`` and ``tmp/``).
"""

from __future__ import annotations

import hashlib
import logging
import os
import posixpath
import stat as stat_mod
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import BinaryIO, Protocol, runtime_checkable

log = logging.getLogger(__name__)

DELETABLE_PREFIXES = ("quarantine/", "tmp/")


class ObjectStoreError(RuntimeError):
    pass


class ObjectExistsError(ObjectStoreError):
    pass


@dataclass(frozen=True)
class ObjectInfo:
    key: str
    size: int
    modified: datetime | None


@dataclass(frozen=True)
class PutResult:
    key: str
    size: int
    sha256: str
    uri: str


def validate_key(key: str) -> str:
    if not key or key.startswith("/") or "\\" in key or "\x00" in key:
        raise ObjectStoreError(f"invalid object key: {key!r}")
    parts = key.split("/")
    if any(p in ("", ".", "..") for p in parts):
        raise ObjectStoreError(f"invalid object key: {key!r}")
    return key


@runtime_checkable
class ObjectStore(Protocol):
    root_uri: str

    def uri_for(self, key: str) -> str: ...
    def exists(self, key: str) -> bool: ...
    def stat(self, key: str) -> ObjectInfo | None: ...
    def list(self, prefix: str) -> list[str]: ...
    def put_stream(self, key: str, chunks: Iterable[bytes], *, overwrite: bool = False) -> PutResult: ...
    def put_bytes(self, key: str, data: bytes, *, overwrite: bool = False) -> PutResult: ...
    def open_read(self, key: str) -> BinaryIO: ...
    def move(self, src: str, dst: str) -> None: ...
    def remove(self, key: str) -> None: ...
    def make_immutable(self, key: str) -> None: ...
    def local_path(self, key: str) -> Path | None: ...
    def prune_empty(self, prefix: str) -> int: ...


def _check_deletable(key: str) -> None:
    if not key.startswith(DELETABLE_PREFIXES):
        raise ObjectStoreError(f"refusing to delete outside scratch prefixes {DELETABLE_PREFIXES}: {key}")


def sha256_of(store: ObjectStore, key: str, chunk: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with store.open_read(key) as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


class LocalObjectStore:
    """Files on local disk under ``root``. Atomic writes via temp file + ``os.replace``."""

    def __init__(self, root: str | Path, immutable: str | None = "chmod") -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.root_uri = self.root.as_uri()
        self.immutable = immutable

    def _path(self, key: str) -> Path:
        p = (self.root / validate_key(key)).resolve()
        if self.root not in p.parents:
            raise ObjectStoreError(f"key escapes store root: {key}")
        return p

    def uri_for(self, key: str) -> str:
        return self._path(key).as_uri()

    def local_path(self, key: str) -> Path | None:
        return self._path(key)

    def exists(self, key: str) -> bool:
        return self._path(key).is_file()

    def stat(self, key: str) -> ObjectInfo | None:
        p = self._path(key)
        if not p.is_file():
            return None
        st = p.stat()
        return ObjectInfo(key, st.st_size, datetime.fromtimestamp(st.st_mtime, tz=UTC))

    def list(self, prefix: str) -> list[str]:
        base = self.root / prefix if prefix else self.root
        if not base.exists():
            return []
        return sorted(
            p.relative_to(self.root).as_posix()
            for p in base.rglob("*")
            if p.is_file() and not p.name.startswith(".tmp-")
        )

    def put_stream(self, key: str, chunks: Iterable[bytes], *, overwrite: bool = False) -> PutResult:
        dst = self._path(key)
        if dst.exists() and not overwrite:
            raise ObjectExistsError(f"object already exists: {key}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.parent / f".tmp-{uuid.uuid4().hex}-{dst.name}"
        h, size = hashlib.sha256(), 0
        try:
            with tmp.open("wb") as f:
                for chunk in chunks:
                    f.write(chunk)
                    h.update(chunk)
                    size += len(chunk)
                f.flush()
                os.fsync(f.fileno())
            if dst.exists() and overwrite:
                os.chmod(dst, stat_mod.S_IWUSR | stat_mod.S_IRUSR)
            os.replace(tmp, dst)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        return PutResult(key, size, h.hexdigest(), dst.as_uri())

    def put_bytes(self, key: str, data: bytes, *, overwrite: bool = False) -> PutResult:
        return self.put_stream(key, [data], overwrite=overwrite)

    def open_read(self, key: str) -> BinaryIO:
        return self._path(key).open("rb")

    def move(self, src: str, dst: str) -> None:
        s, d = self._path(src), self._path(dst)
        if d.exists():
            raise ObjectExistsError(f"object already exists: {dst}")
        d.parent.mkdir(parents=True, exist_ok=True)
        os.replace(s, d)

    def remove(self, key: str) -> None:
        _check_deletable(key)
        p = self._path(key)
        if p.exists():
            os.chmod(p, stat_mod.S_IWUSR | stat_mod.S_IRUSR)
            p.unlink()

    def prune_empty(self, prefix: str) -> int:
        """Delete empty directories under a scratch prefix. Call once, after parallel work has finished."""
        _check_deletable(prefix.rstrip("/") + "/")
        base = self._path(prefix.rstrip("/")) if prefix.strip("/") else self.root
        removed = 0
        if base.is_dir():
            for d in sorted((p for p in base.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
                if not any(d.iterdir()):
                    d.rmdir()
                    removed += 1
            if not any(base.iterdir()):
                base.rmdir()
                removed += 1
        return removed

    def make_immutable(self, key: str) -> None:
        if self.immutable:
            os.chmod(self._path(key), stat_mod.S_IRUSR | stat_mod.S_IRGRP | stat_mod.S_IROTH)  # 0o444


class FsspecObjectStore:
    """Any fsspec filesystem: ``memory://`` (tests), ``s3://`` (s3fs), ``gs://`` (gcsfs), ...

    Object stores publish an object only when the upload completes (multipart/resumable upload), which
    gives the same atomicity as a local rename. Immutability on S3/GCS is enforced by bucket-level
    policy (Object Lock default retention / bucket retention policy), provisioned with infrastructure
    code; ``make_immutable`` therefore only records the intent here.
    """

    def __init__(self, root: str, *, immutable: str | None = None, storage_options: dict | None = None) -> None:
        import fsspec

        self.fs, base = fsspec.core.url_to_fs(root, **(storage_options or {}))
        self.base = base.rstrip("/")
        self.protocol = root.split("://", 1)[0] if "://" in root else "file"
        self.root_uri = root.rstrip("/")
        self.immutable = immutable

    def _path(self, key: str) -> str:
        return posixpath.join(self.base, validate_key(key))

    def uri_for(self, key: str) -> str:
        return f"{self.root_uri}/{validate_key(key)}"

    def local_path(self, key: str) -> Path | None:
        return None

    def exists(self, key: str) -> bool:
        p = self._path(key)
        return self.fs.exists(p) and self.fs.isfile(p)

    def stat(self, key: str) -> ObjectInfo | None:
        p = self._path(key)
        if not self.exists(key):
            return None
        info = self.fs.info(p)
        modified = info.get("LastModified") or info.get("updated") or info.get("created")
        if isinstance(modified, (int, float)):
            modified = datetime.fromtimestamp(modified, tz=UTC)
        elif not isinstance(modified, datetime):
            modified = None
        return ObjectInfo(key, int(info.get("size", 0)), modified)

    def list(self, prefix: str) -> list[str]:
        base = self._path(prefix.rstrip("/")) if prefix else self.base
        if not self.fs.exists(base):
            return []
        out = []
        for p in self.fs.find(base):
            rel = p[len(self.base):].lstrip("/") if p.startswith(self.base) else p
            out.append(rel)
        return sorted(out)

    def put_stream(self, key: str, chunks: Iterable[bytes], *, overwrite: bool = False) -> PutResult:
        p = self._path(key)
        if self.exists(key) and not overwrite:
            raise ObjectExistsError(f"object already exists: {key}")
        parent = posixpath.dirname(p)
        if parent:
            self.fs.makedirs(parent, exist_ok=True)
        h, size = hashlib.sha256(), 0
        try:
            with self.fs.open(p, "wb") as f:
                for chunk in chunks:
                    f.write(chunk)
                    h.update(chunk)
                    size += len(chunk)
        except BaseException:
            if self.fs.exists(p):
                self.fs.rm(p)
            raise
        return PutResult(key, size, h.hexdigest(), self.uri_for(key))

    def put_bytes(self, key: str, data: bytes, *, overwrite: bool = False) -> PutResult:
        return self.put_stream(key, [data], overwrite=overwrite)

    def open_read(self, key: str) -> BinaryIO:
        return self.fs.open(self._path(key), "rb")

    def move(self, src: str, dst: str) -> None:
        if self.exists(dst):
            raise ObjectExistsError(f"object already exists: {dst}")
        parent = posixpath.dirname(self._path(dst))
        if parent:
            self.fs.makedirs(parent, exist_ok=True)
        self.fs.mv(self._path(src), self._path(dst))

    def remove(self, key: str) -> None:
        _check_deletable(key)
        if self.exists(key):
            self.fs.rm(self._path(key))

    def make_immutable(self, key: str) -> None:
        log.debug("immutability for %s is enforced by bucket policy (%s)", key, self.immutable or "none")

    def prune_empty(self, prefix: str) -> int:
        return 0  # object stores have no empty directories to clean up


def copy_object(src_store: ObjectStore, src_key: str, dst_store: ObjectStore, dst_key: str) -> PutResult:
    """Stream an object between two stores (e.g. local landing -> S3 bucket)."""
    with src_store.open_read(src_key) as f:
        return dst_store.put_stream(dst_key, iter(lambda: f.read(8 * 1024 * 1024), b""))


__all__ = [
    "FsspecObjectStore",
    "LocalObjectStore",
    "ObjectExistsError",
    "ObjectInfo",
    "ObjectStore",
    "ObjectStoreError",
    "PutResult",
    "copy_object",
    "sha256_of",
    "validate_key",
]
