"""Safe handling of untrusted zip archives (docs §9.5).

``inspect_archive`` validates the whole central directory before anything is extracted:
  * every member name is safe (no absolute paths, drive letters, '..', backslashes, NUL bytes);
  * no symlink members;
  * member count, total uncompressed size and per-member compression ratio are within limits (zip bombs);
  * nested zips are rejected unless nesting is enabled;
  * ``verify_crc`` re-reads every member and checks its CRC-32.
Junk entries created by macOS (``__MACOSX/``, ``.DS_Store``) are recognised and skipped.
"""

from __future__ import annotations

import posixpath
import stat
import zipfile
from dataclasses import dataclass
from typing import BinaryIO

from ..common.config import ZipLimits


class UnsafeArchiveError(RuntimeError):
    pass


@dataclass(frozen=True)
class MemberInfo:
    path: str            # name inside the archive (normalised to '/')
    file_name: str       # base name
    size: int
    compressed_size: int
    crc32: str
    compress_type: int
    is_junk: bool
    extension: str


def is_junk(path: str) -> bool:
    base = posixpath.basename(path.rstrip("/"))
    return path.startswith("__MACOSX/") or "/__MACOSX/" in path or base == ".DS_Store" or base.startswith("._")


def _check_name(name: str) -> str:
    if "\x00" in name:
        raise UnsafeArchiveError(f"NUL byte in member name: {name!r}")
    norm = name.replace("\\", "/")
    if norm.startswith("/") or (len(norm) > 1 and norm[1] == ":"):
        raise UnsafeArchiveError(f"absolute member path: {name!r}")
    if any(seg == ".." for seg in norm.split("/")):
        raise UnsafeArchiveError(f"path traversal in member name: {name!r}")
    return norm


def _is_symlink(zi: zipfile.ZipInfo) -> bool:
    return stat.S_ISLNK(zi.external_attr >> 16)


def inspect_archive(fileobj: BinaryIO, limits: ZipLimits) -> list[MemberInfo]:
    """Validate an archive and return its non-directory members (junk included, flagged)."""
    try:
        zf = zipfile.ZipFile(fileobj)
    except zipfile.BadZipFile as exc:
        raise UnsafeArchiveError(f"not a valid zip archive: {exc}") from exc
    members: list[MemberInfo] = []
    total = 0
    with zf:
        for zi in zf.infolist():
            path = _check_name(zi.filename)
            if zi.is_dir():
                continue
            if _is_symlink(zi):
                raise UnsafeArchiveError(f"symlink member not allowed: {path}")
            junk = is_junk(path)
            ext = posixpath.splitext(path)[1].lower()
            if not junk:
                if ext == ".zip" and limits.max_nested_depth < 1:
                    raise UnsafeArchiveError(f"nested zip not allowed (max_nested_depth=0): {path}")
                if zi.file_size >= limits.ratio_check_min_bytes and zi.compress_size > 0:
                    ratio = zi.file_size / zi.compress_size
                    if ratio > limits.max_compression_ratio:
                        raise UnsafeArchiveError(f"compression ratio {ratio:.0f}x exceeds limit for {path}")
                total += zi.file_size
            members.append(MemberInfo(
                path=path, file_name=posixpath.basename(path), size=zi.file_size,
                compressed_size=zi.compress_size, crc32=f"{zi.CRC:08x}", compress_type=zi.compress_type,
                is_junk=junk, extension=ext,
            ))
    real = [m for m in members if not m.is_junk]
    if len(real) > limits.max_members:
        raise UnsafeArchiveError(f"{len(real)} members exceeds limit {limits.max_members}")
    if total > limits.max_total_uncompressed_bytes:
        raise UnsafeArchiveError(f"total uncompressed size {total} exceeds limit {limits.max_total_uncompressed_bytes}")
    return members


def verify_crc(fileobj: BinaryIO) -> None:
    """Decompress every member and verify its CRC-32 (zipfile.testzip)."""
    try:
        with zipfile.ZipFile(fileobj) as zf:
            bad = zf.testzip()
    except zipfile.BadZipFile as exc:
        raise UnsafeArchiveError(f"not a valid zip archive: {exc}") from exc
    if bad is not None:
        raise UnsafeArchiveError(f"CRC check failed for member {bad}")


def is_allowed_extension(member: MemberInfo, limits: ZipLimits) -> bool:
    return member.extension in {e.lower() for e in limits.allowed_extensions}
