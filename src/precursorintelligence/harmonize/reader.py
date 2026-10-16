"""Read one landing CSV into an all-text frame (docs/DATA_PIPELINE_SPEC.md §3.1 steps 3 to 5)."""

from __future__ import annotations


class DecodeError(ValueError):
    """The bytes are neither UTF-8 nor cp1252. A data failure (exit 2), never a lossy decode."""


def decode_csv(raw: bytes) -> tuple[str, str]:
    """Return the text and the encoding used: strict UTF-8 first, then cp1252 (2021-01 to 2023-04 files
    hold byte 0x92). The order matters: cp1252 accepts almost any byte string, so trying it first would
    turn UTF-8 multi-byte characters into mojibake without an error."""
    try:
        return raw.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        pass
    try:
        return raw.decode("cp1252"), "cp1252"
    except UnicodeDecodeError as exc:
        raise DecodeError(f"not valid UTF-8 or cp1252: {exc}") from None
