"""Read one landing CSV into an all-text frame (docs/DATA_PIPELINE_SPEC.md §3.1 steps 3 to 5)."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl


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


@dataclass(frozen=True)
class ParsedCsv:
    frame: pl.DataFrame
    blank_rows: int


def parse_csv_text(text: str) -> ParsedCsv:
    """Parse with a real CSV parser (quoted commas and newlines occur), every column as text, original
    header names and order. Rows with every field null are dropped and counted: 39 vintages end with an
    extra blank line that Step 1's row count includes (reconcile as rows + blank_rows == row_count).
    An unquoted empty field is null; Polars cannot tell it from a missing value."""
    frame = pl.read_csv(text.encode("utf-8"), infer_schema=False)
    blank = pl.all_horizontal(pl.all().is_null())
    kept = frame.filter(~blank) if frame.width else frame
    return ParsedCsv(frame=kept, blank_rows=frame.height - kept.height)
