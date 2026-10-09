"""Archive layout knowledge: table classification, layout detection, vintage parsing.

Three layouts exist in the CMS archive (verified by reading every zip's central directory):
  L1 (2019 -> 2020-07)  flat, legacy names, e.g. ProviderInfo_Download.csv; no Processing Date column
  L2 (2020-08 -> 2026-06) NH_<Table>_<MonYYYY>.csv, optionally inside one sub-folder
  L3 (2026-07 ->)       <datasetId>_<YYYY-MM-DD>_NH_<Table>_<MonYYYY>.csv + manifest.json
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
from datetime import date, datetime
from typing import BinaryIO

from ..common.config import TableSpec

OTHER = "other"
_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], start=1)}
_TOKEN = re.compile(r"_([A-Za-z]{3})(\d{4})\.[A-Za-z]+$")
_L3_PREFIX = re.compile(r"^[a-z0-9]{4}-[a-z0-9]{4}_\d{4}-\d{2}-\d{2}_")
_DATE_FORMATS = ("%Y-%m-%d", "%m/%d/%Y", "%Y%m%d", "%m/%d/%y", "%Y/%m/%d")


class TableClassifier:
    def __init__(self, tables: dict[str, TableSpec]) -> None:
        self._rules = [(name, [re.compile(p) for p in spec.patterns]) for name, spec in tables.items()]

    def classify(self, file_name: str) -> str:
        for name, patterns in self._rules:
            if any(p.search(file_name) for p in patterns):
                return name
        return OTHER


def detect_layout(file_names: list[str]) -> str:
    names = [n for n in file_names if n.lower().endswith(".csv")]
    if any(_L3_PREFIX.match(n) for n in names):
        return "L3"
    if any(n.endswith("_Download.csv") for n in names):
        return "L1"
    return "L2"


def filename_month(file_name: str) -> str | None:
    """'NH_ProviderInfo_Nov2021.csv' -> '2021-11' (fallback only; Processing Date is authoritative)."""
    m = _TOKEN.search(file_name)
    if not m or m.group(1).lower() not in _MONTHS:
        return None
    return f"{m.group(2)}-{_MONTHS[m.group(1).lower()]:02d}"


def parse_date(value: str | None) -> date | None:
    if not value:
        return None
    v = value.strip().split(" ")[0].split("T")[0]
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(v, fmt).date()
        except ValueError:
            continue
    return None


def normalize_header(cols: list[str]) -> list[str]:
    return [" ".join(c.replace("﻿", "").split()) for c in cols]


def header_hash(cols: list[str]) -> str:
    return hashlib.sha256("\x1f".join(cols).encode("utf-8")).hexdigest()[:16]


def read_header_and_first_row(f: BinaryIO, max_bytes: int = 4 * 1024 * 1024) -> tuple[list[str], list[str] | None]:
    """Read only the beginning of a CSV (never the whole file)."""
    text = io.TextIOWrapper(f, encoding="utf-8-sig", errors="replace", newline="")
    reader = csv.reader(_bounded_lines(text, max_bytes))
    header = next(reader, None)
    if header is None:
        return [], None
    return normalize_header(header), next(reader, None)


def _bounded_lines(text: io.TextIOBase, max_bytes: int):
    read = 0
    for line in text:
        read += len(line)
        if read > max_bytes:
            return
        yield line


def processing_date_from(header: list[str], row: list[str] | None) -> date | None:
    if not row:
        return None
    for i, col in enumerate(header):
        if "processing" in col.lower() and i < len(row):
            return parse_date(row[i])
    return None


def manifest_filenames(raw: bytes) -> set[str]:
    """File names listed in an L3 manifest.json ([{..., resources: [{filename: ...}]}])."""
    try:
        data = json.loads(raw.decode("utf-8", errors="replace"))
    except json.JSONDecodeError:
        return set()
    names: set[str] = set()

    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                if k == "filename" and isinstance(v, str):
                    names.add(v)
                else:
                    walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
    walk(data)
    return names
