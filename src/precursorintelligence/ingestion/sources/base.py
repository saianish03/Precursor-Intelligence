"""CatalogSource port: anything that can list downloadable archive entries."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Protocol


class CatalogContractError(RuntimeError):
    """The upstream catalog response no longer matches the expected shape."""


@dataclass(frozen=True)
class CatalogEntry:
    entry_id: str          # f"{type}:{zip_date}:{zip_name}" - unique even for the re-used "current" file name
    name: str
    nid: str
    type: str              # theme | annual_theme | current
    theme: str
    zip_date: date
    zip_name: str
    url: str               # relative path as published by the catalog (validated before download)
    size_bytes: int
    source_endpoint: str   # archive | current


@dataclass
class SourceSnapshot:
    entries: list[CatalogEntry]
    raw_payloads: dict[str, bytes] = field(default_factory=dict)   # name -> exact bytes received
    datasets: list[dict[str, Any]] = field(default_factory=list)   # release metadata rows


class CatalogSource(Protocol):
    name: str

    def fetch(self) -> SourceSnapshot: ...
