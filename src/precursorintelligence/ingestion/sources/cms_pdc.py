"""CMS Provider Data Catalog archive source (nursing-homes theme).

Endpoints (verified 2026-10-06):
  GET {api}/archive/aggregate/theme/nursing-homes/relative      -> {"data": [ {name,nid,identifier,type,theme,
                                                                   url,size,date,access_level}, ... ]}
  GET {api}/archive/aggregate/current/theme/all/relative        -> {"data": [ ...one "current" zip per theme ]}
  GET {api}/metastore/schemas/dataset/items/{id}                -> DCAT dataset (modified/released/nextUpdateDate)
The two archive endpoints are undocumented (used by the site's own front end), so their shape is checked
on every call and a CatalogContractError stops the run if it changes.
"""

from __future__ import annotations

import json
import logging
import posixpath
from datetime import date
from typing import Any

from ...common.config import SourcesConfig
from ..http import SafeHttpClient
from .base import CatalogContractError, CatalogEntry, SourceSnapshot

log = logging.getLogger(__name__)

REQUIRED_KEYS = {"name", "type", "url", "size", "date"}


def _entries_from(payload: Any, *, endpoint: str, theme: str) -> list[CatalogEntry]:
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        raise CatalogContractError(f"{endpoint}: expected an object with a 'data' list")
    out: list[CatalogEntry] = []
    for i, item in enumerate(payload["data"]):
        if not isinstance(item, dict) or not REQUIRED_KEYS <= item.keys():
            raise CatalogContractError(f"{endpoint}: item {i} is missing keys {REQUIRED_KEYS - set(item)}")
        if item.get("theme") not in (theme, None):
            continue
        if item.get("access_level", "public") != "public":
            continue
        try:
            zip_date = date.fromisoformat(str(item["date"])[:10])
            size = int(item["size"])
        except (TypeError, ValueError) as exc:
            raise CatalogContractError(f"{endpoint}: item {i} has a bad date/size: {exc}") from exc
        url = str(item["url"]).replace("\\/", "/")
        zip_name = posixpath.basename(url)
        out.append(CatalogEntry(
            entry_id=f"{item['type']}:{zip_date.isoformat()}:{zip_name}",
            name=str(item["name"]), nid=str(item.get("nid") or ""), type=str(item["type"]), theme=theme,
            zip_date=zip_date, zip_name=zip_name, url=url, size_bytes=size, source_endpoint=endpoint,
        ))
    return out


class CmsPdcArchiveSource:
    name = "cms_pdc_nursing_homes"

    def __init__(self, http: SafeHttpClient, cfg: SourcesConfig) -> None:
        self.http, self.cfg = http, cfg

    def fetch(self):
        src = self.cfg.source
        theme = src.theme_slug
        raw: dict[str, bytes] = {}

        archive = self.http.get_json(self.http.api_url(src.endpoints.archive.format(theme=theme)))
        raw["archive.json"] = json.dumps(archive, sort_keys=True).encode()
        entries = _entries_from(archive, endpoint="archive", theme=theme)
        if not any(e.type == "theme" for e in entries):
            raise CatalogContractError("archive endpoint returned no monthly 'theme' entries")

        current = self.http.get_json(self.http.api_url(src.endpoints.current))
        raw["current.json"] = json.dumps(current, sort_keys=True).encode()
        cur = _entries_from(current, endpoint="current", theme=theme)
        if not cur:
            log.warning("current endpoint has no entry for theme %s", theme)
        entries += cur

        datasets: list[dict[str, Any]] = []
        meta_raw: dict[str, Any] = {}
        for table, spec in self.cfg.tables.items():
            if not spec.dataset_id:
                continue
            item = self.http.get_json(self.http.api_url(src.endpoints.metastore_item.format(dataset_id=spec.dataset_id)))
            meta_raw[spec.dataset_id] = item
            dist = (item.get("distribution") or [{}])[0]
            dist = dist.get("data", dist) if isinstance(dist, dict) else {}
            datasets.append({
                "dataset_id": spec.dataset_id, "table_name": table, "title": item.get("title"),
                "modified": item.get("modified"), "released": item.get("released"),
                "next_update": item.get("nextUpdateDate"), "current_download_url": dist.get("downloadURL"),
            })
        raw["metastore.json"] = json.dumps(meta_raw, sort_keys=True).encode()
        log.info("catalog fetched: %d archive entries (%d monthly), %d current, %d dataset records",
                 len(entries) - len(cur), sum(e.type == "theme" for e in entries), len(cur), len(datasets))
        return SourceSnapshot(entries=entries, raw_payloads=raw, datasets=datasets)
