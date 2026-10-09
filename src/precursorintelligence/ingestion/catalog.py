"""Tasks ``discover`` and ``select``.

discover: fetch the archive catalog, persist the raw API responses unchanged (provenance), upsert the
          flattened entries into ``catalog`` and log what changed since the previous run.
select:   choose which catalog entries this run should process (scope dates, types, optional subset).
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from datetime import date

from ..common.context import Context, utc_now
from .sources.base import CatalogEntry, CatalogSource
from .sources.cms_pdc import CmsPdcArchiveSource

log = logging.getLogger(__name__)

CATALOG_PREFIX = "landing/cms_nh/catalog"


@dataclass
class DiscoverOutput:
    n_entries: int
    n_new: int
    n_removed: int
    n_size_changed: int
    snapshot_prefix: str


def default_source(ctx: Context) -> CatalogSource:
    return CmsPdcArchiveSource(ctx.http, ctx.cfg)


def _entry_row(e: CatalogEntry) -> dict:
    row = asdict(e)
    row["zip_date"] = e.zip_date.isoformat()
    return row


def discover(ctx: Context, source: CatalogSource | None = None) -> DiscoverOutput:
    source = source or default_source(ctx)
    snap = source.fetch()
    now = utc_now()
    stamp = ctx.run_id.split("-")[0]
    prefix = f"{CATALOG_PREFIX}/fetched_at={stamp}"
    for name, payload in snap.raw_payloads.items():
        key = f"{prefix}/{name}"
        if not ctx.object_store.exists(key):
            ctx.object_store.put_bytes(key, payload)

    previous = {r["entry_id"]: r for r in ctx.metadata.read("catalog")}
    current_ids = set()
    rows, n_new, n_changed = [], 0, 0
    for e in snap.entries:
        row = _entry_row(e)
        prev = previous.get(e.entry_id)
        current_ids.add(e.entry_id)
        if prev is None:
            n_new += 1
            if previous:
                log.info("catalog: new entry %s (%d bytes)", e.entry_id, e.size_bytes)
        elif prev.get("size_bytes") != e.size_bytes:
            n_changed += 1
            log.warning("catalog: size of %s changed %s -> %s bytes (possible re-publication)",
                        e.entry_id, prev.get("size_bytes"), e.size_bytes)
        row.update(first_seen_at=(prev or {}).get("first_seen_at") or now, last_seen_at=now, active=True)
        rows.append(row)
    removed = [r for eid, r in previous.items() if eid not in current_ids and r.get("active")]
    for r in removed:
        log.warning("catalog: entry %s is no longer advertised by CMS", r["entry_id"])
        r["active"] = False
    ctx.metadata.upsert("catalog", rows + removed)
    if snap.datasets:
        ctx.metadata.upsert("source_datasets", [{**d, "fetched_at": now} for d in snap.datasets])
    out = DiscoverOutput(len(snap.entries), n_new, len(removed), n_changed, prefix)
    log.info("discover: %d entries (%d new, %d removed, %d size-changed); raw payloads at %s",
             out.n_entries, out.n_new, out.n_removed, out.n_size_changed, ctx.object_store.uri_for(prefix + "/archive.json"))
    return out


def _row_to_entry(r: dict) -> CatalogEntry:
    return CatalogEntry(
        entry_id=r["entry_id"], name=r["name"], nid=r["nid"], type=r["type"], theme=r["theme"],
        zip_date=date.fromisoformat(r["zip_date"]), zip_name=r["zip_name"], url=r["url"],
        size_bytes=int(r["size_bytes"]), source_endpoint=r["source_endpoint"],
    )


def select(ctx: Context, *, zip_dates: list[date] | None = None, limit: int | None = None) -> list[CatalogEntry]:
    scope = ctx.cfg.scope
    entries = [_row_to_entry(r) for r in ctx.metadata.read("catalog", {"active": True})]
    chosen = [
        e for e in entries
        if e.type in scope.include_types
        and e.zip_date >= scope.min_zip_date
        and (scope.max_zip_date is None or e.zip_date <= scope.max_zip_date)
        and (not zip_dates or e.zip_date in zip_dates)
    ]
    # deterministic order: oldest first; archive before current on the same date
    chosen.sort(key=lambda e: (e.zip_date, e.source_endpoint != "archive", e.zip_name))
    if limit:
        chosen = chosen[:limit]
    log.info("select: %d of %d catalog entries in scope (types=%s, from %s%s)", len(chosen), len(entries),
             scope.include_types, scope.min_zip_date, f", subset={len(zip_dates)} dates" if zip_dates else "")
    return chosen
