"""Release flags kept in `vintage_flags` (docs/DATA_PIPELINE_SPEC.md §3.1 step 8 and "Republished data")."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from ..common.io.metadata_store import MetadataStore
from .content import PUBLISHED, content_unchanged, previous_content_hash

STALE_REFRESH = "stale_refresh"
REVISED = "revised_after_publication"
# The three tables that carry events (citations, surveys, penalties). A release in which none of them
# changed adds nothing to the ledgers, whatever the reference or quality-measure tables did.
EVENT_TABLES = ("health_citations", "survey_dates", "penalties")


def _upsert(
    meta: MetadataStore,
    vintage: str,
    flag: str,
    table: str,
    value: bool,
    run_id: str,
    now: datetime,
) -> None:
    meta.upsert(
        "vintage_flags",
        [
            {
                "vintage_month": vintage,
                "flag": flag,
                "table_name": table,
                "value": value,
                "run_id": run_id,
                "updated_at": now.isoformat(timespec="seconds"),
            }
        ],
    )


def mark_revised(meta: MetadataStore, table: str, vintage: str, run_id: str, now: datetime) -> None:
    """A content-changing reload rewrote what was published as of this vintage. Never cleared: a later
    reload does not undo that the history was rewritten once."""
    _upsert(meta, vintage, REVISED, table, True, run_id, now)


def refresh_stale_refresh(
    meta: MetadataStore, vintage: str, *, run_id: str, now: datetime
) -> bool | None:
    """stale_refresh = every event table's content equals that table's previous vintage. Returns None and
    writes nothing while any event table has no published file for the vintage. A table's first vintage
    has nothing to compare with, so it is never stale. The comparison is made from the current hashes,
    not from the stored per-file flag, so a republished earlier vintage is reflected here too."""
    latest: dict[str, dict[str, Any]] = {}
    for row in meta.read(
        "bronze_files",
        {"vintage_month": vintage, "table_name": list(EVENT_TABLES), "status": list(PUBLISHED)},
    ):
        if row["table_name"] not in latest or row["id"] > latest[row["table_name"]]["id"]:
            latest[row["table_name"]] = row
    if set(latest) != set(EVENT_TABLES):
        return None
    stale = all(
        content_unchanged(r["content_hash"], previous_content_hash(meta, t, vintage)) is True
        for t, r in latest.items()
    )
    _upsert(meta, vintage, STALE_REFRESH, "", stale, run_id, now)
    return stale
