"""Which anomalies of a run are new: the input of the DAG's anomaly alert (docs/ALERTS_HANDOFF.md §2).

`resolve` recomputes the anomalies over the whole inventory on every run, so an old problem (a file with no
vintage, a download that keeps failing) is recorded again by every run until it is fixed. An anomaly is new
when the previous run whose resolve succeeded had none of the same (entry_id, kind); that run is the last one
an alert could have reported it for. A failed resolve fails its DAG run, which alerts on its own.
"""

from __future__ import annotations

from typing import Any

from ..common.io.metadata_store import MetadataStore

SEVERITIES = ("info", "warning", "error")


class NoResolveError(LookupError):
    """The run has no resolve step in run_log (a wiring error: the check runs after resolve_report)."""


def _last_resolve(metadata: MetadataStore) -> dict[str, tuple[str, str]]:
    """run_id -> (started_at, status) of that run's latest resolve (a retried task logs one per try)."""
    out: dict[str, tuple[str, str]] = {}
    for r in metadata.read("run_log", {"task": "resolve"}):
        if r["run_id"] not in out or r["started_at"] > out[r["run_id"]][0]:
            out[r["run_id"]] = (r["started_at"], r["status"])
    return out


def run_anomalies(
    metadata: MetadataStore, run_id: str, min_severity: str = "error", new_only: bool = False
) -> list[dict[str, Any]]:
    """The run's anomalies at `min_severity` or above, oldest first; with `new_only`, only those the
    previous successful resolve did not have."""
    floor = SEVERITIES.index(min_severity)
    rows = [
        a
        for a in metadata.read("anomalies", {"run_id": run_id})
        if a["severity"] in SEVERITIES and SEVERITIES.index(a["severity"]) >= floor
    ]
    rows.sort(key=lambda a: a["id"])
    if not new_only:
        return rows
    runs = _last_resolve(metadata)
    if run_id not in runs:
        raise NoResolveError(f"run {run_id} has no resolve step in run_log")
    started = runs[run_id][0]
    earlier = [
        (s, r) for r, (s, status) in runs.items() if r != run_id and s < started and status == "ok"
    ]
    if not earlier:
        return rows
    previous = max(earlier)[1]
    seen = {(a["entry_id"], a["kind"]) for a in metadata.read("anomalies", {"run_id": previous})}
    return [a for a in rows if (a["entry_id"], a["kind"]) not in seen]
