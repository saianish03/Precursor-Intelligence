"""Task ``resolve``: duplicates, gaps and anomalies across the whole inventory (guide §2.7).

* Duplicates: when several files map to the same (table, vintage_month), the file from the zip with
  the latest zip_date is kept ('selected'); others become 'duplicate_vintage'. On the same date an
  archive zip wins over the rolling 'current' zip (archive copies are stable and reproducible).
* Gaps: months between scope.vintage_start and V (latest selected vintage) with no file, per table.
* Anomalies: recorded in the ``anomalies`` table with a kind, severity and explanation.
"""

from __future__ import annotations

import logging
import statistics
from collections import defaultdict
from dataclasses import dataclass, field

from ..common.context import Context, utc_now
from .inventory import JUNK
from .layout import OTHER

log = logging.getLogger(__name__)

TYPE_RANK = {"theme": 2, "current": 1, "annual_theme": 0}


def month_range(start: str, end: str) -> list[str]:
    y, m = map(int, start.split("-"))
    ey, em = map(int, end.split("-"))
    out = []
    while (y, m) <= (ey, em):
        out.append(f"{y:04d}-{m:02d}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


@dataclass
class ResolveOutput:
    latest_vintage: str | None
    months: list[str]
    coverage: dict[str, dict[str, str]]            # table -> month -> zip_name of the selected file
    gaps: dict[str, list[str]]
    duplicates: list[dict]
    anomalies: list[dict] = field(default_factory=list)
    n_zips_processed: int = 0
    n_zips_in_scope: int = 0


def resolve(ctx: Context) -> ResolveOutput:
    tables = list(ctx.cfg.tables)
    start = ctx.cfg.scope.vintage_start
    inv = ctx.metadata.read("inventory")
    catalog = {r["entry_id"]: r for r in ctx.metadata.read("catalog")}
    dlog = ctx.metadata.read("download_log")
    anomalies: list[dict] = []

    def anomaly(entry_id: str | None, kind: str, severity: str, detail: str) -> None:
        anomalies.append({"run_id": ctx.run_id, "entry_id": entry_id, "kind": kind, "severity": severity,
                          "detail": detail, "created_at": utc_now()})

    scoped = [r for r in inv if r["table_name"] not in (OTHER, JUNK) and r["file_name"].lower().endswith(".csv")]
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in scoped:
        if not r.get("vintage_month"):
            r["status"], r["status_detail"] = "unresolved", "no vintage could be determined"
            anomaly(r["entry_id"], "no_vintage", "error", f"{r['file_name']}: vintage unknown")
            continue
        if r["vintage_month"] < start:
            r["status"], r["status_detail"] = "out_of_scope", f"vintage {r['vintage_month']} < {start}"
            continue
        groups[(r["table_name"], r["vintage_month"])].append(r)

    duplicates = []
    coverage: dict[str, dict[str, str]] = {t: {} for t in tables}
    for (table, month), rows in groups.items():
        rows.sort(key=lambda r: (r["zip_date"], TYPE_RANK.get(r["zip_type"], 0)), reverse=True)
        keep = rows[0]
        keep["status"], keep["status_detail"] = "selected", None
        coverage[table][month] = keep["zip_name"]
        for other in rows[1:]:
            identical = other["crc32"] == keep["crc32"] and other["file_size_bytes"] == keep["file_size_bytes"]
            other["status"] = "duplicate_vintage"
            other["status_detail"] = f"superseded by {keep['zip_name']} ({'identical' if identical else 'different'} content)"
            duplicates.append({"table": table, "vintage_month": month, "kept": keep["zip_name"],
                               "dropped": other["zip_name"], "identical": identical})

    selected_months = sorted({m for t in coverage.values() for m in t})
    latest = coverage.get("provider_info") and max(coverage["provider_info"]) or (selected_months[-1] if selected_months else None)
    months = month_range(start, latest) if latest else []
    gaps = {t: [m for m in months if m not in coverage[t]] for t in tables}

    # ---- per-zip anomalies
    expected_fallbacks: dict[str, int] = defaultdict(int)
    by_entry: dict[str, list[dict]] = defaultdict(list)
    for r in inv:
        by_entry[r["entry_id"]].append(r)
    real_counts: dict[str, list[int]] = defaultdict(list)
    for eid, rows in by_entry.items():
        real_counts[rows[0]["layout"]].append(sum(r["table_name"] != JUNK for r in rows))
    for eid, rows in by_entry.items():
        cat = catalog.get(eid, {})
        zip_name, layout = rows[0]["zip_name"], rows[0]["layout"]
        present = {r["table_name"] for r in rows if r["table_name"] not in (OTHER, JUNK)}
        vintages = {r["vintage_month"] for r in rows if r["table_name"] not in (OTHER, JUNK) and r.get("vintage_month")}
        if vintages and max(vintages) >= start and (missing := sorted(set(tables) - present)):
            anomaly(eid, "partial_zip", "warning",
                    f"{zip_name} lacks in-scope tables {missing}; contains only {sorted(present) or 'none'}")
        n_real = sum(r["table_name"] != JUNK for r in rows)
        med = statistics.median(real_counts[layout])
        if med and abs(n_real - med) / med > 0.25:
            anomaly(eid, "member_count", "info", f"{zip_name} has {n_real} files vs median {med:.0f} for layout {layout}")
        junk = [r for r in rows if r["table_name"] == JUNK]
        if junk:
            junk_bytes = sum(r["compressed_size_bytes"] or 0 for r in junk)
            size = cat.get("size_bytes") or 0
            share = junk_bytes / size if size else 0
            anomaly(eid, "junk_bloat" if share > 0.5 else "junk_members", "warning" if share > 0.5 else "info",
                    f"{zip_name} contains {len(junk)} macOS metadata entries (__MACOSX/.DS_Store), "
                    f"{junk_bytes / 1e6:.1f} MB = {share:.0%} of the zip; ignored")
        if cat.get("type") == "theme" and "/dataset-archives/theme/" not in (cat.get("url") or ""):
            anomaly(eid, "nonstandard_path", "info", f"{zip_name} is served from {cat.get('url')}")
        for r in rows:
            if r["table_name"] not in (OTHER, JUNK) and r.get("vintage_source") not in (None, "processing_date"):
                if ctx.cfg.tables[r["table_name"]].has_processing_date:
                    anomaly(eid, "vintage_fallback", "warning",
                            f"{r['file_name']}: vintage {r['vintage_month']} from {r['vintage_source']}")
                else:
                    expected_fallbacks[r["table_name"]] += 1

    # ---- zip size outliers among monthly zips: real stored size, neighbours from the whole catalog
    last: dict[str, dict] = {}
    for r in sorted(dlog, key=lambda r: r["id"]):
        last[r["entry_id"]] = r

    def real_size(c: dict) -> int:
        d = last.get(c["entry_id"])
        return int(d["actual_size"]) if d and d.get("actual_size") else int(c["size_bytes"])

    monthly = sorted((c for c in catalog.values() if c["type"] == "theme" and c.get("active")),
                     key=lambda c: c["zip_date"])
    for i, c in enumerate(monthly):
        if c["entry_id"] not in by_entry:
            continue
        neigh = [real_size(m) for m in monthly[max(0, i - 3): i] + monthly[i + 1: i + 4]]
        if len(neigh) >= 2:
            med = statistics.median(neigh)
            ratio = real_size(c) / med
            if ratio > 1.5 or ratio < 1 / 1.5:
                anomaly(c["entry_id"], "size_outlier", "warning",
                        f"{c['zip_name']} is {real_size(c) / 1e6:.1f} MB vs neighbour median {med / 1e6:.1f} MB")

    # ---- download outcomes
    for eid, r in last.items():
        if r["status"] in ("failed", "quarantined", "rejected"):
            anomaly(eid, f"download_{r['status']}", "error", f"{r['zip_name']}: {r['detail']}")
        if r["status"] == "republished":
            anomaly(eid, "republished", "warning", r["detail"])
        if r.get("expected_size") and r.get("catalog_size") and r["expected_size"] != r["catalog_size"]:
            anomaly(eid, "catalog_size_mismatch", "info",
                    f"{r['zip_name']}: catalog {r['catalog_size']} vs server {r['expected_size']} bytes")

    for d in duplicates:
        anomaly(None, "duplicate_vintage", "info",
                f"{d['table']} {d['vintage_month']}: kept {d['kept']}, dropped {d['dropped']} "
                f"({'identical' if d['identical'] else 'different'} content)")
    for t, n in expected_fallbacks.items():
        anomaly(None, "expected_vintage_from_name", "info",
                f"{t}: {n} files have no Processing Date column by design; vintage taken from the file name")
    in_scope = [c for c in catalog.values() if c.get("active") and c["type"] in ctx.cfg.scope.include_types
                and c["zip_date"] >= ctx.cfg.scope.min_zip_date.isoformat()]
    subset = len(by_entry) < len(in_scope)
    for t, ms in gaps.items():
        if ms:
            anomaly(None, "missing_vintage", "info" if subset else "warning",
                    f"{t}: no file for {', '.join(ms)}" + (" (subset run: gaps include unprocessed zips)" if subset else ""))

    ctx.metadata.upsert("inventory", scoped)
    ctx.metadata.delete_where("anomalies", {"run_id": ctx.run_id})
    ctx.metadata.append("anomalies", anomalies)
    n_sel = sum(r["status"] == "selected" for r in scoped)
    log.info("resolve: V=%s, %d selected files, %d duplicates, gaps=%s, %d anomalies",
             latest, n_sel, len(duplicates), {t: len(g) for t, g in gaps.items() if g}, len(anomalies))
    return ResolveOutput(latest, months, coverage, gaps, duplicates, anomalies,
                         n_zips_processed=len(by_entry), n_zips_in_scope=len(in_scope))
