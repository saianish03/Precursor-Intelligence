"""Task ``report``: write Checkpoint 1 artifacts (guide §2.6 and the CHECKPOINT 1 list).

  reports/step1/run=<run_id>/01_inventory.csv          one row per (zip, file)
  reports/step1/run=<run_id>/01_inventory_summary.md   coverage, gaps, header sets, anomalies
  reports/step1/latest/...                             copy of the most recent run
"""

from __future__ import annotations

import csv
import io
import json
import logging
from collections import defaultdict

import polars as pl

from ..common.context import Context, utc_now
from .inventory import JUNK
from .layout import OTHER, parse_date
from .resolve import ResolveOutput

log = logging.getLogger(__name__)

INVENTORY_COLUMNS = [
    # columns required by the guide (§2.6)
    "zip_name", "zip_sha256", "file_name", "table", "vintage_month", "processing_date", "row_count",
    "n_columns", "header_hash", "file_size_bytes",
    # extra provenance columns
    "zip_date", "zip_type", "layout", "member_path", "vintage_source", "status", "status_detail", "extracted_key",
]

EXPLANATIONS = {
    "partial_zip": "The zip does not contain all in-scope tables. CMS sometimes issues small corrective "
                   "re-releases (e.g. only Provider Info). Its files still compete in duplicate resolution.",
    "junk_bloat": "Most of the zip's bytes are macOS metadata (__MACOSX/AppleDouble). The archive was re-packed "
                  "on a Mac; the data members are normal and the junk is ignored.",
    "junk_members": "macOS metadata entries are present; they are ignored.",
    "size_outlier": "Zip size differs by more than 1.5x from neighbouring months; see partial_zip / junk_bloat.",
    "member_count": "Number of files differs by more than 25% from other zips of the same layout.",
    "nonstandard_path": "Served from a different archive folder than other monthly zips; content is normal.",
    "vintage_fallback": "No Processing Date column; the vintage was inferred (file-name token or zip date).",
    "duplicate_vintage": "Two zips carry the same data month; the later-published one is used.",
    "missing_vintage": "CMS published no file for this data month (the archive skips some months).",
    "republished": "CMS replaced a previously downloaded zip with different bytes; both copies are kept.",
    "catalog_size_mismatch": "The catalog's advertised size differs from the server's Content-Length.",
    "download_failed": "Download failed after retries.",
    "download_quarantined": "Downloaded bytes failed validation and were kept in quarantine.",
    "download_rejected": "The catalog URL failed the security allow-list and was not requested.",
    "no_vintage": "No vintage could be determined for this file.",
    "expected_vintage_from_name": "These files never carry a Processing Date column (configured with "
                                  "has_processing_date: false); the MonYYYY token in the file name is used.",
}


def _escape_cell(v):
    """Neutralise spreadsheet formula injection for values opened in Excel/Sheets."""
    if isinstance(v, str) and v[:1] in ("=", "+", "-", "@", "\t", "\r"):
        return "'" + v
    return "" if v is None else v


def _csv_bytes(rows: list[dict]) -> bytes:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=INVENTORY_COLUMNS, extrasaction="ignore", lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow({k: _escape_cell(r.get(k)) for k in INVENTORY_COLUMNS})
    return buf.getvalue().encode("utf-8")


def _earliest_survey_date(ctx: Context, inv: list[dict]) -> tuple[str | None, str]:
    start = ctx.cfg.scope.vintage_start
    rows = [r for r in inv if r["table_name"] == "health_citations" and r["vintage_month"] == start
            and r["status"] == "selected" and r.get("extracted_key")]
    if not rows:
        return None, f"no selected health_citations file for {start}"
    key = rows[0]["extracted_key"]
    header = json.loads(rows[0]["header_json"] or "[]")
    col = next((c for c in header if c.lower() == "survey date"), None) or \
        next((c for c in header if "survey" in c.lower() and "date" in c.lower()), None)
    if col is None:
        return None, f"no survey-date column in {rows[0]['file_name']}"
    path = ctx.object_store.local_path(key)
    src = path if path is not None else ctx.object_store.open_read(key)
    values = (pl.scan_csv(src, infer_schema=False, encoding="utf8-lossy", truncate_ragged_lines=True)
              .select(pl.col(col)).unique().collect()[col].to_list()) if path is not None else \
        pl.read_csv(src, infer_schema=False, encoding="utf8-lossy", columns=[col])[col].unique().to_list()
    dates = [d for d in (parse_date(v) for v in values) if d]
    return (min(dates).isoformat() if dates else None), f"column '{col}' of {rows[0]['file_name']}"


def _header_sets(inv: list[dict]) -> dict[str, list[dict]]:
    per: dict[tuple[str, str], dict] = {}
    for r in inv:
        if r["table_name"] in (OTHER, JUNK) or not r.get("header_hash") or r["status"] not in ("selected", "duplicate_vintage"):
            continue
        k = (r["table_name"], r["header_hash"])
        s = per.setdefault(k, {"hash": r["header_hash"], "n_columns": r["n_columns"], "vintages": set(),
                               "header": json.loads(r["header_json"] or "[]")})
        s["vintages"].add(r["vintage_month"])
    out: dict[str, list[dict]] = defaultdict(list)
    for (table, _), s in per.items():
        out[table].append(s)
    for table, sets in out.items():
        sets.sort(key=lambda s: min(s["vintages"]))
        prev = None
        for s in sets:
            if prev is not None:
                a, b = set(prev["header"]), set(s["header"])
                s["added"], s["removed"] = sorted(b - a), sorted(a - b)
                s["renamed_or_reordered"] = not s["added"] and not s["removed"]
            prev = s
    return out


def build_summary(ctx: Context, res: ResolveOutput, inv: list[dict]) -> str:
    tables = list(ctx.cfg.tables)
    catalog = ctx.metadata.read("catalog")
    datasets = ctx.metadata.read("source_datasets")
    dlog = ctx.metadata.read("download_log")
    anomalies = [a for a in ctx.metadata.read("anomalies", {"run_id": ctx.run_id})]
    zips = sorted({(r["zip_date"], r["zip_name"], r["zip_type"]) for r in inv})
    earliest, earliest_src = _earliest_survey_date(ctx, inv)
    L: list[str] = []
    a = L.append

    a("# Step 1 - Archive inventory summary (CHECKPOINT 1)\n")
    a(f"- **Run:** `{ctx.run_id}` - env `{ctx.settings.env.env}` - generated {utc_now()}")
    a(f"- **Object store:** `{ctx.object_store.root_uri}`")
    a(f"- **Scope:** vintages `{ctx.cfg.scope.vintage_start}` -> V = **`{res.latest_vintage}`**, "
      f"zip types {ctx.cfg.scope.include_types}, zips dated >= {ctx.cfg.scope.min_zip_date}")
    by_type = defaultdict(int)
    for c in catalog:
        if c.get("active"):
            by_type[c["type"]] += 1
    a(f"- **CMS catalog:** {sum(by_type.values())} active entries ({dict(by_type)}); "
      f"**{len(zips)} zips inventoried**, {sum(1 for r in inv if r['table_name'] not in (OTHER, JUNK))} in-scope files")
    last = {}
    for r in sorted(dlog, key=lambda r: r["id"]):
        last[r["entry_id"]] = r
    st = defaultdict(int)
    for r in last.values():
        st[r["status"]] += 1
    if res.n_zips_processed < res.n_zips_in_scope:
        a(f"- **SUBSET RUN:** {res.n_zips_processed} of {res.n_zips_in_scope} in-scope zips processed - "
          f"gaps below include months whose zips were not processed in this run.")
    a(f"- **Download outcomes (latest per zip):** {dict(st)}; total stored "
      f"{sum((r['actual_size'] or 0) for r in last.values() if r['status'] != 'failed') / 1e9:.2f} GB\n")

    if datasets:
        a("## Current CMS release (documented metastore API)\n")
        a("| table | dataset id | data month (modified) | released | next update |")
        a("|---|---|---|---|---|")
        for d in sorted(datasets, key=lambda d: d["table_name"] or ""):
            a(f"| {d['table_name']} | `{d['dataset_id']}` | {d['modified']} | {d['released']} | {d['next_update']} |")
        a("")

    a("## 1. Vintages found per table\n")
    a("`OK` = one file; `OKx2` = duplicates resolved (later zip kept); `-` = gap.\n")
    dup_count = defaultdict(int)
    for d in res.duplicates:
        dup_count[(d["table"], d["vintage_month"])] += 1
    a("| data month | " + " | ".join(tables) + " | source zip (provider_info) |")
    a("|---|" + "---|" * len(tables) + "---|")
    for m in res.months:
        cells = []
        for t in tables:
            if m in res.coverage[t]:
                n = dup_count.get((t, m), 0)
                cells.append("OK" if not n else f"OKx{n + 1}")
            else:
                cells.append("-")
        a(f"| {m} | " + " | ".join(cells) + f" | {res.coverage.get('provider_info', {}).get(m, '')} |")
    a("")
    a("**Gaps per table:**\n")
    for t in tables:
        a(f"- `{t}`: {', '.join(res.gaps[t]) if res.gaps[t] else 'none'}")
    a("")

    a("## 2. Zip date vs data month\n")
    a("The zip's date is when CMS built it; the data month comes from `Processing Date` inside the files. "
      "They differ often, which is why the vintage is never derived from the zip name.\n")
    a("| zip date | zip | type | data month(s) | vintage source |")
    a("|---|---|---|---|---|")
    for zd, zn, zt in zips:
        rows = [r for r in inv if r["zip_name"] == zn and r["zip_date"] == zd and r["table_name"] not in (OTHER, JUNK)]
        vm = sorted({r["vintage_month"] for r in rows if r.get("vintage_month")})
        vs = sorted({r["vintage_source"] for r in rows if r.get("vintage_source")})
        a(f"| {zd} | `{zn}` | {zt} | {', '.join(vm)} | {', '.join(vs)} |")
    a("")

    a("## 3. Duplicate vintages resolved\n")
    if res.duplicates:
        a("| table | data month | kept | dropped | identical |")
        a("|---|---|---|---|---|")
        for d in sorted(res.duplicates, key=lambda d: (d["vintage_month"], d["table"])):
            a(f"| {d['table']} | {d['vintage_month']} | `{d['kept']}` | `{d['dropped']}` | {d['identical']} |")
    else:
        a("None.")
    a("")

    a("## 4. Earliest survey date in the first Health Deficiencies vintage\n")
    a(f"- **{earliest or 'n/a'}** ({earliest_src}). Citation history before this date is not observable "
      f"in the archive (each snapshot keeps roughly three inspection cycles).\n")

    a("## 5. Distinct header sets per table\n")
    a("Each `header_hash` is a distinct column list. Step 2's column map must cover every set.\n")
    for t, sets in _header_sets(inv).items():
        a(f"### `{t}`\n")
        a("| header_hash | columns | vintage range | n vintages | change vs previous set |")
        a("|---|---|---|---|---|")
        for s in sets:
            change = ""
            if "added" in s:
                parts = []
                if s["added"]:
                    parts.append("added: " + ", ".join(f"`{c}`" for c in s["added"]))
                if s["removed"]:
                    parts.append("removed: " + ", ".join(f"`{c}`" for c in s["removed"]))
                change = "; ".join(parts) or "same columns, different order/spelling"
            a(f"| `{s['hash']}` | {s['n_columns']} | {min(s['vintages'])} -> {max(s['vintages'])} | "
              f"{len(s['vintages'])} | {change} |")
        a("")

    a("## 6. Anomalies\n")
    if anomalies:
        by_kind: dict[str, list[dict]] = defaultdict(list)
        for x in anomalies:
            by_kind[x["kind"]].append(x)
        order = {"error": 0, "warning": 1, "info": 2}
        for kind, items in sorted(by_kind.items(), key=lambda kv: (order.get(kv[1][0]["severity"], 3), kv[0])):
            a(f"### `{kind}` ({items[0]['severity']}, {len(items)})\n")
            a(f"_{EXPLANATIONS.get(kind, '')}_\n")
            for x in items[:40]:
                a(f"- {x['detail']}")
            if len(items) > 40:
                a(f"- ... and {len(items) - 40} more (see the anomalies table)")
            a("")
    else:
        a("None.\n")

    a("## 7. Checkpoint\n")
    a("Stop here. A human reviews this report (gaps, duplicate rule, anomalies, header sets) and approves "
      "before Step 2 (bronze load + schema harmonisation) starts.")
    return "\n".join(L) + "\n"


def report(ctx: Context, res: ResolveOutput) -> dict[str, str]:
    inv = ctx.metadata.read("inventory")
    rows = sorted((dict(r, table=r["table_name"]) for r in inv if r["table_name"] != JUNK),
                  key=lambda r: (r["zip_date"], r["zip_name"], r["member_path"]))
    csv_bytes = _csv_bytes(rows)
    md = build_summary(ctx, res, inv).encode("utf-8")
    out = {}
    for base in (f"reports/step1/run={ctx.run_id}", "reports/step1/latest"):
        overwrite = base.endswith("latest")
        ctx.object_store.put_bytes(f"{base}/01_inventory.csv", csv_bytes, overwrite=overwrite)
        ctx.object_store.put_bytes(f"{base}/01_inventory_summary.md", md, overwrite=overwrite)
        out[base] = ctx.object_store.uri_for(f"{base}/01_inventory_summary.md")
    log.info("report: %d inventory rows; summary at %s", len(rows), out["reports/step1/latest"])
    return out


__all__ = ["INVENTORY_COLUMNS", "build_summary", "report"]
