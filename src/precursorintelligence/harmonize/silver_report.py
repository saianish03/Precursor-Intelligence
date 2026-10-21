"""Checkpoint 2 report: 02_harmonization.md, 02_column_availability.csv and 02_unmapped_headers.csv
(docs/DATA_PIPELINE_SPEC.md §3.5). Built from the current state (latest bronze and silver rows), not only
from this run, so a rerun that builds nothing still writes the full report.

Also the markdown of the Checkpoint 3 and 4 reports (lag_report, universe_report; spec §5-§6), which
silver_steps.run_calibration writes."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from typing import Any

import polars as pl

from ..ingestion.context import Context
from . import calibrate as cal
from .columns import CalibrationConfig, SilverConfig
from .flags import REVISED, STALE_REFRESH
from .silver import BUILT, PartitionResult, latest_bronze

AVAILABLE = (
    0.5  # a column counts as available in a vintage when more than half its rows are non-null
)


def _current_stats(ctx: Context) -> dict[tuple[str, str], dict[str, Any]]:
    """(silver table, vintage) -> stats of its latest built row."""
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for r in ctx.metadata.read("silver_partitions", {"status": BUILT}):
        k = (r["table_name"], r["vintage_month"])
        if k not in latest or r["id"] > latest[k]["id"]:
            latest[k] = r
    return {k: json.loads(r["stats"]) | {"row_count": r["row_count"]} for k, r in latest.items()}


def availability(stats: dict[tuple[str, str], dict[str, Any]]) -> pl.DataFrame:
    rows = []
    for (table, vintage), s in stats.items():
        for col, n in s["non_null"].items():
            rate = n / s["row_count"] if s["row_count"] else 0.0
            rows.append({"table": table, "column": col, "vintage": vintage, "rate": rate})
    if not rows:
        return pl.DataFrame()
    df = pl.DataFrame(rows)
    return (
        df.group_by("table", "column")
        .agg(
            pl.col("vintage").filter(pl.col("rate") > AVAILABLE).min().alias("first_vintage"),
            pl.col("vintage").filter(pl.col("rate") > AVAILABLE).max().alias("last_vintage"),
            pl.col("rate").max().round(4).alias("max_non_null_rate"),
            pl.col("vintage").n_unique().alias("n_vintages"),
        )
        .sort("table", "column")
    )


def md_table(header: list[str], rows: list[list[Any]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def write_reports(
    ctx: Context, cfg: SilverConfig, partitions: list[PartitionResult], lookup_versions: int
) -> list[str]:
    stats = _current_stats(ctx)
    avail = availability(stats)
    problems = [
        {"table": p.source, "vintage": p.vintage, "header": h, "problem": why}
        for p in partitions
        for h, why in p.problems
    ]
    md = render(ctx, cfg, partitions, stats, avail, problems, lookup_versions)
    files = {
        "02_harmonization.md": md.encode("utf-8"),
        "02_column_availability.csv": avail.write_csv().encode("utf-8") if avail.height else b"",
        "02_unmapped_headers.csv": pl.DataFrame(
            problems, schema=["table", "vintage", "header", "problem"], orient="row"
        )
        .write_csv()
        .encode("utf-8"),
    }
    keys = []
    for folder in (f"reports/step2/run={ctx.run_id}", "reports/step2/latest"):
        for name, data in files.items():
            key = f"{folder}/{name}"
            ctx.object_store.put_bytes(key, data, overwrite=True)
            keys.append(key)
    return keys


def render(
    ctx: Context,
    cfg: SilverConfig,
    partitions: list[PartitionResult],
    stats: dict[tuple[str, str], dict[str, Any]],
    avail: pl.DataFrame,
    problems: list[dict[str, str]],
    lookup_versions: int,
) -> str:
    counts = Counter(p.status for p in partitions)
    bronze_rows = _latest_bronze_rows(ctx)
    vintages = sorted({v for _, v in stats})
    lines = [
        "# Step 2 - Harmonisation (CHECKPOINT 2)",
        "",
        f"- **Run:** `{ctx.run_id}` - env `{ctx.settings.env.env}`",
        f"- **Partitions this run:** {dict(counts)}; config hash `{cfg.config_hash[:12]}`",
        f"- **Silver partitions current:** {len(stats)} over {len(vintages)} vintages"
        + (f" ({vintages[0]} .. {vintages[-1]})" if vintages else ""),
        f"- **citation_lookup_versions:** {lookup_versions} rows (spec: about 660)",
        "",
        "## 1. Column map",
        "",
        md_table(
            ["bronze table", "canonical columns", "renamed (>1 alias)", "ignored headers"],
            [
                [
                    t,
                    len(m.columns),
                    sum(len(c.aliases) > 1 for c in m.columns.values()),
                    len(m.ignored),
                ]
                for t, m in cfg.sources.items()
            ],
        ),
        "",
        f"**Header problems (unmapped, out-of-range, missing): {len(problems)}** (must be 0)."
        + (" See `02_unmapped_headers.csv`." if problems else ""),
        "",
        "## 2. Bronze inputs: encodings and blank rows",
        "",
        md_table(
            ["table", "files", "cp1252", "utf-8", "files with a blank row"],
            [
                [
                    t,
                    len(rs),
                    sum(r["encoding"] == "cp1252" for r in rs),
                    sum(r["encoding"] == "utf-8" for r in rs),
                    sum((r["blank_rows"] or 0) > 0 for r in rs),
                ]
                for t, rs in sorted(bronze_rows.items())
            ],
        ),
        "",
        "## 3. Rows per vintage",
        "",
        md_table(
            ["silver table", "vintages", "min rows", "max rows", "total rows"],
            [[t, len(r), min(r), max(r), sum(r)] for t, r in sorted(_rows_by_table(stats).items())],
        ),
        "",
        "## 4. Uniqueness",
        "",
        "Every built partition passed its within-vintage key check (a duplicate key fails the partition):",
        "",
        md_table(
            ["silver table", "key"], [[t.name, ", ".join(t.key)] for t in cfg.tables.values()]
        ),
        "",
        *_penalties(stats),
        "",
        "## 5. Normalisation: values that did not convert (set to null)",
        "",
        *_invalid(stats),
        "",
        "## 6. Column availability (non-null rate > 50%)",
        "",
        "Columns not available in every vintage (full list in `02_column_availability.csv`):",
        "",
        *_partial(avail, len(vintages)),
        "",
        "## 7. Events after the cutoff (kept; spec: 217 rows in 3 vintages)",
        "",
        *_post_cutoff(stats),
        "",
        "## 8. Release flags",
        "",
        *_flags(ctx),
        "",
        "## 9. Decisions carried",
        "",
        "- 2026-07 Provider Info comes from the 2026-08-06 re-release (later zip wins, spec §14 #2).",
        "- Penalty exact duplicates are kept with `dup_seq`; the row key never uses `fine_id` (spec §14 #1).",
        "- Citation lookup is kept per vintage plus `citation_lookup_versions` (spec §14 #5).",
        "- Open: keep-list extras (§14 #3) and the `not_corrected_values` list (§14 #4).",
        "",
    ]
    return "\n".join(lines)


def _latest_bronze_rows(ctx: Context) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in latest_bronze(ctx.metadata):
        out[r["table_name"]].append(r)
    return out


def _rows_by_table(stats: dict[tuple[str, str], dict[str, Any]]) -> dict[str, list[int]]:
    out: dict[str, list[int]] = defaultdict(list)
    for (table, _), s in stats.items():
        out[table].append(s["row_count"])
    return out


def _penalties(stats: dict[tuple[str, str], dict[str, Any]]) -> list[str]:
    pen = {v: s for (t, v), s in stats.items() if t == "penalties_by_vintage"}
    dup = {v: s.get("exact_duplicate_rows", 0) for v, s in pen.items()}
    seq = sum(s.get("dup_seq_gt_1", 0) for s in pen.values())
    with_dups = sorted(v for v, n in dup.items() if n)
    return [
        f"**Penalties:** {sum(dup.values())} exact-duplicate rows (extra copies; spec: 2,209), all retained, "
        f"in {len(with_dups)} vintages"
        + (f" ({with_dups[0]} .. {with_dups[-1]})" if with_dups else "")
        + f"; {seq} rows have `dup_seq` > 1 (from 2026-06 these differ only in `Fine ID`)."
    ]


def _invalid(stats: dict[tuple[str, str], dict[str, Any]]) -> list[str]:
    total: Counter[tuple[str, str]] = Counter()
    for (table, _), s in stats.items():
        for col, n in s.get("invalid", {}).items():
            total[(table, col)] += n
    if not total:
        return [
            "None: every non-empty value converted (CCNs are checked strictly; a bad one fails the partition)."
        ]
    return [
        md_table(
            ["silver table", "column", "rows"], [[t, c, n] for (t, c), n in sorted(total.items())]
        )
    ]


def _partial(avail: pl.DataFrame, n_vintages: int) -> list[str]:
    if avail.is_empty():
        return ["(no silver partitions yet)"]
    part = avail.filter(
        pl.col("first_vintage").is_null()
        | (pl.col("n_vintages") < n_vintages)
        | (pl.col("max_non_null_rate") <= AVAILABLE)
        | (pl.col("first_vintage") != avail["first_vintage"].drop_nulls().min())
        | (pl.col("last_vintage") != avail["last_vintage"].drop_nulls().max())
    )
    return [
        md_table(
            ["silver table", "column", "first", "last", "max non-null rate"],
            [
                list(r)
                for r in part.select(
                    "table", "column", "first_vintage", "last_vintage", "max_non_null_rate"
                ).iter_rows()
            ],
        )
    ]


def _post_cutoff(stats: dict[tuple[str, str], dict[str, Any]]) -> list[str]:
    rows = [
        [t, v, s["post_cutoff_rows"]]
        for (t, v), s in sorted(stats.items())
        if s.get("post_cutoff_rows")
    ]
    total = sum(r[2] for r in rows)
    return (
        [
            f"{total} rows in {len(rows)} partitions.",
            "",
            md_table(["silver table", "vintage", "rows"], rows),
        ]
        if rows
        else ["None."]
    )


def _flags(ctx: Context) -> list[str]:
    flags = [r for r in ctx.metadata.read("vintage_flags") if r["value"]]
    stale = sorted(r["vintage_month"] for r in flags if r["flag"] == STALE_REFRESH)
    revised = sorted(
        f"{r['vintage_month']} {r['table_name']}" for r in flags if r["flag"] == REVISED
    )
    return [
        f"- `stale_refresh`: {', '.join(stale) or 'none'} (spec: 2025-09 only). Data kept.",
        f"- `revised_after_publication`: {', '.join(revised) or 'none'}.",
    ]


def lag_report(
    ctx: Context,
    cfg: CalibrationConfig,
    lags: dict[str, dict[str, Any]],
    buffers: dict[str, int | None],
    cross: pl.DataFrame,
    p1: float,
    p0: float,
    vintages: list[str],
    stale: list[str],
) -> str:
    rows = [[e, f"{r['n']:,}", r["median"], r["p90"], r["p95"], r["B"]] for e, r in lags.items()]
    cdf_rows = [
        [a, *[f"{lags[e]['cdf'][a]:.3f}" if lags[e]["cdf"] else "" for e in lags]]
        for a in range(0, 13)
    ]
    return "\n".join(
        [
            "# Step 4 - Posting lag and maturity buffers (CHECKPOINT 3)",
            "",
            f"- **Run:** `{ctx.run_id}`; V = `{vintages[-1]}`; stale vintages: {', '.join(stale) or 'none'}",
            f"- Events: citations from {cfg.citations_from[:7]}, penalties from {cfg.penalties_from[:7]}, "
            f"dated at least {cfg.fully_observed_months} months before V (fully observed lags only).",
            "- Lag = calendar months from the event date to the cutoff of the first vintage that shows it. "
            f"B = smallest lag with c(B) >= {cfg.coverage}. Penalties are counted per revision group.",
            "",
            "## Lag by event type",
            "",
            md_table(["event", "n", "median", "p90", "p95", "B"], rows),
            "",
            "## Proposed buffers (spec measured: B_cit = 8, B_pen = 12)",
            "",
            f"- **B_cit = {buffers['B_cit']}** (covers every G+ subgroup; labels count all of them)",
            f"- **B_pen = {buffers['B_pen']}** (covers fines and payment denials)",
            "",
            "Proposal only: a human commits the values to `configs/labels.yaml` after this checkpoint.",
            "",
            "## Resulting last labelable month (spec §9.3)",
            "",
            md_table(
                ["label", "H", "buffer", "last_labeled_t", "last published t"],
                cal.last_labeled(vintages[-1], buffers, vintages, stale),
            ),
            "",
            "## CDF c(a) for a = 0..12 (full 0..24 in `04_lag_cdf.csv`)",
            "",
            md_table(["a", *lags], cdf_rows),
            "",
            "## Cross-check on the latest vintage (sanity view, not used to set B_pen)",
            "",
            f"Mature-period rates (standard surveys aged 18-36 months): p1 = P(fine | G+) = {p1:.3f}, "
            f"p0 = P(fine | no G+) = {p0:.3f} (spec: 0.735 / 0.028). obs/exp reaches about 1 late because recent "
            "cohorts are fined less often (spec §5 caveat).",
            "",
            md_table(
                ["age (months)", "surveys", "obs fine share", "exp fine share", "obs/exp"],
                [
                    [
                        r[0],
                        r[1],
                        f"{r[2]:.3f}",
                        f"{r[4]:.3f}",
                        f"{r[5]:.2f}" if r[5] is not None else "",
                    ]
                    for r in cross.iter_rows()
                ],
            ),
            "",
        ]
    )


def universe_report(
    ctx: Context, frames: dict[str, pl.DataFrame], proposal: pl.DataFrame, vintages: list[str]
) -> str:
    per_v = frames["facility_universe"].group_by("vintage_month").len().sort("vintage_month")
    presence = frames["facility_presence"]
    exited = presence.filter(pl.col("exited")).height
    led_c = frames["ledger_citations"]
    status = proposal.group_by("status").len().sort("status")
    show = proposal.select(
        "status",
        "table",
        "measure_code",
        "resident_type",
        "feature",
        "first_vintage",
        "last_vintage",
        "reason",
        "measure_description",
    )
    return "\n".join(
        [
            "# Step 5 - Facility universe and quality measures (CHECKPOINT 4)",
            "",
            f"- **Run:** `{ctx.run_id}`; V = `{vintages[-1]}`",
            "",
            "## Facility universe",
            "",
            f"- Facilities per vintage: {min(per_v['len'].to_list()):,} - {max(per_v['len'].to_list()):,} (spec: about 14.5k - 15.4k)",
            f"- Distinct facilities: {presence.height:,}; **exited** (last vintage before V): {exited:,}; "
            f"with gaps: {presence.filter(pl.col('gap_vintages') > 0).height:,}",
            f"- `gold/dim_facility_latest`: {frames['dim_facility_latest'].height:,} rows, "
            f"{frames['dim_facility_latest'].filter(pl.col('is_active')).height:,} active",
            "",
            md_table(["vintage", "facilities"], [list(r) for r in per_v.iter_rows()]),
            "",
            "## Ledgers (Step 3)",
            "",
            md_table(
                ["ledger", "rows", "notes"],
                [
                    [
                        "ledger_citations",
                        f"{led_c.height:,}",
                        f"{led_c.filter(pl.col('final_scope_severity').is_in(cal.G_PLUS)).height:,} G+; "
                        f"{led_c.filter(pl.col('severity_changed')).height:,} severity changed; {led_c.filter(pl.col('removed_by_cms')).height:,} removed by CMS",
                    ],
                    [
                        "ledger_penalties",
                        f"{frames['ledger_penalties'].height:,}",
                        f"{frames['ledger_penalties']['revision_group'].n_unique():,} revision groups",
                    ],
                    [
                        "ledger_surveys",
                        f"{frames['ledger_surveys'].height:,}",
                        f"{frames['ledger_surveys'].filter(~pl.col('in_inspection_dates')).height:,} known only from citations",
                    ],
                ],
            ),
            "",
            "## Proposed `configs/qm_measures.yaml` (approve or edit, then commit)",
            "",
            md_table(["status", "measures"], [list(r) for r in status.iter_rows()]),
            "",
            md_table(list(show.columns), [list(r) for r in show.iter_rows()]),
            "",
            "Bridges get `qm_def_post_2025_02` / `qm_def_post_2026_02` indicators in gold. **405 -> 480 is doubtful** "
            "(different definitions); decide here whether to bridge or drop it.",
            "",
        ]
    )
