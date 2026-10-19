"""Silver Steps 3-5 after the stacked tables: ledgers, universe tables, and (with --calibrate) the
Checkpoint 3 and 4 proposals (docs/DATA_PIPELINE_SPEC.md §4-§6)."""

from __future__ import annotations

import logging
from typing import Any

import polars as pl

from ..ingestion.context import Context
from . import calibrate as cal
from . import ledgers as led
from .flags import STALE_REFRESH
from .silver_report import md_table

log = logging.getLogger(__name__)


def build_ledgers_and_universe(ctx: Context, vintages: list[str]) -> dict[str, Any]:
    """Rebuild every all-vintage table and return the frames the calibration needs plus row counts."""
    s = led.SilverReader(ctx.object_store, vintages)
    frames: dict[str, pl.DataFrame] = {}
    frames["ledger_citations"] = led.ledger_citations(s)
    frames["ledger_penalties"] = led.ledger_penalties(s)
    frames["ledger_surveys"] = led.ledger_surveys(s, frames["ledger_citations"])
    frames["facility_presence"] = led.facility_presence(s)
    frames["facility_universe"] = led.facility_universe(s)
    frames["dim_facility_latest"] = led.dim_facility_latest(s)
    presence = led.qm_presence(s)
    frames["qm_inventory"] = led.qm_inventory(presence)
    for name, frame in frames.items():
        led.write_single(ctx.object_store, led.KEYS[name], frame)
        log.info("silver wrote %s (%d rows)", name, frame.height)
    return {
        "frames": frames,
        "qm_presence": presence,
        "counts": {n: f.height for n, f in frames.items()},
    }


def _write(ctx: Context, step: str, files: dict[str, bytes]) -> list[str]:
    keys = []
    for folder in (f"reports/{step}/run={ctx.run_id}", f"reports/{step}/latest"):
        for name, data in files.items():
            key = f"{folder}/{name}"
            ctx.object_store.put_bytes(key, data, overwrite=True)
            keys.append(key)
    return keys


def run_calibration(ctx: Context, vintages: list[str], built: dict[str, Any]) -> list[str]:
    """Checkpoint 3 (posting lag -> buffers) and Checkpoint 4 (universe and QM selection). Proposals only."""
    frames = built["frames"]
    s = led.SilverReader(ctx.object_store, vintages)
    stale = sorted(
        r["vintage_month"]
        for r in ctx.metadata.read("vintage_flags", {"flag": STALE_REFRESH})
        if r["value"]
    )

    lags = cal.lag_analysis(frames["ledger_citations"], frames["ledger_penalties"], s.latest)
    buffers = cal.proposed_buffers(lags)
    cross, p1, p0 = cal.fine_crosscheck(s)
    cdf = pl.DataFrame(
        [
            {"event": e, "lag_months": a, "c": c}
            for e, r in lags.items()
            for a, c in enumerate(r["cdf"])
        ]
    )
    keys = _write(
        ctx,
        "step4",
        {
            "04_lag_analysis.md": _lag_md(
                ctx, lags, buffers, cross, p1, p0, vintages, stale
            ).encode("utf-8"),
            "04_lag_cdf.csv": cdf.write_csv().encode("utf-8"),
        },
    )

    proposal = cal.qm_proposal(built["qm_presence"], vintages)
    keys += _write(
        ctx,
        "step5",
        {
            "05_universe_and_qm.md": _universe_md(ctx, frames, proposal, vintages).encode("utf-8"),
            "05_qm_inventory.csv": frames["qm_inventory"].write_csv().encode("utf-8"),
            "05_qm_measures.proposed.csv": proposal.write_csv().encode("utf-8"),
        },
    )
    return keys


def _lag_md(
    ctx: Context,
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
            f"- Events: citations from {cal.CITATIONS_FROM[:7]}, penalties from {cal.PENALTIES_FROM[:7]}, "
            f"dated at least {cal.FULLY_OBSERVED_MONTHS} months before V (fully observed lags only).",
            "- Lag = calendar months from the event date to the cutoff of the first vintage that shows it. "
            f"B = smallest lag with c(B) >= {cal.COVERAGE}. Penalties are counted per revision group.",
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


def _universe_md(
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
