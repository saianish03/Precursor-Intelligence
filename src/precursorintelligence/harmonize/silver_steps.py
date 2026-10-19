"""Silver Steps 3-5 after the stacked tables: ledgers, universe tables, and (with --calibrate) the
Checkpoint 3 and 4 proposals (docs/DATA_PIPELINE_SPEC.md §4-§6)."""

from __future__ import annotations

import logging
from typing import Any

import polars as pl

from ..ingestion.context import Context
from . import calibrate as cal
from . import ledgers as led
from . import silver_report as rep
from .flags import STALE_REFRESH

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


def run_calibration(
    ctx: Context, vintages: list[str], built: dict[str, Any], cfg: cal.CalibrationConfig
) -> list[str]:
    """Checkpoint 3 (posting lag -> buffers) and Checkpoint 4 (universe and QM selection). Proposals only."""
    frames = built["frames"]
    s = led.SilverReader(ctx.object_store, vintages)
    stale = sorted(
        r["vintage_month"]
        for r in ctx.metadata.read("vintage_flags", {"flag": STALE_REFRESH})
        if r["value"]
    )

    lags = cal.lag_analysis(frames["ledger_citations"], frames["ledger_penalties"], s.latest, cfg)
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
            "04_lag_analysis.md": rep.lag_report(
                ctx, cfg, lags, buffers, cross, p1, p0, vintages, stale
            ).encode("utf-8"),
            "04_lag_cdf.csv": cdf.write_csv().encode("utf-8"),
        },
    )

    proposal = cal.qm_proposal(built["qm_presence"], vintages, cfg)
    keys += _write(
        ctx,
        "step5",
        {
            "05_universe_and_qm.md": rep.universe_report(ctx, frames, proposal, vintages).encode(
                "utf-8"
            ),
            "05_qm_inventory.csv": frames["qm_inventory"].write_csv().encode("utf-8"),
            "05_qm_measures.proposed.csv": proposal.write_csv().encode("utf-8"),
        },
    )
    return keys
