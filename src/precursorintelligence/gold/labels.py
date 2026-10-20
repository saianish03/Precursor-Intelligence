"""Step 8 (spec §9): labels per spine row, from the event ledgers.

For each label with horizon H and buffer B, the window is W_H = (cutoff_date, cutoff_date + H months],
and the status is decided in this order (spec §9.2):
1. ``immature``: t + H + B > V. The window is not fully observed; y is null even if an event is visible.
2. ``positive``: an event in W_H (G+ citation not removed by CMS, or any penalty revision group).
3. ``censored``: no event, and the facility's last vintage < t + H + B (t + H when not strict).
4. ``negative``: everything else.
y is 1 for positive, 0 for negative and null otherwise. Only ledgers are read, never feature tables.
"""

from __future__ import annotations

import polars as pl

from ..common.config import LabelsConfig
from . import GoldCheckError


def _plus_months(months: int) -> pl.Expr:
    return pl.col("cutoff_date").dt.offset_by(f"{months}mo").dt.strftime("%Y-%m")


def _events(
    cfg: LabelsConfig, ledger_citations: pl.DataFrame, ledger_penalties: pl.DataFrame
) -> dict[str, pl.DataFrame]:
    gplus = ledger_citations.filter(
        pl.col("final_scope_severity").is_in(cfg.gplus_severities), ~pl.col("removed_by_cms")
    )
    return {
        "citations": gplus.select("ccn", event_date=pl.col("survey_date")),
        # The penalty event is the revision_group (ccn|penalty_date|penalty_type), never the row key:
        # revised amounts and identical fines are one event (SILVER_CONTRACT §8).
        "penalties": ledger_penalties.unique("revision_group").select(
            "ccn", event_date=pl.col("penalty_date")
        ),
    }


def _has_event(spine: pl.DataFrame, events: pl.DataFrame, horizon: int) -> pl.DataFrame:
    """(ccn, t) rows with at least one event strictly after the cutoff and within H months."""
    window_end = pl.col("cutoff_date").dt.offset_by(f"{horizon}mo")
    return (
        spine.select("ccn", "t", "cutoff_date")
        .join(events, on="ccn")
        .filter(pl.col("event_date") > pl.col("cutoff_date"), pl.col("event_date") <= window_end)
        .select("ccn", "t")
        .unique()
        .with_columns(hit=pl.lit(True))
    )


def build_labels(
    spine: pl.DataFrame,
    *,
    ledger_citations: pl.DataFrame,
    ledger_penalties: pl.DataFrame,
    facility_presence: pl.DataFrame,
    cfg: LabelsConfig,
    v: str,
) -> pl.DataFrame:
    """One row per spine row: ``ccn, t`` plus ``y_<label>, status_<label>`` for each configured label.

    ``v`` is the latest vintage whose event tables are fresh: pass the last non-stale vintage when V is
    a ``stale_refresh`` vintage (spec §9.2), so maturity is not overstated.
    """
    out = spine.join(
        facility_presence.select("ccn", "last_vintage"), on="ccn", how="left", maintain_order="left"
    )
    if out["last_vintage"].is_null().any():
        raise GoldCheckError("labels: spine CCNs missing from facility_presence")
    events = _events(cfg, ledger_citations, ledger_penalties)
    label_cols: list[str] = []
    for name, spec in cfg.labels.items():
        h, b = spec.horizon_months, cfg.buffer(name)
        hits = _has_event(spine, events[spec.source], h)
        observable_until = _plus_months(h + b if cfg.censor_requires_buffer else h)
        status = (
            pl.when(_plus_months(h + b) > v)
            .then(pl.lit("immature"))
            .when(pl.col("hit"))
            .then(pl.lit("positive"))
            .when(pl.col("last_vintage") < observable_until)
            .then(pl.lit("censored"))
            .otherwise(pl.lit("negative"))
        )
        out = (
            out.join(hits, on=["ccn", "t"], how="left", maintain_order="left")
            .with_columns(pl.col("hit").fill_null(False))
            .with_columns(status.alias(f"status_{name}"))
            .with_columns(
                pl.when(pl.col(f"status_{name}") == "positive")
                .then(1)
                .when(pl.col(f"status_{name}") == "negative")
                .then(0)
                .cast(pl.Int8)
                .alias(f"y_{name}")
            )
            .drop("hit")
        )
        label_cols += [f"y_{name}", f"status_{name}"]
    labels = out.select("ccn", "t", *label_cols)
    if labels.height != spine.height or labels.select("ccn", "t").is_duplicated().any():
        raise GoldCheckError("labels: not exactly one row per spine row")
    return labels
