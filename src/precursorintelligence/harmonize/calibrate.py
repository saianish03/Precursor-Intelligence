"""Silver calibration tasks (backfill or on demand; docs/DATA_PIPELINE_SPEC.md §5 and §6 item 3).

They only write proposals to reports/. `configs/labels.yaml` (buffers) and `configs/qm_measures.yaml` change
only through a human commit after Checkpoints 3 and 4. Their settings live in `configs/calibration.yaml`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import polars as pl

from ..common.config import ConfigError
from .ledgers import SilverReader

G_PLUS = list("GHIJKL")
HORIZONS = {"y_g12": (12, "B_cit"), "y_g6": (6, "B_cit"), "y_pen12": (12, "B_pen")}


@dataclass(frozen=True)
class CalibrationConfig:
    """configs/calibration.yaml: what the Checkpoint 3 and 4 proposals are computed with."""

    coverage: float
    max_lag: int
    citations_from: str
    penalties_from: str
    fully_observed_months: int
    stable_from: str
    bridges: dict[tuple[str, str], tuple[str, str]]  # (old code, new code) -> (feature, note)
    drop: dict[str, str]  # code -> reason

    @classmethod
    def from_yaml(cls, raw: dict[str, Any]) -> CalibrationConfig:
        try:
            lag, qm = raw["lag"], raw["qm"]
            return cls(
                coverage=float(lag["coverage"]),
                max_lag=int(lag["max_lag_months"]),
                citations_from=str(lag["citations_from"]),
                penalties_from=str(lag["penalties_from"]),
                fully_observed_months=int(lag["fully_observed_months"]),
                stable_from=str(qm["stable_from"]),
                bridges={
                    (str(b["old"]), str(b["new"])): (b["feature"], b["note"]) for b in qm["bridges"]
                },
                drop={str(code): why for code, why in qm["drop"].items()},
            )
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise ConfigError(f"calibration.yaml: missing or bad setting {exc}") from None


def month_index(col: pl.Expr) -> pl.Expr:
    """Calendar months since year 0 of a date column (same scale as vintage_index)."""
    return col.dt.year().cast(pl.Int64) * 12 + col.dt.month().cast(pl.Int64)


def vintage_index(col: pl.Expr) -> pl.Expr:
    return col.str.slice(0, 4).cast(pl.Int64) * 12 + col.str.slice(5, 2).cast(pl.Int64)


def _shift(vintage: str, months: int) -> str:
    y, m = int(vintage[:4]), int(vintage[5:7])
    n = y * 12 + (m - 1) + months
    return f"{n // 12:04d}-{n % 12 + 1:02d}"


def lag_table(events: pl.DataFrame, cfg: CalibrationConfig) -> dict[str, Any]:
    """events: one row per event with `lag_months`. Returns n, median, p90, p95, B and the CDF c(0..max_lag)."""
    n = events.height
    if n == 0:
        return {"n": 0, "median": None, "p90": None, "p95": None, "B": None, "cdf": []}
    lag = pl.col("lag_months")
    row = events.select(
        *[(lag <= a).mean().alias(f"c{a}") for a in range(cfg.max_lag + 1)],
        *[lag.quantile(p, interpolation="higher").alias(f"q{p}") for p in (0.5, 0.9, 0.95)],
    ).row(0)
    cdf = [float(c) for c in row[: cfg.max_lag + 1]]
    median, p90, p95 = (int(q) for q in row[cfg.max_lag + 1 :])
    b = next((a for a, c in enumerate(cdf) if c >= cfg.coverage), None)
    return {"n": n, "median": median, "p90": p90, "p95": p95, "B": b, "cdf": cdf}


def lag_analysis(
    citations: pl.DataFrame, penalties: pl.DataFrame, latest: str, cfg: CalibrationConfig
) -> dict[str, dict[str, Any]]:
    """Spec §5. Lag = calendar months from the event date to the cutoff of the vintage it first appeared in.
    Only events old enough at V to have a fully observed lag are used."""
    cutoff_idx = vintage_index(pl.lit(_shift(latest, -cfg.fully_observed_months)))
    cit = (
        citations.filter(pl.col("survey_date") >= pl.lit(cfg.citations_from).str.to_date())
        .with_columns(
            (
                vintage_index(pl.col("first_seen_vintage")) - month_index(pl.col("survey_date"))
            ).alias("lag_months")
        )
        .filter(month_index(pl.col("survey_date")) <= cutoff_idx)
    )
    gplus = cit.filter(pl.col("final_scope_severity").is_in(G_PLUS) & ~pl.col("removed_by_cms"))
    # The penalty event is the revision group: CMS revises amounts, and a key with the amount would date a
    # fine from its last amount (spec §5).
    groups = (
        penalties.filter(pl.col("penalty_date") >= pl.lit(cfg.penalties_from).str.to_date())
        .group_by("revision_group")
        .agg(
            pl.col("penalty_type").first(),
            pl.col("penalty_date").first(),
            pl.col("first_seen_vintage").min(),
        )
        .with_columns(
            (
                vintage_index(pl.col("first_seen_vintage")) - month_index(pl.col("penalty_date"))
            ).alias("lag_months")
        )
        .filter(month_index(pl.col("penalty_date")) <= cutoff_idx)
    )
    return {
        "G+ citations": lag_table(gplus, cfg),
        "G+ citations, complaint-linked": lag_table(
            gplus.filter(pl.col("final_is_complaint")), cfg
        ),
        "G+ citations, standard survey": lag_table(gplus.filter(pl.col("final_is_standard")), cfg),
        "All citations": lag_table(cit, cfg),
        "Fines (revision group)": lag_table(groups.filter(pl.col("penalty_type") == "Fine"), cfg),
        "Payment denials": lag_table(
            groups.filter(pl.col("penalty_type") == "Payment Denial"), cfg
        ),
    }


def proposed_buffers(lags: dict[str, dict[str, Any]]) -> dict[str, int | None]:
    """B_cit covers every G+ subgroup (the labels count all of them); B_pen covers fines and denials."""

    def worst(names: list[str]) -> int | None:
        values = [lags[n]["B"] for n in names if lags[n]["B"] is not None]
        return max(values) if values else None

    return {
        "B_cit": worst(
            ["G+ citations", "G+ citations, complaint-linked", "G+ citations, standard survey"]
        ),
        "B_pen": worst(["Fines (revision group)", "Payment denials"]),
    }


def last_labeled(
    latest: str, buffers: dict[str, int | None], vintages: list[str], stale: list[str]
) -> list[list[Any]]:
    """Spec §9.3: last_labeled_t = V - (H + B); a stale V counts as its last non-stale vintage."""
    v = next((x for x in reversed(vintages) if x not in stale), latest)
    out = []
    for label, (h, b_name) in HORIZONS.items():
        b = buffers[b_name]
        t = _shift(v, -(h + b)) if b is not None else None
        real = max((x for x in vintages if t and x <= t), default=None)
        out.append([label, h, f"{b_name} = {b}", t, real])
    return out


def fine_crosscheck(s: SilverReader) -> tuple[pl.DataFrame, float, float]:
    """Spec §5 cross-check on the latest vintage: per survey-age bucket, observed share of standard surveys
    with a same-day penalty vs the share expected from mature-period rates p1 (G+) and p0 (no G+). A sanity
    view only: recent cohorts have lower fine rates, so obs/exp lags behind B_pen (spec §5 caveat)."""
    v = s.latest
    surveys = s.read("surveys_by_vintage", v, ["ccn", "survey_date", "survey_type"]).filter(
        pl.col("survey_type") == "HEALTH_STANDARD"
    )
    cit = s.read("citations_by_vintage", v, ["ccn", "survey_date", "scope_severity"])
    g = (
        cit.filter(pl.col("scope_severity").is_in(G_PLUS))
        .select("ccn", "survey_date")
        .unique()
        .with_columns(pl.lit(True).alias("g"))
    )
    pen = (
        s.read("penalties_by_vintage", v, ["ccn", "penalty_date"])
        .unique()
        .rename({"penalty_date": "survey_date"})
        .with_columns(pl.lit(True).alias("fine"))
    )
    vi = vintage_index(pl.lit(v))
    df = (
        surveys.join(g, on=["ccn", "survey_date"], how="left")
        .join(pen, on=["ccn", "survey_date"], how="left")
        .with_columns(
            pl.col("g").fill_null(False),
            pl.col("fine").fill_null(False),
            (vi - month_index(pl.col("survey_date"))).alias("age"),
        )
    )
    mature = df.filter(pl.col("age").is_between(18, 36))
    p1, p0 = (
        float("nan") if r is None else float(r)  # no mature surveys: unmeasured, not 0
        for r in mature.select(
            pl.col("fine").filter(pl.col("g")).mean().alias("p1"),
            pl.col("fine").filter(~pl.col("g")).mean().alias("p0"),
        ).row(0)
    )
    table = (
        df.filter(pl.col("age").is_between(0, 18))
        .group_by("age")
        .agg(
            pl.len().alias("surveys"),
            pl.col("fine").mean().alias("obs_fine_share"),
            pl.col("g").mean().alias("g_share"),
        )
        .with_columns(
            (pl.col("g_share") * p1 + (1 - pl.col("g_share")) * p0).alias("exp_fine_share")
        )
        .with_columns((pl.col("obs_fine_share") / pl.col("exp_fine_share")).alias("obs_over_exp"))
        .sort("age")
    )
    return table, p1, p0


def qm_proposal(
    inventory: pl.DataFrame, vintages: list[str], cfg: CalibrationConfig
) -> pl.DataFrame:
    """Spec §6 item 3: stable (every published vintage stable_from -> V), bridged, dropped (the rest).
    `inventory` needs one row per (table, code, resident type, vintage): see qm_presence."""
    latest = vintages[-1]
    needed = sum(v >= cfg.stable_from for v in vintages)
    by_code = inventory.group_by("table", "measure_code", "resident_type").agg(
        pl.col("vintage_month").min().alias("first_vintage"),
        pl.col("vintage_month").max().alias("last_vintage"),
        pl.col("vintage_month")
        .filter(pl.col("vintage_month") >= cfg.stable_from)
        .n_unique()
        .alias("n_from_stable"),
        pl.col("measure_description").sort_by("vintage_month").last(),
    )
    bridged = {
        c: (name, why) for (old, new), (name, why) in cfg.bridges.items() for c in (old, new)
    }
    rows = []
    for r in by_code.sort("table", "measure_code").iter_rows(named=True):
        code = r["measure_code"]
        if code in cfg.drop:
            status, feature, why = "dropped", None, cfg.drop[code]
        elif code in bridged:
            status, (feature, why) = "bridged", bridged[code]
        elif r["n_from_stable"] == needed and r["last_vintage"] == latest:
            status, feature, why = (
                "stable",
                None,
                f"present {r['first_vintage']} -> {r['last_vintage']}",
            )
        else:
            status, feature, why = (
                "dropped",
                None,
                f"not stable: {r['first_vintage']} -> {r['last_vintage']}",
            )
        rows.append(r | {"status": status, "feature": feature, "reason": why})
    return pl.DataFrame(rows).sort("status", "table", "measure_code")
