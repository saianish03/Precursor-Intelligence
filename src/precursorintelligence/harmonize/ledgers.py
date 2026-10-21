"""Step 3 ledgers and the Step 5 universe tables (docs/DATA_PIPELINE_SPEC.md §4, §6 items 1-3).

A ledger records each real-world event once, with the vintages in which it was visible. Ledgers feed labels
and the posting-lag analysis, never features. Everything here is rebuilt over all vintages on every run.
To fit an 8 GB machine the vintages are folded in one at a time, oldest first, so only one row per distinct
event is held in memory (about 1-2 million citations, not the 24 million stacked rows).
"""

from __future__ import annotations

import io
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import polars as pl

from ..common.io.object_store import ObjectStore
from ..common.io.table_sink import TableSink

CITATION_KEY = ["ccn", "survey_date", "deficiency_prefix", "deficiency_tag"]
CITATION_FINAL = [
    "scope_severity",
    "deficiency_category",
    "is_standard",
    "is_complaint",
    "is_infection_control",
]
PENALTY_ATTRS = [
    "ccn",
    "penalty_date",
    "penalty_type",
    "fine_amount",
    "payment_denial_start_date",
    "payment_denial_length_days",
    "dup_seq",
]
SURVEY_KEY = ["ccn", "survey_date", "survey_type"]
# Inspection Dates lists complaint surveys only if cited, so surveys are completed from the citation flags.
CITATION_SURVEY_TYPES = {
    "is_standard": "HEALTH_STANDARD",
    "is_complaint": "HEALTH_COMPLAINT",
    "is_infection_control": "INFECTION_CONTROL",
}
IDENTITY = [
    "provider_name",
    "provider_address",
    "city",
    "state",
    "zip",
    "latitude",
    "longitude",
    "chain_id",
    "chain_name",
]

KEYS = {
    "ledger_citations": "silver/ledger_citations/part-0.parquet",
    "ledger_penalties": "silver/ledger_penalties/part-0.parquet",
    "ledger_surveys": "silver/ledger_surveys/part-0.parquet",
    "facility_presence": "silver/facility_presence/part-0.parquet",
    "facility_universe": "silver/facility_universe/part-0.parquet",
    "qm_inventory": "silver/qm_inventory/part-0.parquet",
    "dim_facility_latest": "gold/dim_facility_latest/part-0.parquet",
}


@dataclass
class SilverReader:
    """Reads single silver partitions, a few columns at a time."""

    store: ObjectStore
    vintages: list[str]  # published vintages, ascending

    def read(self, table: str, vintage: str, columns: list[str]) -> pl.DataFrame:
        with self.store.open_read(f"silver/{table}/vintage={vintage}/part-0.parquet") as f:
            return pl.read_parquet(f, columns=columns)

    def each(self, table: str, columns: list[str]) -> Iterator[tuple[str, pl.DataFrame]]:
        for v in self.vintages:
            yield v, self.read(table, v, columns)

    @property
    def latest(self) -> str:
        return self.vintages[-1]


def write_single(store: ObjectStore, key: str, frame: pl.DataFrame) -> None:
    buf = io.BytesIO()
    frame.write_parquet(buf, compression="zstd")
    store.put_bytes(key, buf.getvalue(), overwrite=True)


def fold(
    frames: Iterator[tuple[str, pl.DataFrame]],
    key: list[str],
    carry: Sequence[str] = (),
    carry_non_null: Sequence[str] = (),
    track: str | None = None,
) -> pl.DataFrame:
    """One row per key over all vintages: first/last vintage seen, number of vintages, the newest value of
    each `carry` column (as `final_<c>`), the newest non-null value of each `carry_non_null` column, and
    `_changed` when `track` ever differed between vintages."""
    state: pl.DataFrame | None = None
    for vintage, raw in frames:
        new = raw.unique(subset=key, keep="first", maintain_order=True).with_columns(
            pl.lit(True).alias("_in")
        )
        if state is None:
            state = new.select(
                *key,
                pl.lit(vintage).alias("first_seen_vintage"),
                pl.lit(vintage).alias("last_seen_vintage"),
                pl.lit(1, pl.Int64).alias("n_vintages_seen"),
                *[pl.col(c).alias(f"final_{c}") for c in [*carry, *carry_non_null]],
                pl.lit(False).alias("_changed"),
            )
            continue
        j = state.join(new, on=key, how="full", coalesce=True, nulls_equal=True)
        now = pl.col("_in").fill_null(False)
        old = pl.col("first_seen_vintage").is_not_null()
        changed = pl.col("_changed").fill_null(False)
        if track:
            changed = changed | (now & old & pl.col(track).ne_missing(pl.col(f"final_{track}")))
        state = j.select(
            *key,
            pl.col("first_seen_vintage").fill_null(pl.lit(vintage)),
            pl.when(now)
            .then(pl.lit(vintage))
            .otherwise(pl.col("last_seen_vintage"))
            .alias("last_seen_vintage"),
            (pl.col("n_vintages_seen").fill_null(0) + now.cast(pl.Int64)).alias("n_vintages_seen"),
            *[
                pl.when(now).then(pl.col(c)).otherwise(pl.col(f"final_{c}")).alias(f"final_{c}")
                for c in carry
            ],
            *[
                pl.when(now & pl.col(c).is_not_null())
                .then(pl.col(c))
                .otherwise(pl.col(f"final_{c}"))
                .alias(f"final_{c}")
                for c in carry_non_null
            ],
            changed.alias("_changed"),
        )
    if state is None:
        raise ValueError("no vintages to fold")
    return state


def _next_vintage(vintages: list[str]) -> dict[str, str]:
    return dict(zip(vintages, vintages[1:], strict=False))


def ledger_citations(s: SilverReader) -> pl.DataFrame:
    """§4.1. removed_by_cms: the citation vanished in the next published vintage while the facility and at
    least one other citation of the same survey are still there (aging out removes a whole survey at once)."""
    led = fold(
        s.each("citations_by_vintage", CITATION_KEY + CITATION_FINAL),
        CITATION_KEY,
        CITATION_FINAL,
        track="scope_severity",
    )
    led = led.rename(
        {"_changed": "severity_changed", "final_deficiency_category": "final_category"}
    )
    nxt = _next_vintage(s.vintages)
    cand = led.filter(pl.col("last_seen_vintage") < s.latest).with_columns(
        pl.col("last_seen_vintage").replace_strict(nxt, return_dtype=pl.String).alias("_next")
    )
    removed = []
    for w in s.vintages[1:]:
        c = cand.filter(pl.col("_next") == w)
        if c.is_empty():
            continue
        surveys = s.read("citations_by_vintage", w, ["ccn", "survey_date"]).unique()
        present = s.read("provider_by_vintage", w, ["ccn"])
        removed.append(
            c.join(present, on="ccn", how="semi")
            .join(surveys, on=["ccn", "survey_date"], how="semi")
            .select(CITATION_KEY)
        )
    gone = (
        pl.concat(removed).with_columns(pl.lit(True).alias("removed_by_cms")) if removed else None
    )
    if gone is None:
        led = led.with_columns(pl.lit(False).alias("removed_by_cms"))
    else:
        led = led.join(gone, on=CITATION_KEY, how="left", nulls_equal=True).with_columns(
            pl.col("removed_by_cms").fill_null(False)
        )
    return led.sort(CITATION_KEY)


def ledger_penalties(s: SilverReader) -> pl.DataFrame:
    """§4.2. One row per penalty_event_key (5 fields + dup_seq) with the latest non-null fine_id.
    revision_group = (ccn, penalty_date, penalty_type) collects revised amounts of one penalty."""
    led = fold(
        s.each("penalties_by_vintage", ["penalty_event_key", *PENALTY_ATTRS, "fine_id"]),
        ["penalty_event_key"],
        PENALTY_ATTRS,
        ["fine_id"],
    ).drop("_changed")
    led = led.rename({f"final_{c}": c for c in [*PENALTY_ATTRS, "fine_id"]})
    group = pl.concat_str(
        [pl.col("ccn"), pl.col("penalty_date").cast(pl.String), pl.col("penalty_type")],
        separator="|",
    )
    return led.with_columns(group.alias("revision_group")).sort("penalty_event_key")


def ledger_surveys(s: SilverReader, citations: pl.DataFrame) -> pl.DataFrame:
    """§4.3. Inspection Dates rows plus the surveys known only from citations, typed from the citation flags."""
    listed = fold(s.each("surveys_by_vintage", SURVEY_KEY), SURVEY_KEY).drop(
        "_changed", "n_vintages_seen"
    )
    from_citations = pl.concat(
        [
            citations.filter(pl.col(f"final_{flag}"))
            .group_by("ccn", "survey_date")
            .agg(pl.col("first_seen_vintage").min(), pl.col("last_seen_vintage").max())
            .with_columns(pl.lit(kind).alias("survey_type"))
            for flag, kind in CITATION_SURVEY_TYPES.items()
        ]
    ).select(*SURVEY_KEY, "first_seen_vintage", "last_seen_vintage")
    both = pl.concat(
        [
            listed.with_columns(pl.lit(True).alias("in_inspection_dates")),
            from_citations.with_columns(pl.lit(False).alias("in_inspection_dates")),
        ]
    )
    return (
        both.group_by(SURVEY_KEY)
        .agg(
            pl.col("first_seen_vintage").min(),
            pl.col("last_seen_vintage").max(),
            pl.col("in_inspection_dates").any(),
        )
        .sort(SURVEY_KEY)
    )


def facility_presence(s: SilverReader) -> pl.DataFrame:
    """§4.4. gap_vintages = published vintages between first and last in which the facility is missing."""
    p = fold(s.each("provider_by_vintage", ["ccn"]), ["ccn"]).drop("_changed")
    idx = {v: i for i, v in enumerate(s.vintages)}
    span = (
        pl.col("last_seen_vintage").replace_strict(idx, return_dtype=pl.Int64)
        - pl.col("first_seen_vintage").replace_strict(idx, return_dtype=pl.Int64)
        + 1
    )
    return p.select(
        "ccn",
        pl.col("first_seen_vintage").alias("first_vintage"),
        pl.col("last_seen_vintage").alias("last_vintage"),
        pl.col("n_vintages_seen").alias("n_vintages"),
        (span - pl.col("n_vintages_seen")).alias("gap_vintages"),
        (pl.col("last_seen_vintage") < s.latest).alias("exited"),
    ).sort("ccn")


def facility_universe(s: SilverReader) -> pl.DataFrame:
    """§6.1: (ccn, vintage_month) for every facility in every published vintage."""
    return pl.concat(
        [
            f.with_columns(pl.lit(v).alias("vintage_month"))
            for v, f in s.each("provider_by_vintage", ["ccn"])
        ]
    ).sort("vintage_month", "ccn")


def dim_facility_latest(s: SilverReader) -> pl.DataFrame:
    """§6.2: name, address and chain from each facility's last vintage. Product display only, never a feature."""
    d = fold(s.each("provider_identity_by_vintage", ["ccn", *IDENTITY]), ["ccn"], IDENTITY).drop(
        "_changed", "first_seen_vintage", "n_vintages_seen"
    )
    return (
        d.rename({f"final_{c}": c for c in IDENTITY} | {"last_seen_vintage": "last_vintage"})
        .with_columns((pl.col("last_vintage") == s.latest).alias("is_active"))
        .sort("ccn")
    )


QM_COLS = ["measure_code", "resident_type", "measure_description"]


def qm_presence(s: SilverReader) -> pl.DataFrame:
    """One row per (table, measure_code, resident_type, measure_description, vintage_month) present."""
    return pl.concat(
        [
            f.unique().with_columns(pl.lit(table).alias("table"), pl.lit(v).alias("vintage_month"))
            for table in ("mds_qm_by_vintage", "claims_qm_by_vintage")
            for v, f in s.each(table, QM_COLS)
        ]
    )


def qm_inventory(presence: pl.DataFrame) -> pl.DataFrame:
    """§6.3: every (table, measure_code, resident_type, measure_description) with its first and last vintage."""
    return (
        presence.group_by("table", *QM_COLS)
        .agg(
            pl.col("vintage_month").min().alias("first_vintage"),
            pl.col("vintage_month").max().alias("last_vintage"),
            pl.col("vintage_month").n_unique().alias("n_vintages"),
        )
        .sort("table", *QM_COLS[:2], "first_vintage")
    )


def published_vintages(sink: TableSink) -> list[str]:
    """Vintages with a provider partition, ascending: the published months (gaps are never-published months)."""
    return sorted(p["vintage"] for p in sink.list_partitions("provider_by_vintage"))
