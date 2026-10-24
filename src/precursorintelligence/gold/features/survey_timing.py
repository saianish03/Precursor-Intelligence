"""``features_survey_timing`` (spec §8.2): when the facility was last inspected, from ``surveys_by_vintage`` at t.

Health survey types only. Inspection Dates lists a complaint survey only if it produced a citation (§4.3),
hence the ``*_cited`` names.
"""

from __future__ import annotations

import polars as pl

from ...common.config import FeaturesConfig
from .common import attach, days_since, fill_zero, within
from .reader import FeatureReader

STANDARD, COMPLAINT, INFECTION = "HEALTH_STANDARD", "HEALTH_COMPLAINT", "INFECTION_CONTROL"
COLUMNS = [
    "days_since_last_std", "days_to_std_due", "std_overdue", "n_std_36m",
    "n_complaint_surveys_cited_12m", "n_complaint_surveys_cited_24m", "n_ic_surveys_12m",
    "days_since_last_complaint_survey_cited", "no_std_on_record",
]  # fmt: skip
COUNTS = [
    "n_std_36m", "n_complaint_surveys_cited_12m", "n_complaint_surveys_cited_24m",
    "n_ic_surveys_12m",
]  # fmt: skip


def build(reader: FeatureReader, rows: pl.DataFrame, t: str, cfg: FeaturesConfig) -> pl.DataFrame:
    surveys = reader.read("surveys_by_vintage", t)
    assert surveys is not None  # t is a spine month, so it was published
    s = surveys.filter(
        pl.col("survey_type").is_in([STANDARD, COMPLAINT, INFECTION]),
        pl.col("survey_date") <= pl.col("cutoff_date"),
    )
    kind = pl.col("survey_type")
    date = pl.col("survey_date")
    per_ccn = s.group_by("ccn").agg(
        last_std=date.filter(kind == STANDARD).max(),
        last_complaint=date.filter(kind == COMPLAINT).max(),
        n_std_36m=date.filter(kind == STANDARD, within("survey_date", 36)).n_unique(),
        n_complaint_surveys_cited_12m=date.filter(
            kind == COMPLAINT, within("survey_date", 12)
        ).n_unique(),
        n_complaint_surveys_cited_24m=date.filter(
            kind == COMPLAINT, within("survey_date", 24)
        ).n_unique(),
        n_ic_surveys_12m=date.filter(kind == INFECTION, within("survey_date", 12)).n_unique(),
    )
    since_std = days_since(pl.col("last_std"))
    return (
        attach(rows, per_ccn)
        .with_columns(
            *fill_zero(*COUNTS),
            days_since_last_std=since_std,
            days_to_std_due=cfg.std_survey_due_days - since_std,
            std_overdue=since_std > cfg.std_survey_due_days,
            days_since_last_complaint_survey_cited=days_since(pl.col("last_complaint")),
            no_std_on_record=pl.col("last_std").is_null(),
        )
        .with_columns(pl.col(c).cast(pl.Int64) for c in COUNTS)
        .select("ccn", "t", *COLUMNS)
    )
