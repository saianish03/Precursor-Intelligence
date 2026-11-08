"""``features_fire_safety`` (docs/ALL_TABLES_CONTRACT.md): life-safety-code and emergency-preparedness
citations, from ``fire_citations_by_vintage`` at t (``survey_date <= cutoff``).

Fire citations are a sign of how well a building is run, which tends to go with how well care is run. The
file lists a fire survey only if it produced a citation, so "last fire survey" means the last cited one.
"""

from __future__ import annotations

import polars as pl

from ...common.config import FeaturesConfig
from .citations import not_corrected_values
from .common import CUTOFF, SEVERITIES, attach, days_since, fill_zero, gplus_min_ordinal, within
from .reader import FeatureReader

SEVERE_FROM = "F"  # F = widespread potential for harm; G+ comes from the labels config
EMERGENCY_PREPAREDNESS = "E"  # deficiency_prefix: K = life safety code, E = emergency preparedness
COLUMNS = [
    "n_fire_cit_12m", "n_fire_cit_36m", "n_fire_cit_fplus_36m", "n_fire_cit_gplus_36m",
    "n_fire_complaint_cit_36m", "n_fire_ep_cit_36m", "n_fire_surveys_36m",
    "n_fire_cit_last_survey", "max_fire_sev_last_survey", "n_fire_repeat_tags_last_survey",
    "share_fire_uncorrected_36m", "avg_fire_days_to_correct_36m", "days_since_last_fire_cit",
]  # fmt: skip
COUNTS = [
    "n_fire_cit_12m", "n_fire_cit_36m", "n_fire_cit_fplus_36m", "n_fire_cit_gplus_36m",
    "n_fire_complaint_cit_36m", "n_fire_ep_cit_36m", "n_fire_surveys_36m", "n_fire_cit_last_survey",
]  # fmt: skip


def build(reader: FeatureReader, rows: pl.DataFrame, t: str, cfg: FeaturesConfig) -> pl.DataFrame:
    # Optional: CMS left the fire-safety file out of some releases (2026-08). That month is unknown, not 0.
    fire = reader.read("fire_citations_by_vintage", t, optional=True)
    if fire is None:
        return rows.select("ccn", "t").with_columns(
            *(pl.lit(None, pl.Int64).alias(c) for c in COUNTS),
            *(pl.lit(None, pl.Float64).alias(c) for c in COLUMNS if c not in COUNTS),
        ).select("ccn", "t", *COLUMNS)
    ordinal = {s: i + 1 for i, s in enumerate(SEVERITIES)}
    c = fire.filter(pl.col("survey_date") <= CUTOFF).with_columns(
        ordinal=pl.col("scope_severity").replace_strict(ordinal, return_dtype=pl.Int64)
    )
    w12, w36 = within("survey_date", 12), within("survey_date", 36)
    sev = pl.col("ordinal")

    def count(cond: pl.Expr) -> pl.Expr:
        return cond.fill_null(False).sum()

    corrected_by_cutoff = pl.col("correction_date").is_not_null() & (
        pl.col("correction_date") <= CUTOFF
    )
    days_to_correct = (pl.col("correction_date") - pl.col("survey_date")).dt.total_days()
    per_ccn = (
        c.group_by("ccn")
        .agg(
            n_fire_cit_12m=count(w12),
            n_fire_cit_36m=count(w36),
            n_fire_cit_fplus_36m=count(w36 & (sev >= ordinal[SEVERE_FROM])),
            n_fire_cit_gplus_36m=count(w36 & (sev >= gplus_min_ordinal(cfg))),
            n_fire_complaint_cit_36m=count(w36 & pl.col("is_complaint")),
            n_fire_ep_cit_36m=count(w36 & (pl.col("deficiency_prefix") == EMERGENCY_PREPAREDNESS)),
            n_fire_surveys_36m=pl.col("survey_date").filter(w36).n_unique(),
            n_uncorrected_36m=count(
                w36 & pl.col("deficiency_corrected").is_in(not_corrected_values())
            ),
            avg_fire_days_to_correct_36m=days_to_correct.filter(w36 & corrected_by_cutoff)
            .mean()
            .cast(pl.Float64),
            dates=pl.col("survey_date").unique().sort(descending=True).head(2),
        )
        .with_columns(
            last=pl.col("dates").list.get(0, null_on_oob=True),
            prev=pl.col("dates").list.get(1, null_on_oob=True),
        )
    )

    on_last_two = (
        c.join(per_ccn.select("ccn", "last", "prev"), on="ccn")
        .group_by("ccn")
        .agg(
            n_fire_cit_last_survey=count(pl.col("survey_date") == pl.col("last")),
            max_last=sev.filter(pl.col("survey_date") == pl.col("last")).max(),
            tags_last=pl.col("deficiency_tag")
            .filter(pl.col("survey_date") == pl.col("last"))
            .unique(),
            tags_prev=pl.col("deficiency_tag")
            .filter(pl.col("survey_date") == pl.col("prev"))
            .unique(),
        )
    )
    return (
        attach(rows, per_ccn)
        .pipe(attach, on_last_two)
        .with_columns(*fill_zero(*COUNTS, "n_uncorrected_36m"))
        .with_columns(pl.col(*COUNTS).cast(pl.Int64))
        .with_columns(
            # null when there is no fire survey on record (or no previous one, for repeats)
            max_fire_sev_last_survey=pl.col("max_last").cast(pl.Float64),
            n_fire_repeat_tags_last_survey=pl.when(pl.col("prev").is_not_null()).then(
                pl.col("tags_last").list.set_intersection("tags_prev").list.len().cast(pl.Float64)
            ),
            share_fire_uncorrected_36m=pl.when(pl.col("n_fire_cit_36m") > 0).then(
                pl.col("n_uncorrected_36m") / pl.col("n_fire_cit_36m")
            ),
            days_since_last_fire_cit=days_since(pl.col("last")),
        )
        .select("ccn", "t", *COLUMNS)
    )
