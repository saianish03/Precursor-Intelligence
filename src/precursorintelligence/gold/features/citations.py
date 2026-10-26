"""``features_citations`` (spec §8.3): how many and how serious the facility's citations were.

Reads ``citations_by_vintage`` at t (``survey_date <= cutoff``), ``surveys_by_vintage`` at t for the last and
previous standard surveys, and ``provider_by_vintage`` at t for the beds of the per-bed rate.
"""

from __future__ import annotations

from functools import cache

import polars as pl

from ...common.config import FeaturesConfig
from ...harmonize.columns import load_silver_config
from .common import SEVERITIES, attach, days_since, fill_zero, gplus_min_ordinal, within
from .reader import FeatureReader
from .survey_timing import STANDARD

COLUMNS = [
    "n_cit_12m", "n_cit_24m", "n_cit_AC_24m", "n_cit_DF_24m", "n_cit_GI_24m", "n_cit_JL_24m",
    "sev_points_12m", "sev_points_24m", "sev_points_per_bed_24m",
    "max_sev_last_std", "gplus_last_std", "max_sev_prev_std", "gplus_prev_std",
    "n_gplus_surveys_36m", "days_since_last_gplus", "n_complaint_cit_12m", "n_complaint_gplus_24m",
    "n_ic_cit_24m", "n_cat_abuse_24m", "n_cat_qol_care_24m", "n_cat_assessment_24m",
    "n_cat_nursing_physician_24m", "n_cat_resident_rights_24m", "n_cat_administration_24m",
    "n_repeat_tags_last_std", "n_uncorrected",
]  # fmt: skip
INT_COUNTS = [
    c for c in COLUMNS if c.startswith("n_") and c != "n_repeat_tags_last_std"
]  # n_repeat_tags_last_std is null unless both surveys exist, so it is a float


@cache
def not_corrected_values() -> tuple[str, ...]:
    """``value_maps.yaml: not_corrected_values``, the Deficiency Corrected values that count as uncorrected."""
    return tuple(load_silver_config().value_maps["not_corrected_values"])


def _last_two_standard_surveys(surveys: pl.DataFrame) -> pl.DataFrame:
    """Per ccn, the latest two HEALTH_STANDARD survey dates on or before the cutoff (null when missing)."""
    dates = (
        surveys.filter(
            pl.col("survey_type") == STANDARD, pl.col("survey_date") <= pl.col("cutoff_date")
        )
        .group_by("ccn")
        .agg(pl.col("survey_date").unique().sort(descending=True).head(2).alias("d"))
    )
    return dates.select(
        "ccn",
        last_std=pl.col("d").list.get(0, null_on_oob=True),
        prev_std=pl.col("d").list.get(1, null_on_oob=True),
    )


def build(reader: FeatureReader, rows: pl.DataFrame, t: str, cfg: FeaturesConfig) -> pl.DataFrame:
    citations, surveys, provider = (
        reader.read(name, t)
        for name in ("citations_by_vintage", "surveys_by_vintage", "provider_by_vintage")
    )
    assert citations is not None and surveys is not None and provider is not None

    ordinal = {s: i + 1 for i, s in enumerate(SEVERITIES)}
    band = {s: b for b, letters in cfg.severity_bands.items() for s in letters}
    c = citations.filter(pl.col("survey_date") <= pl.col("cutoff_date")).with_columns(
        points=pl.col("scope_severity").replace_strict(
            cfg.severity_points, return_dtype=pl.Float64
        ),
        ordinal=pl.col("scope_severity").replace_strict(ordinal, return_dtype=pl.Int64),
        band=pl.col("scope_severity").replace_strict(band, return_dtype=pl.String),
        complaint=pl.col("is_complaint").fill_null(False),
    )
    w12, w24, w36 = (within("survey_date", m) for m in (12, 24, 36))
    gplus_from = gplus_min_ordinal(cfg)  # G with labels.yaml's G-L
    gplus = pl.col("ordinal") >= gplus_from
    category = pl.col("deficiency_category")

    def count(cond: pl.Expr) -> pl.Expr:
        return cond.fill_null(False).sum()

    per_ccn = c.group_by("ccn").agg(
        n_cit_12m=count(w12),
        n_cit_24m=count(w24),
        **{f"n_cit_{b}_24m": count(w24 & (pl.col("band") == b)) for b in cfg.severity_bands},
        sev_points_12m=pl.col("points").filter(w12).sum(),
        sev_points_24m=pl.col("points").filter(w24).sum(),
        n_gplus_surveys_36m=pl.col("survey_date").filter(gplus & w36).n_unique(),
        last_gplus=pl.col("survey_date").filter(gplus).max(),
        n_complaint_cit_12m=count(w12 & pl.col("complaint")),
        n_complaint_gplus_24m=count(w24 & pl.col("complaint") & gplus),
        n_ic_cit_24m=count(
            w24 & ((category == cfg.infection_control_category) | pl.col("is_infection_control"))
        ),
        **{
            f"n_cat_{name}_24m": count(w24 & (category == value))
            for name, value in cfg.citation_categories.items()
        },
        n_uncorrected=count(w24 & pl.col("deficiency_corrected").is_in(not_corrected_values())),
    )

    std = _last_two_standard_surveys(surveys)
    on_std = (
        c.filter(pl.col("is_standard").fill_null(False))
        .join(std, on="ccn")
        .group_by("ccn")
        .agg(
            max_last=pl.col("ordinal").filter(pl.col("survey_date") == pl.col("last_std")).max(),
            max_prev=pl.col("ordinal").filter(pl.col("survey_date") == pl.col("prev_std")).max(),
            tags_last=pl.col("ftag").filter(pl.col("survey_date") == pl.col("last_std")).unique(),
            tags_prev=pl.col("ftag").filter(pl.col("survey_date") == pl.col("prev_std")).unique(),
        )
    )
    beds = provider.select("ccn", "certified_beds")

    def max_sev(survey: str, worst: str) -> pl.Expr:
        # 0 when that survey has no standard citation; null when there is no such survey
        return (
            pl.when(pl.col(survey).is_not_null()).then(pl.col(worst).fill_null(0)).cast(pl.Float64)
        )

    both_cited = (pl.col("tags_last").list.len() > 0) & (pl.col("tags_prev").list.len() > 0)
    return (
        attach(rows, per_ccn)
        .pipe(attach, std)
        .pipe(attach, on_std)
        .pipe(attach, beds)
        .with_columns(*fill_zero(*INT_COUNTS, "sev_points_12m", "sev_points_24m"))
        .with_columns(pl.col(c).cast(pl.Int64) for c in INT_COUNTS)
        .with_columns(
            sev_points_per_bed_24m=pl.when(pl.col("certified_beds") > 0).then(
                pl.col("sev_points_24m") / pl.col("certified_beds")
            ),
            max_sev_last_std=max_sev("last_std", "max_last"),
            max_sev_prev_std=max_sev("prev_std", "max_prev"),
            days_since_last_gplus=days_since(pl.col("last_gplus")),
            n_repeat_tags_last_std=pl.when(both_cited.fill_null(False)).then(
                pl.col("tags_last").list.set_intersection("tags_prev").list.len().cast(pl.Float64)
            ),
        )
        .with_columns(
            gplus_last_std=pl.col("max_sev_last_std") >= gplus_from,
            gplus_prev_std=pl.col("max_sev_prev_std") >= gplus_from,
        )
        .select("ccn", "t", *COLUMNS)
    )
