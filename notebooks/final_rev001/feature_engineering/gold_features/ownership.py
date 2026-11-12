"""``features_ownership`` (docs/ALL_TABLES_CONTRACT.md): who owns and manages the facility, from
``ownership_by_vintage`` at t, and at t−12 for owner turnover.

Roles are matched case-insensitively on CMS's role text: an owner has a role containing "ownership" (5% or
greater direct / indirect ownership interest), a manager one containing "manag" (operational/managerial control,
managing employee). People and organisations are counted by distinct ``owner_name``. A facility with no
ownership rows gets 0 counts, null shares / ages and ``own_on_record`` False. The t−12 Δ counts are null when
t−12 was never published or the facility had no ownership rows at t or t−12; no fallback month. Association
dates after the cutoff are ignored.
"""

from __future__ import annotations

import polars as pl

from ...common.config import FeaturesConfig
from .common import CUTOFF, attach, fill_zero, within
from .reader import FeatureReader

DAYS_PER_MONTH = 30.4375
COUNTS = ["own_n_owners", "own_n_managers", "own_n_new_assoc_12m"]
COLUMNS = [
    "own_on_record", *COUNTS, "own_share_org_owners", "own_max_pct", "own_has_indirect",
    "own_months_since_newest_assoc", "own_months_since_oldest_assoc", "own_n_added_12m",
    "own_n_removed_12m",
]  # fmt: skip

_ROLE = pl.col("owner_role").str.to_lowercase()
_IS_OWNER = _ROLE.str.contains("ownership")
_IS_MANAGER = _ROLE.str.contains("manag")
_NAME = pl.col("owner_name").str.strip_chars().str.to_uppercase()


def _months(col: str) -> pl.Expr:
    return (pl.col("_cutoff") - pl.col(col)).dt.total_days() / DAYS_PER_MONTH


def _names(frame: pl.DataFrame, alias: str) -> pl.DataFrame:
    """One row per ccn with the set of distinct owner / manager names."""
    return frame.group_by("ccn").agg(_NAME.drop_nulls().unique().alias(alias))


def build(reader: FeatureReader, rows: pl.DataFrame, t: str, cfg: FeaturesConfig) -> pl.DataFrame:
    owners = reader.read("ownership_by_vintage", t)
    assert owners is not None
    assoc = pl.col("association_date")
    valid_assoc = assoc.filter(assoc <= CUTOFF)
    per_ccn = owners.group_by("ccn").agg(
        own_on_record=pl.lit(True),
        own_n_owners=_NAME.filter(_IS_OWNER).drop_nulls().n_unique(),
        own_n_managers=_NAME.filter(_IS_MANAGER).drop_nulls().n_unique(),
        own_n_new_assoc_12m=_NAME.filter(within("association_date", 12)).drop_nulls().n_unique(),
        _n_owner_rows=_IS_OWNER.sum(),
        _n_org_owner_rows=(
            _IS_OWNER & (pl.col("owner_type").str.to_lowercase() == "organization")
        ).sum(),
        own_max_pct=pl.col("ownership_pct").max(),
        own_has_indirect=(_IS_OWNER & _ROLE.str.contains("indirect")).any(),
        _newest=valid_assoc.max(),
        _oldest=valid_assoc.min(),
        _cutoff=CUTOFF.first(),
    )
    per_ccn = per_ccn.with_columns(
        own_share_org_owners=pl.when(pl.col("_n_owner_rows") > 0).then(
            pl.col("_n_org_owner_rows") / pl.col("_n_owner_rows")
        ),
        own_months_since_newest_assoc=_months("_newest"),
        own_months_since_oldest_assoc=_months("_oldest"),
    )
    out = attach(rows, per_ccn).with_columns(
        *fill_zero(*COUNTS),
        pl.col("own_on_record", "own_has_indirect").fill_null(False),
    )

    before = reader.read("ownership_by_vintage", t, months_back=12)
    if before is None:  # t−12 never published: every Δ is null
        out = out.with_columns(
            own_n_added_12m=pl.lit(None, pl.Int64), own_n_removed_12m=pl.lit(None, pl.Int64)
        )
    else:
        out = attach(attach(out, _names(owners, "_now")), _names(before, "_then")).with_columns(
            own_n_added_12m=pl.col("_now").list.set_difference("_then").list.len(),
            own_n_removed_12m=pl.col("_then").list.set_difference("_now").list.len(),
        )
    return out.with_columns(
        pl.col(*COUNTS, "own_n_added_12m", "own_n_removed_12m").cast(pl.Int64),
        pl.col("own_share_org_owners", "own_max_pct").cast(pl.Float64),
        pl.col("own_months_since_newest_assoc", "own_months_since_oldest_assoc").cast(pl.Float64),
        pl.col("own_has_indirect").cast(pl.Boolean),
    ).select("ccn", "t", *COLUMNS)
