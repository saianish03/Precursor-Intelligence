"""Step 7 (spec §8): gold's feature tables, one row per spine row, keyed by (ccn, t).

Every family reads silver only through ``FeatureReader`` (reader.py). ``build_features`` builds one family for
every spine month; a family not built yet raises NotImplementedError.
"""

from __future__ import annotations

from collections.abc import Callable

import polars as pl

from ...common.config import FeaturesConfig, load_features_config
from . import citations, context, enforcement, facility_staffing, survey_timing
from .reader import FeatureReader

# family -> builder(reader, spine rows of one t, t, cfg) -> one row per spine row, keyed by (ccn, t)
FamilyBuilder = Callable[[FeatureReader, pl.DataFrame, str, FeaturesConfig], pl.DataFrame]
FAMILIES: dict[str, FamilyBuilder] = {
    "survey_timing": survey_timing.build,
    "citations": citations.build,
    "enforcement": enforcement.build,
    "facility_staffing": facility_staffing.build,
    "context": context.build,
}
# spec §10 join order; quality waits for configs/qm_measures.yaml (Checkpoint 4)
FAMILY_ORDER = (
    "survey_timing", "citations", "enforcement", "facility_staffing", "quality", "context",
)  # fmt: skip


def build_family(
    family: str,
    reader: FeatureReader,
    spine: pl.DataFrame,
    cfg: FeaturesConfig | None = None,
) -> pl.DataFrame:
    """``family`` for every spine row, month by month, in the spine's order."""
    if family not in FAMILY_ORDER:
        raise ValueError(f"unknown feature family {family!r}; expected one of {FAMILY_ORDER}")
    if family not in FAMILIES:
        raise NotImplementedError(f"feature family {family!r} is not built yet")
    cfg = cfg or load_features_config()
    builder = FAMILIES[family]
    # Each month gets a reader limited to it: row t can read nothing after t, whatever the family asks.
    parts = [
        builder(reader.limited_to(str(t)), rows, str(t), cfg)
        for (t,), rows in spine.group_by("t", maintain_order=True)
    ]
    out = spine.select("ccn", "t").join(
        pl.concat(parts), on=["ccn", "t"], how="left", validate="1:1", maintain_order="left"
    )
    if out.height != spine.height:
        raise AssertionError(f"features_{family}: {out.height} rows for {spine.height} spine rows")
    return out


def build_features(
    family: str,
    silver: dict[str, pl.DataFrame],
    spine: pl.DataFrame,
    cfg: FeaturesConfig | None = None,
) -> pl.DataFrame:
    """``family`` from stacked in-memory silver tables (the toy world and tests)."""
    return build_family(family, FeatureReader.from_frames(silver), spine, cfg)
