"""Time travel (spec §11.3, in-gate form): every feature family rebuilt for t from a world that holds only
vintages <= t, through a reader that refuses anything later, equals the family built from the full world."""

import sys
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from precursorintelligence.gold.build import build_spine
from precursorintelligence.gold.features import FAMILIES, build_family
from precursorintelligence.gold.features.reader import FeatureReader

sys.path.insert(0, str(Path(__file__).parents[1] / "gold"))  # the toy world lives in tests/gold
from silver_fakes import VINTAGES, V, build_silver  # noqa: E402

# one early month (no t−12), the month after the gap, the month a late citation is posted, and V
MONTHS = ["2023-02", "2024-01", "2024-06", V]


@pytest.fixture(scope="module")
def world():
    silver = build_silver()
    spine = build_spine(
        silver["facility_universe"], published_vintages=VINTAGES, spine_start="2023-01", v=V
    )
    return silver, spine


@pytest.mark.parametrize("family", sorted(FAMILIES))
@pytest.mark.parametrize("t", MONTHS)
def test_features_at_t_need_nothing_after_t(world, family, t):
    silver, spine = world
    rows = spine.filter(pl.col("t") == t)
    full = build_family(family, FeatureReader.from_frames(silver), rows)

    past = {
        name: df.filter(pl.col("vintage_month") <= t) if "vintage_month" in df.columns else df
        for name, df in silver.items()
    }
    reader = FeatureReader.from_frames(past, limit=t)
    rebuilt = build_family(family, reader, rows)

    assert_frame_equal(rebuilt, full)
    assert reader.reads and all(v <= t for _, v in reader.reads)
