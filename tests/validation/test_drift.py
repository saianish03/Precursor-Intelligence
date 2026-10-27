"""This month's statistics against the previous month's: drift is a warning, small changes stay clean."""

import polars as pl
from validation_fakes import citations

from precursorintelligence.common.config import DriftConfig
from precursorintelligence.validation.stats import table_stats
from precursorintelligence.validation.validate import WARNING, check_drift

CFG = DriftConfig(
    row_count_change_pct=10, null_rate_jump_pts=5, mean_shift_std=0.5, skip_columns=["y_*"]
)


def drift(prev_df, df, cfg=CFG):
    prev = table_stats(prev_df, table="t", month="2026-06", top_values=5)
    stats = table_stats(df, table="t", month="2026-07", top_values=5)
    return check_drift(stats, prev, cfg, table="citations_by_vintage")


def test_the_same_month_again_has_no_drift():
    assert drift(citations(n=40), citations(n=40)) == []


def test_a_row_count_drop_is_a_warning():
    (a,) = drift(citations(n=40), citations(n=28))  # -30%
    assert (a.check, a.severity, a.column) == ("row_count_change", WARNING, None)
    assert (a.observed, a.expected, a.month) == (28, 40, "2026-07")


def test_a_small_row_count_change_is_clean():
    assert drift(citations(n=40), citations(n=38)) == []  # -5%


def test_a_null_spike_is_a_warning():
    cycles = [None] * 10 + [1, 2, 3, 1] * 7 + [1, 2]
    found = drift(citations(n=40), citations(n=40, inspection_cycle=cycles))
    a = next(a for a in found if a.check == "null_rate_jump")
    assert (a.column, a.observed, a.expected, a.severity) == (
        "inspection_cycle",
        25.0,
        0.0,
        WARNING,
    )


def test_fewer_nulls_is_not_a_warning():
    cycles = [None] * 10 + [1, 2, 3, 1] * 7 + [1, 2]
    assert drift(citations(n=40, inspection_cycle=cycles), citations(n=40)) == []


def test_a_mean_shift_is_a_warning():
    df = citations(n=40).with_columns(pl.col("inspection_cycle") + 1)
    (a,) = drift(citations(n=40), df)
    assert (a.check, a.column, a.severity) == ("mean_shift", "inspection_cycle", WARNING)
    assert a.observed == a.expected + 1


def test_skipped_columns_never_drift():
    prev = citations(n=40).with_columns(pl.Series("y_g12", [0, 1] * 20))
    df = citations(n=40).with_columns(pl.lit(1).alias("y_g12"))
    assert drift(prev, df) == []
    assert any(
        a.column == "y_g12" for a in drift(prev, df, CFG.model_copy(update={"skip_columns": []}))
    )


def test_a_new_column_has_nothing_to_compare_to():
    df = citations(n=40).with_columns(pl.lit(None, pl.Float64).alias("new_col"))
    assert drift(citations(n=40), df) == []


def test_a_downward_mean_shift_is_a_warning():
    df = citations(n=40).with_columns(pl.col("inspection_cycle") - 1)
    (a,) = drift(citations(n=40), df)
    assert (a.check, a.column) == ("mean_shift", "inspection_cycle")
    assert a.observed == a.expected - 1


def test_a_row_change_of_exactly_the_limit_is_clean():
    assert drift(citations(n=40), citations(n=44)) == []  # +10%, limit 10%
    assert [a.check for a in drift(citations(n=40), citations(n=45))] == ["row_count_change"]


def test_a_constant_column_last_month_gives_no_mean_shift_and_no_crash():
    before = citations(n=40, inspection_cycle=[1] * 40)  # std 0
    assert drift(before, citations(n=40)) == []
