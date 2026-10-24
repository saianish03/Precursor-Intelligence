"""The split windows (spec §9.3, §9.5): calendar months back from last_labeled_t = V - (H + B)."""

import polars as pl
import pytest

from precursorintelligence.common.config import SplitsConfig, load_labels_config
from precursorintelligence.gold.splits import (
    SPLITS,
    add_splits,
    last_labeled_t,
    shift,
    split_windows,
)

CFG = SplitsConfig(
    spine_start="2023-01", test_months=3, calib_months=3, required_labels=["g12", "g6"]
)


def _months(first: str, last: str) -> int:
    y1, m1 = map(int, first.split("-"))
    y2, m2 = map(int, last.split("-"))
    return (y2 * 12 + m2) - (y1 * 12 + m1) + 1


@pytest.mark.parametrize(
    ("month", "n", "expected"),
    [
        ("2025-03", -20, "2023-07"),
        ("2024-01", -1, "2023-12"),
        ("2023-12", 1, "2024-01"),
        ("2026-08", 0, "2026-08"),
    ],
)
def test_shift_counts_calendar_months_across_years(month, n, expected):
    assert shift(month, n) == expected


def test_last_labeled_t_is_v_minus_horizon_and_buffer():
    assert last_labeled_t("2026-08", 12, 8) == "2024-12"  # spec §9.3: y_g12
    assert last_labeled_t("2026-08", 6, 8) == "2025-06"  # y_g6
    assert last_labeled_t("2026-08", 12, 12) == "2024-08"  # y_pen12


@pytest.mark.parametrize(("horizon", "buffer"), [(12, 8), (6, 8), (12, 12)])
def test_windows_are_back_to_back_with_the_configured_lengths(horizon, buffer):
    w = split_windows("2026-08", horizon, buffer, CFG)
    assert _months(*w["test"]) == CFG.test_months
    assert _months(*w["purge"]) == horizon
    assert _months(*w["calib"]) == CFG.calib_months
    assert w["test"][1] == w["last_labeled_t"] and w["score"] == [
        shift(w["last_labeled_t"], 1),
        None,
    ]
    for earlier, later in (("train", "calib"), ("calib", "purge"), ("purge", "test")):
        if w[earlier] is not None:
            assert shift(w[earlier][1], 1) == w[later][0]
    assert w["train"][0] == CFG.spine_start


def test_every_window_moves_with_v():
    a, b = split_windows("2026-08", 12, 8, CFG), split_windows("2026-09", 12, 8, CFG)
    for split in ("calib", "purge", "test"):
        assert b[split] == [shift(m, 1) for m in a[split]]
    assert b["train"] == [CFG.spine_start, shift(a["train"][1], 1)]  # train grows by one month


def test_train_is_empty_when_calib_starts_at_or_before_spine_start():
    w = split_windows("2025-03", 12, 8, CFG)  # the toy world's V
    assert w["train"] is None and w["calib"] == ["2022-02", "2022-04"]


def test_without_calibration_months_train_runs_up_to_purge():
    w = split_windows("2026-08", 12, 8, CFG.model_copy(update={"calib_months": 0}))
    assert w["calib"] is None and shift(w["train"][1], 1) == w["purge"][0]


def test_add_splits_gives_every_row_one_split_and_score_after_last_labeled_t():
    months = [shift("2023-01", i) for i in range(44)]  # 2023-01 -> 2026-08
    labels = pl.DataFrame({"ccn": ["015001"] * len(months), "t": months})
    out = add_splits(labels, "2026-08", load_labels_config(), CFG)
    assert out.columns == ["ccn", "t", "split_g12", "split_g6", "split_pen12"]
    for name in ("g12", "g6", "pen12"):
        col = out[f"split_{name}"]
        assert col.null_count() == 0 and set(col) <= set(SPLITS)
        last = split_windows("2026-08", *_hb(name), CFG)["last_labeled_t"]
        assert out.filter(pl.col("t") > last)[f"split_{name}"].unique().to_list() == ["score"]


def _hb(name: str) -> tuple[int, int]:
    cfg = load_labels_config()
    return cfg.labels[name].horizon_months, cfg.buffer(name)
