"""training/data.py: model frame, folds, windows, input checks and flags (docs/ML_PIPELINE_SPEC.md §1.4-§1.5, §3)."""

from datetime import date

import numpy as np
import polars as pl
import pytest
from make_ml_gold import CMS_GAPS, build_frame, ml_config, published_months

from precursorintelligence.common.config import (
    ConfigError,
    FeatureRegistry,
    FeatureSpec,
    load_feature_registry,
    load_labels_config,
    load_splits_config,
)
from precursorintelligence.gold.splits import split_windows
from precursorintelligence.training import MlInputError
from precursorintelligence.training.data import (
    MISSING_LEVEL,
    add_folds,
    average_precision,
    build_frames,
    check_inputs,
    compute_windows,
    feature_list,
    fold_of,
    leakage_flags,
    regime_flags,
    to_model_frame,
)

LABELS, SPLITS = load_labels_config(), load_splits_config()


def spec(name, dtype):
    return FeatureSpec(
        name=name,
        family="x",
        source_table="x",
        formula="x",
        first_available_vintage="2021-01",
        dtype=dtype,
        model_use="model",
    )


# ------------------------------------------------------------------ to_model_frame


def test_model_frame_casts_every_dtype_the_same_way():
    df = pl.DataFrame(
        {
            "a": pl.Series([1.5, None], dtype=pl.Float64),
            "b": pl.Series(
                [3, 4], dtype=pl.Int64
            ),  # no nulls here, a null later: float64 either way
            "c": pl.Series([True, None], dtype=pl.Boolean),
            "d": pl.Series(["x", None], dtype=pl.String),
            "extra": [1, 2],
        }
    )
    feats = [spec("d", "str"), spec("a", "f64"), spec("b", "i64"), spec("c", "bool")]
    out = to_model_frame(df, feats)
    assert list(out.columns) == ["d", "a", "b", "c"]  # feature order, nothing else
    assert [str(out[c].dtype) for c in ("a", "b", "c")] == ["float64"] * 3
    assert out["d"].dtype == object and out["d"].tolist() == ["x", MISSING_LEVEL]
    assert np.isnan(out["a"][1]) and np.isnan(out["c"][1]) and out["c"][0] == 1.0


def test_model_frame_refuses_rows_without_a_feature():
    with pytest.raises(MlInputError, match="missing"):
        to_model_frame(pl.DataFrame({"a": [1.0]}), [spec("a", "f64"), spec("b", "f64")])


# ------------------------------------------------------------------ folds


def test_folds_are_a_stable_hash_of_the_ccn():
    # sha256 of the CCN, first 8 hex digits, modulo k (§3.4): the same on every machine and in every run
    assert (fold_of("015001", 5), fold_of("01A193", 5), fold_of("T00001", 5)) == (2, 3, 1)
    assert fold_of("015001", 3) == 0


def test_every_row_of_a_facility_gets_the_same_fold():
    df = add_folds(build_frame(200, "2026-10", "2026-10", seed=1), 5)
    per_ccn = df.group_by("ccn").agg(pl.col("fold").n_unique())
    assert per_ccn["fold"].max() == 1
    assert sorted(df["fold"].unique().to_list()) == [0, 1, 2, 3, 4]
    assert df["fold"].dtype == pl.Int8


# ------------------------------------------------------------------ average precision


def test_average_precision_matches_scikit_learn():
    sklearn_metrics = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(0)
    y = (rng.random(500) < 0.3).astype(float)
    for score in (rng.random(500), rng.integers(0, 4, 500).astype(float), y + rng.random(500)):
        assert average_precision(y, score) == pytest.approx(
            sklearn_metrics.average_precision_score(y, score), abs=1e-12
        )


# ------------------------------------------------------------------ windows


def windows(v_labels, *, v=None, label="g12", cfg=None, splits=SPLITS, months=None):
    months = months or published_months(splits.spine_start, v or v_labels)
    h, b = LABELS.labels[label].horizon_months, LABELS.buffer(label)
    test_from, test_to = split_windows(v_labels, h, b, splits)["test"]
    test = [m for m in months if test_from <= m <= test_to]
    return compute_windows(
        months,
        test,
        label=label,
        v=v or v_labels,
        v_labels=v_labels,
        labels_cfg=LABELS,
        splits_cfg=splits,
        cfg=cfg or ml_config(),
    )


def test_windows_match_the_worked_example():
    # spec §3.2 at V_L = 2026-10 (the 2026-10-28 release)
    w = windows("2026-10")
    assert w.last_labeled_t == "2025-02"
    assert (w.test.first, w.test.last, w.test.months) == ("2024-12", "2025-02", ["2025-02"])
    assert w.eval.months == [
        "2022-05", "2022-06", "2022-07", "2022-08", "2022-09", "2022-10",
        "2022-11", "2023-01", "2023-02", "2023-03", "2023-04",
    ]  # fmt: skip
    assert w.refit.months == [
        "2024-02", "2024-03", "2024-04", "2024-05", "2024-06", "2024-07",
        "2024-08", "2024-09", "2024-10", "2024-11", "2025-02",
    ]  # fmt: skip
    assert (w.score_month, w.prev_score_month) == ("2026-10", "2026-09")
    assert w.development_months == ["2025-02"]


def test_the_purge_gap_reproduces_pr_49s_window():
    cfg = ml_config()
    cfg = cfg.model_copy(update={"windows": cfg.windows.model_copy(update={"eval_gap": "purge"})})
    w = windows("2026-10", cfg=cfg)
    assert (w.eval.first, w.eval.last, len(w.eval.months)) == ("2023-01", "2023-11", 11)


def test_windows_count_only_months_with_rows():
    w = windows("2026-10")
    assert not set(CMS_GAPS) & set(w.eval.months + w.refit.months + w.test.months)
    assert len(w.eval.months) == len(w.refit.months) == 11


def test_a_late_spine_start_leaves_too_few_eval_months():
    # with gold starting in 2023-01 the eval window holds 4 months (§3.2): below min_train_months
    late = SPLITS.model_copy(update={"spine_start": "2023-01"})
    with pytest.raises(MlInputError, match="eval window holds 4 months"):
        windows("2026-10", splits=late)


def test_a_stale_refresh_scores_v_but_takes_windows_from_v_labels():
    w = windows("2026-09", v="2026-10")
    assert (w.v, w.v_labels, w.score_month, w.prev_score_month) == (
        "2026-10",
        "2026-09",
        "2026-10",
        "2026-09",
    )
    assert w.last_labeled_t == "2025-01" and w.test.months == ["2024-11"]


def test_gold_split_months_that_disagree_with_splits_yaml_are_refused():
    months = published_months("2022-01", "2026-10")
    with pytest.raises(MlInputError, match="other split settings"):
        compute_windows(
            months,
            ["2024-11", "2025-02"],  # gold built with test_months 4
            label="g12",
            v="2026-10",
            v_labels="2026-10",
            labels_cfg=LABELS,
            splits_cfg=SPLITS,
            cfg=ml_config(),
        )


def test_an_unknown_label_is_a_config_error():
    months = published_months("2022-01", "2026-10")
    with pytest.raises(ConfigError, match="g24"):
        compute_windows(
            months,
            ["2025-02"],
            label="g24",
            v="2026-10",
            v_labels="2026-10",
            labels_cfg=LABELS,
            splits_cfg=SPLITS,
            cfg=ml_config(),
        )


def test_a_v_without_rows_is_refused():
    months = [m for m in published_months("2022-01", "2026-10") if m != "2026-10"]
    with pytest.raises(MlInputError, match="no rows for V"):
        windows("2026-10", months=months)


# ------------------------------------------------------------------ checks


@pytest.fixture(scope="module")
def frame():
    return build_frame(120, "2026-10", "2026-10", seed=3)


@pytest.fixture(scope="module")
def features():
    return feature_list(load_feature_registry(), ml_config())


def check(df, features, **kw):
    args = {"min_score_facilities": 50, "v": "2026-10", "last_scored_v": None} | kw
    check_inputs(df, features, "g12", **args)


def test_the_fixture_passes_the_input_checks(frame, features):
    check(frame, features)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda df: pl.concat([df, df.head(1)]), "not unique"),
        (
            lambda df: df.with_columns(pl.col("ccn").str.replace("^T00001$", "t00001")),
            "6 characters",
        ),
        (lambda df: df.drop("n_cit_12m"), "missing"),
        (lambda df: df.with_columns(pl.col("n_cit_12m").cast(pl.Float64)), "dtypes differ"),
        (
            lambda df: df.with_columns(
                pl.when(pl.col("status_g12") == "positive")
                .then(0)
                .otherwise(pl.col("y_g12"))
                .cast(pl.Int8)
                .alias("y_g12")
            ),
            "disagrees with their status",
        ),
    ],
)
def test_broken_gold_is_refused(frame, features, change, message):
    with pytest.raises(MlInputError, match=message):
        check(change(frame), features)


def test_too_few_facilities_to_score_is_refused(frame, features):
    with pytest.raises(MlInputError, match="min_score_facilities"):
        check(frame, features, min_score_facilities=10_000)


def test_an_older_gold_than_the_last_scored_one_is_refused(frame, features):
    with pytest.raises(MlInputError, match="earlier than the last scored V"):
        check(frame, features, last_scored_v="2026-11")
    check(frame, features, last_scored_v="2026-10")  # rescoring the same month is fine


# ------------------------------------------------------------------ flags


def frames_of(df, cfg=None):
    w = windows("2026-10", cfg=cfg)
    return build_frames(df, w, 5)


def test_the_fixture_raises_no_leakage_flag(frame, features):
    assert leakage_flags(frames_of(frame), features, "g12", ml_config()) == []


def test_a_feature_that_copies_the_label_is_flagged(frame, features):
    leaky = frame.with_columns(
        (pl.col("y_g12").fill_null(0).cast(pl.Int64) * 7).alias("n_cit_JL_24m"),
        pl.when(pl.col("y_g12") == 1)
        .then(pl.lit("yes"))
        .otherwise(pl.lit("no"))
        .alias("sff_status"),
    )
    flags = leakage_flags(frames_of(leaky), features, "g12", ml_config())
    assert {(f.frame, f.feature) for f in flags} == {
        (frame_name, feature)
        for frame_name in ("eval_train", "refit_train")
        for feature in ("n_cit_JL_24m", "sff_status")
    }
    assert all(f.pr_auc > f.threshold and not f.cleared for f in flags)
    cleared = ml_config(leakage_cleared=[{"feature": "sff_status", "reason": "test"}])
    again = leakage_flags(frames_of(leaky), features, "g12", cleared)
    assert {f.feature for f in again if f.cleared} == {"sff_status"}


def test_a_score_month_level_unseen_in_the_refit_window_is_a_regime_flag(frame, features):
    shifted = frame.with_columns(
        pl.when(pl.col("t") == "2026-10")
        .then(pl.lit(9, dtype=pl.Int64))
        .otherwise(pl.col("staffing_rating"))
        .alias("staffing_rating")
    )
    flags = regime_flags(frames_of(shifted), features, ml_config())
    assert [f.feature for f in flags] == ["staffing_rating"] and "9" in flags[0].detail
    assert regime_flags(frames_of(frame), features, ml_config()) == []


# ------------------------------------------------------------------ feature list


def test_an_excluded_name_the_registry_lacks_is_a_config_error():
    cfg = ml_config()
    cfg = cfg.model_copy(update={"features": cfg.features.model_copy(update={"exclude": ["eraa"]})})
    with pytest.raises(ConfigError, match="eraa"):
        feature_list(load_feature_registry(), cfg)


def test_a_forbidden_column_marked_model_is_a_config_error():
    registry = load_feature_registry()
    broken = FeatureRegistry.model_validate(
        {"_meta": registry.meta, "features": [*registry.features, spec("y_g12_copy", "f64")]}
    )
    with pytest.raises(ConfigError, match="forbidden"):
        feature_list(broken, ml_config())


def test_cutoff_date_is_a_date_column(frame):
    assert frame.schema["cutoff_date"] == pl.Date and frame["cutoff_date"][0] == date(2022, 1, 1)
