"""Gold spine and labels by month t: the committed baselines match gold's code, and labels never drift."""

import polars as pl
import pytest

from precursorintelligence.common.config import load_labels_config, load_validation_config
from precursorintelligence.gold.splits import SPLITS
from precursorintelligence.ingestion.pipeline import EXIT_OK, EXIT_VALIDATION
from precursorintelligence.validation.run import run_validation
from precursorintelligence.validation.schema import load_schema

CFG = load_validation_config()
LABELS = list(load_labels_config().labels)


def labels_month(t, n=20, y=0, status="negative", split="train"):
    rows = {"ccn": [f"0150{i:02d}" for i in range(n)], "t": [t] * n}
    for name in LABELS:
        rows |= {
            f"y_{name}": pl.Series([y] * n, dtype=pl.Int8),
            f"status_{name}": [status] * n,
        }
    return pl.DataFrame(rows).with_columns(pl.lit(split).alias(f"split_{lab}") for lab in LABELS)


@pytest.mark.parametrize("table", CFG.layers["gold"].tables)
def test_each_gold_table_has_a_baseline_with_six_character_ccns(table):
    schema = load_schema(CFG.schema_path(table))
    assert schema.layer == "gold" and schema.columns["ccn"].pattern == r"^[0-9A-Z]{6}$"


def test_the_labels_baseline_lists_gold_s_split_and_status_values():
    columns = load_schema(CFG.schema_path("labels")).columns
    for name in LABELS:
        assert columns[f"split_{name}"].allowed == list(SPLITS)
        assert set(columns[f"status_{name}"].allowed) == {
            "positive",
            "negative",
            "censored",
            "immature",
        }


@pytest.fixture
def gold_cfg(cfg, tmp_path):
    """The test config, but with the committed gold baselines and drift skips."""
    for table in ("spine", "labels"):
        (tmp_path / "contracts").mkdir(exist_ok=True)
        (tmp_path / "contracts" / f"{table}.yaml").write_bytes(CFG.schema_path(table).read_bytes())
    cfg.layers["gold"].tables = ["labels"]
    cfg.drift.skip_columns = CFG.drift.skip_columns
    cfg.layers["gold"].all_months = CFG.layers["gold"].all_months
    return cfg


def test_immature_labels_and_moving_splits_are_not_drift(put_gold, ctx, gold_cfg):
    before = labels_month("2025-01", y=0, status="negative", split="train")
    after = labels_month("2025-02", y=None, status="immature", split="score")
    put_gold("labels", pl.concat([before, after]))
    res = run_validation(ctx, gold_cfg, layer="gold")
    assert res.exit_code == EXIT_OK and res.anomalies == []
    assert res.checked == {"labels": ["2025-01", "2025-02"]}  # gold checks every month by default


def test_a_censored_row_in_the_score_split_is_an_error(put_gold, ctx, gold_cfg):
    """Gold can never write it: score rows are exactly the immature ones (gold/splits.py)."""
    put_gold("labels", labels_month("2025-01", y=None, status="censored", split="score"))
    res = run_validation(ctx, gold_cfg, layer="gold")
    assert res.exit_code == EXIT_VALIDATION
    assert {(a.check, a.column) for a in res.anomalies} == {
        ("score_iff_immature", f"status_{n}") for n in LABELS
    }


@pytest.mark.parametrize(
    "y, status, split, rules",
    [
        (1, "negative", "train", {"y_matches_status"}),
        (0, "positive", "test", {"y_matches_status"}),
        (None, "negative", "train", {"y_null_iff_unlabelled", "y_matches_status"}),
        (1, "censored", "train", {"y_null_iff_unlabelled", "y_matches_status"}),
        (None, "immature", "train", {"score_iff_immature"}),
    ],
)
def test_a_label_that_contradicts_its_status_or_split_is_an_error(
    put_gold, ctx, gold_cfg, y, status, split, rules
):
    good = labels_month("2025-01", n=19)
    bad = labels_month("2025-01", n=1, y=y, status=status, split=split).with_columns(
        pl.lit("015099").alias("ccn")
    )
    put_gold("labels", pl.concat([good, bad]))
    found = {
        (a.check, a.column, a.observed["rows"])
        for a in run_validation(ctx, gold_cfg, layer="gold").anomalies
    }
    assert found == {(r, f"status_{n}", 1) for r in rules for n in LABELS}


@pytest.mark.parametrize(
    "y, status, split",
    [
        (1, "positive", "train"),
        (0, "negative", "calib"),
        (None, "censored", "test"),
        (None, "immature", "score"),
    ],
)
def test_every_label_gold_can_write_passes(put_gold, ctx, gold_cfg, y, status, split):
    put_gold("labels", labels_month("2025-01", y=y, status=status, split=split))
    assert run_validation(ctx, gold_cfg, layer="gold").anomalies == []


def test_one_month_can_still_be_chosen_in_gold(put_gold, ctx, gold_cfg):
    put_gold("labels", pl.concat([labels_month("2025-01"), labels_month("2025-02")]))
    assert run_validation(ctx, gold_cfg, layer="gold", month="2025-01").checked == {
        "labels": ["2025-01"]
    }


def test_a_bad_split_value_is_an_error(put_gold, ctx, gold_cfg):
    put_gold("labels", labels_month("2025-01", split="holdout"))
    res = run_validation(ctx, gold_cfg, layer="gold")
    assert res.exit_code == EXIT_VALIDATION
    assert {(a.column, a.check) for a in res.anomalies} == {
        (f"split_{n}", "allowed_values") for n in LABELS
    }


def test_a_bad_status_value_is_an_error(put_gold, ctx, gold_cfg):
    put_gold("labels", labels_month("2025-01", status="unknown"))
    found = {(a.column, a.check) for a in run_validation(ctx, gold_cfg, layer="gold").anomalies}
    assert {(f"status_{n}", "allowed_values") for n in LABELS} <= found


def test_a_label_stored_as_text_is_a_wrong_dtype_error_not_a_crash(put_gold, ctx, gold_cfg):
    name = LABELS[0]
    put_gold("labels", labels_month("2025-01").with_columns(pl.col(f"y_{name}").cast(pl.String)))
    res = run_validation(ctx, gold_cfg, layer="gold")
    assert res.exit_code == EXIT_VALIDATION
    assert [(a.check, a.column) for a in res.anomalies] == [("wrong_dtype", f"y_{name}")]
