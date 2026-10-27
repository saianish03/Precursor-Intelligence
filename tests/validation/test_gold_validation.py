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
    assert schema.layer == "gold" and schema.columns["ccn"].str_length == 6


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
    return cfg


def test_censored_labels_and_moving_splits_are_not_drift(put_gold, ctx, gold_cfg):
    before = labels_month("2025-01", y=0, status="negative", split="train")
    after = labels_month("2025-02", status="censored", split="score").with_columns(
        pl.lit(None, pl.Int8).alias(f"y_{n}") for n in LABELS
    )
    put_gold("labels", pl.concat([before, after]))
    res = run_validation(ctx, gold_cfg, layer="gold")
    assert res.exit_code == EXIT_OK and res.anomalies == []


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
    assert found == {(f"status_{n}", "allowed_values") for n in LABELS}
