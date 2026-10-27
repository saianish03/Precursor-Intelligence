"""One validate run: what it checks, what it writes, and its exit code."""

import json

import polars as pl
import pytest
from validation_fakes import citations

from precursorintelligence.common.config import ConfigError
from precursorintelligence.ingestion.pipeline import EXIT_OK, EXIT_VALIDATION
from precursorintelligence.validation.run import run_validation, write_schemas

TABLE = "citations_by_vintage"


@pytest.fixture
def silver(put_silver, ctx, cfg):
    """Three clean months, and a schema inferred from them."""
    for v in ("2026-06", "2026-07", "2026-08"):
        put_silver(TABLE, v, citations(n=40, vintage=v))
    write_schemas(ctx, cfg, layer="silver")
    return put_silver


def read_json(store, key):
    with store.open_read(key) as f:
        return json.load(f)


def test_clean_months_pass_and_the_reports_are_written(silver, ctx, cfg, store):
    res = run_validation(ctx, cfg, layer="silver")
    assert res.exit_code == EXIT_OK and res.anomalies == []
    assert res.checked == {TABLE: ["2026-08"]}
    stats = read_json(store, f"reports/validation/run=R1/stats_{TABLE}.json")
    assert [m["month"] for m in stats["months"]] == ["2026-07", "2026-08"]  # checked + previous
    report = read_json(store, "reports/validation/run=R1/anomalies.json")
    assert report["counts"] == {"error": 0, "warning": 0} and report["anomalies"] == []
    assert read_json(store, res.manifest_key)["exit_code"] == EXIT_OK
    assert res.manifest_key.startswith("metadata/validation/run_date=")
    assert ctx.metadata.read("dq_results") == []


def test_a_schema_violation_is_an_error_and_exits_2(silver, ctx, cfg, store):
    cycles = [1, 2, 3, 1] * 9 + [1, 2, 3, 9]  # one 1 became a 9
    silver(TABLE, "2026-08", citations(n=40, vintage="2026-08", inspection_cycle=cycles))
    res = run_validation(ctx, cfg, layer="silver")
    assert res.exit_code == EXIT_VALIDATION
    (a,) = read_json(store, "reports/validation/run=R1/anomalies.json")["anomalies"]
    assert (a["table"], a["month"], a["column"], a["check"], a["severity"]) == (
        TABLE,
        "2026-08",
        "inspection_cycle",
        "out_of_range",
        "error",
    )
    (row,) = ctx.metadata.read("dq_results")
    assert (row["run_id"], row["node"], row["layer"], row["severity"]) == (
        "R1",
        "validation",
        "silver",
        "error",
    )
    assert (row["table_name"], row["vintage_month"], row["column_name"]) == (
        TABLE,
        "2026-08",
        "inspection_cycle",
    )


def test_drift_is_a_warning_that_fails_only_with_strict(silver, ctx, cfg, store):
    silver(TABLE, "2026-08", citations(n=28, vintage="2026-08"))  # -30% rows
    res = run_validation(ctx, cfg, layer="silver")
    assert res.exit_code == EXIT_OK
    (a,) = read_json(store, "reports/validation/run=R1/anomalies.json")["anomalies"]
    assert (a["check"], a["severity"]) == ("row_count_change", "warning")
    assert ctx.metadata.read("dq_results")[0]["severity"] == "warning"
    assert run_validation(ctx, cfg, layer="silver", strict=True).exit_code == EXIT_VALIDATION


def test_all_months_checks_each_month_against_the_one_before(silver, ctx, cfg):
    silver(TABLE, "2026-07", citations(n=28, vintage="2026-07"))
    res = run_validation(ctx, cfg, layer="silver", all_months=True)
    assert res.checked == {TABLE: ["2026-06", "2026-07", "2026-08"]}
    drops = [(a.month, a.check) for a in res.anomalies]
    assert drops == [("2026-07", "row_count_change"), ("2026-08", "row_count_change")]  # down, up


def test_one_month_can_be_chosen(silver, ctx, cfg):
    assert run_validation(ctx, cfg, layer="silver", month="2026-06").checked == {TABLE: ["2026-06"]}
    with pytest.raises(ConfigError, match="no month 2025-01"):
        run_validation(ctx, cfg, layer="silver", month="2025-01")


def test_a_missing_schema_file_is_a_config_error_and_writes_nothing(put_silver, ctx, cfg, store):
    put_silver(TABLE, "2026-08", citations(vintage="2026-08"))
    with pytest.raises(ConfigError, match="schema file not found"):
        run_validation(ctx, cfg, layer="silver")
    assert store.list("reports/") == [] and ctx.metadata.read("dq_results") == []


def test_an_unknown_table_is_a_config_error(silver, ctx, cfg):
    with pytest.raises(ConfigError, match="no configured table"):
        run_validation(ctx, cfg, layer="silver", tables=["ledger_citations"])


def test_a_configured_table_missing_from_the_store_is_an_error(silver, ctx, cfg, tmp_path):
    cfg.layers["silver"].tables.append("surveys_by_vintage")
    (tmp_path / "contracts" / "surveys_by_vintage.yaml").write_text(
        "table: surveys_by_vintage\nlayer: silver\ncolumns: {}\n"
    )
    res = run_validation(ctx, cfg, layer="silver")
    (a,) = res.anomalies
    assert (a.table, a.check, a.severity) == ("surveys_by_vintage", "table_missing", "error")
    assert res.exit_code == EXIT_VALIDATION


def test_validation_never_writes_to_the_layer_it_reads(silver, ctx, cfg, store):
    before = {k: store.stat(k).size for k in store.list("silver/")}
    run_validation(ctx, cfg, layer="silver", all_months=True)
    assert {k: store.stat(k).size for k in store.list("silver/")} == before


def test_gold_months_come_from_t(put_gold, ctx, cfg):
    spine = pl.DataFrame(
        {"ccn": [f"0150{i:02d}" for i in range(20)] * 2, "t": ["2025-01"] * 20 + ["2025-02"] * 20}
    )
    put_gold("spine", spine)
    write_schemas(ctx, cfg, layer="gold")
    put_gold("spine", spine.head(30))  # 2025-02 lost half its rows
    res = run_validation(ctx, cfg, layer="gold")
    assert res.checked == {"spine": ["2025-02"]}
    assert [(a.check, a.severity) for a in res.anomalies] == [("row_count_change", "warning")]


def test_an_empty_month_fails_the_run(silver, ctx, cfg):
    silver(TABLE, "2026-08", citations(n=40, vintage="2026-08").clear())
    res = run_validation(ctx, cfg, layer="silver")
    assert res.exit_code == EXIT_VALIDATION
    assert {(a.check, a.severity) for a in res.anomalies} == {
        ("empty_month", "error"),
        ("row_count_change", "warning"),
    }


def test_drift_across_a_gap_compares_with_the_previous_vintage_present(put_silver, ctx, cfg):
    """Silver has no 2024-12 or 2025-01: 2025-02 is compared with 2024-11, not with an empty month."""
    for v, n in (("2024-10", 40), ("2024-11", 40), ("2025-02", 28)):
        put_silver(TABLE, v, citations(n=n, vintage=v))
    write_schemas(ctx, cfg, layer="silver")
    (a,) = run_validation(ctx, cfg, layer="silver").anomalies
    assert (a.month, a.check, a.expected) == ("2025-02", "row_count_change", 40)
    assert "in 2024-11" in a.detail
