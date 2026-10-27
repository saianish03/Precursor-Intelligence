"""Inferring a baseline schema and keeping it as hand-editable YAML."""

import polars as pl
import pytest
from validation_fakes import citations

from precursorintelligence.common.config import ConfigError
from precursorintelligence.validation.schema import infer_schema, load_schema, write_schema


def infer(*frames, low_cardinality_max=25):
    months = [(f"2026-{i + 1:02d}", df) for i, df in enumerate(frames)]
    return infer_schema(
        months,
        table="citations_by_vintage",
        layer="silver",
        low_cardinality_max=low_cardinality_max,
    )


def test_dtypes_nullability_and_ranges_cover_every_month():
    jan = citations(n=4)
    feb = citations(n=4, inspection_cycle=[None, 2, 7, 1])
    schema = infer(jan, feb)
    cycle = schema.columns["inspection_cycle"]
    assert (cycle.dtype, cycle.nullable, cycle.min, cycle.max) == ("Int64", True, 1, 7)
    assert schema.columns["ccn"].nullable is False
    assert schema.inferred_from == {"first": "2026-01", "last": "2026-02", "months": 2}


def test_low_cardinality_text_gets_its_allowed_values():
    schema = infer(citations(n=4), citations(n=4, scope_severity=["J", "D", "D", "D"]))
    assert schema.columns["scope_severity"].allowed == ["D", "E", "G", "J"]


def test_high_cardinality_text_gets_no_allowed_list():
    schema = infer(citations(n=8), low_cardinality_max=5)  # 8 distinct CCNs
    assert schema.columns["ccn"].allowed is None


def test_booleans_and_dates_get_no_range_or_allowed_list():
    schema = infer(citations(n=4))
    for name in ("is_complaint", "survey_date"):
        rule = schema.columns[name]
        assert rule.min is None and rule.max is None and rule.allowed is None


def test_write_then_load_gives_the_same_schema(tmp_path):
    schema = infer(citations(n=4))
    path = tmp_path / "citations_by_vintage.yaml"
    write_schema(schema, path)
    assert load_schema(path) == schema
    assert (
        "  scope_severity: {dtype: String, nullable: false, allowed: [D, E, G]}" in path.read_text()
    )


def test_a_hand_edit_is_kept_on_load(tmp_path):
    path = tmp_path / "citations_by_vintage.yaml"
    write_schema(infer(citations(n=4)), path)
    text = path.read_text().replace("allowed: [D, E, G]", "allowed: [D, E]")
    text = text.replace(
        "  ccn: {dtype: String, nullable: false",
        "  ccn: {dtype: String, nullable: false, str_length: 6",
    )
    path.write_text(text)
    schema = load_schema(path)
    assert schema.columns["scope_severity"].allowed == ["D", "E"]
    assert schema.columns["ccn"].str_length == 6


def test_a_missing_or_mistyped_schema_file_is_a_config_error(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_schema(tmp_path / "nope.yaml")
    bad = tmp_path / "bad.yaml"
    bad.write_text("table: x\nlayer: silver\ncolumns:\n  ccn: {dtype: String, alowed: [A]}\n")
    with pytest.raises(ConfigError, match="alowed"):
        load_schema(bad)


def test_the_key_and_a_pattern_survive_write_and_load(tmp_path):
    schema = infer(citations(n=4)).model_copy(update={"unique": ["ccn", "survey_date"]})
    schema.columns["ccn"].pattern = r"^[0-9A-Z]{6}$"
    path = tmp_path / "citations_by_vintage.yaml"
    write_schema(schema, path)
    assert load_schema(path) == schema
    assert "unique: [ccn, survey_date]" in path.read_text()


def test_a_month_column_never_gets_an_allowed_list():
    schema = infer(citations(n=4), citations(n=4, vintage="2026-08"))  # 2 months, under the limit
    assert schema.columns["vintage_month"].allowed is None


LABELS = pl.DataFrame({"ccn": ["015001"], "t": ["2025-01"], "y_g12": [0], "status_g12": ["negative"],
                       "split_g12": ["train"]})  # fmt: skip


def test_rules_survive_write_and_load_and_an_unknown_rule_is_a_config_error(tmp_path):
    schema = infer(LABELS).model_copy(update={"rules": ["score_iff_immature"]})
    path = tmp_path / "labels.yaml"
    write_schema(schema, path)
    assert load_schema(path) == schema
    path.write_text(path.read_text().replace("score_iff_immature", "score_if_immature"))
    with pytest.raises(ConfigError, match="score_if_immature"):
        load_schema(path)


def test_nulls_and_extremes_only_in_the_first_month_are_kept():
    jan = citations(n=4, inspection_cycle=[None, 9, 1, 1])
    feb = citations(n=4, inspection_cycle=[2, 3, 2, 3])  # no nulls, narrower range
    cycle = infer(jan, feb).columns["inspection_cycle"]
    assert (cycle.nullable, cycle.min, cycle.max) == (True, 1, 9)


def test_exactly_low_cardinality_max_values_still_get_an_allowed_list():
    schema = infer(citations(n=4), low_cardinality_max=3)  # D, E, G
    assert schema.columns["scope_severity"].allowed == ["D", "E", "G"]
    assert infer(citations(n=4), low_cardinality_max=2).columns["scope_severity"].allowed is None


@pytest.mark.parametrize(
    "pattern", ["^[A-", "^(?=0)[0-9]{6}$"]
)  # invalid; a look-ahead Polars can't run
def test_a_pattern_polars_cannot_run_is_a_config_error(tmp_path, pattern):
    path = tmp_path / "citations_by_vintage.yaml"
    write_schema(infer(citations(n=4)), path)
    text = path.read_text().replace(
        "  ccn: {dtype: String", f"  ccn: {{pattern: '{pattern}', dtype: String"
    )
    path.write_text(text)
    with pytest.raises(ConfigError, match="not a regex Polars can run"):
        load_schema(path)


def contract(tmp_path, schema, extra):
    path = tmp_path / "t.yaml"
    write_schema(schema, path)
    path.write_text(path.read_text().replace("columns:\n", f"{extra}\ncolumns:\n", 1))
    return path


def test_a_unique_column_not_in_the_contract_is_a_config_error(tmp_path):
    with pytest.raises(ConfigError, match=r"unique names columns not in columns: \['month'\]"):
        load_schema(contract(tmp_path, infer(citations(n=4)), "unique: [ccn, month]"))


@pytest.mark.parametrize(
    "drop, match",
    [
        (["y_g12", "status_g12", "split_g12"], "at least one status_<n>"),
        (["y_g12"], r"\['y_g12'\]"),
        (["split_g12"], r"\['split_g12'\]"),  # needed by score_iff_immature
    ],
)
def test_rules_on_a_table_without_their_label_columns_are_a_config_error(tmp_path, drop, match):
    schema = infer(LABELS.drop(drop))
    with pytest.raises(ConfigError, match=match):
        load_schema(
            contract(tmp_path, schema, "rules: [y_null_iff_unlabelled, score_iff_immature]")
        )


def test_rules_without_score_iff_immature_need_no_split_column(tmp_path):
    path = contract(tmp_path, infer(LABELS.drop("split_g12")), "rules: [y_matches_status]")
    assert load_schema(path).rules == ["y_matches_status"]


def test_allowed_severity_without_an_allowed_list_is_a_config_error(tmp_path):
    path = tmp_path / "t.yaml"
    write_schema(infer(citations(n=4)), path)
    text = path.read_text().replace(
        "  is_complaint: {", "  is_complaint: {allowed_severity: warning, "
    )
    path.write_text(text)
    with pytest.raises(ConfigError, match="allowed_severity without an allowed list"):
        load_schema(path)
