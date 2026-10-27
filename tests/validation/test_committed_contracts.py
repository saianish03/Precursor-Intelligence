"""The committed contracts in configs/contracts/: one per configured silver table, with the hand-tightened rules."""

import re

import polars as pl
import pytest

from precursorintelligence.common.config import load_features_config, load_validation_config
from precursorintelligence.harmonize.columns import load_silver_config
from precursorintelligence.harmonize.normalize import CCN_PATTERN, POLARS_TYPES
from precursorintelligence.validation.schema import TableSchema, load_schema
from precursorintelligence.validation.validate import check_schema, pandera_schema

CFG = load_validation_config()
SILVER = CFG.layers["silver"].tables
ALL = SILVER + CFG.layers["gold"].tables


@pytest.mark.parametrize("table", SILVER)
def test_each_silver_table_has_a_loadable_baseline(table):
    schema = load_schema(CFG.schema_path(table))
    assert schema.table == table and schema.layer == "silver"
    pandera_schema(schema)  # every dtype and rule turns into a Pandera check


@pytest.mark.parametrize("table", ALL)
def test_ccn_is_six_character_text_everywhere(table):
    ccn = load_schema(CFG.schema_path(table)).columns["ccn"]
    assert (ccn.dtype, ccn.nullable, ccn.pattern) == ("String", False, CCN_PATTERN)


@pytest.mark.parametrize("table", ALL)
def test_every_month_column_is_yyyy_mm(table):
    columns = load_schema(CFG.schema_path(table)).columns
    month = columns.get("vintage_month") or columns["t"]
    assert month.pattern is not None
    assert re.fullmatch(month.pattern, "2026-08") and not re.fullmatch(month.pattern, "2026/08")
    assert not re.fullmatch(month.pattern, "2026-13")


@pytest.mark.parametrize("table", SILVER)
def test_silver_key_is_the_keep_columns_key(table):
    schema = load_schema(CFG.schema_path(table))
    assert schema.unique == list(load_silver_config().tables[table].key)


@pytest.mark.parametrize("table", CFG.layers["gold"].tables)
def test_gold_key_is_one_row_per_facility_and_month(table):
    assert load_schema(CFG.schema_path(table)).unique == ["ccn", "t"]


def test_severity_is_the_full_a_to_l_grid():
    rule = load_schema(CFG.schema_path("citations_by_vintage")).columns["scope_severity"]
    assert rule.allowed == list("ABCDEFGHIJKL")


def test_star_ratings_are_checked_only_as_data_in_1_to_5():
    rule = load_schema(CFG.schema_path("provider_by_vintage")).columns["overall_rating"]
    assert (rule.min, rule.max) == (1.0, 5.0)


# Columns silver derives in code (configs/keep_columns.yaml header), with the dtype it writes them as.
DERIVED = {
    "ownership_group": "String",
    "sprinkler_status_not_available": "Boolean",
    "ftag": "String",
    "dup_seq": "Int64",
    "penalty_event_key": "String",
    "vintage_month": "String",
    "cutoff_date": "Date",
}


@pytest.mark.parametrize("table", SILVER)
def test_silver_contract_columns_and_dtypes_match_the_silver_configs(table):
    """The contract lists exactly what silver writes, so a silver config change can't fail every validate run."""
    silver = load_silver_config()
    spec = silver.tables[table]
    source = silver.sources[spec.source].columns
    contract = load_schema(CFG.schema_path(table)).columns
    assert list(contract) == [*spec.columns, "vintage_month", "cutoff_date"]
    for name, rule in contract.items():
        expected = str(POLARS_TYPES[source[name].dtype]) if name in source else DERIVED[name]
        assert rule.dtype == expected, name


def _contract_rules():
    silver = load_silver_config()
    for table in SILVER:
        source = silver.sources[silver.tables[table].source].columns
        for name, rule in load_schema(CFG.schema_path(table)).columns.items():
            yield table, name, rule, source[name].dtype if name in source else None


def test_every_footnote_column_allows_exactly_the_codes_gold_maps():
    codes = sorted(load_features_config().footnote_reasons)
    fn = [(t, n, r) for t, n, r, dtype in _contract_rules() if dtype == "footnote"]
    assert {n for _, n, _ in fn} >= {"overall_rating_fn", "survey_fn"}
    for table, name, rule in fn:
        assert (rule.allowed, rule.allowed_severity) == (codes, None), (table, name)


def test_mapped_columns_allow_exactly_what_silver_maps_to():
    value_maps = load_silver_config().value_maps
    for _table, name, rule, dtype in _contract_rules():
        if dtype == "enum":
            enum = value_maps["enums"][name]
            expected = set(enum["map"].values()) | ({enum["blank"]} if "blank" in enum else set())
            assert (set(rule.allowed or []), rule.allowed_severity) == (expected, None), name
        elif dtype == "survey_type_enum":
            # silver writes OTHER for an unknown type on purpose: listed nowhere, so it is a warning
            assert set(rule.allowed or []) == set(value_maps["survey_type"]["map"].values())
            assert rule.allowed_severity == "warning"


@pytest.mark.parametrize(
    "table, column",
    [
        ("citations_by_vintage", "deficiency_category"),
        ("citations_by_vintage", "scope_severity"),
        ("penalties_by_vintage", "penalty_type"),
        ("provider_by_vintage", "provider_type"),
    ],
)
def test_values_gold_logic_depends_on_stay_hard_errors(table, column):
    rule = load_schema(CFG.schema_path(table)).columns[column]
    assert rule.allowed and rule.allowed_severity is None


@pytest.mark.parametrize(
    "table, column",
    [("provider_by_vintage", "ownership_type"), ("citations_by_vintage", "deficiency_corrected")],
)
def test_free_cms_text_only_warns_on_a_new_value(table, column):
    assert load_schema(CFG.schema_path(table)).columns[column].allowed_severity == "warning"


@pytest.mark.parametrize(
    "table, column, good, bad",
    [
        ("provider_by_vintage", "state", "CA", "ca"),
        ("citations_by_vintage", "ftag", "F0880", "F880"),
        ("citations_by_vintage", "deficiency_tag", "0880", "880A"),
        ("labels", "t", "2025-01", "2025-1"),
    ],
)
def test_committed_patterns_accept_the_format_and_reject_a_near_miss(table, column, good, bad):
    pattern = load_schema(CFG.schema_path(table)).columns[column].pattern
    assert pattern and re.fullmatch(pattern, good) and not re.fullmatch(pattern, bad)


@pytest.mark.parametrize("table", ALL)
def test_a_ccn_with_a_valid_prefix_and_extra_characters_fails_the_committed_pattern(table):
    """Pandera runs the pattern from the start only: without the end anchor, 0150011 would pass."""
    ccn = load_schema(CFG.schema_path(table)).columns["ccn"]
    schema = TableSchema(table=table, layer="x", columns={"ccn": ccn})
    found = check_schema(pl.DataFrame({"ccn": ["015001", "0150011"]}), schema, month="2026-07")
    assert [(a.check, a.observed) for a in found] == [
        ("pattern", {"rows": 1, "examples": ["0150011"]})
    ]
