"""The committed contracts in configs/contracts/: one per configured silver table, with the hand-tightened rules."""

import pytest

from precursorintelligence.common.config import load_validation_config
from precursorintelligence.harmonize.columns import load_silver_config
from precursorintelligence.harmonize.normalize import POLARS_TYPES
from precursorintelligence.validation.schema import load_schema
from precursorintelligence.validation.validate import pandera_schema

CFG = load_validation_config()
SILVER = CFG.layers["silver"].tables


@pytest.mark.parametrize("table", SILVER)
def test_each_silver_table_has_a_loadable_baseline(table):
    schema = load_schema(CFG.schema_path(table))
    assert schema.table == table and schema.layer == "silver"
    pandera_schema(schema)  # every dtype and rule turns into a Pandera check


@pytest.mark.parametrize("table", SILVER)
def test_ccn_is_six_character_text_everywhere(table):
    ccn = load_schema(CFG.schema_path(table)).columns["ccn"]
    assert (ccn.dtype, ccn.nullable, ccn.str_length) == ("String", False, 6)


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
