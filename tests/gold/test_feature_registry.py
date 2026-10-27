"""configs/feature_registry.yaml lists exactly the built features, with the built dtypes and the column
contract's model_use (spec §8, §11.1)."""

import csv

import polars as pl
import pytest
from silver_fakes import VINTAGES, V

from precursorintelligence.common.config import load_feature_registry, project_root
from precursorintelligence.gold.build import build_spine
from precursorintelligence.gold.features import FAMILIES, build_features

DTYPES = {"f64": pl.Float64, "i64": pl.Int64, "bool": pl.Boolean, "str": pl.String}
CONTRACT = project_root() / "configs" / "contracts" / "facility_month_dataset_spec_columns.csv"


@pytest.fixture(scope="module")
def registry():
    return load_feature_registry()


@pytest.fixture(scope="module")
def contract():
    with CONTRACT.open(encoding="utf-8") as f:
        return {r["column"]: r for r in csv.DictReader(f)}


@pytest.fixture(scope="module")
def built(silver):
    spine = build_spine(
        silver["facility_universe"], published_vintages=VINTAGES, spine_start="2023-01", v=V
    )
    return {family: build_features(family, silver, spine) for family in FAMILIES}


def test_every_built_family_is_registered_column_for_column(registry, built):
    for family, df in built.items():
        assert [f.name for f in registry.family(family)] == df.columns[2:], family


def test_the_registry_lists_only_built_families(registry):
    assert {f.family for f in registry.features} == set(FAMILIES)


def test_registered_dtypes_are_the_built_dtypes(registry, built):
    for family, df in built.items():
        for spec in registry.family(family):
            assert df.schema[spec.name] == DTYPES[spec.dtype], spec.name


def test_registry_matches_the_column_contract(registry, contract):
    for spec in registry.features:
        row = contract[spec.name]
        assert (spec.family, spec.dtype, spec.model_use) == (
            row["family"], row["dtype"], row["model_use"],
        ), spec.name  # fmt: skip
    built_families = {f.family for f in registry.features}
    expected = {c for c, r in contract.items() if r["family"] in built_families}
    assert {f.name for f in registry.features} == expected


def test_star_ratings_and_chain_follow_the_contract(registry):
    use = {f.name: f.model_use for f in registry.features}
    assert use["staffing_rating"] == use["qm_rating"] == "model"
    assert use["health_inspection_rating"] == use["overall_rating"] == "ablation_only"
    assert use["health_inspection_rating_fn"] == use["overall_rating_fn"] == "ablation_only"
    assert use["chain_id"] == "product_only"
