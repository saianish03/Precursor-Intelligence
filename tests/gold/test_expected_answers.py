"""Fixture tests for the fake silver world in silver_fakes.py: it says what its story says (the
facilities, the late-posted G citation, the identical fines, the 419 -> 481 code change) and uses the names of
the spec keep-lists and SILVER_CONTRACT.
"""

import re
from pathlib import Path

import polars as pl
import pytest
from silver_fakes import BY_VINTAGE, KEYS, VINTAGES, V

from precursorintelligence.common.io.table_sink import ParquetTableSink

SPEC = Path(__file__).parents[2] / "docs" / "DATA_PIPELINE_SPEC.md"

# Spine rows per facility (§7): every published t the facility appears in.
EXPECTED_SPINE_ROWS = {"015001": 26, "015002": 26, "01A193": 8, "015004": 26, "015005": 23}

MAPLE_G = (pl.col("ccn") == "015004") & (pl.col("survey_date") == pl.date(2024, 2, 10))


# ------------------------------------------------------------------ fixture tests


def _spec_columns(heading: str) -> list[str]:
    m = re.search(rf"\*\*{re.escape(heading)}:\*\*\s*`([^`]+)`", SPEC.read_text(encoding="utf-8"))
    assert m, f"keep-list {heading!r} not found in the spec"
    return m.group(1).split(", ")


@pytest.mark.parametrize(
    ("table", "heading"),
    [
        ("provider_by_vintage", "Provider Information"),
        ("citations_by_vintage", "Health Deficiencies"),
        ("surveys_by_vintage", "Inspection Dates"),
        ("penalties_by_vintage", "Penalties"),
        ("mds_qm_by_vintage", "MDS Quality Measures"),
        ("claims_qm_by_vintage", "Medicare Claims Quality Measures"),
        ("citation_lookup", "Citation Code Look-up (all vintages)"),
    ],
)
def test_by_vintage_columns_are_the_spec_keep_lists(silver, table, heading):
    expected = _spec_columns(heading)
    # Silver also writes two derived provider columns (SILVER_CONTRACT §4, configs/keep_columns.yaml).
    for derived, after in DERIVED_COLUMNS.get(table, []):
        expected.insert(expected.index(after) + 1, derived)
    assert silver[table].columns == [*expected, "vintage_month", "cutoff_date"]


DERIVED_COLUMNS = {
    "provider_by_vintage": [
        ("ownership_group", "ownership_type"),
        ("sprinkler_status_not_available", "sprinkler_status"),
    ]
}


def test_identity_table_holds_the_product_identity_columns(silver):
    expected = {*_spec_columns("Product identity only"), "ccn", "state", "chain_id", "chain_name"}
    assert set(silver["provider_identity_by_vintage"].columns) == expected | {
        "vintage_month",
        "cutoff_date",
    }


def test_reports_run_2023_01_to_2025_03_without_2023_12(silver):
    assert len(VINTAGES) == 26
    assert (VINTAGES[0], VINTAGES[-1], V) == ("2023-01", "2025-03", "2025-03")
    for name in BY_VINTAGE:
        months = set(silver[name]["vintage_month"])
        assert "2023-12" not in months
        if name != "penalties_by_vintage":  # the first fine is posted in 2023-09
            assert months == set(VINTAGES), name


def test_ccn_is_six_character_text(silver):
    for name, df in silver.items():
        if "ccn" in df.columns:
            assert df.schema["ccn"] == pl.String, name
            assert df["ccn"].str.contains(r"^[0-9A-Z]{6}$").all(), name
    assert "01A193" in set(silver["citations_by_vintage"]["ccn"])


def test_cutoff_date_is_first_of_vintage_month(silver):
    for name in BY_VINTAGE:
        df = silver[name]
        assert (df["cutoff_date"].dt.strftime("%Y-%m-%d") == df["vintage_month"] + "-01").all()


def test_within_vintage_keys_are_unique_and_sorted(silver):
    for name, key in KEYS.items():
        df = silver[name]
        assert not df.select(key).is_duplicated().any(), name
        assert df.equals(df.sort(key)), name


def test_maple_house_g_citation_is_posted_late(silver):
    cit = silver["citations_by_vintage"]
    seen = cit.filter(MAPLE_G)["vintage_month"].to_list()
    assert seen == [v for v in VINTAGES if v >= "2024-06"]
    # The 2024-03 report has Maple House (and its other citation) but not the G citation.
    march = cit.filter(pl.col("vintage_month") == "2024-03", pl.col("ccn") == "015004")
    assert march.height == 1 and march["survey_date"][0].isoformat() == "2022-12-07"
    assert silver["ledger_citations"].filter(MAPLE_G)["first_seen_vintage"].to_list() == ["2024-06"]


def test_gplus_citations_in_the_ledger(silver):
    gplus = silver["ledger_citations"].filter(pl.col("final_scope_severity").is_in(list("GHIJKL")))
    assert sorted(
        (r["ccn"], r["survey_date"].isoformat(), r["first_seen_vintage"])
        for r in gplus.iter_rows(named=True)
    ) == [("015001", "2023-06-15", "2023-08"), ("015004", "2024-02-10", "2024-06")]
    oak = silver["citations_by_vintage"].filter(pl.col("ccn") == "015002")
    assert set(oak["scope_severity"]) == {"D", "E"}
    assert not silver["ledger_citations"]["removed_by_cms"].any()


def test_facility_presence(silver):
    rows = {r["ccn"]: r for r in silver["facility_presence"].iter_rows(named=True)}
    assert {
        c: (r["first_vintage"], r["last_vintage"], r["n_vintages"]) for c, r in rows.items()
    } == {
        "015001": ("2023-01", "2025-03", 26),
        "015002": ("2023-01", "2025-03", 26),
        "01A193": ("2023-01", "2023-08", 8),
        "015004": ("2023-01", "2025-03", 26),
        "015005": ("2023-04", "2025-03", 23),
    }
    universe = silver["facility_universe"].group_by("ccn").len()
    assert dict(universe.iter_rows()) == EXPECTED_SPINE_ROWS
    dim = silver["dim_facility_latest"]
    assert dim.filter(~pl.col("is_active"))["ccn"].to_list() == ["01A193"]


def test_sunny_acres_fine_is_on_its_g_citation_day(silver):
    fines = silver["ledger_penalties"].filter(pl.col("ccn") == "015001")
    assert [
        (d.isoformat(), s) for d, s in fines.select("penalty_date", "first_seen_vintage").rows()
    ] == [("2023-06-15", "2023-09")]
    g_days = silver["ledger_citations"].filter(
        pl.col("ccn") == "015001", pl.col("final_scope_severity") == "G"
    )["survey_date"]
    assert fines["penalty_date"].to_list() == g_days.to_list()


def test_oak_grove_identical_fines_get_dup_seq_1_and_2(silver):
    pen = silver["penalties_by_vintage"].filter(pl.col("ccn") == "015002")
    for _, rows in pen.group_by("vintage_month"):
        assert rows["dup_seq"].to_list() == [1, 2]
        assert rows["penalty_event_key"].n_unique() == 2
        assert rows.drop("dup_seq", "penalty_event_key").n_unique() == 1  # otherwise identical
    assert sorted(set(pen["vintage_month"])) == [v for v in VINTAGES if v >= "2024-01"]
    ledger = silver["ledger_penalties"].filter(pl.col("ccn") == "015002")
    assert sorted(ledger["dup_seq"]) == [1, 2]  # two events, not one


def test_antipsychotic_code_changes_from_419_to_481_in_2024_06(silver):
    mds = silver["mds_qm_by_vintage"]
    for code, expected in [
        ("419", [v for v in VINTAGES if v < "2024-06"]),
        ("481", [v for v in VINTAGES if v >= "2024-06"]),
    ]:
        assert sorted(set(mds.filter(pl.col("measure_code") == code)["vintage_month"])) == expected
    both = mds.filter(pl.col("measure_code").is_in(["419", "481"]))
    assert not both.select("ccn", "vintage_month").is_duplicated().any()  # never both in one report
    inv = silver["qm_inventory"].filter(pl.col("measure_code").is_in(["419", "481"]))
    assert inv.select("measure_code", "first_vintage", "last_vintage").rows() == [
        ("419", "2023-01", "2024-05"),
        ("481", "2024-06", "2025-03"),
    ]


def test_quality_rows_per_facility_and_measure_period(silver):
    per_report = {"mds_qm_by_vintage": 3, "claims_qm_by_vintage": 2}
    for name, n in per_report.items():
        counts = silver[name].group_by("ccn", "vintage_month").len()["len"]
        assert set(counts) == {n}, name
    aug = silver["mds_qm_by_vintage"].filter(pl.col("vintage_month") == "2024-08")
    assert set(aug["measure_period"]) == {"2023Q2-2024Q1"}  # the spec's §6 example


def test_write_silver_uses_the_spec_layout(silver, silver_store):
    assert silver_store.exists("silver/citations_by_vintage/vintage=2024-03/part-0.parquet")
    assert not silver_store.exists("silver/citations_by_vintage/vintage=2023-12/part-0.parquet")
    assert silver_store.exists("silver/ledger_citations/part-0.parquet")
    assert silver_store.exists(
        "silver/penalties_by_vintage/vintage=2023-01/part-0.parquet"
    )  # empty
    assert silver_store.exists("silver/qm_inventory/part-0.parquet")
    assert silver_store.exists("gold/dim_facility_latest/part-0.parquet")
    back = ParquetTableSink(silver_store, prefix="silver").read("citations_by_vintage")
    key = KEYS["citations_by_vintage"]
    assert back.drop("vintage").sort(key).equals(silver["citations_by_vintage"])
