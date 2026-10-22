"""The feature reader refuses every input a feature may not read (spec §8, §11.3; SILVER_CONTRACT §6)."""

import sys
from pathlib import Path

import pytest

from precursorintelligence.common.io.object_store import LocalObjectStore
from precursorintelligence.gold.features.reader import (
    FEATURE_INPUTS,
    FeatureReader,
    FeatureReadError,
    shift_month,
)

sys.path.insert(0, str(Path(__file__).parents[1] / "gold"))  # the toy world lives in tests/gold
from silver_fakes import VINTAGES, build_silver, write_silver  # noqa: E402

FORBIDDEN = [
    "ledger_citations",
    "ledger_penalties",
    "ledger_surveys",
    "facility_presence",
    "facility_universe",
    "citation_lookup_versions",
    "qm_inventory",
    "provider_identity_by_vintage",
    "dim_facility_latest",
    "labels",
]


@pytest.fixture(scope="module")
def reader() -> FeatureReader:
    return FeatureReader.from_frames(build_silver())


def test_feature_inputs_are_the_by_vintage_tables_and_the_lookup():
    assert set(FEATURE_INPUTS) == {
        "provider_by_vintage", "citations_by_vintage", "surveys_by_vintage",
        "penalties_by_vintage", "mds_qm_by_vintage", "claims_qm_by_vintage", "citation_lookup",
    }  # fmt: skip


@pytest.mark.parametrize("table", FORBIDDEN)
def test_reader_refuses_tables_outside_the_whitelist(reader, table):
    with pytest.raises(FeatureReadError, match=table):
        reader.read(table, "2024-06")


@pytest.mark.parametrize("months_back", [-1, 1, 3, 24])
def test_reader_refuses_any_offset_but_t_t6_t12(reader, months_back):
    with pytest.raises(FeatureReadError):
        reader.read("provider_by_vintage", "2024-06", months_back=months_back)


def test_reader_returns_only_the_asked_vintage(reader):
    df = reader.read("citations_by_vintage", "2024-07", months_back=6)
    assert set(df["vintage_month"]) == {"2024-01"}


def test_a_never_published_vintage_reads_as_none(reader):
    # 2023-12 was never published in the toy world, so t−6 of 2024-06 does not exist.
    assert reader.read("provider_by_vintage", "2024-06", months_back=6) is None
    assert set(reader.read("provider_by_vintage", "2024-06", months_back=12)["vintage_month"]) == {
        "2023-06"
    }


def test_reader_records_every_read():
    r = FeatureReader.from_frames(build_silver())
    r.read("penalties_by_vintage", "2024-06")
    r.read("provider_by_vintage", "2024-06", months_back=12)
    assert r.reads == [("penalties_by_vintage", "2024-06"), ("provider_by_vintage", "2023-06")]


def test_a_limited_reader_refuses_anything_after_its_limit():
    r = FeatureReader.from_frames(build_silver(), limit="2024-05")
    r.read("provider_by_vintage", "2024-05")
    with pytest.raises(FeatureReadError, match="limit"):
        r.read("provider_by_vintage", "2024-06")
    with pytest.raises(FeatureReadError, match="limit"):  # t−6 is allowed, but t itself is too late
        r.read("provider_by_vintage", "2024-11", months_back=6)


def test_store_reader_opens_only_the_partition_it_needs(tmp_path):
    class Recording(LocalObjectStore):
        opened: list[str] = []

        def open_read(self, key):  # type: ignore[no-untyped-def]
            self.opened.append(key)
            return super().open_read(key)

    store = Recording(tmp_path, immutable=None)
    write_silver(store, build_silver())
    r = FeatureReader.from_store(store, published=VINTAGES)
    df = r.read("surveys_by_vintage", "2024-06")
    assert store.opened == ["silver/surveys_by_vintage/vintage=2024-06/part-0.parquet"]
    assert set(df["vintage_month"]) == {"2024-06"}


@pytest.mark.parametrize(
    ("t", "months", "expected"),
    [("2024-06", -6, "2023-12"), ("2024-01", -12, "2023-01"), ("2023-12", 1, "2024-01")],
)
def test_shift_month(t, months, expected):
    assert shift_month(t, months) == expected
