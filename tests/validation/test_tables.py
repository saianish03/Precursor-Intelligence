"""Reading silver and gold tables month by month, read-only."""

import polars as pl
from validation_fakes import citations

from precursorintelligence.common.config import ValidationLayer
from precursorintelligence.validation.tables import TableReader

SILVER = ValidationLayer(partition_key="vintage", tables=["citations_by_vintage"])
GOLD = ValidationLayer(month_column="t", tables=["spine"])


def test_silver_months_come_from_the_partition_folders_in_order(store, put_silver):
    for v in ("2026-08", "2026-06", "2026-07"):
        put_silver("citations_by_vintage", v, citations(vintage=v))
    reader = TableReader(store, "silver", SILVER, "citations_by_vintage")
    assert reader.months() == ["2026-06", "2026-07", "2026-08"]
    assert reader.month("2026-07")["vintage_month"].unique().to_list() == ["2026-07"]


def test_other_files_in_the_table_folder_are_not_months(store, put_silver):
    put_silver("citations_by_vintage", "2026-07", citations())
    store.put_bytes("silver/citations_by_vintage/_manifest.json", b"{}")
    assert TableReader(store, "silver", SILVER, "citations_by_vintage").months() == ["2026-07"]


def test_gold_months_are_split_by_the_month_column(store, put_gold):
    spine = pl.DataFrame(
        {"ccn": ["015001", "015002", "015001"], "t": ["2025-02", "2025-01", "2025-01"]}
    )
    put_gold("spine", spine)
    reader = TableReader(store, "gold", GOLD, "spine")
    assert reader.months() == ["2025-01", "2025-02"]
    assert reader.month("2025-01").height == 2


def test_a_table_missing_from_the_store_has_no_months(store):
    assert TableReader(store, "silver", SILVER, "citations_by_vintage").months() == []
    assert TableReader(store, "gold", GOLD, "spine").months() == []


def test_reading_writes_nothing(store, put_silver):
    put_silver("citations_by_vintage", "2026-07", citations())
    before = store.list("")
    reader = TableReader(store, "silver", SILVER, "citations_by_vintage")
    reader.month(reader.months()[0])
    assert store.list("") == before
