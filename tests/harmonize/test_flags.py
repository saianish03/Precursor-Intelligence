"""stale_refresh and revised_after_publication (docs/DATA_PIPELINE_SPEC.md §3.1 step 8 and "Republished data")."""

from datetime import UTC, datetime

from bronze_fakes import csv_text

from precursorintelligence.harmonize.bronze import load_file
from precursorintelligence.harmonize.flags import (
    EVENT_TABLES,
    REVISED,
    STALE_REFRESH,
    refresh_stale_refresh,
)

NOW = datetime(2026, 10, 16, 12, 0, tzinfo=UTC)
ROWS = [("015009", "Oak Manor"), ("15E123", "Pine Court")]
OTHER_ROWS = [("015009", "Oak Manor"), ("15E123", "Changed")]


def load(store, meta, sink, inv, run_id="R1"):
    return load_file(store, meta, sink, inv, run_id=run_id, now=NOW)


def load_release(
    store, meta, sink, put_landing, vintage, rows_by_table, run_id="R1", zip_sha="a" * 64
):
    for table, rows in rows_by_table.items():
        inv = put_landing(
            csv_text(rows, processing=f"{vintage}-01"),
            table=table,
            vintage=vintage,
            processing_date=f"{vintage}-01",
            member=f"NH_{table}_{vintage}.csv",
            zip_sha=zip_sha,
        )
        load(store, meta, sink, inv, run_id=run_id)


def flag(meta, vintage, name=STALE_REFRESH, table=""):
    rows = meta.read("vintage_flags", {"vintage_month": vintage, "flag": name, "table_name": table})
    return rows[0]["value"] if rows else None


def same(rows):
    return {t: rows for t in EVENT_TABLES}


def test_a_release_whose_three_event_tables_are_unchanged_is_a_stale_refresh(
    store, meta, sink, put_landing
):
    load_release(store, meta, sink, put_landing, "2025-07", same(ROWS))
    load_release(store, meta, sink, put_landing, "2025-09", same(ROWS), zip_sha="b" * 64)

    assert refresh_stale_refresh(meta, "2025-09", run_id="R1", now=NOW) is True
    assert flag(meta, "2025-09") is True


def test_one_changed_event_table_means_the_release_is_not_stale(store, meta, sink, put_landing):
    load_release(store, meta, sink, put_landing, "2025-07", same(ROWS))
    changed = same(ROWS) | {"penalties": OTHER_ROWS}
    load_release(store, meta, sink, put_landing, "2025-08", changed, zip_sha="b" * 64)

    assert refresh_stale_refresh(meta, "2025-08", run_id="R1", now=NOW) is False


def test_the_first_vintage_has_nothing_to_be_stale_against(store, meta, sink, put_landing):
    load_release(store, meta, sink, put_landing, "2025-07", same(ROWS))
    assert refresh_stale_refresh(meta, "2025-07", run_id="R1", now=NOW) is False


def test_unchanged_reference_tables_do_not_make_a_release_stale(store, meta, sink, put_landing):
    load_release(store, meta, sink, put_landing, "2025-07", same(ROWS) | {"citation_lookup": ROWS})
    changed = same(OTHER_ROWS) | {"citation_lookup": ROWS}
    load_release(store, meta, sink, put_landing, "2025-08", changed, zip_sha="b" * 64)

    assert refresh_stale_refresh(meta, "2025-08", run_id="R1", now=NOW) is False


def test_a_release_missing_an_event_table_gets_no_flag(store, meta, sink, put_landing):
    partial = {t: ROWS for t in EVENT_TABLES[:2]}
    load_release(store, meta, sink, put_landing, "2025-07", partial)

    assert refresh_stale_refresh(meta, "2025-07", run_id="R1", now=NOW) is None
    assert flag(meta, "2025-07") is None


def test_the_flag_is_one_row_that_follows_the_latest_load(store, meta, sink, put_landing):
    load_release(store, meta, sink, put_landing, "2025-07", same(ROWS))
    load_release(store, meta, sink, put_landing, "2025-09", same(ROWS), zip_sha="b" * 64)
    refresh_stale_refresh(meta, "2025-09", run_id="R1", now=NOW)
    # CMS republishes 2025-09 with real changes: the flag must flip, not duplicate.
    load_release(
        store, meta, sink, put_landing, "2025-09", same(OTHER_ROWS), run_id="R2", zip_sha="c" * 64
    )
    refresh_stale_refresh(meta, "2025-09", run_id="R2", now=NOW)

    assert len(meta.read("vintage_flags", {"vintage_month": "2025-09", "flag": STALE_REFRESH})) == 1
    assert flag(meta, "2025-09") is False


def test_a_reload_with_different_content_is_flagged_revised_for_that_table_only(
    store, meta, sink, put_landing
):
    load_release(store, meta, sink, put_landing, "2024-03", same(ROWS))
    revised = put_landing(
        csv_text(OTHER_ROWS, processing="2024-03-01"),
        table="penalties",
        vintage="2024-03",
        processing_date="2024-03-01",
        member="NH_penalties_2024-03.csv",
        zip_sha="b" * 64,
    )
    result = load(store, meta, sink, revised, run_id="R2")

    assert result.status == "reloaded" and result.revised is True
    assert flag(meta, "2024-03", REVISED, "penalties") is True
    assert flag(meta, "2024-03", REVISED, "survey_dates") is None


def test_a_reload_with_identical_content_is_not_a_revision(store, meta, sink, put_landing):
    load_release(store, meta, sink, put_landing, "2024-03", same(ROWS))
    republished = put_landing(
        csv_text(ROWS, processing="2024-03-01"),
        table="penalties",
        vintage="2024-03",
        processing_date="2024-03-01",
        member="NH_penalties_2024-03.csv",
        zip_sha="b" * 64,
    )
    result = load(store, meta, sink, republished, run_id="R2")

    assert result.status == "reloaded" and result.revised is False
    assert flag(meta, "2024-03", REVISED, "penalties") is None


def test_the_revised_flag_stays_set_once_raised(store, meta, sink, put_landing):
    load_release(store, meta, sink, put_landing, "2024-03", same(ROWS))
    for sha, rows in (("b" * 64, OTHER_ROWS), ("c" * 64, OTHER_ROWS)):
        inv = put_landing(
            csv_text(rows, processing="2024-03-01"),
            table="penalties",
            vintage="2024-03",
            processing_date="2024-03-01",
            member="NH_penalties_2024-03.csv",
            zip_sha=sha,
        )
        load(store, meta, sink, inv, run_id=f"R-{sha[0]}")

    assert flag(meta, "2024-03", REVISED, "penalties") is True
