"""Step 1 on the fake CMS archive, then bronze on what Step 1 recorded: real inventory rows, no hand-built ones.

Lives next to the ingestion tests because it needs the fake server and the `make_ctx` fixture."""

import json

import pytest
from cms_fakes import TABLE_FILES, csv_bytes, make_zip
from test_pipeline_offline import build_archive

from precursorintelligence.harmonize.pipeline import run_bronze
from precursorintelligence.ingestion.pipeline import EXIT_OK, run_step1


def selected_files(ctx):
    return ctx.metadata.read("inventory", {"status": "selected"})


def statuses(result):
    return {(f.table, f.vintage): f.status for f in result.files}


@pytest.fixture(params=["local", "memory"])
def landed(request, fake_cms, make_ctx):
    """The full fake archive, downloaded, extracted and inventoried."""
    build_archive(fake_cms)
    ctx = make_ctx(fake_cms, backend=request.param)
    assert not run_step1(ctx).failures
    return ctx


def test_bronze_loads_every_file_step1_selected(landed):
    ctx = landed
    result = run_bronze(ctx)

    assert result.exit_code == EXIT_OK, result.failures
    selected = selected_files(ctx)
    assert len(result.files) == len(selected) == 14  # 2021-01 and 2021-03, 7 tables each
    assert {f.status for f in result.files} == {"loaded"}
    for inv in selected:
        part = ctx.table_sink.read(inv["table_name"], {"vintage": inv["vintage_month"]})
        assert part.height == inv["row_count"] == 3
        assert set(part["_source_zip"]) == {inv["zip_name"]}
        assert set(part["_zip_sha256"]) == {inv["zip_sha256"]}
        assert set(part["_header_hash"]) == {inv["header_hash"]}
        assert part["_row_number"].to_list() == [1, 2, 3]
        assert part["CMS Certification Number (CCN)"].to_list() == ["000000", "000001", "000002"]


def test_the_stale_month_is_the_one_whose_event_tables_repeat_the_previous_release(landed):
    result = run_bronze(landed)
    # The fake archive repeats the same data every month (only Processing Date moves).
    assert result.stale_vintages == ["2021-03"]


def test_a_second_run_is_a_no_op(landed):
    ctx = landed
    run_bronze(ctx)
    rows_before = len(ctx.metadata.read("bronze_files"))
    again = run_bronze(ctx)

    assert again.exit_code == EXIT_OK
    assert {f.status for f in again.files} == {"skipped"}
    assert len(ctx.metadata.read("bronze_files")) == rows_before


def test_a_republished_zip_reloads_that_vintage_and_flags_only_what_changed(fake_cms, make_ctx):
    rel = fake_cms.add("2021-01-27", make_zip("2021-01", "Jan2021"))
    ctx = make_ctx(fake_cms)
    run_step1(ctx)
    first = run_bronze(ctx)
    assert {f.status for f in first.files} == {"loaded"}

    others = [t for t in TABLE_FILES if t != "provider_info"]
    republished = make_zip(
        "2021-01",
        "Jan2021",
        tables=others,
        extra={"NH_ProviderInfo_Jan2021.csv": csv_bytes("2021-01", extra_rows=5)},
    )
    fake_cms.replace(rel, republished)
    assert run_step1(ctx).downloads == {"republished": 1}
    second = run_bronze(ctx)

    assert second.exit_code == EXIT_OK
    assert set(statuses(second).values()) == {
        "reloaded"
    }  # a new zip: every file of the vintage reloads
    assert [(f.table, f.vintage) for f in second.files if f.revised] == [
        ("provider_info", "2021-01")
    ]
    assert ctx.table_sink.read("provider_info", {"vintage": "2021-01"}).height == 5
    manifest = json.loads(ctx.object_store.open_read(second.manifest_key).read())
    assert manifest["revised"] == [{"table": "provider_info", "vintage": "2021-01"}]
    flags = ctx.metadata.read("vintage_flags", {"flag": "revised_after_publication"})
    assert [(r["table_name"], r["vintage_month"], r["value"]) for r in flags] == [
        ("provider_info", "2021-01", True)
    ]
