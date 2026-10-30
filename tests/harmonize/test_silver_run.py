"""The silver node on bronze built by the real bronze node: build, skip, rebuild, fail closed (spec §3.5)."""

import json
from dataclasses import replace
from datetime import UTC, date, datetime
from shutil import copy

import polars as pl
import pytest
import yaml

from precursorintelligence.common.config import project_root
from precursorintelligence.harmonize import silver_pipeline
from precursorintelligence.harmonize.columns import Alias, Column, load_silver_config
from precursorintelligence.harmonize.pipeline import run_bronze
from precursorintelligence.harmonize.silver import harmonize
from precursorintelligence.harmonize.silver_pipeline import run_silver, silver_sink
from precursorintelligence.harmonize.silver_report import NEW_HEADER
from precursorintelligence.ingestion.pipeline import EXIT_OK, EXIT_VALIDATION

NOW = datetime(2026, 10, 18, 12, 0, tzinfo=UTC)


def write_config(folder, *, extra_ignored=(), name_dtype="str"):
    """A tiny column map for the 3-column provider CSVs of the bronze fixtures."""
    folder.mkdir(exist_ok=True)
    alias = lambda raw: [{"raw": raw, "from": "2021-01", "to": None}]  # noqa: E731
    cmap = {
        "provider_info": {
            "columns": {
                "ccn": {"dtype": "ccn", "aliases": alias("CMS Certification Number (CCN)")},
                "provider_name": {"dtype": name_dtype, "aliases": alias("Provider Name")},
                "processing_date": {"dtype": "date", "aliases": alias("Processing Date")},
            },
            "ignored": list(extra_ignored),
        }
    }
    keep = {
        "provider_by_vintage": {
            "source": "provider_info",
            "key": ["ccn"],
            "columns": ["ccn", "provider_name"],
        }
    }
    (folder / "column_map.yaml").write_text(yaml.safe_dump(cmap), encoding="utf-8")
    (folder / "keep_columns.yaml").write_text(yaml.safe_dump(keep), encoding="utf-8")
    (folder / "value_maps.yaml").write_text(
        yaml.safe_dump({"yes_no": {"Y": True, "N": False}}), encoding="utf-8"
    )
    copy(project_root() / "configs" / "calibration.yaml", folder / "calibration.yaml")
    return load_silver_config(folder)


@pytest.fixture(autouse=True)
def no_all_vintage_tables(monkeypatch):
    """Steps 3-5 need all 7 tables; these tests stage provider files only (the ledgers have their own tests)."""
    monkeypatch.setattr(
        silver_pipeline,
        "build_ledgers_and_universe",
        lambda ctx, v: {"frames": {}, "qm_presence": None, "counts": {}},
    )


def statuses(meta):
    return [
        (r["vintage_month"], r["status"])
        for r in sorted(meta.read("silver_partitions"), key=lambda r: r["id"])
    ]


def test_bronze_becomes_typed_silver_with_time_columns(ctx, meta, stage, tmp_path):
    stage("provider_info", "2025-06")
    run_bronze(ctx, now=NOW)
    res = run_silver(ctx, cfg=write_config(tmp_path / "cfg"), now=NOW)

    assert res.exit_code == EXIT_OK and [p.status for p in res.partitions] == ["built"]
    out = silver_sink(ctx).read("provider_by_vintage", {"vintage": "2025-06"})
    assert out.sort("ccn").select(
        "ccn", "provider_name", "vintage_month", "cutoff_date"
    ).rows() == [
        ("015009", "Oak Manor", "2025-06", date(2025, 6, 1)),
        ("15E123", "Pine Court", "2025-06", date(2025, 6, 1)),
    ]
    assert ctx.object_store.exists("reports/step2/latest/02_harmonization.md")


def test_a_rerun_skips_and_a_config_change_rebuilds(ctx, meta, stage, tmp_path):
    stage("provider_info", "2025-06")
    run_bronze(ctx, now=NOW)
    cfg = write_config(tmp_path / "a")
    run_silver(ctx, cfg=cfg, now=NOW)
    again = run_silver(ctx, cfg=cfg, now=NOW)
    changed = run_silver(ctx, cfg=write_config(tmp_path / "b", extra_ignored=["Unused"]), now=NOW)

    assert [p.status for p in again.partitions] == ["skipped"]
    assert [p.status for p in changed.partitions] == ["built"]
    assert statuses(meta) == [("2025-06", "built"), ("2025-06", "built")]


def test_a_renamed_header_fails_closed_and_is_retried_after_the_fix(
    ctx, meta, stage, put_landing, tmp_path
):
    stage("provider_info", "2025-06")
    run_bronze(ctx, now=NOW)
    cfg = write_config(tmp_path / "cfg")
    # a CMS rename the column map does not know yet: it expects a name the file no longer has
    narrow = load_silver_config(tmp_path / "cfg")
    narrow.sources["provider_info"].columns["provider_name"] = Column(
        "provider_name", "str", (Alias("Facility Name", "2021-01", None),)
    )
    res = run_silver(ctx, cfg=narrow, now=NOW)

    assert res.exit_code == EXIT_VALIDATION
    assert res.partitions[0].problems == [
        ("provider_name", "expected header 'Facility Name' is missing"),
        ("Provider Name", "unmapped header"),
    ]
    assert silver_sink(ctx).list_partitions("provider_by_vintage") == []
    assert ctx.object_store.exists("reports/step2/latest/02_unmapped_headers.csv")

    fixed = run_silver(ctx, cfg=cfg, now=NOW)
    assert fixed.exit_code == EXIT_OK and [p.status for p in fixed.partitions] == ["built"]
    assert statuses(meta) == [("2025-06", "failed"), ("2025-06", "built")]
    # silver returns its failures instead of raising them; run_log must still show the failed step
    steps = [r for r in meta.read("run_log") if r["task"] == "silver"]
    assert [(r["status"], json.loads(r["detail"])["n_failures"]) for r in steps] == [
        ("failed", 1),
        ("ok", 0),
    ]


def config_without_provider_name(folder):
    """A column map that doesn't know the fixtures' `Provider Name`: a new CMS column (spec §3.2)."""
    write_config(folder)
    cfg = load_silver_config(folder)
    cfg.sources["provider_info"].columns.pop("provider_name")
    cfg.tables["provider_by_vintage"] = replace(cfg.tables["provider_by_vintage"], columns=("ccn",))
    return cfg


def unmapped_report(ctx):
    csv = ctx.object_store.open_read("reports/step2/latest/02_unmapped_headers.csv")
    return pl.read_csv(csv).select("table", "vintage", "header", "problem").rows()


def test_a_new_header_is_dropped_with_a_warning_and_the_partition_is_built(
    ctx, meta, stage, tmp_path
):
    stage("provider_info", "2025-06")
    run_bronze(ctx, now=NOW)
    # CMS adds a column the map doesn't know, and every expected column is still there
    res = run_silver(ctx, cfg=config_without_provider_name(tmp_path / "cfg"), now=NOW)

    assert res.exit_code == EXIT_OK and res.failures == []
    (part,) = res.partitions
    assert (part.status, part.problems) == ("built", [])
    assert res.new_headers == ["provider_info 2025-06: Provider Name"]  # the CLI prints these
    built = silver_sink(ctx).read("provider_by_vintage", {"vintage": "2025-06"})
    assert built.height == 2  # both facilities, without the dropped column
    assert unmapped_report(ctx) == [("provider_info", "2025-06", "Provider Name", NEW_HEADER)]
    manifest = json.loads(ctx.object_store.open_read(res.manifest_key).read())
    assert manifest["new_headers"] == res.new_headers


def test_a_dropped_header_is_still_reported_after_a_retry_and_a_later_run(
    ctx, meta, stage, tmp_path
):
    stage("provider_info", "2025-06")
    run_bronze(ctx, now=NOW)
    cfg = config_without_provider_name(tmp_path / "cfg")
    run_silver(ctx, cfg=cfg, now=NOW)
    # a retry of the same DAG run (same run id) skips the built partition but must still alert on it
    retry = run_silver(ctx, cfg=cfg, now=NOW)
    later = run_silver(replace(ctx, run_id="R2"), cfg=cfg, now=NOW)

    assert [p.status for p in retry.partitions + later.partitions] == ["skipped", "skipped"]
    assert retry.new_headers == ["provider_info 2025-06: Provider Name"]
    assert later.new_headers == []  # alerted once per build, not on every run
    # the latest report describes the current silver, so the dropped column stays listed
    assert unmapped_report(ctx) == [("provider_info", "2025-06", "Provider Name", NEW_HEADER)]


def test_a_partition_that_fails_for_another_reason_also_names_its_new_header(
    ctx, meta, stage, tmp_path
):
    stage("provider_info", "2025-06", rows=[("015009", "Oak Manor"), ("015009", "Oak Annex")])
    run_bronze(ctx, now=NOW)
    res = run_silver(ctx, cfg=config_without_provider_name(tmp_path / "cfg"), now=NOW)

    assert res.exit_code == EXIT_VALIDATION and "share a key" in res.partitions[0].detail
    assert res.partitions[0].problems == [("Provider Name", "unmapped header")]  # the clue
    assert res.new_headers == []  # nothing was built without it


def test_silver_refuses_to_start_on_incomplete_bronze(ctx, meta, stage, tmp_path):
    stage("provider_info", "2025-06")  # staged in landing, but bronze never ran
    res = run_silver(ctx, cfg=write_config(tmp_path / "cfg"), now=NOW)
    assert res.exit_code == EXIT_VALIDATION and "bronze is incomplete" in res.failures[0]
    assert res.partitions == []


def _penalty_frame(rows):
    cfg = load_silver_config()
    smap = cfg.sources["penalties"]
    headers = {
        c: next(a.raw for a in col.aliases if a.valid_at("2026-08"))
        for c, col in smap.columns.items()
    }
    data = {headers[c]: [r.get(c, "") for r in rows] for c in headers}
    frame = pl.DataFrame(data, schema={h: pl.String for h in data}).with_columns(
        pl.int_range(1, len(rows) + 1, dtype=pl.Int64).alias("_row_number")
    )
    return frame, cfg


def test_identical_penalties_are_kept_and_numbered_in_file_order():
    fine = {
        "ccn": "015009",
        "penalty_date": "2026-05-04",
        "penalty_type": "Fine",
        "fine_amount": "6500",
        "processing_date": "2026-08-01",
    }
    frame, cfg = _penalty_frame(
        [
            fine | {"fine_id": "A1"},
            fine | {"fine_id": "A2"},
            fine | {"fine_amount": "7000", "fine_id": "A3"},
        ]
    )
    outputs, stats, _ = harmonize(frame, "penalties", "2026-08", cfg)
    pen = outputs["penalties_by_vintage"].sort("fine_id")

    assert pen["dup_seq"].to_list() == [1, 2, 1]
    assert pen["penalty_event_key"].to_list()[:2] == [
        "015009|2026-05-04|Fine|6500.0||1",
        "015009|2026-05-04|Fine|6500.0||2",
    ]
    assert (
        stats["dup_seq_gt_1"] == 1 and stats["exact_duplicate_rows"] == 0
    )  # Fine ID tells them apart


def test_a_wrong_processing_date_fails_the_partition():
    from precursorintelligence.harmonize.silver import PartitionError

    frame, cfg = _penalty_frame(
        [
            {
                "ccn": "015009",
                "penalty_date": "2026-05-04",
                "penalty_type": "Fine",
                "processing_date": "2026-07-01",
            }
        ]
    )
    with pytest.raises(PartitionError, match="Processing Date"):
        harmonize(frame, "penalties", "2026-08", cfg)


@pytest.mark.parametrize(
    "bad",
    [
        {"penalty_date": "06/01/2026"},  # unparseable date
        {"penalty_date": ""},  # missing date
        {"penalty_type": ""},  # missing type
        {"fine_amount": "lots"},  # unparseable amount
    ],
)
def test_a_penalty_whose_key_does_not_parse_fails_the_partition(bad):
    from precursorintelligence.harmonize.silver import PartitionError

    row = {
        "ccn": "015009",
        "penalty_date": "2026-05-04",
        "penalty_type": "Fine",
        "fine_amount": "6500",
        "processing_date": "2026-08-01",
        "fine_id": "A1",
    }
    frame, cfg = _penalty_frame([row | bad])
    with pytest.raises(PartitionError, match="key field"):
        harmonize(frame, "penalties", "2026-08", cfg)


def test_a_payment_denial_without_a_fine_amount_still_builds():
    frame, cfg = _penalty_frame(
        [
            {
                "ccn": "015009",
                "penalty_date": "2026-05-04",
                "penalty_type": "Payment Denial",
                "payment_denial_start_date": "2026-05-10",
                "processing_date": "2026-08-01",
                "fine_id": "A1",
            }
        ]
    )
    outputs, _, _ = harmonize(frame, "penalties", "2026-08", cfg)
    assert outputs["penalties_by_vintage"].height == 1
