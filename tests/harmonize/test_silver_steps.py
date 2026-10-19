"""Silver Steps 3-5 end to end on three small staged vintages: build_ledgers_and_universe, then
run_calibration (spec §4-§6). test_silver_run.py stubs these out, and test_silver_ledgers.py covers the
pure functions one at a time; this file runs them together through the real Context and object store."""

from datetime import date

import polars as pl

from precursorintelligence.harmonize import ledgers as led
from precursorintelligence.harmonize.flags import STALE_REFRESH
from precursorintelligence.harmonize.silver_steps import build_ledgers_and_universe, run_calibration

# 2021-06 events are fully observed at V = 2023-09 (24 months back is 2021-09). 2023-09 is a stale
# re-publication of 2023-08: same events, flagged stale_refresh.
VINTAGES = ["2021-07", "2023-08", "2023-09"]
A, B = "015009", "015010"  # A is in every vintage; B leaves after 2023-08
OLD_SURVEY, NEW_SURVEY = date(2021, 6, 15), date(2023, 7, 10)

PENALTY_SCHEMA = {
    "penalty_event_key": pl.String,
    "ccn": pl.String,
    "penalty_date": pl.Date,
    "penalty_type": pl.String,
    "fine_amount": pl.Float64,
    "payment_denial_start_date": pl.Date,
    "payment_denial_length_days": pl.Int64,
    "dup_seq": pl.Int64,
    "fine_id": pl.String,
}


def citations(vintage):
    rows = [(OLD_SURVEY, "0880", "G")]  # G+, first seen one month after the survey
    if vintage >= "2023-08":
        rows.append((NEW_SURVEY, "0689", "D"))
    return pl.DataFrame(
        [
            {
                "ccn": A,
                "survey_date": d,
                "deficiency_prefix": "F",
                "deficiency_tag": tag,
                "scope_severity": sev,
                "deficiency_category": "x",
                "is_standard": True,
                "is_complaint": False,
                "is_infection_control": False,
            }
            for d, tag, sev in rows
        ]
    )


def surveys(vintage):
    days = [OLD_SURVEY] + ([NEW_SURVEY] if vintage >= "2023-08" else [])
    return pl.DataFrame({"ccn": [A] * len(days), "survey_date": days}).with_columns(
        pl.lit("HEALTH_STANDARD").alias("survey_type")
    )


def penalties(vintage):
    if vintage < "2023-08":
        return pl.DataFrame(schema=PENALTY_SCHEMA)
    fine = {
        "penalty_event_key": f"{A}|2023-07-10|Fine|6500.0||1",
        "ccn": A,
        "penalty_date": NEW_SURVEY,
        "penalty_type": "Fine",
        "fine_amount": 6500.0,
        "payment_denial_start_date": None,
        "payment_denial_length_days": None,
        "dup_seq": 1,
        "fine_id": "F1",
    }
    return pl.DataFrame([fine], schema=PENALTY_SCHEMA)


def stage_silver(ctx):
    for v in VINTAGES:
        ccns = [A, B] if v <= "2023-08" else [A]
        tables = {
            "citations_by_vintage": citations(v),
            "surveys_by_vintage": surveys(v),
            "penalties_by_vintage": penalties(v),
            "provider_by_vintage": pl.DataFrame({"ccn": ccns}),
            "provider_identity_by_vintage": pl.DataFrame(
                {"ccn": ccns, **{c: [f"{c} {ccn}" for ccn in ccns] for c in led.IDENTITY}}
            ),
            "mds_qm_by_vintage": pl.DataFrame(
                {
                    "measure_code": ["410"],
                    "resident_type": ["Long Stay"],
                    "measure_description": ["falls"],
                }
            ),
            "claims_qm_by_vintage": pl.DataFrame(
                {
                    "measure_code": ["551"],
                    "resident_type": ["Short Stay"],
                    "measure_description": ["rehosp"],
                }
            ),
        }
        for table, frame in tables.items():
            led.write_single(ctx.object_store, f"silver/{table}/vintage={v}/part-0.parquet", frame)
    ctx.metadata.upsert(
        "vintage_flags",
        [
            {
                "vintage_month": "2023-09",
                "flag": STALE_REFRESH,
                "table_name": "",
                "value": True,
                "run_id": "R0",
                "updated_at": "2026-10-18T12:00:00+00:00",
            }
        ],
    )


def test_steps_3_to_5_build_the_ledgers_universe_and_checkpoint_reports(ctx):
    stage_silver(ctx)
    built = build_ledgers_and_universe(ctx, VINTAGES)
    f = built["frames"]

    cit = f["ledger_citations"].sort("survey_date")
    assert cit["first_seen_vintage"].to_list() == ["2021-07", "2023-08"]
    assert cit["last_seen_vintage"].to_list() == ["2023-09", "2023-09"]
    assert cit["n_vintages_seen"].to_list() == [3, 2]  # the stale 2023-09 copy moves no first_seen
    assert not cit["removed_by_cms"].any()

    pen = f["ledger_penalties"]
    assert pen["revision_group"].to_list() == [f"{A}|2023-07-10|Fine"]
    assert pen["first_seen_vintage"].to_list() == ["2023-08"]

    assert f["ledger_surveys"].height == 2
    presence = dict(f["facility_presence"].select("ccn", "exited").rows())
    assert presence == {A: False, B: True}
    assert f["facility_universe"].height == 5  # A in 3 vintages, B in 2
    active = dict(f["dim_facility_latest"].select("ccn", "is_active").rows())
    assert active == {A: True, B: False}
    assert f["qm_inventory"].height == 2
    assert built["counts"]["facility_universe"] == 5
    for name in f:  # every all-vintage table is written where gold expects it
        with ctx.object_store.open_read(led.KEYS[name]) as fh:
            assert pl.read_parquet(fh).height == f[name].height

    keys = run_calibration(ctx, VINTAGES, built)
    written = {k.rsplit("/", 1)[-1] for k in keys if "/latest/" in k}
    assert written == {
        "04_lag_analysis.md",
        "04_lag_cdf.csv",
        "05_universe_and_qm.md",
        "05_qm_inventory.csv",
        "05_qm_measures.proposed.csv",
    }
    with ctx.object_store.open_read("reports/step4/latest/04_lag_analysis.md") as fh:
        lag_md = fh.read().decode("utf-8")
    assert "stale vintages: 2023-09" in lag_md
    g_plus_row = next(line for line in lag_md.splitlines() if line.startswith("| G+ citations |"))
    assert g_plus_row.rstrip(" |").endswith(
        "| 1"
    )  # B = 1: the 2021-06 citation posted one month later
