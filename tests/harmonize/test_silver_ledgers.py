"""Steps 3-5 on hand-made silver partitions: the vintage fold, removed_by_cms, presence gaps, lookup versions,
and the calibration maths (spec §4-§6, §9.3)."""

import uuid
from datetime import date

import polars as pl

from precursorintelligence.common.io.object_store import FsspecObjectStore
from precursorintelligence.harmonize import calibrate as cal
from precursorintelligence.harmonize import ledgers as led
from precursorintelligence.harmonize.silver_pipeline import lookup_versions

V = ["2024-01", "2024-02", "2024-04"]  # 2024-03 was never published


def reader(tables: dict[str, dict[str, pl.DataFrame]]) -> led.SilverReader:
    store = FsspecObjectStore(f"memory://silver-{uuid.uuid4().hex}")
    for table, parts in tables.items():
        for vintage, frame in parts.items():
            led.write_single(store, f"silver/{table}/vintage={vintage}/part-0.parquet", frame)
    return led.SilverReader(store, sorted({v for p in tables.values() for v in p}))


def cit(rows):
    """Citation rows: (ccn, survey_date, tag, severity)."""
    return pl.DataFrame(
        [
            {
                "ccn": c,
                "survey_date": date.fromisoformat(d),
                "deficiency_prefix": "F",
                "deficiency_tag": t,
                "scope_severity": s,
                "deficiency_category": "x",
                "is_standard": True,
                "is_complaint": False,
                "is_infection_control": False,
            }
            for c, d, t, s in rows
        ]
    )


def providers(*ccns):
    return pl.DataFrame({"ccn": list(ccns)})


def test_a_citation_ledger_keeps_first_and_last_sighting_and_the_final_severity():
    s = reader(
        {
            "citations_by_vintage": {
                V[0]: cit([("015009", "2023-12-01", "0880", "G")]),
                V[1]: cit([("015009", "2023-12-01", "0880", "D")]),  # downgraded after a dispute
                V[2]: cit([("015009", "2023-12-01", "0880", "D")]),
            },
            "provider_by_vintage": {v: providers("015009") for v in V},
        }
    )
    row = led.ledger_citations(s).row(0, named=True)
    assert (row["first_seen_vintage"], row["last_seen_vintage"], row["n_vintages_seen"]) == (
        V[0],
        V[2],
        3,
    )
    assert row["final_scope_severity"] == "D" and row["severity_changed"] is True
    assert row["removed_by_cms"] is False


def test_removed_by_cms_only_when_the_survey_and_facility_are_still_there():
    s = reader(
        {
            "citations_by_vintage": {
                V[0]: cit(
                    [
                        ("015009", "2023-12-01", "0880", "G"),  # deleted: its sibling stays
                        ("015009", "2023-12-01", "0689", "D"),
                        ("15E123", "2022-01-05", "0880", "D"),  # aged out: the whole survey goes
                        ("15E124", "2023-11-01", "0880", "G"),  # facility closed
                    ]
                ),
                V[1]: cit([("015009", "2023-12-01", "0689", "D")]),
                V[2]: cit([("015009", "2023-12-01", "0689", "D")]),
            },
            "provider_by_vintage": {
                V[0]: providers("015009", "15E123", "15E124"),
                V[1]: providers("015009", "15E123"),
                V[2]: providers("015009", "15E123"),
            },
        }
    )
    removed = dict(
        led.ledger_citations(s)
        .select(pl.concat_str("ccn", "deficiency_tag"), "removed_by_cms")
        .rows()
    )
    assert removed == {
        "0150090880": True,
        "0150090689": False,
        "15E1230880": False,
        "15E1240880": False,
    }


def test_facility_presence_counts_gaps_in_published_vintages_only():
    s = reader(
        {
            "provider_by_vintage": {
                V[0]: providers("A00001", "B00002"),
                V[1]: providers("B00002"),
                V[2]: providers("A00001"),
            }
        }
    )
    rows = {r["ccn"]: r for r in led.facility_presence(s).iter_rows(named=True)}
    assert (rows["A00001"]["gap_vintages"], rows["A00001"]["exited"]) == (
        1,
        False,
    )  # missing in 2024-02
    assert (rows["B00002"]["gap_vintages"], rows["B00002"]["exited"]) == (0, True)


def test_penalty_ledger_keeps_the_latest_fine_id():
    pen = lambda fine_id: pl.DataFrame(  # noqa: E731
        {
            "penalty_event_key": ["015009|2024-01-02|Fine|500.0||1"],
            "ccn": ["015009"],
            "penalty_date": [date(2024, 1, 2)],
            "penalty_type": ["Fine"],
            "fine_amount": [500.0],
            "payment_denial_start_date": pl.Series([None], dtype=pl.Date),
            "payment_denial_length_days": pl.Series([None], dtype=pl.Float64),
            "dup_seq": [1],
            "fine_id": pl.Series([fine_id], dtype=pl.String),
        }
    )
    s = reader({"penalties_by_vintage": {V[0]: pen(None), V[1]: pen("F9"), V[2]: pen(None)}})
    row = led.ledger_penalties(s).row(0, named=True)
    assert row["fine_id"] == "F9" and row["revision_group"] == "015009|2024-01-02|Fine"
    assert (row["first_seen_vintage"], row["n_vintages_seen"]) == (V[0], 3)


def test_lookup_versions_split_on_changes_and_gaps():
    rows = [("A", V[0]), ("B", V[1]), ("A", V[2])]
    lookup = pl.DataFrame(
        {
            "deficiency_prefix": ["F"] * 3,
            "deficiency_tag": ["0880"] * 3,
            "deficiency_prefix_and_number": ["F-0880"] * 3,
            "deficiency_description": [d for d, _ in rows],
            "deficiency_category": ["c"] * 3,
            "vintage_month": [v for _, v in rows],
        }
    )
    versions = lookup_versions(lookup, V)
    assert versions.select(
        "deficiency_description", "valid_from_vintage", "valid_to_vintage"
    ).rows() == [
        ("A", V[0], V[0]),
        ("B", V[1], V[1]),
        ("A", V[2], V[2]),
    ]


def test_lag_table_picks_the_smallest_lag_with_95_percent_coverage():
    lags = pl.DataFrame({"lag_months": [2] * 80 + [5] * 15 + [9] * 5})
    t = cal.lag_table(lags)
    assert (t["n"], t["median"], t["B"]) == (100, 2, 5)
    assert t["cdf"][4] == 0.80 and t["cdf"][5] == 0.95


def test_last_labeled_month_uses_the_last_non_stale_vintage():
    vintages = ["2025-07", "2025-09"]
    rows = cal.last_labeled("2025-09", {"B_cit": 8, "B_pen": 12}, vintages, stale=["2025-09"])
    assert rows[0][:4] == [
        "y_g12",
        12,
        "B_cit = 8",
        "2023-11",
    ]  # from 2025-07, not the stale 2025-09


def test_qm_proposal_classifies_stable_bridged_and_dropped():
    vs = ["2022-01", "2022-02", "2025-02"]
    presence = pl.DataFrame(
        [
            {
                "table": "mds",
                "measure_code": c,
                "resident_type": "Long Stay",
                "measure_description": c,
                "vintage_month": v,
            }
            for c, months in {
                "410": vs,
                "453": vs[:2],
                "479": vs[2:],
                "415": vs,
                "999": vs[1:],
            }.items()
            for v in months
        ]
    )
    status = dict(cal.qm_proposal(presence, vs).select("measure_code", "status").rows())
    assert status == {
        "410": "stable",
        "453": "bridged",
        "479": "bridged",
        "415": "dropped",
        "999": "dropped",
    }
