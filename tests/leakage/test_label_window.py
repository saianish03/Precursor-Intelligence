"""Label rule (spec §0.4): a label counts only events dated strictly after the cutoff and within H months."""

from datetime import date

import polars as pl
import pytest

from precursorintelligence.common.config import LabelsConfig, LabelSpec
from precursorintelligence.gold.build import events_published_by
from precursorintelligence.gold.labels import build_labels

CFG = LabelsConfig(
    gplus_severities=list("GHIJKL"),
    buffers_months={"citations": 0, "penalties": 0},
    labels={"g12": LabelSpec(source="citations", horizon_months=12)},
)
SPINE = pl.DataFrame({"ccn": ["015001"], "t": ["2023-01"], "cutoff_date": [date(2023, 1, 1)]})
PRESENCE = pl.DataFrame({"ccn": ["015001"], "last_vintage": ["2025-03"]})
NO_PENALTIES = pl.DataFrame(
    schema={"ccn": pl.String, "penalty_date": pl.Date, "revision_group": pl.String}
)


def _y_g12(survey_date: date, severity: str = "G", removed: bool = False) -> int | None:
    ledger = pl.DataFrame(
        {
            "ccn": ["015001"],
            "survey_date": [survey_date],
            "final_scope_severity": [severity],
            "removed_by_cms": [removed],
        }
    )
    labels = build_labels(
        SPINE,
        ledger_citations=ledger,
        ledger_penalties=NO_PENALTIES,
        facility_presence=PRESENCE,
        cfg=CFG,
        v="2025-03",
    )
    return labels["y_g12"][0]


@pytest.mark.parametrize(
    ("survey_date", "expected"),
    [
        (date(2022, 12, 31), 0),  # before the cutoff: known at t, so never a label
        (date(2023, 1, 1), 0),  # on the cutoff day: not strictly after
        (date(2023, 1, 2), 1),
        (date(2024, 1, 1), 1),  # cutoff + 12 months: the window end is inclusive
        (date(2024, 1, 2), 0),  # past the window
    ],
)
def test_window_is_strictly_after_cutoff_and_within_horizon(survey_date, expected):
    assert _y_g12(survey_date) == expected


def test_only_gplus_not_removed_citations_count():
    assert _y_g12(date(2023, 6, 1), severity="F") == 0
    assert _y_g12(date(2023, 6, 1), removed=True) == 0


def _status(survey_date: date | None, *, last_vintage: str, v: str) -> tuple[str, int | None]:
    ledger = pl.DataFrame(
        {
            "ccn": ["015001"] if survey_date else [],
            "survey_date": [survey_date] if survey_date else [],
            "final_scope_severity": ["G"] if survey_date else [],
            "removed_by_cms": [False] if survey_date else [],
        },
        schema={"ccn": pl.String, "survey_date": pl.Date, "final_scope_severity": pl.String,
                "removed_by_cms": pl.Boolean},
    )  # fmt: skip
    labels = build_labels(
        SPINE,
        ledger_citations=ledger,
        ledger_penalties=NO_PENALTIES,
        facility_presence=PRESENCE.with_columns(last_vintage=pl.lit(last_vintage)),
        cfg=CFG.model_copy(update={"buffers_months": {"citations": 8, "penalties": 12}}),
        v=v,
    )
    return labels["status_g12"][0], labels["y_g12"][0]


def test_positive_holds_even_if_the_facility_exits_before_window_plus_buffer():
    # Exits in 2023-08, long before t + 12 + 8 = 2024-09, but the G on 2023-06-01 was seen.
    assert _status(date(2023, 6, 1), last_vintage="2023-08", v="2025-03") == ("positive", 1)
    assert _status(None, last_vintage="2023-08", v="2025-03") == ("censored", None)


def test_window_not_yet_observed_is_immature_even_with_a_visible_event():
    # With fresh data only to 2024-08 (e.g. V is a stale refresh), t + 12 + 8 = 2024-09 > v.
    assert _status(date(2023, 6, 1), last_vintage="2025-03", v="2024-08") == ("immature", None)


def test_an_event_published_after_v_does_not_label_earlier_rows():
    # A G dated 2023-06-01 but first published in 2025-06 is invisible to a gold run at V = 2025-03.
    ledger = pl.DataFrame(
        {
            "ccn": ["015001"],
            "survey_date": [date(2023, 6, 1)],
            "final_scope_severity": ["G"],
            "removed_by_cms": [False],
            "first_seen_vintage": ["2025-06"],
        }
    )
    assert events_published_by(ledger, "2025-03").is_empty()
    assert events_published_by(ledger, "2025-06").height == 1
