"""E3, E4: typed records with stable evidence IDs, the month and the gold run; counts done in code."""

import asyncio
import time
from datetime import date
from typing import get_args

import polars as pl
import pytest
from evidence_fakes import GOLD, PREFIX, V, toy_tables, write_evidence

from precursorintelligence.agents.tools.evidence import (
    Chain,
    Citation,
    EvidenceTools,
    FacilityScore,
    Penalty,
    Reason,
    Record,
    ToolInputError,
    ToolTimeout,
)
from precursorintelligence.agents.tools.source import (
    NULLABLE,
    EvidenceError,
    ParquetEvidenceSource,
)


@pytest.fixture
def tools(store):
    write_evidence(store)
    return EvidenceTools(ParquetEvidenceSource.open(store, PREFIX), timeout_s=5)


def run(coro):
    return asyncio.run(coro)


def test_facility_score_keeps_the_ccn_as_text_and_stamps_the_month(tools):
    s = run(tools.get_facility_score("015009"))
    assert s is not None
    assert (s.ccn, s.p, s.risk_percentile, s.risk_decile) == ("015009", 0.42, 100.0, 1)
    assert (s.evidence_id, s.v, s.gold_run_id, s.model_version) == (f"score:015009:{V}", V, GOLD, 3)


def test_ccn_is_normalized_and_checked(tools):
    assert run(tools.get_facility_score(" 01a193 ")).ccn == "01A193"
    assert run(tools.get_facility_score("999999")) is None  # well-formed, not scored
    for bad in ("15009", "01-193", "0150090", ""):
        with pytest.raises(ToolInputError):
            run(tools.get_facility_score(bad))


def test_reasons_raise_first_and_ids_include_the_direction(tools):
    reasons = run(tools.get_reasons("015009"))
    assert [(r.direction, r.rank, r.feature) for r in reasons] == [
        ("raises", 1, "complaints_12m"),
        ("raises", 2, "cit_g_36m"),
        ("lowers", 1, "rn_hours"),
    ]
    ids = [r.evidence_id for r in reasons]
    assert ids[0] == f"reason:015009:{V}:raises:1"
    assert ids[2] == f"reason:015009:{V}:lowers:1"  # rank 1 again, a different ID
    assert all(r.v == V and r.gold_run_id == GOLD for r in reasons)
    assert run(tools.get_reasons("450124")) == []


def test_citations_newest_first(tools):
    cits = run(tools.get_citations("015009"))
    assert [c.evidence_id for c in cits] == [
        "citation:015009:2026-01-20:F0880",
        "citation:015009:2025-03-04:F0689",
    ]
    assert cits[1].scope_severity == "G" and cits[1].correction_date == date(2025, 4, 1)
    assert all(c.v == V and c.gold_run_id == GOLD for c in cits)


def test_identical_penalties_get_distinct_ids(tools):
    pens = run(tools.get_penalties("015009"))
    assert [p.evidence_id for p in pens] == [
        "penalty:015009:2025-09-01:payment_denial:1",
        "penalty:015009:2025-05-01:fine:1",
        "penalty:015009:2025-05-01:fine:2",
    ]
    assert pens[0].payment_denial_length_days == 30.0
    assert [p.fine_amount for p in pens[1:]] == [6500.0, 6500.0]


def test_chain(tools):
    c = run(tools.get_chain("C1"))
    assert c is not None
    assert (c.evidence_id, c.n_facilities, c.worst_ccn, c.v) == (f"chain:C1:{V}", 2, "015009", V)
    assert run(tools.get_chain("C9")) is None
    with pytest.raises(ToolInputError):
        run(tools.get_chain("  "))


def test_portfolio_counts(tools):
    stats = run(tools.portfolio_stats(["450123", "015009", "015009", "01a193", "123456"], top_n=2))
    assert stats.n_requested == 4  # duplicates count once
    assert stats.n_scored == 3
    assert stats.unscored_ccns == ["123456"]
    assert stats.decile_counts == {1: 1, 2: 0, 3: 1, 4: 0, 5: 1, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0}
    assert stats.mean_p == pytest.approx((0.42 + 0.10 + 0.25) / 3)
    assert stats.expected_facilities_with_event == pytest.approx(0.77)
    assert stats.state_counts == {"AL": 2, "TX": 1}
    assert stats.chain_counts == {"C1": 2}
    assert [t.ccn for t in stats.top] == ["015009", "450123"]
    assert [t.evidence_id for t in stats.top] == [f"score:015009:{V}", f"score:450123:{V}"]
    assert (stats.v, stats.gold_run_id) == (V, GOLD)


def test_portfolio_id_ignores_input_order_but_not_top_n(tools):
    a = run(tools.portfolio_stats(["015009", "450124"], top_n=0))
    b = run(tools.portfolio_stats(["450124", "015009"], top_n=0))
    c = run(tools.portfolio_stats(["015009", "450124"], top_n=1))
    assert a.evidence_id == b.evidence_id and a.evidence_id.startswith(f"portfolio:{V}:")
    assert c.evidence_id != a.evidence_id
    assert a.top == [] and a.state_counts == {"AL": 1, "unknown": 1}


def test_empty_portfolio(tools):
    stats = run(tools.portfolio_stats([], top_n=5))
    assert (stats.n_scored, stats.mean_p, stats.expected_facilities_with_event) == (0, None, 0.0)


def test_a_slow_read_times_out(store):
    write_evidence(store)

    class Slow(ParquetEvidenceSource):
        def rows(self, table, key, values):
            time.sleep(0.5)
            return super().rows(table, key, values)

    slow = Slow.open(store, PREFIX)
    with pytest.raises(ToolTimeout, match="facility_score took over"):
        run(EvidenceTools(slow, timeout_s=0.05).get_facility_score("015009"))


def test_a_repeated_citation_id_is_refused(store):
    tables = toy_tables()
    tables["citations"] = pl.concat([tables["citations"], tables["citations"].head(1)])
    write_evidence(store, tables)
    tools = EvidenceTools(ParquetEvidenceSource.open(store, PREFIX), timeout_s=5)
    with pytest.raises(EvidenceError, match="repeats the IDs"):
        run(tools.get_citations("015009"))


def test_every_id_in_a_full_set_of_answers_is_unique(tools):
    records = [
        run(tools.get_facility_score("015009")),
        *run(tools.get_reasons("015009")),
        *run(tools.get_citations("015009")),
        *run(tools.get_penalties("015009")),
        run(tools.get_chain("C1")),
        run(tools.portfolio_stats(["015009", "01A193"], top_n=0)),
    ]
    ids = [r.evidence_id for r in records]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize(
    ("model", "table"),
    [
        (FacilityScore, "facility_scores"),
        (Reason, "facility_reasons"),
        (Citation, "citations"),
        (Penalty, "penalties"),
        (Chain, "chain_scores"),
    ],
)
def test_nullable_columns_match_the_optional_fields(model, table):
    # The source refuses a null in any other column at open(), so a record never fails to build inside a tool.
    for name, field in model.model_fields.items():
        if name in Record.model_fields:
            continue
        optional = type(None) in get_args(field.annotation)
        assert optional == (name in NULLABLE[table]), name
