"""Fake silver tables for the gold-layer tests: five facilities, monthly reports 2023-01 → 2025-03.

Table and column names follow docs/DATA_PIPELINE_SPEC.md §3.3/§3.5 (by-vintage tables), §4 (ledgers)
and §6 (facility universe, dim). Every value is invented. The story:

- 2023-12 was never published, so no table has rows for it. V = 2025-03.
- 015001 Sunny Acres: G citation on a standard survey dated 2023-06-15 (first posted in 2023-08).
- 015002 Oak Grove: only D and E citations, never G-L.
- 01A193 Pine View: last appears in the 2023-08 report (closes).
- 015004 Maple House: in every report, but its G citation on a complaint survey dated 2024-02-10 is
  posted late: absent before the 2024-06 report, present from it onward.
- 015005 Cedar Hill: first appears in the 2023-04 report.
- Penalties: Sunny Acres is fined on 2023-06-15 (posted 2023-09); Oak Grove gets two identical fines
  on 2023-10-05 (posted 2024-01), so they carry dup_seq 1 and 2.
- Quality measures: three MDS and two claims measures per facility; the long-stay antipsychotic
  measure is code 419 up to the 2024-05 report and code 481 from 2024-06.

A citation stays visible from its first report until the facility's last report. The expected gold
answers live in test_expected_answers.py.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from datetime import date, timedelta

import polars as pl

from precursorintelligence.common.io.metadata_store import MetadataStore
from precursorintelligence.common.io.object_store import ObjectStore
from precursorintelligence.common.io.table_sink import ParquetTableSink
from precursorintelligence.harmonize.columns import load_silver_config

NEVER_PUBLISHED = ("2023-12",)
V = "2025-03"


def _month_range(start: str, end: str) -> list[str]:
    y, m = map(int, start.split("-"))
    out: list[str] = []
    while f"{y:04d}-{m:02d}" <= end:
        out.append(f"{y:04d}-{m:02d}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


VINTAGES = [v for v in _month_range("2023-01", V) if v not in NEVER_PUBLISHED]


def cutoff(vintage: str) -> date:
    return date.fromisoformat(f"{vintage}-01")


# ------------------------------------------------------------------ schemas
# Column names, order and types follow configs/keep_columns.yaml and docs/SILVER_CONTRACT.md: the spec §3.3
# keep-lists, the columns silver derives (ownership_group, sprinkler_status_not_available, ...) and the
# columns CMS added in 2026-09 (high_performing_icon, survey_fn), which are null in this 2023-2025 world.

S, F, B, D, N = pl.String, pl.Float64, pl.Boolean, pl.Date, pl.Int64
LINEAGE = {"vintage_month": S, "cutoff_date": D}

PROVIDER = {
    "ccn": S, "state": S, "county_name": S, "ownership_type": S, "ownership_group": S,
    "certified_beds": F,
    "avg_residents_per_day": F, "avg_residents_per_day_fn": S, "provider_type": S,
    "provider_in_hospital": B, "date_first_approved": D, "ccrc": B, "special_focus_status": S,
    "abuse_icon": B, "high_performing_icon": B, "most_recent_inspection_over_2y": B,
    "changed_ownership_12m": B,
    "resident_family_council": S, "sprinkler_status": S, "sprinkler_status_not_available": B,
    "overall_rating": F,
    "overall_rating_fn": S, "health_inspection_rating": F, "health_inspection_rating_fn": S,
    "qm_rating": F, "qm_rating_fn": S, "staffing_rating": F, "staffing_rating_fn": S,
    "reported_staffing_fn": S, "pt_staffing_fn": S, "hprd_aide": F, "hprd_lpn": F, "hprd_rn": F,
    "hprd_licensed": F, "hprd_total": F, "hprd_weekend_total": F, "hprd_weekend_rn": F,
    "hprd_pt": F, "adj_hprd_aide": F, "adj_hprd_lpn": F, "adj_hprd_rn": F, "adj_hprd_total": F,
    "turnover_total": F, "turnover_total_fn": S, "turnover_rn": F, "turnover_rn_fn": S,
    "admin_departures": F, "admin_departures_fn": S, "chain_id": S, "chain_name": S,
} | LINEAGE  # fmt: skip
PROVIDER_IDENTITY = {
    "ccn": S, "provider_name": S, "provider_address": S, "city": S, "state": S, "zip": S,
    "latitude": F, "longitude": F, "chain_id": S, "chain_name": S,
} | LINEAGE  # fmt: skip
CITATIONS = {
    "ccn": S, "survey_date": D, "deficiency_prefix": S, "deficiency_category": S,
    "deficiency_tag": S, "ftag": S, "scope_severity": S, "deficiency_corrected": S,
    "correction_date": D, "is_standard": B, "is_complaint": B, "is_infection_control": B,
    "inspection_cycle": N, "citation_under_idr": B, "citation_under_iidr": B, "survey_fn": S,
} | LINEAGE  # fmt: skip
SURVEYS = {
    "ccn": S, "survey_date": D, "survey_type": S, "survey_cycle": N, "survey_fn": S,
} | LINEAGE  # fmt: skip
PENALTIES = {
    "ccn": S, "penalty_date": D, "penalty_type": S, "fine_amount": F,
    "payment_denial_start_date": D, "payment_denial_length_days": F, "fine_id": S, "dup_seq": N,
    "penalty_event_key": S,
} | LINEAGE  # fmt: skip
MDS_QM = {
    "ccn": S, "measure_code": S, "measure_description": S, "resident_type": S,
    "four_quarter_avg": F, "four_quarter_avg_fn": S, "used_in_5star": B, "measure_period": S,
    "measure_period_start": D, "measure_period_end": D,
} | LINEAGE  # fmt: skip
CLAIMS_QM = {
    "ccn": S, "measure_code": S, "measure_description": S, "resident_type": S,
    "adjusted_score": F, "observed_score": F, "expected_score": F, "measure_fn": S,
    "used_in_5star": B, "measure_period": S, "measure_period_start": D, "measure_period_end": D,
} | LINEAGE  # fmt: skip
CITATION_LOOKUP = {
    "deficiency_prefix": S, "deficiency_tag": S, "deficiency_prefix_and_number": S,
    "deficiency_description": S, "deficiency_category": S,
} | LINEAGE  # fmt: skip

# §4 ledgers and §6 tables
LEDGER_CITATIONS = {
    "ccn": S, "survey_date": D, "deficiency_prefix": S, "deficiency_tag": S,
    "first_seen_vintage": S, "last_seen_vintage": S, "n_vintages_seen": N,
    "final_scope_severity": S, "final_category": S, "final_is_standard": B,
    "final_is_complaint": B, "final_is_infection_control": B, "severity_changed": B,
    "removed_by_cms": B,
}  # fmt: skip
LEDGER_PENALTIES = {
    "penalty_event_key": S, "first_seen_vintage": S, "last_seen_vintage": S,
    "n_vintages_seen": N, "ccn": S, "penalty_date": D, "penalty_type": S, "fine_amount": F,
    "payment_denial_start_date": D, "payment_denial_length_days": F, "dup_seq": N, "fine_id": S,
    "revision_group": S,
}  # fmt: skip
LEDGER_SURVEYS = {
    "ccn": S, "survey_date": D, "survey_type": S, "first_seen_vintage": S,
    "last_seen_vintage": S, "in_inspection_dates": B,
}  # fmt: skip
FACILITY_PRESENCE = {
    "ccn": S, "first_vintage": S, "last_vintage": S, "n_vintages": N, "gap_vintages": N,
    "exited": B,
}  # fmt: skip
FACILITY_UNIVERSE = {"ccn": S, "vintage_month": S}
DIM_FACILITY_LATEST = {
    "ccn": S, "last_vintage": S, "provider_name": S, "provider_address": S, "city": S, "state": S,
    "zip": S, "latitude": F, "longitude": F, "chain_id": S, "chain_name": S, "is_active": B,
}  # fmt: skip

# Within-vintage keys (contract §4), plus vintage_month so the stacked toy frame splits into sorted partitions.
KEYS = {
    "provider_by_vintage": ["ccn", "vintage_month"],
    "provider_identity_by_vintage": ["ccn", "vintage_month"],
    "citations_by_vintage": [
        "ccn", "survey_date", "deficiency_prefix", "deficiency_tag", "vintage_month",
    ],
    "surveys_by_vintage": ["ccn", "survey_date", "survey_type", "vintage_month"],
    "penalties_by_vintage": ["penalty_event_key", "vintage_month"],
    "mds_qm_by_vintage": ["ccn", "measure_code", "vintage_month"],
    "claims_qm_by_vintage": ["ccn", "measure_code", "vintage_month"],
    "citation_lookup": ["deficiency_prefix", "deficiency_tag", "vintage_month"],
}  # fmt: skip
BY_VINTAGE = list(KEYS)

# ------------------------------------------------------------------ the fake world


@dataclass(frozen=True)
class Facility:
    ccn: str
    name: str
    first: str
    last: str
    beds: float
    chain: tuple[str, str] | None


FACILITIES = (
    Facility("015001", "Sunny Acres", "2023-01", V, 120.0, ("900001", "Fake Care Group")),
    Facility("015002", "Oak Grove", "2023-01", V, 90.0, ("900001", "Fake Care Group")),
    Facility("01A193", "Pine View", "2023-01", "2023-08", 60.0, None),
    Facility("015004", "Maple House", "2023-01", V, 100.0, None),
    Facility("015005", "Cedar Hill", "2023-04", V, 80.0, ("900002", "Test Living LLC")),
)
BY_CCN = {f.ccn: f for f in FACILITIES}

# tag -> (category, description); all deficiency_prefix "F"
TAGS = {
    "0600": ("Freedom from Abuse, Neglect, and Exploitation Deficiencies", "Fake: protect from abuse"),
    "0689": ("Quality of Life and Care Deficiencies", "Fake: prevent accident hazards"),
    "0761": ("Pharmacy Service Deficiencies", "Fake: label and store drugs"),
    "0812": ("Nutrition and Dietary Deficiencies", "Fake: store and serve food safely"),
    "0880": ("Infection Control Deficiencies", "Fake: run an infection control program"),
}  # fmt: skip


@dataclass(frozen=True)
class Citation:
    ccn: str
    survey_date: date
    tag: str
    severity: str
    standard: bool
    first_seen: str  # first report that posts it


CITATION_EVENTS = (
    Citation("015001", date(2022, 5, 10), "0880", "D", True, "2023-01"),
    Citation("015001", date(2023, 6, 15), "0689", "G", True, "2023-08"),
    Citation("015001", date(2023, 6, 15), "0880", "D", True, "2023-08"),
    Citation("015002", date(2022, 9, 14), "0812", "E", True, "2023-01"),
    Citation("015002", date(2022, 9, 14), "0880", "D", True, "2023-01"),
    Citation("015002", date(2023, 10, 5), "0689", "D", True, "2023-11"),
    Citation("015002", date(2023, 10, 5), "0761", "E", True, "2023-11"),
    Citation("015002", date(2024, 4, 18), "0600", "D", False, "2024-06"),
    Citation("01A193", date(2022, 11, 2), "0880", "E", True, "2023-01"),
    Citation("015004", date(2022, 12, 7), "0880", "D", True, "2023-01"),
    Citation("015004", date(2024, 2, 10), "0689", "G", False, "2024-06"),  # posted late
    Citation("015005", date(2023, 3, 22), "0880", "D", True, "2023-04"),
    Citation("015005", date(2024, 5, 15), "0812", "E", True, "2024-07"),
)


@dataclass(frozen=True)
class Penalty:
    ccn: str
    penalty_date: date
    fine_amount: float
    first_seen: str


PENALTY_EVENTS = (
    Penalty("015001", date(2023, 6, 15), 15000.0, "2023-09"),  # same day as its G citation
    Penalty("015002", date(2023, 10, 5), 5000.0, "2024-01"),
    Penalty("015002", date(2023, 10, 5), 5000.0, "2024-01"),  # identical twin: dup_seq 2
)

# (code, resident_type, description, base value, first vintage, last vintage)
MDS_MEASURES = (
    ("410", "Long Stay", "Fake: falls with major injury", 3.0, "2023-01", V),
    ("419", "Long Stay", "Fake: antipsychotic medication (old code)", 15.0, "2023-01", "2024-05"),
    ("481", "Long Stay", "Fake: antipsychotic medication (new code)", 12.0, "2024-06", V),
    ("434", "Short Stay", "Fake: newly received antipsychotic medication", 1.5, "2023-01", V),
)
CLAIMS_MEASURES = (
    ("521", "Short Stay", "Fake: re-hospitalized after admission", 22.0, "2023-01", V),
    ("551", "Long Stay", "Fake: hospitalizations per 1,000 resident days", 1.8, "2023-01", V),
)


def _visible(c: Citation | Penalty, vintage: str) -> bool:
    return c.first_seen <= vintage <= BY_CCN[c.ccn].last


def _present(f: Facility, vintage: str) -> bool:
    return f.first <= vintage <= f.last


def _cycle(d: date, std_dates: set[date], standard: bool) -> int:
    """Simplified CMS cycle: 1 = most recent standard survey; a complaint joins the cycle it follows."""
    later = sum(s > d for s in std_dates)
    return later + 1 if standard else max(later, 1)


def _frame(
    rows: list[dict[str, object]], schema: dict[str, pl.DataType], key: list[str]
) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=schema).sort(key) if rows else pl.DataFrame(schema=schema)


def _lineage(vintage: str) -> dict[str, object]:
    return {"vintage_month": vintage, "cutoff_date": cutoff(vintage)}


def _provider_rows() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    provider, identity = [], []
    for v in VINTAGES:
        for i, f in enumerate(FACILITIES):
            if not _present(f, v):
                continue
            chain = (
                f.chain if f.chain and v >= "2023-06" else (None, None)
            )  # chain exists from 2023-06
            row: dict[str, object] = dict.fromkeys(PROVIDER) | {
                "ccn": f.ccn, "state": "AL", "county_name": "Fake County",
                "ownership_type": "For profit - Corporation", "ownership_group": "for_profit",
                "certified_beds": f.beds,
                "avg_residents_per_day": f.beds * 0.8, "provider_type": "Medicare and Medicaid",
                "provider_in_hospital": False, "date_first_approved": date(1990 + i, 1, 15),
                "ccrc": False, "special_focus_status": "NONE", "abuse_icon": False,
                "most_recent_inspection_over_2y": False, "changed_ownership_12m": False,
                "resident_family_council": "RESIDENT", "sprinkler_status": "YES",
                "sprinkler_status_not_available": False,
                "overall_rating": 3.0, "health_inspection_rating": 3.0, "qm_rating": 3.0,
                "staffing_rating": 3.0, "hprd_aide": 2.3, "hprd_lpn": 0.8, "hprd_rn": 0.6,
                "hprd_licensed": 1.4, "hprd_total": 3.7, "hprd_weekend_total": 3.3,
                "hprd_weekend_rn": 0.4, "hprd_pt": 0.05, "adj_hprd_aide": 2.2, "adj_hprd_lpn": 0.8,
                "adj_hprd_rn": 0.6, "adj_hprd_total": 3.6, "turnover_total": 48.0,
                "turnover_rn": 45.0, "admin_departures": 1.0, "chain_id": chain[0],
                "chain_name": chain[1],
            } | _lineage(v)  # fmt: skip
            provider.append(row)
            lat_long = (32.0 + i / 10, -86.0 - i / 10) if v >= "2023-08" else (None, None)
            identity.append({
                "ccn": f.ccn, "state": "AL", "chain_id": chain[0], "chain_name": chain[1],
                "provider_name": f.name.upper(), "provider_address": f"{100 + i} Fake Street",
                "city": "Testville", "zip": f"3500{i}", "latitude": lat_long[0],
                "longitude": lat_long[1],
            } | _lineage(v))  # fmt: skip
    return provider, identity


def _citation_and_survey_rows() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    citations, surveys = [], []
    for v in VINTAGES:
        shown = [c for c in CITATION_EVENTS if _visible(c, v)]
        for c in shown:
            std = {x.survey_date for x in shown if x.ccn == c.ccn and x.standard}
            cyc = _cycle(c.survey_date, std, c.standard)
            citations.append({
                "ccn": c.ccn, "survey_date": c.survey_date, "deficiency_prefix": "F",
                "deficiency_category": TAGS[c.tag][0], "deficiency_tag": c.tag, "ftag": f"F{c.tag}",
                "scope_severity": c.severity,
                "deficiency_corrected": "Deficient, Provider has date of correction",
                "correction_date": c.survey_date + timedelta(days=30), "is_standard": c.standard,
                "is_complaint": not c.standard, "is_infection_control": False,
                "inspection_cycle": cyc, "citation_under_idr": False, "citation_under_iidr": False,
            } | _lineage(v))  # fmt: skip
        # Inspection Dates lists a complaint survey only if it produced a citation (§4.3).
        for ccn, d, standard in sorted({(c.ccn, c.survey_date, c.standard) for c in shown}):
            std = {x.survey_date for x in shown if x.ccn == ccn and x.standard}
            surveys.append({
                "ccn": ccn, "survey_date": d,
                "survey_type": "HEALTH_STANDARD" if standard else "HEALTH_COMPLAINT",
                "survey_cycle": _cycle(d, std, standard),
            } | _lineage(v))  # fmt: skip
    return citations, surveys


def _penalty_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for v in VINTAGES:
        seen: dict[tuple[object, ...], int] = {}
        for p in PENALTY_EVENTS:
            if not _visible(p, v):
                continue
            five = (p.ccn, p.penalty_date, "Fine", p.fine_amount, None)
            dup_seq = seen[five] = seen.get(five, 0) + 1  # occurrence in file order (§3.5)
            rows.append({
                "ccn": p.ccn, "penalty_date": p.penalty_date, "penalty_type": "Fine",
                "fine_amount": p.fine_amount, "payment_denial_start_date": None,
                "payment_denial_length_days": None, "fine_id": None,  # Fine ID exists from 2026-06
                "dup_seq": dup_seq,
                "penalty_event_key": "|".join("" if x is None else str(x) for x in (*five, dup_seq)),
            } | _lineage(v))  # fmt: skip
    return rows


def _measure_period(vintage: str) -> tuple[str, str, date, date]:
    """4-quarter window ending in the quarter 5 months before the cutoff (2024-08 → 2023Q2-2024Q1)."""
    y, m = map(int, vintage.split("-"))
    end_q = (y * 12 + m - 1 - 5) // 3  # quarters since year 0
    start_q = end_q - 3
    start = date(start_q // 4, start_q % 4 * 3 + 1, 1)
    end = date((end_q + 1) // 4, (end_q + 1) % 4 * 3 + 1, 1) - timedelta(days=1)
    mds = f"{start_q // 4}Q{start_q % 4 + 1}-{end_q // 4}Q{end_q % 4 + 1}"
    return mds, f"{start:%Y%m%d}-{end:%Y%m%d}", start, end


def _qm_rows() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    mds, claims = [], []
    for n, v in enumerate(VINTAGES):
        mds_period, claims_period, start, end = _measure_period(v)
        for i, f in enumerate(FACILITIES):
            if not _present(f, v):
                continue
            for code, rtype, desc, base, first, last in MDS_MEASURES:
                if first <= v <= last:
                    mds.append({
                        "ccn": f.ccn, "measure_code": code, "measure_description": desc,
                        "resident_type": rtype, "four_quarter_avg": round(base + i + n / 20, 2),
                        "four_quarter_avg_fn": None, "used_in_5star": True,
                        "measure_period": mds_period, "measure_period_start": start,
                        "measure_period_end": end,
                    } | _lineage(v))  # fmt: skip
            for code, rtype, desc, base, first, last in CLAIMS_MEASURES:
                if first <= v <= last:
                    score = round(base + i / 2 + n / 50, 2)
                    claims.append({
                        "ccn": f.ccn, "measure_code": code, "measure_description": desc,
                        "resident_type": rtype, "adjusted_score": score,
                        "observed_score": score + 0.4, "expected_score": score - 0.3,
                        "measure_fn": None, "used_in_5star": True,
                        "measure_period": claims_period, "measure_period_start": start,
                        "measure_period_end": end,
                    } | _lineage(v))  # fmt: skip
    return mds, claims


def _ledger_citations(cit: pl.DataFrame) -> pl.DataFrame:
    key = ["ccn", "survey_date", "deficiency_prefix", "deficiency_tag"]
    last = pl.col("vintage_month").arg_max()
    return (
        cit.group_by(key)
        .agg(
            first_seen_vintage=pl.col("vintage_month").min(),
            last_seen_vintage=pl.col("vintage_month").max(),
            n_vintages_seen=pl.len().cast(pl.Int64),
            final_scope_severity=pl.col("scope_severity").get(last),
            final_category=pl.col("deficiency_category").get(last),
            final_is_standard=pl.col("is_standard").get(last),
            final_is_complaint=pl.col("is_complaint").get(last),
            final_is_infection_control=pl.col("is_infection_control").get(last),
            severity_changed=pl.col("scope_severity").n_unique() > 1,
        )
        # No citation in this world is deleted by CMS: Pine View's vanish only because it closed.
        .with_columns(removed_by_cms=pl.lit(False))
        .select(list(LEDGER_CITATIONS))
        .sort(key)
    )


def build_silver() -> dict[str, pl.DataFrame]:
    """Every fake silver table (plus gold/dim_facility_latest) by its spec name."""
    provider, identity = _provider_rows()
    citations, surveys = _citation_and_survey_rows()
    mds, claims = _qm_rows()
    lookup = [
        {"deficiency_prefix": "F", "deficiency_tag": tag, "deficiency_prefix_and_number": f"F{tag}",
         "deficiency_description": desc, "deficiency_category": cat} | _lineage(v)
        for v in VINTAGES for tag, (cat, desc) in TAGS.items()
    ]  # fmt: skip
    t: dict[str, pl.DataFrame] = {
        "provider_by_vintage": _frame(provider, PROVIDER, KEYS["provider_by_vintage"]),
        "provider_identity_by_vintage": _frame(
            identity, PROVIDER_IDENTITY, KEYS["provider_identity_by_vintage"]
        ),
        "citations_by_vintage": _frame(citations, CITATIONS, KEYS["citations_by_vintage"]),
        "surveys_by_vintage": _frame(surveys, SURVEYS, KEYS["surveys_by_vintage"]),
        "penalties_by_vintage": _frame(_penalty_rows(), PENALTIES, KEYS["penalties_by_vintage"]),
        "mds_qm_by_vintage": _frame(mds, MDS_QM, KEYS["mds_qm_by_vintage"]),
        "claims_qm_by_vintage": _frame(claims, CLAIMS_QM, KEYS["claims_qm_by_vintage"]),
        "citation_lookup": _frame(lookup, CITATION_LOOKUP, KEYS["citation_lookup"]),
    }
    # The look-up never changes in this world, so each tag has one version (§3.5).
    t["citation_lookup_versions"] = (
        t["citation_lookup"]
        .group_by(list(CITATION_LOOKUP)[:5])
        .agg(
            valid_from_vintage=pl.col("vintage_month").min(),
            valid_to_vintage=pl.col("vintage_month").max(),
        )
        .sort("deficiency_prefix", "deficiency_tag")
    )
    t["ledger_citations"] = _ledger_citations(t["citations_by_vintage"])
    t["ledger_penalties"] = (
        t["penalties_by_vintage"]
        .sort("vintage_month")
        .group_by("penalty_event_key")
        .agg(
            pl.col("vintage_month").min().alias("first_seen_vintage"),
            pl.col("vintage_month").max().alias("last_seen_vintage"),
            pl.len().cast(pl.Int64).alias("n_vintages_seen"),
            *(pl.col(c).last() for c in list(LEDGER_PENALTIES)[4:11]),
            pl.col("fine_id").drop_nulls().last(),
        )
        .with_columns(  # the penalty event for labels and lag (contract §8, §10)
            revision_group=pl.concat_str(
                [pl.col("ccn"), pl.col("penalty_date").cast(pl.String), pl.col("penalty_type")],
                separator="|",
            )
        )
        .select(list(LEDGER_PENALTIES))
        .sort("penalty_event_key")
    )
    t["qm_inventory"] = pl.concat(
        [
            t[table]
            .group_by("measure_code", "resident_type", "measure_description")
            .agg(
                first_vintage=pl.col("vintage_month").min(),
                last_vintage=pl.col("vintage_month").max(),
                n_vintages=pl.col("vintage_month").n_unique(),
            )
            .select(pl.lit(table).alias("table"), pl.all())
            for table in ("mds_qm_by_vintage", "claims_qm_by_vintage")
        ]
    ).sort("table", "measure_code", "resident_type", "first_vintage")
    # Inspection Dates rows, plus surveys known only from citation flags (contract §5, spec §4.3).
    led = t["ledger_citations"]
    from_citations = pl.concat(
        [
            led.filter(pl.col(flag))
            .group_by("ccn", "survey_date")
            .agg(pl.col("first_seen_vintage").min(), pl.col("last_seen_vintage").max())
            .with_columns(survey_type=pl.lit(kind), in_inspection_dates=pl.lit(False))
            for flag, kind in (
                ("final_is_standard", "HEALTH_STANDARD"),
                ("final_is_complaint", "HEALTH_COMPLAINT"),
                ("final_is_infection_control", "INFECTION_CONTROL"),
            )
        ]
    )
    listed = (
        t["surveys_by_vintage"]
        .group_by("ccn", "survey_date", "survey_type")
        .agg(
            first_seen_vintage=pl.col("vintage_month").min(),
            last_seen_vintage=pl.col("vintage_month").max(),
        )
        .with_columns(in_inspection_dates=pl.lit(True))
    )
    t["ledger_surveys"] = (
        pl.concat([listed, from_citations.select(listed.columns)])
        .group_by("ccn", "survey_date", "survey_type")
        .agg(
            pl.col("first_seen_vintage").min(),
            pl.col("last_seen_vintage").max(),
            pl.col("in_inspection_dates").any(),
        )
        .select(list(LEDGER_SURVEYS))
        .sort("ccn", "survey_date", "survey_type")
    )
    t["facility_presence"] = pl.DataFrame(
        [
            {"ccn": f.ccn, "first_vintage": f.first, "last_vintage": f.last,
             "n_vintages": sum(_present(f, v) for v in VINTAGES),
             "gap_vintages": 0,  # present in every published vintage between first and last
             "exited": f.last < V}
            for f in FACILITIES
        ],
        schema=FACILITY_PRESENCE,
    ).sort("ccn")  # fmt: skip
    t["facility_universe"] = (
        t["provider_by_vintage"].select(list(FACILITY_UNIVERSE)).sort("vintage_month", "ccn")
    )
    at_last_vintage = t["provider_identity_by_vintage"].join(
        t["facility_presence"].select("ccn", vintage_month="last_vintage"),
        on=["ccn", "vintage_month"],
    )
    t["dim_facility_latest"] = (
        at_last_vintage.with_columns(
            last_vintage=pl.col("vintage_month"), is_active=pl.col("vintage_month") == V
        )
        .select(list(DIM_FACILITY_LATEST))
        .sort("ccn")
    )
    return t


def write_silver(store: ObjectStore, tables: dict[str, pl.DataFrame]) -> None:
    """Write the tables at their spec §0.6 paths (one partition per published vintage, else one file).

    Rewrites existing files, so running it again refreshes a toy folder.
    """
    sink = ParquetTableSink(store, prefix="silver")
    for name, df in tables.items():
        if name in BY_VINTAGE:
            for v in VINTAGES:
                part = df.filter(pl.col("vintage_month") == v)
                sink.write_partition(name, {"vintage": v}, part, overwrite=True)
            continue
        layer = "gold" if name == "dim_facility_latest" else "silver"
        buf = io.BytesIO()
        df.write_parquet(buf)
        store.put_bytes(f"{layer}/{name}/part-0.parquet", buf.getvalue(), overwrite=True)


def write_vintage_flags(meta: MetadataStore, *, stale: tuple[str, ...] = ()) -> None:
    """The release-level stale_refresh rows bronze would have written (harmonize/flags.py), one per vintage."""
    meta.upsert(
        "vintage_flags",
        [
            {
                "vintage_month": v,
                "flag": "stale_refresh",
                "table_name": "",
                "value": v in stale,
                "run_id": "toy",
                "updated_at": "2026-10-19T00:00:00",
            }
            for v in VINTAGES
        ],
    )


def write_bronze_and_silver_records(meta: MetadataStore) -> None:
    """The bronze_files and silver_partitions rows a real run leaves, one per (table, vintage), all current,
    so gold's freshness check (partitions_without_current_silver) passes on the toy world."""
    cfg = load_silver_config()
    bronze, silver = [], []
    for v in VINTAGES:
        for source in cfg.sources:
            content, zip_sha = f"toy-{source}-{v}", "0" * 64
            bronze.append({
                "run_id": "toy", "table_name": source, "vintage_month": v, "status": "loaded",
                "content_hash": content, "zip_sha256": zip_sha,
            })  # fmt: skip
            silver += [
                {"run_id": "toy", "table_name": t.name, "source_table": source, "vintage_month": v,
                 "status": "built", "content_hash": content, "zip_sha256": zip_sha,
                 "config_hash": cfg.config_hash}
                for t in cfg.tables_for(source)
            ]  # fmt: skip
    meta.append("bronze_files", bronze)
    meta.append("silver_partitions", silver)
