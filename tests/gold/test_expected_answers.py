"""Expected gold answers for the fake silver world in silver_fakes.py.

Two kinds of test:
- fixture tests check that the fake silver data says what the story says, and uses silver's own column
  names (configs/keep_columns.yaml);
- gold tests run the real spine and labels builders on it and compare with hand-computed answers
  (spec §7, §9). A feature test is xfail while its family is not built; then a wrong answer fails
  for real (only NotImplementedError is expected).

Label arithmetic, V = 2025-03, H = 12, W_12 = (cutoff_date, cutoff_date + 12 months]:
- y_g12, B_cit = 8: mature only if t + 20 months <= V, so the last labelable t is 2023-07;
- y_g6, B_cit = 8: mature only if t + 14 months <= V, so the last labelable t is 2024-01;
- y_pen12, B_pen = 12: mature only if t + 24 months <= V, so the last labelable t is 2023-03.
"""

from pathlib import Path

import polars as pl
import pytest
import yaml
from silver_fakes import BY_VINTAGE, KEYS, VINTAGES, V

from precursorintelligence.common.config import load_labels_config, load_splits_config
from precursorintelligence.common.io.table_sink import ParquetTableSink
from precursorintelligence.gold import GoldCheckError
from precursorintelligence.gold.labels import build_labels
from precursorintelligence.gold.spine import build_spine
from precursorintelligence.gold.splits import add_splits

LAST_LABELED_G12 = "2023-07"

# (ccn, t, status_g12, y_g12) for every mature spine row. Every later row is immature with y_g12 null.
EXPECTED_G12 = [
    # Sunny Acres: G on 2023-06-15 lies in W_12 for t <= 2023-06 (cutoff 2023-06-01 is before it).
    ("015001", "2023-01", "positive", 1),
    ("015001", "2023-02", "positive", 1),
    ("015001", "2023-03", "positive", 1),
    ("015001", "2023-04", "positive", 1),
    ("015001", "2023-05", "positive", 1),
    ("015001", "2023-06", "positive", 1),
    ("015001", "2023-07", "negative", 0),
    # Oak Grove: D and E only.
    ("015002", "2023-01", "negative", 0),
    ("015002", "2023-02", "negative", 0),
    ("015002", "2023-03", "negative", 0),
    ("015002", "2023-04", "negative", 0),
    ("015002", "2023-05", "negative", 0),
    ("015002", "2023-06", "negative", 0),
    ("015002", "2023-07", "negative", 0),
    # Pine View: no G, last report 2023-08 < t + 12 + 8, so censored, never negative.
    ("01A193", "2023-01", "censored", None),
    ("01A193", "2023-02", "censored", None),
    ("01A193", "2023-03", "censored", None),
    ("01A193", "2023-04", "censored", None),
    ("01A193", "2023-05", "censored", None),
    ("01A193", "2023-06", "censored", None),
    ("01A193", "2023-07", "censored", None),
    # Maple House: G on 2024-02-10, posted late (2024-06). In W_12 from t = 2023-03 (window ends
    # 2024-03-01); for t = 2023-02 the window ends 2024-02-01, before it.
    ("015004", "2023-01", "negative", 0),
    ("015004", "2023-02", "negative", 0),
    ("015004", "2023-03", "positive", 1),
    ("015004", "2023-04", "positive", 1),
    ("015004", "2023-05", "positive", 1),
    ("015004", "2023-06", "positive", 1),
    ("015004", "2023-07", "positive", 1),
    # Cedar Hill: first report 2023-04; only D and E.
    ("015005", "2023-04", "negative", 0),
    ("015005", "2023-05", "negative", 0),
    ("015005", "2023-06", "negative", 0),
    ("015005", "2023-07", "negative", 0),
]

# Spine rows per facility (§7): every published t the facility appears in.
EXPECTED_SPINE_ROWS = {"015001": 26, "015002": 26, "01A193": 8, "015004": 26, "015005": 23}

LAST_LABELED_PEN12 = "2023-03"

# (ccn, t, status_pen12, y_pen12) for every mature spine row. Cedar Hill has none (first t 2023-04).
EXPECTED_PEN12 = [
    # Sunny Acres: fine on 2023-06-15 lies in W_12.
    ("015001", "2023-01", "positive", 1),
    ("015001", "2023-02", "positive", 1),
    ("015001", "2023-03", "positive", 1),
    # Oak Grove: the two identical fines on 2023-10-05 are one revision group, so one positive.
    ("015002", "2023-01", "positive", 1),
    ("015002", "2023-02", "positive", 1),
    ("015002", "2023-03", "positive", 1),
    # Pine View: no fine, last report 2023-08 < t + 12 + 12.
    ("01A193", "2023-01", "censored", None),
    ("01A193", "2023-02", "censored", None),
    ("01A193", "2023-03", "censored", None),
    # Maple House: never fined, present to V.
    ("015004", "2023-01", "negative", 0),
    ("015004", "2023-02", "negative", 0),
    ("015004", "2023-03", "negative", 0),
]

# features_enforcement (§8.4) uses only the report at t: a fine counts once it is posted, and every
# row counts, so Oak Grove's dup_seq 2 twin makes 2 fines.
EXPECTED_N_FINES_24M = {
    ("015001", "2023-08"): 0,  # dated 2023-06-15 but not posted until the 2023-09 report
    ("015001", "2023-09"): 1,
    ("015002", "2023-11"): 0,  # posted in 2024-01
    ("015002", "2024-01"): 2,
    ("015004", "2024-01"): 0,
}

# features_quality (§8.6): d6 for the antipsychotic measure (bridged 419 → 481) at Sunny Acres.
# The feature name assumes qm_measures.yaml calls the bridge `ls_antipsychotic`.
D6_ANTIPSYCHOTIC_NULL = (
    [v for v in VINTAGES if v <= "2023-06"]  # t-6 is before the first toy report
    + ["2024-06"]  # t-6 = 2023-12, never published
    + ["2024-07", "2024-08", "2024-09", "2024-10", "2024-11"]  # t-6 has 419, t has 481
)

MAPLE_G = (pl.col("ccn") == "015004") & (pl.col("survey_date") == pl.date(2024, 2, 10))


# ------------------------------------------------------------------ fixture tests


KEEP_COLUMNS = yaml.safe_load(
    (Path(__file__).parents[2] / "configs" / "keep_columns.yaml").read_text(encoding="utf-8")
)


@pytest.mark.parametrize("table", BY_VINTAGE)
def test_by_vintage_columns_are_silvers_keep_columns(silver, table):
    # The same names and order silver writes (configs/keep_columns.yaml), so a silver column change shows here.
    assert silver[table].columns == [
        *KEEP_COLUMNS[table]["columns"],
        "vintage_month",
        "cutoff_date",
    ]


def test_reports_run_2023_01_to_2025_03_without_2023_12(silver):
    assert len(VINTAGES) == 26
    assert (VINTAGES[0], VINTAGES[-1], V) == ("2023-01", "2025-03", "2025-03")
    for name in BY_VINTAGE:
        months = set(silver[name]["vintage_month"])
        assert "2023-12" not in months
        if name != "penalties_by_vintage":  # the first fine is posted in 2023-09
            assert months == set(VINTAGES), name


def test_ccn_is_six_character_text(silver):
    for name, df in silver.items():
        if "ccn" in df.columns:
            assert df.schema["ccn"] == pl.String, name
            assert df["ccn"].str.contains(r"^[0-9A-Z]{6}$").all(), name
    assert "01A193" in set(silver["citations_by_vintage"]["ccn"])


def test_cutoff_date_is_first_of_vintage_month(silver):
    for name in BY_VINTAGE:
        df = silver[name]
        assert (df["cutoff_date"].dt.strftime("%Y-%m-%d") == df["vintage_month"] + "-01").all()


def test_within_vintage_keys_are_unique_and_sorted(silver):
    for name, key in KEYS.items():
        df = silver[name]
        assert not df.select(key).is_duplicated().any(), name
        assert df.equals(df.sort(key)), name


def test_maple_house_g_citation_is_posted_late(silver):
    cit = silver["citations_by_vintage"]
    seen = cit.filter(MAPLE_G)["vintage_month"].to_list()
    assert seen == [v for v in VINTAGES if v >= "2024-06"]
    # The 2024-03 report has Maple House (and its other citation) but not the G citation.
    march = cit.filter(pl.col("vintage_month") == "2024-03", pl.col("ccn") == "015004")
    assert march.height == 1 and march["survey_date"][0].isoformat() == "2022-12-07"
    assert silver["ledger_citations"].filter(MAPLE_G)["first_seen_vintage"].to_list() == ["2024-06"]


def test_gplus_citations_in_the_ledger(silver):
    gplus = silver["ledger_citations"].filter(pl.col("final_scope_severity").is_in(list("GHIJKL")))
    assert sorted(
        (r["ccn"], r["survey_date"].isoformat(), r["first_seen_vintage"])
        for r in gplus.iter_rows(named=True)
    ) == [("015001", "2023-06-15", "2023-08"), ("015004", "2024-02-10", "2024-06")]
    oak = silver["citations_by_vintage"].filter(pl.col("ccn") == "015002")
    assert set(oak["scope_severity"]) == {"D", "E"}
    assert not silver["ledger_citations"]["removed_by_cms"].any()


def test_facility_presence(silver):
    rows = {r["ccn"]: r for r in silver["facility_presence"].iter_rows(named=True)}
    assert {
        c: (r["first_vintage"], r["last_vintage"], r["n_vintages"]) for c, r in rows.items()
    } == {
        "015001": ("2023-01", "2025-03", 26),
        "015002": ("2023-01", "2025-03", 26),
        "01A193": ("2023-01", "2023-08", 8),
        "015004": ("2023-01", "2025-03", 26),
        "015005": ("2023-04", "2025-03", 23),
    }
    universe = silver["facility_universe"].group_by("ccn").len()
    assert dict(universe.iter_rows()) == EXPECTED_SPINE_ROWS
    dim = silver["dim_facility_latest"]
    assert dim.filter(~pl.col("is_active"))["ccn"].to_list() == ["01A193"]


def test_sunny_acres_fine_is_on_its_g_citation_day(silver):
    fines = silver["ledger_penalties"].filter(pl.col("ccn") == "015001")
    assert [
        (d.isoformat(), s) for d, s in fines.select("penalty_date", "first_seen_vintage").rows()
    ] == [("2023-06-15", "2023-09")]
    g_days = silver["ledger_citations"].filter(
        pl.col("ccn") == "015001", pl.col("final_scope_severity") == "G"
    )["survey_date"]
    assert fines["penalty_date"].to_list() == g_days.to_list()


def test_oak_grove_identical_fines_get_dup_seq_1_and_2(silver):
    pen = silver["penalties_by_vintage"].filter(pl.col("ccn") == "015002")
    for _, rows in pen.group_by("vintage_month"):
        assert rows["dup_seq"].to_list() == [1, 2]
        assert rows["penalty_event_key"].n_unique() == 2
        assert rows.drop("dup_seq", "penalty_event_key").n_unique() == 1  # otherwise identical
    assert sorted(set(pen["vintage_month"])) == [v for v in VINTAGES if v >= "2024-01"]
    ledger = silver["ledger_penalties"].filter(pl.col("ccn") == "015002")
    assert sorted(ledger["dup_seq"]) == [1, 2]  # two events, not one


def test_antipsychotic_code_changes_from_419_to_481_in_2024_06(silver):
    mds = silver["mds_qm_by_vintage"]
    for code, expected in [
        ("419", [v for v in VINTAGES if v < "2024-06"]),
        ("481", [v for v in VINTAGES if v >= "2024-06"]),
    ]:
        assert sorted(set(mds.filter(pl.col("measure_code") == code)["vintage_month"])) == expected
    both = mds.filter(pl.col("measure_code").is_in(["419", "481"]))
    assert not both.select("ccn", "vintage_month").is_duplicated().any()  # never both in one report
    inv = silver["qm_inventory"].filter(pl.col("measure_code").is_in(["419", "481"]))
    assert inv.select("measure_code", "first_vintage", "last_vintage").rows() == [
        ("419", "2023-01", "2024-05"),
        ("481", "2024-06", "2025-03"),
    ]


def test_quality_rows_per_facility_and_measure_period(silver):
    per_report = {"mds_qm_by_vintage": 3, "claims_qm_by_vintage": 2}
    for name, n in per_report.items():
        counts = silver[name].group_by("ccn", "vintage_month").len()["len"]
        assert set(counts) == {n}, name
    aug = silver["mds_qm_by_vintage"].filter(pl.col("vintage_month") == "2024-08")
    assert set(aug["measure_period"]) == {"2023Q2-2024Q1"}  # the spec's §6 example


def test_write_silver_uses_the_spec_layout(silver, silver_store):
    assert silver_store.exists("silver/citations_by_vintage/vintage=2024-03/part-0.parquet")
    assert not silver_store.exists("silver/citations_by_vintage/vintage=2023-12/part-0.parquet")
    assert silver_store.exists("silver/ledger_citations/part-0.parquet")
    assert silver_store.exists(
        "silver/penalties_by_vintage/vintage=2023-01/part-0.parquet"
    )  # empty
    assert silver_store.exists("silver/qm_inventory/part-0.parquet")
    assert silver_store.exists("gold/dim_facility_latest/part-0.parquet")
    back = ParquetTableSink(silver_store, prefix="silver").read("citations_by_vintage")
    key = KEYS["citations_by_vintage"]
    assert back.drop("vintage").sort(key).equals(silver["citations_by_vintage"])


# ------------------------------------------------------------------ gold tests

LAST_LABELED_G6 = "2024-01"  # t + 6 + 8 <= 2025-03


def _months(start: str, end: str) -> list[str]:
    return [v for v in VINTAGES if start <= v <= end]


def _same(ccn: str, months: list[str], status: str) -> list[tuple[str, str, str, int | None]]:
    return [(ccn, t, status, {"positive": 1, "negative": 0}.get(status)) for t in months]


# (ccn, t, status_g6, y_g6) for every mature spine row (W_6 = (cutoff, cutoff + 6 months]).
EXPECTED_G6 = [
    # Sunny Acres: G on 2023-06-15 is in W_6 for t = 2023-01 … 2023-06.
    *_same("015001", _months("2023-01", "2023-06"), "positive"),
    *_same("015001", _months("2023-07", "2024-01"), "negative"),
    *_same("015002", _months("2023-01", "2024-01"), "negative"),
    # Pine View: last report 2023-08 < t + 6 + 8 for every t it has.
    *_same("01A193", _months("2023-01", "2023-08"), "censored"),
    # Maple House: G on 2024-02-10 is in W_6 once the cutoff is on or after 2023-08-10.
    *_same("015004", _months("2023-01", "2023-08"), "negative"),
    *_same("015004", _months("2023-09", "2024-01"), "positive"),
    *_same("015005", _months("2023-04", "2024-01"), "negative"),
]

LABEL_CASES = [
    ("g12", LAST_LABELED_G12, EXPECTED_G12),
    ("g6", LAST_LABELED_G6, EXPECTED_G6),
    ("pen12", LAST_LABELED_PEN12, EXPECTED_PEN12),
]


@pytest.fixture(scope="module")
def spine(silver) -> pl.DataFrame:
    return build_spine(
        silver["facility_universe"],
        published_vintages=VINTAGES,
        spine_start=load_splits_config().spine_start,
        v=V,
    )


@pytest.fixture(scope="module")
def labels(silver, spine) -> pl.DataFrame:
    return build_labels(
        spine,
        ledger_citations=silver["ledger_citations"],
        ledger_penalties=silver["ledger_penalties"],
        facility_presence=silver["facility_presence"],
        cfg=load_labels_config(),
        v=V,
    )


def test_spine_has_one_row_per_facility_per_published_month(spine):
    assert spine.columns == ["ccn", "t", "cutoff_date"]
    assert spine.height == 109
    assert dict(spine.group_by("ccn").len().iter_rows()) == EXPECTED_SPINE_ROWS
    assert "2023-12" not in set(spine["t"])
    assert spine.filter(pl.col("ccn") == "015005")["t"].min() == "2023-04"
    assert spine.filter(pl.col("ccn") == "01A193")["t"].max() == "2023-08"
    assert spine.filter(pl.col("ccn") == "015004")["t"].min() == "2023-01"


def test_spine_refuses_a_published_month_without_rows(silver):
    universe = silver["facility_universe"].filter(pl.col("vintage_month") != "2024-03")
    with pytest.raises(GoldCheckError, match="2024-03"):
        build_spine(universe, published_vintages=VINTAGES, spine_start="2023-01", v=V)


@pytest.mark.parametrize(("label", "last_labeled", "expected"), LABEL_CASES)
def test_label_answers_for_mature_months(labels, label, last_labeled, expected):
    mature = labels.filter(pl.col("t") <= last_labeled)
    got = sorted(mature.select("ccn", "t", f"status_{label}", f"y_{label}").iter_rows())
    assert got == sorted(expected)


@pytest.mark.parametrize(("label", "last_labeled", "expected"), LABEL_CASES)
def test_rows_after_last_labelable_month_are_immature(labels, label, last_labeled, expected):
    late = labels.filter(pl.col("t") > last_labeled)
    assert late.height == 109 - len(expected)
    assert set(late[f"status_{label}"]) == {"immature"}
    assert late[f"y_{label}"].is_null().all()


def test_labels_have_one_row_per_spine_row(labels, spine):
    assert labels.select("ccn", "t").equals(spine.select("ccn", "t"))
    assert labels.columns == [
        "ccn", "t", "y_g12", "status_g12", "y_g6", "status_g6", "y_pen12", "status_pen12",
    ]  # fmt: skip


def test_late_posted_citation_still_labels_earlier_months(labels):
    # Maple House's G (2024-02-10) was first published in 2024-06, after these cutoffs; the ledger
    # still knows it, so the label sees it.
    maple = labels.filter(pl.col("ccn") == "015004", pl.col("t").is_in(["2023-02", "2023-03"]))
    assert maple.select("t", "y_g12").rows() == [("2023-02", 0), ("2023-03", 1)]


# ------------------------------------------------------------------ gold features (not built yet)

not_built = pytest.mark.xfail(
    raises=NotImplementedError, strict=True, reason="gold feature family not implemented yet"
)


def _features(silver, spine, family: str) -> pl.DataFrame:
    """gold/features_<family> (§8). Point this at the real builder when features land."""
    from precursorintelligence.gold.features import build_features

    return build_features(family, silver, spine)


def test_fine_counts_use_only_what_was_posted_at_t(silver, spine):
    enf = _features(silver, spine, "enforcement")
    got = {
        (r["ccn"], r["t"]): r["n_fines_24m"]
        for r in enf.iter_rows(named=True)
        if (r["ccn"], r["t"]) in EXPECTED_N_FINES_24M
    }
    assert got == EXPECTED_N_FINES_24M


@not_built
def test_d6_antipsychotic_is_null_across_the_code_change(silver, spine):
    qual = _features(silver, spine, "quality").filter(pl.col("ccn") == "015001").sort("t")
    null_months = qual.filter(pl.col("d6_qm_ls_antipsychotic").is_null())["t"].to_list()
    assert null_months == D6_ANTIPSYCHOTIC_NULL


def test_spine_refuses_rows_for_a_never_published_month(silver):
    with pytest.raises(GoldCheckError, match="never published"):
        build_spine(
            silver["facility_universe"],
            published_vintages=[v for v in VINTAGES if v != "2024-03"],
            spine_start="2023-01",
            v=V,
        )


@pytest.mark.parametrize(
    "ccn",
    [pl.Series("ccn", [15001] * 1), pl.Series("ccn", ["15001"])],
    ids=["int", "zero-stripped"],
)
def test_spine_refuses_a_ccn_that_is_not_six_character_text(ccn):
    universe = pl.DataFrame({"ccn": ccn, "vintage_month": ["2023-01"]})
    with pytest.raises(GoldCheckError, match="6-character text"):
        build_spine(universe, published_vintages=["2023-01"], spine_start="2023-01", v="2023-01")


# ------------------------------------------------------------------ splits (spec §9.3, §9.5)

# Toy V = 2025-03, test_months = 3, calib_months = 3, spine from 2023-01. Calendar months; 2023-12 has no rows.
#   g6:    last_labeled_t 2024-01 -> train 2023-01, calib 2023-02..04, purge 2023-05..10, test 2023-11..2024-01
#   g12:   last_labeled_t 2023-07 -> test 2023-05..07, purge 2022-05..2023-04 (spine: 2023-01..04), no calib/train
#   pen12: last_labeled_t 2023-03 -> test 2023-01..03, the purge, calib and train windows fall before the spine
EXPECTED_SPLIT_MONTHS = {
    "g6": {"train": _months("2023-01", "2023-01"), "calib": _months("2023-02", "2023-04"),
           "purge": _months("2023-05", "2023-10"), "test": _months("2023-11", "2024-01"),
           "score": _months("2024-02", "2025-03")},
    "g12": {"purge": _months("2023-01", "2023-04"), "test": _months("2023-05", "2023-07"),
            "score": _months("2023-08", "2025-03")},
    "pen12": {"test": _months("2023-01", "2023-03"), "score": _months("2023-04", "2025-03")},
}  # fmt: skip
EXPECTED_SPLIT_ROWS = {
    "g6": {"train": 4, "calib": 13, "purge": 28, "test": 8, "score": 56},
    "g12": {"purge": 17, "test": 15, "score": 77},
    "pen12": {"test": 12, "score": 97},
}


@pytest.fixture(scope="module")
def split_labels(labels) -> pl.DataFrame:
    return add_splits(labels, V, load_labels_config(), load_splits_config())


@pytest.mark.parametrize("label", ["g6", "g12", "pen12"])
def test_split_months_on_the_toy_world(split_labels, label):
    got = {
        split: sorted(set(rows["t"])) for (split,), rows in split_labels.group_by(f"split_{label}")
    }
    assert got == EXPECTED_SPLIT_MONTHS[label]


@pytest.mark.parametrize("label", ["g6", "g12", "pen12"])
def test_split_rows_on_the_toy_world(split_labels, label):
    counts = dict(split_labels[f"split_{label}"].value_counts().iter_rows())
    assert counts == EXPECTED_SPLIT_ROWS[label]
    assert sum(counts.values()) == 109


def test_g6_trains_on_sunny_positive_oak_and_maple_negative_and_pine_censored(split_labels):
    train = split_labels.filter(pl.col("split_g6") == "train").sort("ccn")
    assert train.select("ccn", "status_g6").rows() == [
        ("015001", "positive"), ("015002", "negative"), ("015004", "negative"), ("01A193", "censored"),
    ]  # fmt: skip
