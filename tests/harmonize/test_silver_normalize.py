"""Type and value normalisation (spec §3.4): what converts, what is counted, what fails."""

from datetime import date

import polars as pl
import pytest

from precursorintelligence.common.config import project_root
from precursorintelligence.harmonize.normalize import (
    NormalizeError,
    measure_period_bounds,
    normalize,
    ownership_group,
)

VM = __import__("yaml").safe_load(
    (project_root() / "configs" / "value_maps.yaml").read_text(encoding="utf-8")
)


def run(values: dict[str, list], dtypes: dict[str, str]):
    return normalize(pl.DataFrame(values, schema={k: pl.String for k in values}), dtypes, VM)


def test_ccn_keeps_letters_and_leading_zeros():
    out, bad = run({"ccn": ["015009", " 15e123 "]}, {"ccn": "ccn"})
    assert out["ccn"].to_list() == ["015009", "15E123"] and bad == {}


@pytest.mark.parametrize("ccn", ["15009", "0150091", "01-509", None, ""])
def test_a_missing_or_malformed_ccn_fails_the_partition(ccn):
    with pytest.raises(NormalizeError, match="CCN"):
        run({"ccn": ["015009", ccn]}, {"ccn": "ccn"})


def test_values_that_do_not_convert_become_null_and_are_counted():
    out, bad = run(
        {
            "d": ["2025-06-01", "06/01/2025", "", None],
            "n": ["1.5", "abc", " ", None],
            "f": ["Y", "N", "maybe", ""],
            "s": ["G", "l", "Z", ""],
        },
        {"d": "date", "n": "float", "f": "bool_yn", "s": "severity"},
    )
    assert out["d"].to_list() == [date(2025, 6, 1), None, None, None]
    assert out["n"].to_list() == [1.5, None, None, None]
    assert out["f"].to_list() == [True, False, None, None]
    assert out["s"].to_list() == ["G", "L", None, None]
    assert bad == {"d": 1, "n": 1, "f": 1, "s": 1}  # empty cells are missing, not invalid


def test_empty_text_is_null_but_text_is_kept_verbatim():
    out, _ = run({"t": ["  Oak  Manor ", "", None]}, {"t": "str"})
    assert out["t"].to_list() == ["  Oak  Manor ", None, None]


def test_survey_types_map_to_the_enum_and_unknown_ones_to_other():
    out, bad = run(
        {"st": ["Health Standard", "Fire Safety Complaint", "Life Safety", None]},
        {"st": "survey_type_enum"},
    )
    assert out["st"].to_list() == ["HEALTH_STANDARD", "FIRE_COMPLAINT", "OTHER", None]
    assert bad == {"st": 1}


def test_enums_blank_is_a_category_and_the_sprinkler_sentinel_is_flagged():
    out, bad = run(
        {
            "special_focus_status": ["SFF", "SFF Candidate", ""],
            "sprinkler_status": ["Yes", "Data Not Available", "Odd"],
        },
        {"special_focus_status": "enum", "sprinkler_status": "enum"},
    )
    assert out["special_focus_status"].to_list() == ["SFF", "SFF_CANDIDATE", "NONE"]
    assert out["sprinkler_status"].to_list() == ["YES", None, None]
    assert out["sprinkler_status__na"].to_list() == [False, True, False]
    assert bad == {"sprinkler_status": 1}  # the sentinel is expected; "Odd" is not


def test_tags_keep_four_digit_text():
    out, bad = run({"tag": ["0880", "880", "F0880"]}, {"tag": "tag"})
    assert out["tag"].to_list() == ["0880", None, None] and bad == {"tag": 2}


def test_ownership_group_is_the_prefix():
    df = pl.DataFrame(
        {"ownership_type": ["For profit - Corporation", "Government - County", "Other"]}
    )
    assert df.select(ownership_group(VM))["ownership_type"].to_list() == [
        "for_profit",
        "government",
        None,
    ]


def test_measure_periods_become_dates():
    mds = pl.DataFrame({"measure_period": ["2024Q2-2025Q1", "bad"]})
    start, end = measure_period_bounds("mds_qm")
    assert mds.select(start.alias("s"), end.alias("e")).rows() == [
        (date(2024, 4, 1), date(2025, 3, 31)),
        (None, None),
    ]
    claims = pl.DataFrame({"measure_period": ["20240101-20241231"]})
    start, end = measure_period_bounds("claims_qm")
    assert claims.select(start.alias("s"), end.alias("e")).rows() == [
        (date(2024, 1, 1), date(2024, 12, 31))
    ]
