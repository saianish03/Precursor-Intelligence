"""Silver configs and header resolution (spec §3.2): the hard rule and the alias ranges."""

import json
from pathlib import Path

import pytest

from precursorintelligence.common.config import ConfigError
from precursorintelligence.harmonize.columns import (
    Alias,
    Column,
    SourceMap,
    load_silver_config,
    resolve_headers,
)

CCN = Column(
    "ccn",
    "ccn",
    (
        Alias("Federal Provider Number", "2021-01", "2023-05"),
        Alias("CMS Certification Number (CCN)", "2023-06", None),
    ),
)
FINE_ID = Column("fine_id", "str", (Alias("Fine ID", "2026-06", None),))
SOURCE = SourceMap("penalties", {"ccn": CCN, "fine_id": FINE_ID}, frozenset({"Location"}))


def test_the_alias_valid_in_the_vintage_is_used():
    res = resolve_headers(SOURCE, ["Federal Provider Number", "Location"], "2023-05")
    assert res.problems == [] and res.rename == {"ccn": "Federal Provider Number"}
    assert res.absent == ["fine_id"]  # not published yet: null in silver, not an error


def test_an_open_ended_alias_stays_valid_for_new_releases():
    res = resolve_headers(SOURCE, ["CMS Certification Number (CCN)", "Fine ID"], "2027-03")
    assert res.problems == [] and res.rename == {
        "ccn": "CMS Certification Number (CCN)",
        "fine_id": "Fine ID",
    }


def test_an_unknown_header_fails():
    res = resolve_headers(
        SOURCE, ["CMS Certification Number (CCN)", "Fine ID", "Brand New"], "2026-07"
    )
    assert ("Brand New", "unmapped header") in res.problems


def test_an_alias_outside_its_range_fails():
    res = resolve_headers(SOURCE, ["Federal Provider Number"], "2024-01")
    problems = dict(res.problems)
    assert "alias of ccn used outside its range" in problems["Federal Provider Number"]
    assert "is missing" in problems["ccn"]


def test_a_missing_expected_header_fails():
    res = resolve_headers(SOURCE, ["CMS Certification Number (CCN)"], "2026-07")
    assert res.problems == [("fine_id", "expected header 'Fine ID' is missing")]


def test_the_repo_configs_load_and_cover_every_silver_table():
    cfg = load_silver_config()
    assert set(cfg.sources) == {
        "provider_info",
        "health_citations",
        "survey_dates",
        "penalties",
        "mds_qm",
        "claims_qm",
        "citation_lookup",
    }
    assert {t.name for t in cfg.tables_for("provider_info")} == {
        "provider_by_vintage",
        "provider_identity_by_vintage",
    }
    # current aliases are open-ended, so the next CMS release needs no config edit
    assert cfg.sources["penalties"].columns["ccn"].aliases[-1].end is None


def test_the_config_hash_ignores_comments_and_line_endings(tmp_path):
    from shutil import copy

    from precursorintelligence.common.config import project_root

    for name in ("column_map.yaml", "keep_columns.yaml", "value_maps.yaml", "calibration.yaml"):
        copy(project_root() / "configs" / name, tmp_path / name)
    vm = tmp_path / "value_maps.yaml"
    vm.write_bytes(b"# a new comment\r\n" + vm.read_bytes().replace(b"\n", b"\r\n"))
    assert load_silver_config(tmp_path).config_hash == load_silver_config().config_hash


def test_editing_the_calibration_settings_does_not_change_the_config_hash(tmp_path):
    from shutil import copy

    from precursorintelligence.common.config import project_root

    for name in ("column_map.yaml", "keep_columns.yaml", "value_maps.yaml", "calibration.yaml"):
        copy(project_root() / "configs" / name, tmp_path / name)
    cal = tmp_path / "calibration.yaml"
    cal.write_text(
        cal.read_text(encoding="utf-8").replace("coverage: 0.95", "coverage: 0.9"), encoding="utf-8"
    )
    edited = load_silver_config(tmp_path)
    assert edited.calibration.coverage == 0.9
    assert edited.config_hash == load_silver_config().config_hash


def test_a_keep_list_key_that_is_not_kept_is_a_config_error(tmp_path):
    from shutil import copy

    from precursorintelligence.common.config import project_root

    for name in ("column_map.yaml", "value_maps.yaml", "calibration.yaml"):
        copy(project_root() / "configs" / name, tmp_path / name)
    (tmp_path / "keep_columns.yaml").write_text(
        "t:\n  source: penalties\n  key: [fine_id]\n  columns: [ccn]\n", encoding="utf-8"
    )
    with pytest.raises(ConfigError, match="not kept"):
        load_silver_config(tmp_path)


@pytest.mark.parametrize(
    "entry", ["  source: penalties\n  columns: [ccn]\n", "  source: penalties\n  key: [ccn]\n"]
)
def test_a_keep_list_entry_without_key_or_columns_is_a_config_error(tmp_path, entry):
    from shutil import copy

    from precursorintelligence.common.config import project_root

    for name in ("column_map.yaml", "value_maps.yaml", "calibration.yaml"):
        copy(project_root() / "configs" / name, tmp_path / name)
    (tmp_path / "keep_columns.yaml").write_text("t:\n" + entry, encoding="utf-8")
    with pytest.raises(ConfigError, match="missing"):
        load_silver_config(tmp_path)


def test_a_calibration_config_with_a_missing_setting_is_a_config_error(tmp_path):
    from shutil import copy

    from precursorintelligence.common.config import project_root

    for name in ("column_map.yaml", "keep_columns.yaml", "value_maps.yaml"):
        copy(project_root() / "configs" / name, tmp_path / name)
    (tmp_path / "calibration.yaml").write_text("lag:\n  coverage: 0.95\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="calibration.yaml"):
        load_silver_config(tmp_path)


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("coverage: 0.95", "coverage: 95"),
        ("max_lag_months: 24", "max_lag_months: 0"),
        ('citations_from: "2021-06-01"', 'citations_from: "2021-06"'),
        ('stable_from: "2022-01"', 'stable_from: "2022-01-01"'),
    ],
)
def test_a_calibration_setting_out_of_range_is_a_config_error(tmp_path, old, new):
    from shutil import copy

    from precursorintelligence.common.config import project_root

    for name in ("column_map.yaml", "keep_columns.yaml", "value_maps.yaml", "calibration.yaml"):
        copy(project_root() / "configs" / name, tmp_path / name)
    cal = tmp_path / "calibration.yaml"
    text = cal.read_text(encoding="utf-8")
    assert old in text
    cal.write_text(text.replace(old, new), encoding="utf-8")
    with pytest.raises(ConfigError, match="calibration.yaml"):
        load_silver_config(tmp_path)


def test_the_calibration_settings_are_read_from_the_config():
    cal = load_silver_config().calibration
    assert cal.coverage == 0.95 and cal.fully_observed_months == 24
    assert cal.bridges[("453", "479")][0] == "ls_pressure_ulcer"
    assert cal.drop["471"] == "ended 2024-11"


def test_the_sep_2026_release_headers_are_all_mapped_or_ignored():
    """The real header sets of the 2026-10-08 release (Sep 2026 data, recorded by Step 1). CMS added
    `Survey Footnote` (citations, surveys) and three provider columns; an unknown header fails silver."""
    cfg = load_silver_config()
    headers = json.loads((Path(__file__).parent / "data" / "headers_2026-09.json").read_text())
    for table, names in headers.items():
        assert resolve_headers(cfg.sources[table], names, "2026-09").problems == [], table


def test_the_sep_2026_columns_are_mapped_from_2026_09_and_absent_before():
    cfg = load_silver_config()
    headers = json.loads((Path(__file__).parent / "data" / "headers_2026-09.json").read_text())
    new = {
        "provider_info": {"high_performing_icon": "High Performing Icon"},
        "health_citations": {"survey_fn": "Survey Footnote"},
        "survey_dates": {"survey_fn": "Survey Footnote"},
    }
    for table, cols in new.items():
        res = resolve_headers(cfg.sources[table], headers[table], "2026-09")
        assert {c: res.rename[c] for c in cols} == cols, table
        older = [h for h in headers[table] if h not in cols.values()]
        assert set(cols) <= set(resolve_headers(cfg.sources[table], older, "2026-08").absent), table
