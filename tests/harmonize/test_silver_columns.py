"""Silver configs and header resolution (spec §3.2): the hard rule and the alias ranges."""

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

    for name in ("column_map.yaml", "keep_columns.yaml", "value_maps.yaml"):
        copy(project_root() / "configs" / name, tmp_path / name)
    vm = tmp_path / "value_maps.yaml"
    vm.write_bytes(b"# a new comment\r\n" + vm.read_bytes().replace(b"\n", b"\r\n"))
    assert load_silver_config(tmp_path).config_hash == load_silver_config().config_hash


def test_a_keep_list_key_that_is_not_kept_is_a_config_error(tmp_path):
    from shutil import copy

    from precursorintelligence.common.config import project_root

    for name in ("column_map.yaml", "value_maps.yaml"):
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

    for name in ("column_map.yaml", "value_maps.yaml"):
        copy(project_root() / "configs" / name, tmp_path / name)
    (tmp_path / "keep_columns.yaml").write_text("t:\n" + entry, encoding="utf-8")
    with pytest.raises(ConfigError, match="missing"):
        load_silver_config(tmp_path)
