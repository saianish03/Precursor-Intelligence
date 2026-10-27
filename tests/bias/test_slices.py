"""Slice construction and the leakage guard (bias spec §5.1)."""

import logging

import polars as pl
import pytest

from precursorintelligence.bias.slices import (
    MISSING,
    LeakageError,
    SliceDef,
    build_slice_columns,
    check_no_leakage,
    enumerate_slices,
    slice_col,
)


def _one_col(cfg, **cols):
    """A frame with just the slice source columns (others filled so every slice can be built)."""
    base = {
        "state": ["MA"],
        "ownership_group": ["for_profit"],
        "certified_beds": [100.0],
        "chain_id": ["C1"],
        "provider_in_hospital": [False],
        "ccrc": [False],
        "sff_status": ["NONE"],
        "overall_rating": [3.0],
    }
    n = len(next(iter(cols.values())))
    data = {k: v * n for k, v in base.items()} | cols
    return build_slice_columns(pl.DataFrame(data, strict=False), cfg)


def test_bed_band_edges(cfg):
    out = _one_col(cfg, certified_beds=[49.0, 50.0, 99.0, 100.0, 199.0, 200.0, 0.0, None])
    assert out[slice_col("bed_band")].to_list() == [
        "<50",
        "50-99",
        "50-99",
        "100-149",
        "150-199",
        "200+",
        MISSING,  # 0 beds is a data error, not a small home
        MISSING,
    ]


def test_cms_region_map(cfg, caplog):
    with caplog.at_level(logging.WARNING, logger="precursorintelligence.bias.slices"):
        out = _one_col(cfg, state=["MA", "CA", "ZZ", None])
    assert out[slice_col("cms_region")].to_list() == ["1", "9", MISSING, MISSING]
    assert out[slice_col("state")].to_list() == ["MA", "CA", "ZZ", MISSING]
    assert "ZZ" in caplog.text


def test_chain_status(cfg):
    out = _one_col(cfg, chain_id=["C1", None, ""])
    assert out[slice_col("chain_status")].to_list() == ["chain", "independent", "independent"]


def test_bool_and_star_slices(cfg):
    out = _one_col(cfg, ccrc=[True, False, None], overall_rating=[5.0, 1.0, None])
    assert out[slice_col("ccrc")].to_list() == ["true", "false", MISSING]
    assert out[slice_col("star_band")].to_list() == ["5", "1", MISSING]


@pytest.mark.parametrize("bad", ["y_g12", "status_g12", "split_g12", "beds_fwd_6m"])
def test_leakage_guard(cfg, bad):
    with pytest.raises(LeakageError, match=bad):
        check_no_leakage([bad])
    leaky = cfg.model_copy(update={"columns": {**cfg.columns, "certified_beds": bad}})
    with pytest.raises(LeakageError):
        _one_col(leaky, **{bad: [1.0]})


def test_enumerate_includes_config_levels_and_crossings(cfg, gold_frame):
    df = build_slice_columns(gold_frame, cfg)
    slices = enumerate_slices(df, cfg)
    names = {s.name for s in slices}
    assert {"state", "bed_band", "ownership_group x bed_band"} <= names
    # star 1-5 and every bed band are listed even when absent; cms region 2 has no rows here
    assert SliceDef(("cms_region",), ("2",)) in slices
    assert SliceDef(("bed_band",), ("<50",)) in slices
    assert slices == enumerate_slices(df, cfg)  # deterministic
