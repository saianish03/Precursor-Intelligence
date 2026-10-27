"""configs/validation.yaml and its loader."""

import pytest
import yaml

from precursorintelligence.common.config import ConfigError, load_validation_config


def write(tmp_path, **over):
    doc = {
        "schema_dir": str(tmp_path / "contracts"),
        "low_cardinality_max": 25,
        "top_values": 5,
        "layers": {"silver": {"partition_key": "vintage", "tables": ["citations_by_vintage"]}},
        "drift": {"row_count_change_pct": 10, "null_rate_jump_pts": 5, "mean_shift_std": 0.5},
    } | over
    path = tmp_path / "validation.yaml"
    path.write_text(yaml.safe_dump(doc))
    return path


def test_the_committed_config_loads_both_layers():
    cfg = load_validation_config()
    assert set(cfg.layers) == {"silver", "gold"}
    assert cfg.layers["silver"].partition_key == "vintage"
    assert cfg.layers["gold"].month_column == "t"
    assert "y_*" in cfg.drift.skip_columns


def test_schema_path_is_one_yaml_per_table(tmp_path):
    cfg = load_validation_config(write(tmp_path))
    assert (
        cfg.schema_path("citations_by_vintage")
        == tmp_path / "contracts" / "citations_by_vintage.yaml"
    )


def test_a_layer_needs_exactly_one_way_to_find_its_months(tmp_path):
    both = {"silver": {"partition_key": "vintage", "month_column": "t", "tables": ["x"]}}
    with pytest.raises(ConfigError, match="exactly one"):
        load_validation_config(write(tmp_path, layers=both))


@pytest.mark.parametrize(
    "over",
    [
        {"layers": {"bronze": {"partition_key": "vintage", "tables": ["x"]}}},
        {"drift": {"row_count_change_pct": 0, "null_rate_jump_pts": 5, "mean_shift_std": 0.5}},
        {"layers": {"silver": {"partition_key": "vintage", "tables": []}}},
        {"unknown_key": 1},
    ],
)
def test_a_bad_config_is_a_config_error(tmp_path, over):
    with pytest.raises(ConfigError):
        load_validation_config(write(tmp_path, **over))
