"""An earlier ``--v`` ignores events first published after it (spec §0.4), end to end through run_gold.

Maple House's G citation (survey 2024-02-10) is first published in the 2024-06 report. With the buffers set to
0, t = 2023-03 is labelled at V = 2024-05 and at V = 2024-06, and its window (2023-03-01, 2024-03-01] holds
that survey date: the label must ignore the citation at 2024-05 and count it at 2024-06.
"""

import sys
from pathlib import Path

import polars as pl
import pytest

from precursorintelligence.common.config import (
    load_labels_config,
    load_settings,
    load_splits_config,
)
from precursorintelligence.gold.build import run_gold
from precursorintelligence.ingestion.context import build_context

sys.path.insert(0, str(Path(__file__).parents[1] / "gold"))  # the toy world lives in tests/gold
from make_toy_silver import make_toy, no_gate_splits  # noqa: E402

NO_BUFFERS = load_labels_config().model_copy(
    update={"buffers_months": {"citations": 0, "penalties": 0}}
)


@pytest.mark.parametrize(
    ("v", "status", "y"), [("2024-05", "negative", 0), ("2024-06", "positive", 1)]
)
def test_an_earlier_v_ignores_a_citation_first_published_after_it(tmp_path, v, status, y):
    env = make_toy(tmp_path / "toy")
    ctx = build_context(load_settings(env_path=env))
    result = run_gold(
        ctx, labels_cfg=NO_BUFFERS, splits_cfg=load_splits_config(no_gate_splits(env.parent)), v=v
    )
    assert result.exit_code == 0, result.failures
    labels = pl.read_parquet(tmp_path / "toy" / "gold" / "labels" / "part-0.parquet")
    maple = labels.filter(pl.col("ccn") == "015004", pl.col("t") == "2023-03")
    assert maple.select("status_g12", "y_g12").rows() == [(status, y)]
