"""Fixtures for the bias tests: a synthetic gold frame with planted, known values, and the toy world's gold.

The bias checks need the optional "bias" extra (fairlearn, statsmodels, scipy, matplotlib); CI installs every
extra, and a local run without it skips these tests instead of failing.
"""

import sys
from pathlib import Path

import pytest

pytest.importorskip("fairlearn")
pytest.importorskip("statsmodels")
pytest.importorskip("matplotlib")

from bias_fakes import make_gold  # noqa: E402

from precursorintelligence.bias.config import load_bias_config  # noqa: E402

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "gold")
)  # the toy world (make_toy_silver)


@pytest.fixture
def cfg():
    return load_bias_config()


@pytest.fixture
def gold_frame():
    return make_gold()


@pytest.fixture(scope="session")
def toy_gold(tmp_path_factory) -> Path:
    """The toy world's real gold run (tests/gold/make_toy_silver.py → precursor-pipeline gold): its root."""
    from make_toy_silver import TOY_SPLITS, make_toy

    from precursorintelligence.common.config import (
        load_labels_config,
        load_settings,
        load_splits_config,
    )
    from precursorintelligence.gold.build import run_gold
    from precursorintelligence.ingestion.context import build_context, new_run_id

    root = tmp_path_factory.mktemp("biastoy")
    env_file = make_toy(root)
    ctx = build_context(load_settings("toy", env_path=env_file), run_id=new_run_id())
    res = run_gold(
        ctx,
        labels_cfg=load_labels_config(),
        splits_cfg=load_splits_config(root / TOY_SPLITS),
        own_lock=True,
    )
    assert not res.failures, res.failures
    return root
