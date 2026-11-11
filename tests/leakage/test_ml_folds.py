"""Leakage test 4 (docs/ML_PIPELINE_SPEC.md §17.1, §3.4): each facility is in exactly one fold, in every frame and
every run. (That OOF predictions come from fits without the facility is W1's half of the test.)"""

import sys
from pathlib import Path

import polars as pl

from precursorintelligence.common.config import (
    load_feature_registry,
    load_labels_config,
    load_splits_config,
)
from precursorintelligence.common.io.metadata_store import SqlAlchemyMetadataStore
from precursorintelligence.common.io.object_store import LocalObjectStore
from precursorintelligence.training.contracts import FRAMES
from precursorintelligence.training.data import read_frame, run_prepare

sys.path.insert(0, str(Path(__file__).parents[1] / "ml"))  # the synthetic gold lives in tests/ml
from make_ml_gold import make_ml_gold, ml_config  # noqa: E402


def _prepare(tmp_path, name, seed):
    g = make_ml_gold(tmp_path / name, seed=seed)
    store = LocalObjectStore(g.root, immutable=None)
    meta = SqlAlchemyMetadataStore(f"sqlite:///{(g.root / 'ops' / 'metadata.db').as_posix()}")
    meta.ensure_schema()
    res = run_prepare(
        store,
        meta,
        ml_run_id="manual__folds",
        cfg=ml_config(),
        labels_cfg=load_labels_config(),
        splits_cfg=load_splits_config(),
        registry=load_feature_registry(),
    )
    return pl.concat(
        [read_frame(store, res.paths, n).select("ccn", "fold") for n in FRAMES], how="vertical"
    )


def test_each_facility_is_in_exactly_one_fold_across_frames_and_runs(tmp_path):
    first = _prepare(tmp_path, "a", seed=1)
    assert first.group_by("ccn").agg(pl.col("fold").n_unique())["fold"].max() == 1
    assert set(first["fold"].unique().to_list()) == {0, 1, 2, 3, 4}
    # another gold (other rows and labels, same facilities): every facility keeps its fold
    second = _prepare(tmp_path, "b", seed=2)
    joined = first.unique().join(second.unique(), on="ccn", suffix="_b")
    assert joined.height == first["ccn"].n_unique()
    assert (joined["fold"] == joined["fold_b"]).all()
