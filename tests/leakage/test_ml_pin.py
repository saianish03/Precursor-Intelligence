"""Leakage test 12 (docs/ML_PIPELINE_SPEC.md §17.1, §1.2): the run pins exactly the gold `decide` evaluated, or
nothing. Every refusal leaves the run folder empty."""

import json
import sys
from pathlib import Path

import pytest

from precursorintelligence.common.io.object_store import LocalObjectStore
from precursorintelligence.training import GoldPinError
from precursorintelligence.training.contracts import RunPaths
from precursorintelligence.training.data import pin_gold, sha256_hex

sys.path.insert(0, str(Path(__file__).parents[1] / "ml"))  # the synthetic gold lives in tests/ml
from make_ml_gold import make_ml_gold  # noqa: E402

PATHS = RunPaths("", "ml/runs", "manual__pin")


@pytest.fixture
def gold(tmp_path):
    return make_ml_gold(tmp_path / "store", n_facilities=40)


def _store(gold):
    return LocalObjectStore(gold.root, immutable=None)


def _nothing_pinned(store):
    return store.list(PATHS.run_dir) == []


def test_the_pinned_run_is_the_one_decide_evaluated(gold):
    store = _store(gold)
    pinned = pin_gold(store, PATHS, gold.run_id)
    assert pinned.current["run_id"] == gold.run_id
    copy = store.open_read(PATHS.input_file("facility_month_dataset.parquet")).read()
    assert sha256_hex(copy) == gold.merged_sha256 == pinned.files["facility_month_dataset.parquet"]
    assert (
        json.loads(store.open_read(PATHS.input_file("_current.json")).read())["run_id"]
        == gold.run_id
    )


def test_another_gold_run_is_refused(gold):
    store = _store(gold)
    with pytest.raises(GoldPinError, match="not run 20261111T999999Z-other that decide evaluated"):
        pin_gold(store, PATHS, "20261111T999999Z-other")
    assert _nothing_pinned(store)


def _edit_json(path, **changes):
    data = json.loads(path.read_text())
    data.update(changes)
    path.write_text(json.dumps(data))


@pytest.mark.parametrize(
    ("break_it", "message"),
    [
        (lambda g: _edit_json(g.root / "gold/_current.json", state="staging"), "not 'current'"),
        (
            lambda g: _edit_json(
                g.root / "gold/dataset_manifest.json", merged_file={"sha256": "0" * 64}
            ),
            "disagree",
        ),
        (
            lambda g: (g.root / "gold/facility_month_dataset.parquet").write_bytes(b"not parquet"),
            "does not match",
        ),
        (lambda g: (g.root / "gold/_current.json").unlink(), "missing"),
    ],
)
def test_a_gold_that_is_not_one_published_run_is_refused(gold, break_it, message):
    break_it(gold)
    store = _store(gold)
    with pytest.raises(GoldPinError, match=message):
        pin_gold(store, PATHS, None)
    assert _nothing_pinned(store)


def test_a_promotion_during_the_copy_is_refused(gold, monkeypatch):
    store = _store(gold)
    real_open = store.open_read
    reads = {"current": 0}

    def open_read(key):
        if key.endswith("gold/_current.json"):
            reads["current"] += 1
            if reads["current"] == 2:  # the re-read after the copy sees a newer run
                _edit_json(gold.root / "gold/_current.json", run_id="20261112T000000Z-newer")
        return real_open(key)

    monkeypatch.setattr(store, "open_read", open_read)
    with pytest.raises(GoldPinError, match="changed while gold was read"):
        pin_gold(store, PATHS, gold.run_id)
    assert _nothing_pinned(store)


def test_prepare_reads_gold_once_and_builds_from_the_pinned_bytes(gold, monkeypatch):
    from make_ml_gold import ml_config

    from precursorintelligence.common.config import (
        load_feature_registry,
        load_labels_config,
        load_splits_config,
    )
    from precursorintelligence.common.io.metadata_store import SqlAlchemyMetadataStore
    from precursorintelligence.training.data import read_frame, run_prepare

    store = _store(gold)
    real_open = store.open_read
    gold_reads = []

    def open_read(key):
        if key.startswith("gold/"):
            gold_reads.append(key)
        return real_open(key)

    monkeypatch.setattr(store, "open_read", open_read)
    meta = SqlAlchemyMetadataStore(f"sqlite:///{(gold.root / 'ops' / 'metadata.db').as_posix()}")
    meta.ensure_schema()
    res = run_prepare(
        store,
        meta,
        ml_run_id=PATHS.ml_run_id,
        cfg=ml_config(min_score_facilities=10),
        labels_cfg=load_labels_config(),
        splits_cfg=load_splits_config(),
        registry=load_feature_registry(),
    )
    assert gold_reads.count("gold/facility_month_dataset.parquet") == 1
    assert set(gold_reads) <= {
        "gold/_current.json",
        "gold/dataset_manifest.json",
        "gold/facility_month_dataset.parquet",
    }  # never dim_facility_latest or a partition
    (gold.root / "gold/facility_month_dataset.parquet").write_bytes(b"replaced after the pin")
    assert read_frame(store, res.paths, "score").height == res.manifest.rows["score"]
