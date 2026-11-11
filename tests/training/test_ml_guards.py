"""The guards `prepare` keeps around gold and its own run folder (docs/ML_PIPELINE_SPEC.md §1.2, §1.5, §2, §9.2 of the
data spec): decide's gold run, gold built with other settings, label contract, files that drift from the manifest."""

import json

import polars as pl
import pytest
from make_ml_gold import build_frame, make_ml_gold, ml_config

from precursorintelligence.common.config import load_feature_registry
from precursorintelligence.common.io.object_store import LocalObjectStore
from precursorintelligence.training import GoldPinError, MlInputError
from precursorintelligence.training.contracts import Decision, RunPaths
from precursorintelligence.training.data import (
    feature_list,
    leakage_flags,
    read_frame,
    read_windows,
    to_model_frame,
)

PATHS = RunPaths("", "ml/runs", "manual__t")


def _decide(store, **fields):
    decision = Decision(decision="RETRAIN", proceed=True, reason="test", mode="auto", **fields)
    store.put_bytes(PATHS.decision, decision.model_dump_json().encode(), overwrite=True)


# ------------------------------------------------------------------ decide's gold run (leakage test 12)


def test_prepare_pins_the_gold_run_decide_recorded(gold, store, meta, prepare):
    _decide(store, gold_run_id=gold.run_id)
    assert prepare(store, meta).manifest.gold_run_id == gold.run_id


def test_a_decision_for_another_gold_run_is_refused_even_without_the_flag(store, meta, prepare):
    _decide(store, gold_run_id="20261101T000000Z-older")
    with pytest.raises(GoldPinError, match="not run 20261101T000000Z-older"):
        prepare(store, meta)
    assert not store.exists(PATHS.input_manifest)


def test_a_flag_that_disagrees_with_the_decision_is_refused(gold, store, meta, prepare):
    _decide(store, gold_run_id=gold.run_id)
    with pytest.raises(MlInputError, match="disagrees with"):
        prepare(store, meta, gold_run_id="20261101T000000Z-older")


def test_a_decision_without_a_gold_run_is_refused(store, meta, prepare):
    _decide(store)  # e.g. a SKIP
    with pytest.raises(MlInputError, match="names no gold run"):
        prepare(store, meta)


# ------------------------------------------------------------------ gold built with other settings


def _edit(path, **changes):
    data = json.loads(path.read_text())
    data.update(changes)
    path.write_text(json.dumps(data))


@pytest.mark.parametrize(
    ("file", "changes", "message"),
    [
        (
            "dataset_manifest.json",
            {"config_sha256": {"labels.yaml": "0" * 64}},
            "other labels.yaml",
        ),
        ("dataset_manifest.json", {"horizons_months": {"g12": 6}}, "horizon"),
        ("dataset_manifest.json", {"buffers_months": {"citations": 3}}, "buffer"),
        ("dataset_manifest.json", {"spine_start": "2023-01"}, "spine_start"),
        ("dataset_manifest.json", {"v_labels": "2026-09"}, "v_labels"),
        ("_current.json", {"v_labels": "2026-11"}, "after v"),
    ],
)
def test_gold_built_with_other_settings_is_refused(gold, meta, prepare, file, changes, message):
    path = gold.root / "gold" / file
    _edit(path, **changes)
    if (
        file == "dataset_manifest.json"
    ):  # keep the pointer and the manifest agreeing on the merged file
        assert json.loads(path.read_text())["merged_file"]["sha256"] == gold.merged_sha256
    with pytest.raises(MlInputError, match=message):
        prepare(LocalObjectStore(gold.root, immutable=None), meta)


# ------------------------------------------------------------------ label contract


def _gold_with(tmp_path, change, name="g"):
    frame = change(build_frame(80, "2026-10", "2026-10", seed=4))
    return make_ml_gold(tmp_path / name, frame=frame)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (
            lambda df: df.with_columns(
                pl.when(pl.col("status_g12") == "censored")
                .then(0)
                .otherwise(pl.col("y_g12"))
                .cast(pl.Int8)
                .alias("y_g12")
            ),
            "disagrees with their status",  # a censored row must carry no label
        ),
        (
            lambda df: df.with_columns(
                pl.when(pl.col("status_g12") == "censored")
                .then(pl.lit("lost"))
                .otherwise(pl.col("status_g12"))
                .alias("status_g12")
            ),
            "outside",
        ),
        (lambda df: df.with_columns(pl.col("y_g12").cast(pl.Int64)), "y_g12 is Int64"),
        (
            lambda df: df.with_columns(
                pl.when((pl.col("t") == "2026-10") & (pl.col("ccn") == "T00001"))
                .then(pl.lit("negative"))
                .otherwise(pl.col("status_g12"))
                .alias("status_g12"),
                pl.when((pl.col("t") == "2026-10") & (pl.col("ccn") == "T00001"))
                .then(pl.lit(0, dtype=pl.Int8))
                .otherwise(pl.col("y_g12"))
                .alias("y_g12"),
            ),
            "after L = 2025-02",  # a labelled row whose outcome window isn't observed yet
        ),
    ],
)
def test_broken_label_columns_are_refused(tmp_path, meta, prepare, change, message):
    g = _gold_with(tmp_path, change)
    with pytest.raises(MlInputError, match=message):
        prepare(LocalObjectStore(g.root, immutable=None), meta)


# ------------------------------------------------------------------ files that drift from the manifest


def test_a_frame_rewritten_after_its_manifest_is_refused(store, meta, prepare):
    res = prepare(store, meta)
    read_frame(store, res.paths, "test")  # fine as written
    store.put_bytes(
        res.paths.frame("test"), b"another frame", overwrite=True
    )  # a rerun that died half-way
    with pytest.raises(MlInputError, match="does not match the run's input/manifest.json"):
        read_frame(store, res.paths, "test")


def test_windows_rewritten_after_the_manifest_are_refused(store, meta, prepare):
    res = prepare(store, meta)
    read_windows(store, res.paths)
    store.put_bytes(res.paths.windows, b"{}", overwrite=True)
    with pytest.raises(MlInputError, match="does not match"):
        read_windows(store, res.paths)


# ------------------------------------------------------------------ model input


def test_the_model_frame_refuses_a_forbidden_column():
    registry = load_feature_registry()
    forbidden = [f for f in registry.features if f.name in ("overall_rating", "chain_id")]
    df = build_frame(5, "2026-10", "2026-10", seed=1)
    with pytest.raises(MlInputError, match="never a model input"):
        to_model_frame(df, forbidden)


# ------------------------------------------------------------------ the leakage flag's categorical score


def test_a_categorical_unique_to_each_row_is_not_flagged(store, meta, prepare):
    """Out-of-fold level rates: a level seen only on its own row scores the prior, so AP is about the base rate.
    An in-fold rate would score each row by its own label (AP = 1) and flag it."""
    res = prepare(store, meta)
    frames = {name: read_frame(store, res.paths, name) for name in ("eval_train", "refit_train")}
    unique = {
        name: f.with_columns((pl.col("ccn") + pl.col("t")).alias("provider_type"))
        for name, f in frames.items()
    }
    features = feature_list(load_feature_registry(), ml_config())
    assert [f.feature for f in leakage_flags(unique, features, "g12", ml_config())] == []
