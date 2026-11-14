"""Leakage tests of `ml train` (docs/ML_PIPELINE_SPEC.md §17.1, Leakage (1) and (4)).

(1) Changing the test rows (their labels, their base rate, a feature) leaves the eval-fit models, their calibration,
    the selection, the eval OOF predictions and the tuned parameters byte-identical: nothing in training,
    calibration, selection or tuning reads the test frame. A canary shows the comparison does fail when training reads test rows. (The refit trains on the
    test months by design, §3.3, and is never evaluated on them, so it is not part of this check.)
(4) Every OOF prediction comes from a fit that never saw that facility, on both windows and for both members.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "ml"))
from make_ml_gold import RUN_ID, build_frame, make_ml_gold, ml_config  # noqa: E402

from precursorintelligence.common.config import (  # noqa: E402
    load_feature_registry,
    load_labels_config,
    load_splits_config,
)
from precursorintelligence.common.io.metadata_store import SqlAlchemyMetadataStore  # noqa: E402
from precursorintelligence.common.io.object_store import LocalObjectStore  # noqa: E402
from precursorintelligence.training import train as train_mod  # noqa: E402
from precursorintelligence.training.contracts import CANDIDATES, RunPaths  # noqa: E402
from precursorintelligence.training.data import read_frame, run_prepare, to_model_frame  # noqa: E402
from precursorintelligence.training.train import fit_window, run_train  # noqa: E402
from precursorintelligence.training.tune import run_tune  # noqa: E402

INPUT_PARQUET = "facility_month_dataset.parquet"  # the pinned gold copy (prepare reads it, train and tune never)


def small_config():
    cfg = ml_config()
    lgbm = cfg.members.lgbm.model_copy(
        update={"n_estimators": 40, "min_child_samples": 20, "num_threads": 1}
    )
    tuning = cfg.tuning.model_copy(
        update={"timeout_minutes": 5.0, "early_stopping_rounds": 10, "max_rounds": 30}
    )
    return cfg.model_copy(
        update={"members": cfg.members.model_copy(update={"lgbm": lgbm}), "tuning": tuning}
    )


def _proposal_params(store, paths) -> bytes:
    """``tuning_proposal.yaml`` up to its ``source`` block (run ids and the dataset version, which name the input
    rather than anything tuning chose): the parameters, the CV PR-AUC, trials, folds and seed."""
    data = _bytes(store, paths.tuning_proposal)
    head, sep, _ = data.partition(b"\nsource:")
    assert sep, "the proposal has no source block"
    return head


@pytest.fixture
def gold(tmp_path):
    return make_ml_gold(tmp_path / "store")


@pytest.fixture
def store(gold):
    return LocalObjectStore(gold.root, immutable=None)


@pytest.fixture
def meta(gold):
    m = SqlAlchemyMetadataStore(f"sqlite:///{(gold.root / 'ops' / 'metadata.db').as_posix()}")
    m.ensure_schema()
    return m


@pytest.fixture
def prepare():
    def _prepare(store, meta, run_id, cfg):
        return run_prepare(
            store,
            meta,
            ml_run_id=run_id,
            cfg=cfg,
            labels_cfg=load_labels_config(),
            splits_cfg=load_splits_config(),
            registry=load_feature_registry(),
        )

    return _prepare


def _train(root: Path, frame: pl.DataFrame, cfg):
    gold = make_ml_gold(root, frame=frame)
    assert gold.run_id == RUN_ID
    store = LocalObjectStore(gold.root, immutable=None)
    meta = SqlAlchemyMetadataStore(f"sqlite:///{(gold.root / 'ops' / 'metadata.db').as_posix()}")
    meta.ensure_schema()
    run_prepare(
        store,
        meta,
        ml_run_id="leak",
        cfg=cfg,
        labels_cfg=load_labels_config(),
        splits_cfg=load_splits_config(),
        registry=load_feature_registry(),
    )
    out = run_train(
        store, ml_run_id="leak", cfg=cfg, registry=load_feature_registry(), recipe_hash="leak"
    )
    run_tune(store, ml_run_id="leak", cfg=cfg, registry=load_feature_registry(), n_trials=2)
    return store, out


def _bytes(store, key):
    with store.open_read(key) as f:
        return f.read()


def _alter_test_rows(frame: pl.DataFrame, how: str) -> pl.DataFrame:
    """The same gold with the labelled test rows changed: ``permute`` shuffles their labels (base rate kept);
    ``flip`` makes ~80 % of them positive (another base rate) and shifts a feature on every test row."""
    test = pl.col("split_g12") == "test"
    labelled = frame.filter(test & pl.col("status_g12").is_in(["positive", "negative"]))
    assert labelled.height > 10 and labelled["y_g12"].n_unique() == 2
    rng = np.random.default_rng(1)
    if how == "permute":
        y = labelled["y_g12"].to_numpy()[rng.permutation(labelled.height)]
    else:
        y = (rng.random(labelled.height) < 0.8).astype("int8")
        y[:2] = [0, 1]  # both classes stay
    new = labelled.select("ccn", "t").with_columns(
        pl.Series("y_new", y, dtype=pl.Int8),
        pl.Series("status_new", np.where(y == 1, "positive", "negative")),
    )
    out = (
        frame.join(new, on=["ccn", "t"], how="left")
        .with_columns(
            pl.coalesce("y_new", "y_g12").alias("y_g12"),
            pl.coalesce("status_new", "status_g12").alias("status_g12"),
        )
        .drop("y_new", "status_new")
    )
    if how == "flip":
        hprd = pl.col("hprd_total")
        out = out.with_columns(pl.when(test).then(hprd + 50.0).otherwise(hprd).alias("hprd_total"))
    out = out.select(frame.columns)
    assert not out.filter(test)["y_g12"].equals(frame.filter(test)["y_g12"])
    return out


@pytest.mark.parametrize("how", ["permute", "flip"])
def test_changing_the_test_rows_changes_nothing_the_eval_fit_chose(tmp_path, monkeypatch, how):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    cfg = small_config()
    frame = build_frame(300, "2026-10", "2026-10", seed=7)
    store_a, a = _train(tmp_path / "a", frame, cfg)
    store_b, b = _train(tmp_path / "b", _alter_test_rows(frame, how), cfg)
    paths = RunPaths.of(cfg, "leak")
    for c in CANDIDATES:
        assert a.eval_models[c].to_files() == b.eval_models[c].to_files(), c
        assert a.result.calibration[f"eval/{c}"] == b.result.calibration[f"eval/{c}"]
    assert a.selection == b.selection
    assert _bytes(store_a, paths.selection) == _bytes(store_b, paths.selection)
    assert _bytes(store_a, paths.oof("eval")) == _bytes(store_b, paths.oof("eval"))
    assert _proposal_params(store_a, paths) == _proposal_params(store_b, paths)


def test_the_comparison_catches_a_leak(tmp_path, monkeypatch):
    """Canary for leakage test (1): a train that did read the test rows makes the same comparison differ."""
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    cfg = small_config()
    frame = build_frame(300, "2026-10", "2026-10", seed=7)
    real = train_mod.read_frame

    def leaky(store_, paths_, name):
        got = real(store_, paths_, name)
        if name == "eval_train":
            return pl.concat([got, real(store_, paths_, "test").select(got.columns)])
        return got

    monkeypatch.setattr(train_mod, "read_frame", leaky)
    _, a = _train(tmp_path / "a", frame, cfg)
    _, b = _train(tmp_path / "b", _alter_test_rows(frame, "flip"), cfg)
    assert a.eval_models["ensemble"].to_files() != b.eval_models["ensemble"].to_files()


def _row_hashes(x: pd.DataFrame) -> np.ndarray:
    return pd.util.hash_pandas_object(x.reset_index(drop=True), index=False).to_numpy()


@pytest.mark.parametrize("window", ["eval_train", "refit_train"])
def test_oof_predictions_come_from_fits_without_that_facility(
    store, meta, prepare, monkeypatch, window
):
    cfg = small_config()
    prepare(store, meta, run_id="oof", cfg=cfg)
    paths = RunPaths.of(cfg, "oof")
    frame = read_frame(store, paths, window)
    features = train_mod.resolve_features(
        load_feature_registry(), cfg, train_mod.read_input_manifest(store, paths)
    )
    x_all = to_model_frame(frame, features)
    hashes = _row_hashes(x_all)
    assert len(set(hashes)) == len(hashes)  # rows are identifiable by content
    calls: list[tuple[str, set[int]]] = []
    real_fit = train_mod._fit_member

    def recording(member, x, y, feats, cfg_, levels=None):
        calls.append((member, set(_row_hashes(x))))
        return real_fit(member, x, y, feats, cfg_, levels)

    monkeypatch.setattr(train_mod, "_fit_member", recording)
    wf = fit_window("eval", frame, features, cfg, ["lgbm", "lr"], sorted(frame["t"].unique()))
    folds, ccn = frame["fold"].to_numpy(), frame["ccn"].to_numpy()
    ks = np.unique(folds)
    by_hash = dict(zip(hashes, ccn, strict=True))
    for i, member in enumerate(("lgbm", "lr")):
        mine = calls[i * (len(ks) + 1) : (i + 1) * (len(ks) + 1)]
        assert [m for m, _ in mine] == [member] * (len(ks) + 1)
        assert mine[0][1] == set(hashes)  # the full fit
        for k, (_, rows) in zip(ks, mine[1:], strict=True):
            assert rows == set(hashes[folds != k])  # exactly the other folds' rows
            assert {by_hash[h] for h in rows}.isdisjoint(set(ccn[folds == k]))
    for k in ks:  # every OOF value is what a fit on the other facilities gives
        held = folds == k
        for member in ("lgbm", "lr"):
            fitted = real_fit(
                member, x_all[~held].reset_index(drop=True), wf.y[~held], features, cfg
            )
            got = fitted.logit(x_all[held].reset_index(drop=True))
            assert np.allclose(got, wf.oof[member][held]), (member, k)


@pytest.mark.parametrize("status", ["censored", "immature"])
def test_a_training_frame_with_unlabelled_rows_is_refused(store, meta, prepare, status):
    cfg = small_config()
    prepare(store, meta, run_id="st", cfg=cfg)
    frame = read_frame(store, RunPaths.of(cfg, "st"), "eval_train")
    bad = frame.with_columns(
        pl.when(pl.int_range(pl.len()) == 0)
        .then(pl.lit(status))
        .otherwise(pl.col("status_g12"))
        .alias("status_g12")
    )
    features = train_mod.resolve_features(
        load_feature_registry(), cfg, train_mod.read_input_manifest(store, RunPaths.of(cfg, "st"))
    )
    with pytest.raises(train_mod.MlInputError, match="immature or censored"):
        fit_window("eval", bad, features, cfg, ["lr"], ["2022-05"])


def test_train_never_reads_the_test_frame(store, meta, prepare, monkeypatch):
    """Nothing train calls (its own code, selection, the bias check, MLflow logging) opens the test frame: the
    store itself is watched."""
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    cfg = small_config()
    prepare(store, meta, run_id="nt", cfg=cfg)
    paths = RunPaths.of(cfg, "nt")
    opened: list[str] = []
    real_open = store.open_read

    def spy(key):
        opened.append(key)
        return real_open(key)

    monkeypatch.setattr(store, "open_read", spy)
    run_train(store, ml_run_id="nt", cfg=cfg, registry=load_feature_registry(), recipe_hash="t")
    assert paths.frame("test") not in opened
    assert paths.input_file(INPUT_PARQUET) not in opened  # the frames are the only rows train reads
    assert paths.frame("eval_train") in opened and paths.frame("refit_train") in opened


def test_tune_never_reads_the_test_frame(store, meta, prepare, monkeypatch):
    """The store-level guard for tuning: it opens eval_train only, never the test frame or the pinned input (a
    future early-stopping change that read frames/test.parquet would fail here)."""
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    cfg = small_config()
    prepare(store, meta, run_id="ntt", cfg=cfg)
    paths = RunPaths.of(cfg, "ntt")
    opened: list[str] = []
    real_open = store.open_read

    def spy(key):
        opened.append(key)
        return real_open(key)

    monkeypatch.setattr(store, "open_read", spy)
    run_tune(store, ml_run_id="ntt", cfg=cfg, registry=load_feature_registry(), n_trials=1)
    frames = {k for k in opened if k.startswith(f"{paths.run_dir}frames/")}
    assert frames == {paths.frame("eval_train")}
    assert paths.input_file(INPUT_PARQUET) not in opened
