"""evaluation/mitigation.py: the reweighting trade-off on OOF predictions (docs/ML_PIPELINE_SPEC.md §7, leakage
tests 1 and 9)."""

import numpy as np
import polars as pl
import pytest

from precursorintelligence.common.config import load_feature_registry
from precursorintelligence.evaluation.mitigation import eval_weights, tradeoff, weighted_oof
from precursorintelligence.evaluation.slices import slice_config
from precursorintelligence.training.contracts import RunPaths
from precursorintelligence.training.data import read_frame, read_input_manifest
from precursorintelligence.training.train import resolve_features


def setup(run):
    store, p = run.store(), RunPaths.of(run.cfg, run.run_id)
    features = resolve_features(load_feature_registry(), run.cfg, read_input_manifest(store, p))
    frame = read_frame(store, p, "eval_train")
    oof = pl.read_parquet(run.gold.root / run.train.result.oof["eval"])
    chosen = run.train.result.selected
    p_none = (
        frame.select("ccn", "t")
        .join(oof, on=["ccn", "t"], how="left", maintain_order="left")[f"p_{chosen}"]
        .to_numpy()
    )
    return frame, features, chosen, p_none


def test_weights_cover_every_row_and_favour_the_rarer_groups(trained):
    frame, *_ = setup(trained)
    w = eval_weights(frame, slice_config(trained.cfg), load_feature_registry())
    assert len(w) == frame.height and not np.isnan(w).any()
    assert w.mean() == pytest.approx(1.0)
    owner = frame["ownership_group"].to_numpy()
    groups, counts = np.unique(owner, return_counts=True)
    rare, common = groups[counts.argmin()], groups[counts.argmax()]
    assert w[owner == rare].mean() > w[owner == common].mean()


def test_unit_weights_reproduce_trains_oof_predictions(trained):
    # the reweighted variant is built exactly like train's OOF: only the weights differ
    frame, features, chosen, p_none = setup(trained)
    p = weighted_oof(frame, features, trained.cfg, chosen, np.ones(frame.height))
    assert np.allclose(p, p_none, atol=1e-9)


def test_the_tradeoff_compares_reweighting_with_none(trained):
    frame, features, chosen, p_none = setup(trained)
    t = tradeoff(frame, features, trained.cfg, chosen, p_none, bias_cfg=slice_config(trained.cfg))
    assert [v.variant for v in t.variants] == ["none", "reweight"]
    oof_pr = {r.candidate: r.oof_pr_auc for r in trained.train.selection.candidates}[chosen]
    assert t.variant("none").pr_auc == pytest.approx(oof_pr, abs=1e-6)
    assert t.variant("reweight").pr_auc != t.variant("none").pr_auc
    assert t.weight_slice == "ownership_group" and 0 < t.effective_sample_size <= frame.height
    assert {"slice", "value", "oe_none", "oe_reweight"} <= set(t.by_value[0])


def test_the_report_and_card_carry_it(trained):
    from precursorintelligence.evaluation.evaluate import run_evaluate

    out = run_evaluate(
        trained.store(), ml_run_id=trained.run_id, cfg=trained.cfg, shuffled_label=False
    )
    assert out.report.mitigation is not None
    skipped = run_evaluate(
        trained.store(),
        ml_run_id=trained.run_id,
        cfg=trained.cfg,
        shuffled_label=False,
        mitigation=False,
    )
    assert skipped.report.mitigation is None
