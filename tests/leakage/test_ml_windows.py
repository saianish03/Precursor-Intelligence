"""Leakage test 2 (docs/ML_PIPELINE_SPEC.md §17.1): the ML windows never let a training outcome reach the test window.

For g12, g6 and pen12 and every V_L from 2024-06 to 2027-12 (CMS gap months included as V_L and inside windows):
the last eval month plus H + B is before the first test month, the refit window ends at L, every window starts at
or after spine_start, and the training frames of a prepared run hold only positive or negative rows.
"""

import sys
from pathlib import Path

import pytest

from precursorintelligence.common.config import load_labels_config, load_splits_config
from precursorintelligence.gold.splits import shift, split_windows
from precursorintelligence.training import MlInputError
from precursorintelligence.training.data import build_frames, compute_windows

sys.path.insert(0, str(Path(__file__).parents[1] / "ml"))  # the synthetic gold lives in tests/ml
from make_ml_gold import build_frame, ml_config, published_months  # noqa: E402

LABELS, SPLITS = load_labels_config(), load_splits_config()


def _v_labels():
    m, out = "2024-06", []
    while m <= "2027-12":
        out.append(m)
        m = shift(m, 1)
    return out


@pytest.mark.parametrize("label", ["g12", "g6", "pen12"])
@pytest.mark.parametrize("eval_gap", ["production_lag", "purge"])
def test_no_training_month_reaches_the_test_window(label, eval_gap):
    cfg = ml_config()
    cfg = cfg.model_copy(update={"windows": cfg.windows.model_copy(update={"eval_gap": eval_gap})})
    h, b = LABELS.labels[label].horizon_months, LABELS.buffer(label)
    built = 0
    for v_labels in _v_labels():
        months = published_months(SPLITS.spine_start, v_labels)  # V_L itself may be a gap month
        months = months if v_labels in months else [*months, v_labels]
        test_from, test_to = split_windows(v_labels, h, b, SPLITS)["test"]
        test = [m for m in months if test_from <= m <= test_to]
        try:
            w = compute_windows(
                months,
                test,
                label=label,
                v=v_labels,
                v_labels=v_labels,
                labels_cfg=LABELS,
                splits_cfg=SPLITS,
                cfg=cfg,
            )
        except (
            MlInputError
        ) as exc:  # early V_L: too few months before the gap, or an empty test window
            assert "min_train_months" in str(exc) or "holds no rows" in str(exc)
            continue
        built += 1
        gap = h + b if eval_gap == "production_lag" else h
        assert shift(w.eval.last, gap) < w.test.first, (v_labels, w.eval.last, w.test.first)
        assert max(w.eval.months) < min(w.test.months)
        assert w.refit.last <= w.last_labeled_t == shift(v_labels, -(h + b))
        assert min(w.eval.months + w.refit.months) >= SPLITS.spine_start
        assert len(w.eval.months) <= cfg.windows.train_months >= len(w.refit.months)
    assert built > 10  # the loop really checked windows


@pytest.mark.parametrize("label", ["g12", "g6", "pen12"])
def test_a_stale_refresh_takes_every_window_from_v_labels(label):
    """V is a stale refresh (no new labels): L and every window come from V_L, the score month is V."""
    h, b = LABELS.labels[label].horizon_months, LABELS.buffer(label)
    built = 0
    for v_labels in _v_labels():
        months = published_months(SPLITS.spine_start, shift(v_labels, 2))
        later = [m for m in months if m > v_labels]
        if v_labels not in months or not later:
            continue
        v = later[0]  # the next month with rows after V_L
        test_from, test_to = split_windows(v_labels, h, b, SPLITS)["test"]
        test = [m for m in months if test_from <= m <= test_to and m <= v]
        try:
            w = compute_windows(
                [m for m in months if m <= v],
                test,
                label=label,
                v=v,
                v_labels=v_labels,
                labels_cfg=LABELS,
                splits_cfg=SPLITS,
                cfg=ml_config(),
            )
        except MlInputError as exc:
            assert "min_train_months" in str(exc) or "holds no rows" in str(exc)
            continue
        built += 1
        assert w.last_labeled_t == shift(v_labels, -(h + b))  # not from V
        assert w.refit.last <= w.last_labeled_t and shift(w.eval.last, h + b) < w.test.first
        assert (w.score_month, w.v, w.v_labels) == (v, v, v_labels)
    assert built > 10


def test_training_frames_hold_only_labelled_rows():
    df = build_frame(150, "2026-10", "2026-10", seed=11)
    months = df["t"].unique().sort().to_list()
    test = df.filter(df["split_g12"] == "test")["t"].unique().to_list()
    w = compute_windows(
        months,
        test,
        label="g12",
        v="2026-10",
        v_labels="2026-10",
        labels_cfg=LABELS,
        splits_cfg=SPLITS,
        cfg=ml_config(),
    )
    frames = build_frames(df, w, 5)
    for name in ("eval_train", "test", "refit_train"):
        assert set(frames[name]["status_g12"].unique().to_list()) <= {"positive", "negative"}
        assert frames[name]["y_g12"].null_count() == 0
    assert frames["score"]["status_g12"].unique().to_list() == ["immature"]  # never trained on
