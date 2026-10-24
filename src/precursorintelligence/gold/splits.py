"""Step 8 splits (spec §9.3, §9.5): for each label, which months train, calibrate, are purged, test, or are
only scored.

Every window is in calendar months (a never-published month still counts) and ends at
``last_labeled_t = V - (H + B)``. V is the label vintage: the last vintage that is not a stale refresh, as for
maturity (spec §9.2), so the ``score`` rows are exactly the ``immature`` ones.

- ``test``: the last ``test_months`` months up to last_labeled_t;
- ``purge``: the H months before test (label windows overlap, so no training row's outcome reaches test);
- ``calib``: the ``calib_months`` months before purge, held out from fitting;
- ``train``: from ``spine_start`` up to calib (empty when calib already starts at or before spine_start);
- ``score``: every month after last_labeled_t.

Windows that reach back before ``spine_start`` simply hold no rows. Recomputed from V on every run.
"""

from __future__ import annotations

from typing import Any, cast

import polars as pl

from ..common.config import LabelsConfig, SplitsConfig

SPLITS = ("train", "calib", "purge", "test", "score")


def shift(month: str, months: int) -> str:
    """``YYYY-MM`` plus ``months`` calendar months (negative goes back)."""
    y, m = map(int, month.split("-"))
    n = y * 12 + (m - 1) + months
    return f"{n // 12:04d}-{n % 12 + 1:02d}"


def last_labeled_t(v: str, horizon: int, buffer: int) -> str:
    """Spec §9.3: the last t whose window and buffer are fully observed at V."""
    return shift(v, -(horizon + buffer))


def split_windows(v: str, horizon: int, buffer: int, cfg: SplitsConfig) -> dict[str, Any]:
    """The calendar months of each split for one label, as inclusive ``[first, last]`` ranges.

    ``train`` is None when it would be empty; ``score`` runs from the month after last_labeled_t onward.
    """
    last = last_labeled_t(v, horizon, buffer)
    test_from = shift(last, -(cfg.test_months - 1))
    purge_from = shift(test_from, -horizon)
    calib_from = shift(purge_from, -cfg.calib_months)
    train_to = shift(calib_from, -1)
    return {
        "last_labeled_t": last,
        "train": [cfg.spine_start, train_to] if train_to >= cfg.spine_start else None,
        "calib": [calib_from, shift(purge_from, -1)] if cfg.calib_months else None,
        "purge": [purge_from, shift(test_from, -1)],
        "test": [test_from, last],
        "score": [shift(last, 1), None],
    }


def label_windows(
    v: str, labels_cfg: LabelsConfig, splits_cfg: SplitsConfig
) -> dict[str, dict[str, Any]]:
    """``split_windows`` for every configured label."""
    return {
        name: split_windows(v, spec.horizon_months, labels_cfg.buffer(name), splits_cfg)
        for name, spec in labels_cfg.labels.items()
    }


def _split_expr(windows: dict[str, Any]) -> pl.Expr:
    t = pl.col("t")
    branches = [(t > windows["last_labeled_t"], "score")]
    for split in ("test", "purge", "calib", "train"):
        if windows[split] is not None:
            first, last = windows[split]
            branches.append((t.is_between(pl.lit(first), pl.lit(last)), split))
    expr: Any = pl  # pl.when(...) starts the chain, .when(...) continues it
    for condition, split in branches:
        expr = expr.when(condition).then(pl.lit(split))
    return cast(pl.Expr, expr.otherwise(pl.lit(None, dtype=pl.String)))


def add_splits(
    labels: pl.DataFrame, v: str, labels_cfg: LabelsConfig, splits_cfg: SplitsConfig
) -> pl.DataFrame:
    """``labels`` plus ``split_<label>`` for every configured label (values in ``SPLITS``)."""
    windows = label_windows(v, labels_cfg, splits_cfg)
    return labels.with_columns(_split_expr(w).alias(f"split_{name}") for name, w in windows.items())
