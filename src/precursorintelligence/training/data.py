"""Task ``prepare`` and the data rules every ML package shares (docs/ML_PIPELINE_SPEC.md §1-§3).

- ``pin_gold``: copy the published gold into the run folder, hash-checked (§1.2). Later tasks read only that copy.
- ``feature_list``: the registry's ``model`` features minus ``ml.yaml: features.exclude``, in registry order (§1.4).
- ``compute_windows``: the eval, test and refit windows (§3.2), counted in months that hold rows.
- ``build_frames``: the five frames; ``add_folds``: the facility folds (§3.4).
- ``to_model_frame``: the one function that turns rows into model input (§1.4).
- ``check_inputs``, ``leakage_flags``, ``regime_flags``: the input checks of §1.5 and §1.4.
- ``run_prepare``: all of the above for one run, writing the frames, ``windows.json`` and ``input/manifest.json``.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from ..common.config import (
    ConfigError,
    FeatureRegistry,
    FeatureSpec,
    LabelsConfig,
    MlConfig,
    SplitsConfig,
)
from ..common.io.metadata_store import MetadataStore
from ..common.io.object_store import ObjectStore, sha256_of
from ..gold.dataset import MANIFEST_NAME, MERGED, settings_sha256
from ..gold.splits import last_labeled_t, shift, split_windows
from . import GoldPinError, MlInputError
from .contracts import (
    FRAMES,
    Decision,
    Frame,
    InputManifest,
    LeakageFlag,
    MonthRange,
    RegimeFlag,
    RunPaths,
    Windows,
)

log = logging.getLogger(__name__)

CURRENT_NAME = "_current.json"
LABELLED = ("positive", "negative")  # the only statuses a training or test row may have
STATUSES = (*LABELLED, "censored", "immature")  # gold's label statuses (data spec §9.2)
MISSING_LEVEL = "__missing__"  # a null categorical value, as its own level (§1.4)
CCN_PATTERN = r"^[0-9A-Z]{6}$"
_POLARS_DTYPE = {"f64": pl.Float64, "i64": pl.Int64, "bool": pl.Boolean, "str": pl.String}
# Never a model input, whatever the registry says (§1.4, leakage test 5): keys, labels, forward-looking columns, any
# variant of the two star ratings ML-17 forbids (e.g. their _fn footnotes), and the roll-up's chain id.
_FORBIDDEN_PREFIXES = ("y_", "status_", "split_", "slice__")
_FORBIDDEN_PARTS = ("_fwd", "overall_rating", "health_inspection_rating")
_FORBIDDEN_NAMES = frozenset({"ccn", "t", "cutoff_date", "chain_id"})
_LEVEL_CHECK_MAX = (
    20  # a numeric regime-sensitive feature with at most this many values is checked level by level
)


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def gold_config_sha(manifest: InputManifest) -> str:
    """§15.2: sha256 of the canonical JSON of gold's ``{"config_sha256", "git_commit"}``. Gold built with other
    configs or other code is another gold. ``gate`` writes it into ``ml_runs`` and ``decide`` matches on it, so
    both call this one function."""
    payload = {"config_sha256": manifest.gold_config_sha256, "git_commit": manifest.gold_git_commit}
    return sha256_hex(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def _forbidden(name: str) -> bool:
    return (
        name in _FORBIDDEN_NAMES
        or name.startswith(_FORBIDDEN_PREFIXES)
        or any(part in name for part in _FORBIDDEN_PARTS)
    )


# --------------------------------------------------------------------------- features (§1.4)


def feature_list(registry: FeatureRegistry, cfg: MlConfig) -> list[FeatureSpec]:
    """The model input: registry entries with ``model_use: model``, in registry order, minus ``features.exclude``.

    A name in ``features.exclude`` or ``features.regime_sensitive`` that the registry doesn't know, or a forbidden
    column marked ``model``, is a config error (exit 3): a typo must not silently change the model.
    """
    names = {f.name for f in registry.features}
    model = [f for f in registry.features if f.model_use == "model"]
    unknown = [n for n in cfg.features.exclude if n not in {f.name for f in model}]
    if unknown:
        raise ConfigError(
            f"ml.yaml features.exclude names no model feature of the registry: {unknown}"
        )
    unknown = [n for n in cfg.features.regime_sensitive if n not in names]
    if unknown:
        raise ConfigError(f"ml.yaml features.regime_sensitive names no registry feature: {unknown}")
    forbidden = [f.name for f in model if _forbidden(f.name)]
    if forbidden:
        raise ConfigError(
            f"the feature registry marks forbidden columns as model features: {forbidden}"
        )
    return [f for f in model if f.name not in cfg.features.exclude]


def features_sha256(features: Sequence[FeatureSpec]) -> str:
    return sha256_hex(json.dumps([[f.name, f.dtype] for f in features]).encode())


def to_model_frame(df: pl.DataFrame, features: Sequence[FeatureSpec]) -> pd.DataFrame:
    """Rows -> model input, the same way for training, OOF, scoring and the signature (§1.4).

    Numeric and boolean features become float64 (True/False -> 1.0/0.0, null -> NaN); categorical ones become Python
    strings with null as its own level (``MISSING_LEVEL``). Columns come out in ``features`` order, so a model never
    sees a schema that depends on the rows it got (e.g. an all-non-null sample read as integers).
    """
    forbidden = [f.name for f in features if _forbidden(f.name)]
    if forbidden:
        raise MlInputError(f"never a model input: {forbidden}")
    missing = [f.name for f in features if f.name not in df.columns]
    if missing:
        raise MlInputError(f"model features missing from the rows: {missing}")
    exprs = [
        pl.col(f.name).cast(pl.String).fill_null(MISSING_LEVEL)
        if f.dtype == "str"
        else pl.col(f.name).cast(pl.Float64)
        for f in features
    ]
    out = df.select(exprs).to_pandas()
    for f in features:
        if f.dtype == "str":
            out[f.name] = out[f.name].astype(object)
    return out


# --------------------------------------------------------------------------- folds (§3.4)


def fold_of(ccn: str, k: int) -> int:
    """A facility's fold: stable across months, runs and machines (not Python's salted ``hash``)."""
    return int(hashlib.sha256(ccn.encode("utf-8")).hexdigest()[:8], 16) % k


def add_folds(df: pl.DataFrame, k: int) -> pl.DataFrame:
    """``df`` plus ``fold`` (Int8): every row of a facility in the same fold."""
    ccns = df["ccn"].unique().to_list()
    mapping = {c: fold_of(c, k) for c in ccns}
    return df.with_columns(
        pl.col("ccn").replace_strict(mapping, return_dtype=pl.Int8).alias("fold")
    )


# --------------------------------------------------------------------------- windows (§3.2)


def _last_n(months: Sequence[str], first: str, last: str, n: int) -> list[str]:
    return [m for m in months if first <= m <= last][-n:]


def compute_windows(
    months: Sequence[str],
    test_months_in_gold: Sequence[str],
    *,
    label: str,
    v: str,
    v_labels: str,
    labels_cfg: LabelsConfig,
    splits_cfg: SplitsConfig,
    cfg: MlConfig,
) -> Windows:
    """Every window of one run, from the months that hold rows (CMS gap months don't count, §3.2).

    ``months`` are gold's months with rows; ``test_months_in_gold`` those whose ``split_<label>`` is ``test``.
    Gold's test months must be exactly the months with rows of the calendar test window that
    ``configs/splits.yaml`` gives at ``V_L``; otherwise gold was built with other split settings (exit 2).
    """
    if label not in labels_cfg.labels:
        raise ConfigError(f"ml.yaml label {label!r} is not a label in labels.yaml")
    months = sorted(set(months))
    h, b = labels_cfg.labels[label].horizon_months, labels_cfg.buffer(label)
    last = last_labeled_t(v_labels, h, b)
    calendar = split_windows(v_labels, h, b, splits_cfg)
    test_from, test_to = calendar["test"]
    expected_test = [m for m in months if test_from <= m <= test_to]
    if sorted(test_months_in_gold) != expected_test:
        raise MlInputError(
            f"gold's {label} test months {sorted(test_months_in_gold)} are not the months with rows of the test "
            f"window {test_from}..{test_to} that configs/splits.yaml gives at V_L {v_labels}: gold was built with "
            "other split settings"
        )
    if not expected_test:
        raise MlInputError(f"the {label} test window {test_from}..{test_to} holds no rows")
    w = cfg.windows
    if w.eval_gap == "production_lag":
        # Exactly production's lag (ML-20): the refit trains up to L = V_L - (H + B) and scores V_L, so the eval fit
        # trains up to H + B months before the first test month. Its newest labels are known when the test month is,
        # and no training outcome window (t, t + H] reaches the test window.
        eval_last = shift(test_from, -(h + b))
    else:  # "purge": end before gold's purge window (reproducing PR #49)
        eval_last = shift(calendar["purge"][0], -1)
    floor = splits_cfg.spine_start
    eval_months = _last_n(months, floor, eval_last, w.train_months)
    refit_months = _last_n(months, floor, last, w.train_months)
    for name, got in (("eval", eval_months), ("refit", refit_months)):
        if len(got) < w.min_train_months:
            raise MlInputError(
                f"the {name} window holds {len(got)} months with rows ({got}); windows.min_train_months is "
                f"{w.min_train_months}"
            )
    if v not in months:
        raise MlInputError(f"gold has no rows for V = {v}")
    before = [m for m in months if m < v]
    return Windows(
        label=label,
        horizon_months=h,
        buffer_months=b,
        v=v,
        v_labels=v_labels,
        last_labeled_t=last,
        eval_gap=w.eval_gap,
        train_months=w.train_months,
        spine_start=floor,
        eval=MonthRange(first=eval_months[0], last=eval_months[-1], months=eval_months),
        test=MonthRange(first=test_from, last=test_to, months=expected_test),
        refit=MonthRange(first=refit_months[0], last=refit_months[-1], months=refit_months),
        score_month=v,
        prev_score_month=before[-1] if before else None,
        development_months=[m for m in cfg.evaluation.development_months if m in expected_test],
    )


# --------------------------------------------------------------------------- frames and checks (§3.2, §1.5)


def build_frames(df: pl.DataFrame, w: Windows, k: int) -> dict[Frame, pl.DataFrame]:
    """The five frames. Training and test frames hold labelled rows only (status positive or negative)."""
    label = w.label
    labelled = pl.col(f"status_{label}").is_in(LABELLED)
    df = add_folds(df, k)
    t = pl.col("t")
    frames: dict[Frame, pl.DataFrame] = {
        "eval_train": df.filter(t.is_in(w.eval.months) & labelled),
        "test": df.filter((pl.col(f"split_{label}") == "test") & labelled),
        "refit_train": df.filter(t.is_in(w.refit.months) & labelled),
        "score": df.filter(t == w.score_month),
        "prev_score": df.filter(t == w.prev_score_month) if w.prev_score_month else df.clear(),
    }
    return {name: f.sort("t", "ccn") for name, f in frames.items()}


def check_inputs(
    df: pl.DataFrame,
    features: Sequence[FeatureSpec],
    label: str,
    *,
    min_score_facilities: int,
    v: str,
    last_scored_v: str | None,
) -> None:
    """Spec §1.5 on the whole dataset (exit 2 on any failure)."""
    need = ["ccn", "t", f"y_{label}", f"status_{label}", f"split_{label}"]
    absent = [c for c in need if c not in df.columns]
    absent += [f.name for f in features if f.name not in df.columns]
    if absent:
        raise MlInputError(f"columns missing from gold: {absent}")
    contract = {"ccn": pl.String, "t": pl.String, f"y_{label}": pl.Int8}
    contract |= {f"status_{label}": pl.String, f"split_{label}": pl.String}
    wrong = [
        f"{c} is {df.schema[c]}, not {dtype}"
        for c, dtype in contract.items()
        if df.schema[c] != dtype
    ]
    wrong += [
        f"{f.name} is {df.schema[f.name]}, the registry says {f.dtype}"
        for f in features
        if df.schema[f.name] != _POLARS_DTYPE[f.dtype]
    ]
    if wrong:
        raise MlInputError("gold's dtypes differ from the contract: " + "; ".join(wrong))
    if df.select("ccn", "t").is_duplicated().any():
        raise MlInputError("ccn, t is not unique in gold")
    bad_ccn = df.filter(~pl.col("ccn").str.contains(CCN_PATTERN).fill_null(False)).height
    if bad_ccn:
        raise MlInputError(f"{bad_ccn} rows have a ccn that is not 6 characters of 0-9 and A-Z")
    y, status = pl.col(f"y_{label}"), pl.col(f"status_{label}")
    unknown = sorted(set(df[f"status_{label}"].unique().to_list()) - set(STATUSES), key=str)
    if unknown:
        raise MlInputError(f"status_{label} holds values outside {STATUSES}: {unknown}")
    inconsistent = df.filter(
        ((status == "positive") & (y != 1).fill_null(True))
        | ((status == "negative") & (y != 0).fill_null(True))
        | (~status.is_in(LABELLED) & y.is_not_null())  # censored or immature rows carry no label
    ).height
    if inconsistent:
        raise MlInputError(f"{inconsistent} rows have a y_{label} that disagrees with their status")
    newest = str(df["t"].max())
    if newest != v:
        raise MlInputError(f"gold's newest month is {newest}, not V = {v}")
    if last_scored_v is not None and v < last_scored_v:
        raise MlInputError(
            f"gold's V {v} is earlier than the last scored V {last_scored_v}: scoring it would overwrite newer scores"
        )
    score_rows = df.filter(pl.col("t") == v).height
    if score_rows < min_score_facilities:
        raise MlInputError(
            f"V = {v} has {score_rows} rows to score; checks.min_score_facilities is {min_score_facilities}"
        )


def check_maturity(df: pl.DataFrame, w: Windows) -> None:
    """No labelled row after L = V_L - (H + B): its outcome window isn't fully observed yet (exit 2)."""
    late = df.filter(
        pl.col(f"status_{w.label}").is_in(LABELLED) & (pl.col("t") > w.last_labeled_t)
    ).height
    if late:
        raise MlInputError(f"{late} labelled {w.label} rows are after L = {w.last_labeled_t}")


def check_gold_settings(
    current: dict[str, Any],
    manifest: dict[str, Any],
    label: str,
    labels_cfg: LabelsConfig,
    splits_cfg: SplitsConfig,
) -> None:
    """Gold was built with the label and split settings ML uses for its windows, and its pointer and manifest name
    the same months (exit 2): otherwise y_<label>, L and the windows would mean different things."""
    problems = []
    for key in ("v", "v_labels"):
        if current.get(key) != manifest.get(key):
            problems.append(
                f"_current.json {key} {current.get(key)} != manifest {manifest.get(key)}"
            )
    if (current.get("v_labels") or "") > (current.get("v") or ""):
        problems.append(f"v_labels {current.get('v_labels')} is after v {current.get('v')}")
    recorded = manifest.get("config_sha256") or {}
    for name, cfg in (("labels.yaml", labels_cfg), ("splits.yaml", splits_cfg)):
        if recorded.get(name) != settings_sha256(cfg):
            problems.append(f"gold was built with other {name} settings than this run's")
    source = labels_cfg.labels[label].source
    if (manifest.get("horizons_months") or {}).get(label) != labels_cfg.labels[
        label
    ].horizon_months:
        problems.append(f"gold's {label} horizon differs from labels.yaml")
    if (manifest.get("buffers_months") or {}).get(source) != labels_cfg.buffers_months[source]:
        problems.append(f"gold's {source} buffer differs from labels.yaml")
    if manifest.get("spine_start") != splits_cfg.spine_start:
        problems.append(
            f"gold's spine_start {manifest.get('spine_start')} != splits.yaml {splits_cfg.spine_start}"
        )
    if problems:
        raise MlInputError(
            "gold doesn't match this run's settings (rebuild gold): " + "; ".join(problems)
        )


def check_frames(frames: dict[Frame, pl.DataFrame], label: str) -> None:
    """Each training and test frame has positive and negative rows (§1.5; exit 2)."""
    for name in ("eval_train", "test", "refit_train"):
        counts = frames[name][f"y_{label}"].value_counts()
        present = set(counts[f"y_{label}"].to_list())
        if not {0, 1} <= present:
            raise MlInputError(
                f"frame {name} lacks positive or negative rows: y values {sorted(present)}"
            )


def last_scored_v(meta: MetadataStore) -> str | None:
    """The newest ``v`` whose scores were published (``ml_runs``), or None."""
    scored = [r["v"] for r in meta.read("ml_runs") if r.get("scores_published") and r.get("v")]
    return max(scored) if scored else None


# --------------------------------------------------------------------------- flags (§1.5, §1.4)


def average_precision(y: np.ndarray, score: np.ndarray) -> float:
    """Average precision (the area under the step-wise precision-recall curve), as scikit-learn computes it:
    thresholds at the distinct scores, AP = sum over thresholds of (R_n - R_n-1) x P_n."""
    positives = float(y.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-score, kind="mergesort")
    y, score = y[order], score[order]
    last_of_tie = np.r_[np.diff(score) != 0, True]
    tp = np.cumsum(y)[last_of_tie]
    seen = np.arange(1, len(y) + 1)[last_of_tie]
    recall = tp / positives
    precision = tp / seen
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def _numeric_ap(y: np.ndarray, x: np.ndarray) -> float:
    """max AP over both directions, with nulls as their own value at either end."""
    finite = x[~np.isnan(x)]
    if finite.size == 0:
        return float("nan")
    low, high = finite.min() - 1.0, finite.max() + 1.0
    best = float("nan")
    for null_value in (low, high):
        filled = np.where(np.isnan(x), null_value, x)
        for direction in (filled, -filled):
            ap = average_precision(y, direction)
            best = ap if np.isnan(best) else max(best, ap)
    return best


def _categorical_ap(y: np.ndarray, levels: np.ndarray, folds: np.ndarray) -> float:
    """AP of each row's level scored by the positive rate of that level in the other folds."""
    score = np.empty(len(y), dtype=float)
    frame = pd.DataFrame({"y": y, "level": levels, "fold": folds})
    for fold in np.unique(folds):
        rest = frame[frame["fold"] != fold]
        rates = rest.groupby("level")["y"].mean()
        prior = float(rest["y"].mean()) if len(rest) else float(y.mean())
        mask = folds == fold
        score[mask] = frame.loc[mask, "level"].map(rates).fillna(prior).to_numpy(dtype=float)
    return max(average_precision(y, score), average_precision(y, -score))


def leakage_flags(
    frames: dict[Frame, pl.DataFrame],
    features: Sequence[FeatureSpec],
    label: str,
    cfg: MlConfig,
) -> list[LeakageFlag]:
    """Features whose own PR-AUC on the eval or refit frame exceeds min(pr_auc_max, multiple x base rate) (§1.5).

    A flag doesn't stop prepare: it holds promotion at G2 until the feature is listed, with a reason, in
    ``checks.leakage_cleared``.
    """
    rule = cfg.checks.leakage_flag
    cleared = {c.feature for c in cfg.checks.leakage_cleared}
    flags: list[LeakageFlag] = []
    frame_name: Frame
    for frame_name in ("eval_train", "refit_train"):
        frame = frames[frame_name]
        y = frame[f"y_{label}"].cast(pl.Float64).to_numpy()
        base = float(y.mean())
        threshold = min(rule.pr_auc_max, rule.base_rate_multiple * base)
        folds = frame["fold"].to_numpy()
        for f in features:
            if f.dtype == "str":
                levels = frame[f.name].fill_null(MISSING_LEVEL).to_numpy()
                ap = _categorical_ap(y, levels, folds)
            else:
                ap = _numeric_ap(y, frame[f.name].cast(pl.Float64).to_numpy())
            if not np.isnan(ap) and ap > threshold:
                flags.append(
                    LeakageFlag(
                        frame=frame_name,
                        feature=f.name,
                        pr_auc=round(ap, 6),
                        threshold=round(threshold, 6),
                        cleared=f.name in cleared,
                    )
                )
    return flags


def regime_flags(
    frames: dict[Frame, pl.DataFrame], features: Sequence[FeatureSpec], cfg: MlConfig
) -> list[RegimeFlag]:
    """Regime-sensitive features whose score-month values the refit window doesn't cover (§1.4): a monitoring flag
    and a model-card line, not a refusal."""
    sensitive = set(cfg.features.regime_sensitive)
    refit, score = frames["refit_train"], frames["score"]
    flags: list[RegimeFlag] = []
    for f in features:
        if f.name not in sensitive:
            continue
        seen = refit[f.name].drop_nulls()
        now = score[f.name].drop_nulls()
        if now.is_empty():
            continue
        if seen.is_empty():
            flags.append(RegimeFlag(feature=f.name, detail="always null in the refit window"))
        elif f.dtype == "str" or seen.n_unique() <= _LEVEL_CHECK_MAX:
            new = sorted(set(now.cast(pl.String).to_list()) - set(seen.cast(pl.String).to_list()))
            if new:
                flags.append(
                    RegimeFlag(feature=f.name, detail=f"levels not in the refit window: {new}")
                )
        else:
            old, new_values = seen.cast(pl.Float64).to_numpy(), now.cast(pl.Float64).to_numpy()
            lo, hi = float(old.min()), float(old.max())
            share = float(((new_values < lo) | (new_values > hi)).mean())
            if share > 0:
                flags.append(
                    RegimeFlag(
                        feature=f.name,
                        detail=f"{share:.1%} of score values outside the refit range [{lo}, {hi}]",
                    )
                )
    return flags


# --------------------------------------------------------------------------- pin (§1.2)


@dataclass(frozen=True)
class PinnedGold:
    current: dict[str, Any]
    manifest: dict[str, Any]
    merged: bytes
    files: dict[str, str]  # pinned file -> sha256


def _read_json(store: ObjectStore, key: str) -> tuple[dict[str, Any], bytes]:
    if not store.exists(key):
        raise GoldPinError(f"{key} is missing: no gold published under this prefix")
    with store.open_read(key) as f:
        data = f.read()
    try:
        parsed = json.loads(data)
    except ValueError:
        raise GoldPinError(f"{key} is not valid JSON") from None
    if not isinstance(parsed, dict):
        raise GoldPinError(f"{key} does not hold a JSON object")
    return parsed, data


def pin_gold(store: ObjectStore, paths: RunPaths, gold_run_id: str | None) -> PinnedGold:
    """Copy gold's merged dataset, its manifest and ``_current.json`` into ``paths.input_dir`` (§1.2).

    Refuses (GoldPinError, exit 1) unless gold is current, is the run ``decide`` evaluated (``gold_run_id``; None =
    whatever is current, for a run by hand), the manifest and the pointer agree on the merged file, its bytes match,
    and the pointer didn't change while it was read. Nothing is written unless every check passed.
    """
    current_key = f"{paths.gold_prefix}{CURRENT_NAME}"
    current, current_bytes = _read_json(store, current_key)
    if current.get("state") != "current":
        raise GoldPinError(f"{current_key} says state={current.get('state')!r}, not 'current'")
    run_id = current.get("run_id")
    if gold_run_id is not None and run_id != gold_run_id:
        raise GoldPinError(
            f"gold is run {run_id}, not run {gold_run_id} that decide evaluated: a newer gold starts its own run"
        )
    merged_sha = current.get("sha256", {}).get(MERGED)
    if not merged_sha:
        raise GoldPinError(f"{current_key} lists no {MERGED}")
    manifest, manifest_bytes = _read_json(store, f"{paths.gold_prefix}{MANIFEST_NAME}")
    if manifest.get("merged_file", {}).get("sha256") != merged_sha:
        raise GoldPinError(
            f"{MANIFEST_NAME} and {CURRENT_NAME} disagree about {MERGED}: rerun gold"
        )
    with store.open_read(f"{paths.gold_prefix}{MERGED}") as f:
        merged = f.read()
    if sha256_hex(merged) != merged_sha:
        raise GoldPinError(f"{paths.gold_prefix}{MERGED} does not match {CURRENT_NAME}")
    again, _ = _read_json(store, current_key)
    if again != current:
        raise GoldPinError(
            f"{current_key} changed while gold was read (a gold run promoted): retry"
        )
    data = {MERGED: merged, MANIFEST_NAME: manifest_bytes, CURRENT_NAME: current_bytes}
    files: dict[str, str] = {}
    for name, payload in data.items():
        key = paths.input_file(name)
        store.put_bytes(key, payload, overwrite=True)
        files[name] = sha256_of(store, key)  # read back: the stored copy, not the bytes handed over
        if files[name] != sha256_hex(payload):
            raise GoldPinError(f"the stored copy of {name} does not match what was read")
    return PinnedGold(current=current, manifest=manifest, merged=merged, files=files)


# --------------------------------------------------------------------------- reading a run folder (later tasks)


# The manifest is written last and names the hash of every other file, so a prepare that died half-way through a
# rerun (new frames, old manifest) is caught here instead of being trained on.


def read_input_manifest(store: ObjectStore, paths: RunPaths) -> InputManifest:
    with store.open_read(paths.input_manifest) as f:
        return InputManifest.model_validate_json(f.read())


def _read_checked(store: ObjectStore, key: str, expected: str | None) -> bytes:
    with store.open_read(key) as f:
        data = f.read()
    if sha256_hex(data) != expected:
        raise MlInputError(f"{key} does not match the run's input/manifest.json: rerun prepare")
    return data


def read_windows(store: ObjectStore, paths: RunPaths) -> Windows:
    manifest = read_input_manifest(store, paths)
    return Windows.model_validate_json(_read_checked(store, paths.windows, manifest.windows_sha256))


def read_frame(store: ObjectStore, paths: RunPaths, name: Frame) -> pl.DataFrame:
    manifest = read_input_manifest(store, paths)
    data = _read_checked(store, paths.frame(name), manifest.frames_sha256.get(name))
    return pl.read_parquet(io.BytesIO(data))


# --------------------------------------------------------------------------- prepare


@dataclass(frozen=True)
class PrepareResult:
    paths: RunPaths
    manifest: InputManifest
    windows: Windows

    def summary(self) -> dict[str, Any]:
        """The last stdout line of ``ml prepare``."""
        m = self.manifest
        return {
            "ml_run_id": m.ml_run_id,
            "gold_run_id": m.gold_run_id,
            "v": m.v,
            "v_labels": m.v_labels,
            "dataset_version": m.dataset_version,
            "verified": m.verified,
            "rows": m.rows,
            "eval": [self.windows.eval.first, self.windows.eval.last],
            "test": [self.windows.test.first, self.windows.test.last],
            "refit": [self.windows.refit.first, self.windows.refit.last],
            "leakage_flags": len(m.leakage_flags),
            "regime_flags": len(m.regime_flags),
        }


def _put_json(store: ObjectStore, key: str, payload: str) -> None:
    store.put_bytes(key, payload.encode("utf-8"), overwrite=True)


def decided_gold_run(store: ObjectStore, paths: RunPaths, gold_run_id: str | None) -> str | None:
    """The gold run this run must pin (leakage test 12): the one ``decide`` recorded in ``decision.json`` when it
    wrote one, else ``--gold-run-id``, else None (a run by hand pins what is current). Both given and different,
    or a decision that names no gold run, is refused (exit 2): clearing ``prepare`` in an old DAG run must never pin
    a newer gold under it."""
    if not store.exists(paths.decision):
        return gold_run_id
    with store.open_read(paths.decision) as f:
        raw = f.read()
    try:
        decision = Decision.model_validate_json(raw)
    except ValueError as exc:
        raise MlInputError(f"{paths.decision} is not a valid decision: {exc}") from None
    if decision.gold_run_id is None:
        raise MlInputError(f"{paths.decision} names no gold run (decide said {decision.decision})")
    if gold_run_id is not None and gold_run_id != decision.gold_run_id:
        raise MlInputError(
            f"--gold-run-id {gold_run_id} disagrees with {paths.decision}: {decision.gold_run_id}"
        )
    return decision.gold_run_id


def run_prepare(
    store: ObjectStore,
    meta: MetadataStore,
    *,
    ml_run_id: str,
    cfg: MlConfig,
    labels_cfg: LabelsConfig,
    splits_cfg: SplitsConfig,
    registry: FeatureRegistry,
    gold_run_id: str | None = None,
    allow_unverified: bool | None = None,
) -> PrepareResult:
    """Pin gold, check it, build and write the frames, ``windows.json`` and ``input/manifest.json`` (§1.2, §1.5, §3).

    ``allow_unverified`` overrides ``ml.yaml`` for this run (None = the config's value).
    """
    paths = RunPaths.of(cfg, ml_run_id)
    label = cfg.label
    features = feature_list(registry, cfg)
    if label not in labels_cfg.labels:
        raise ConfigError(f"ml.yaml label {label!r} is not a label in labels.yaml")
    pinned = pin_gold(store, paths, decided_gold_run(store, paths, gold_run_id))
    current, manifest = pinned.current, pinned.manifest
    check_gold_settings(current, manifest, label, labels_cfg, splits_cfg)
    verified = current.get("verified") is True and manifest.get("verified") is True
    allowed = cfg.allow_unverified if allow_unverified is None else allow_unverified
    if not verified and not allowed:
        raise MlInputError("gold is not verified (data spec §11) and allow_unverified is false")
    v, v_labels = current.get("v"), current.get("v_labels")
    if not v or not v_labels:
        raise MlInputError("gold's _current.json and manifest name no v or v_labels")
    df = pl.read_parquet(io.BytesIO(pinned.merged))
    check_inputs(
        df,
        features,
        label,
        min_score_facilities=cfg.checks.min_score_facilities,
        v=v,
        last_scored_v=last_scored_v(meta),
    )
    months = df["t"].unique().sort().to_list()
    test_months = df.filter(pl.col(f"split_{label}") == "test")["t"].unique().sort().to_list()
    windows = compute_windows(
        months,
        test_months,
        label=label,
        v=v,
        v_labels=v_labels,
        labels_cfg=labels_cfg,
        splits_cfg=splits_cfg,
        cfg=cfg,
    )
    check_maturity(df, windows)
    frames = build_frames(df, windows, cfg.folds.k)
    check_frames(frames, label)
    leaks = leakage_flags(frames, features, label, cfg)
    regimes = regime_flags(frames, features, cfg)
    frames_sha256: dict[str, str] = {}
    for name in FRAMES:
        buf = io.BytesIO()
        frames[name].write_parquet(buf)
        frames_sha256[name] = store.put_bytes(
            paths.frame(name), buf.getvalue(), overwrite=True
        ).sha256
    windows_json = windows.model_dump_json(indent=2)
    input_manifest = InputManifest(
        ml_run_id=ml_run_id,
        gold_run_id=str(current["run_id"]),
        v=v,
        v_labels=v_labels,
        dataset_version=str(manifest.get("dataset_version", "")),
        verified=verified,
        merged_sha256=pinned.files[MERGED],
        gold_git_commit=manifest.get("git_commit"),
        gold_config_sha256=dict(manifest.get("config_sha256") or {}),
        flags=dict(manifest.get("flags") or {}),
        label=label,
        features=[f.name for f in features],
        excluded_features=list(cfg.features.exclude),
        features_sha256=features_sha256(features),
        folds=cfg.folds.k,
        rows={name: frames[name].height for name in FRAMES},
        files=pinned.files,
        frames_sha256=frames_sha256,
        windows_sha256=sha256_hex(windows_json.encode("utf-8")),
        leakage_flags=leaks,
        regime_flags=regimes,
        pinned_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )
    _put_json(store, paths.windows, windows_json)
    _put_json(
        store, paths.input_manifest, input_manifest.model_dump_json(indent=2)
    )  # last: names all the rest
    log.info(
        "prepared run %s from gold %s (V %s, V_L %s): %s",
        ml_run_id,
        input_manifest.gold_run_id,
        v,
        v_labels,
        input_manifest.rows,
    )
    return PrepareResult(paths=paths, manifest=input_manifest, windows=windows)
