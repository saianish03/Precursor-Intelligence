"""Task ``tune``: the Optuna search over the members' hyperparameters (docs/ML_PIPELINE_SPEC.md §13, §8.3, ML-19).

- ``search_space``: ``tuning.space`` of ``configs/ml.yaml`` (spec §13, PR #49's ``boost_models.py``), LightGBM's
  parameters plus the LR ``C`` (``lr_C``).
- ``tune``: TPE (seeded), ``n_jobs`` 1, on ``eval_train`` only (never the test frame, §20). Each trial fits, for each of
  ``tuning.folds`` facility-grouped folds (``fold_of(ccn, tuning.folds)``, §3.4), LightGBM on the other folds with
  early stopping (``tuning.early_stopping_rounds``, within ``tuning.max_rounds``) on the held-out fold, and LR with
  the trial's ``C``.
- ``run_tune``: ``tune`` on a prepared run folder; writes ``tuning/`` (``tuning_proposal.yaml``, ``trials.csv``,
  ``importances.json``, the plots) and one MLflow run ``tune``.

**Objective.** The mean, over the folds, of the held-out average precision (PR-AUC) of the *ensemble*: the
``ensemble.weights`` mean of the two members' log-odds (ML-16). Spec §13 says "mean PR-AUC over 3 facility-grouped
folds"; the ensemble is the scored default candidate, and only an objective that includes LR can choose LR's ``C``
(which §13's space includes). LightGBM-only and LR-only APs are kept per trial as user attributes, so a reviewer can
see whether a trial won through either member. Early stopping still watches LightGBM's own log loss.

The result is a **proposal**: the parameters reach ``configs/ml.yaml`` only through a reviewed PR (ML-19). For now
the proposal is the only output: ``train`` never reads ``tuning_proposal.yaml``, so no tuned challenger is fitted or
gated yet. Spec §13 lists that challenger as a tune output; it is an open item for the W2 gate and W6's RETUNE path.
"""

from __future__ import annotations

import io
import json
import logging
import tempfile
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import optuna
import pandas as pd
import polars as pl
import yaml

from ..common.config import (
    TUNING_LR_C,
    FeatureRegistry,
    FeatureSpec,
    LgbmParams,
    LrParams,
    MlConfig,
    TuningParamCfg,
)
from ..common.io.object_store import ObjectStore
from . import MlInputError
from .contracts import RunPaths
from .data import (
    average_precision,
    feature_list,
    fold_of,
    read_frame,
    read_input_manifest,
    to_model_frame,
)
from .ensemble import combine
from .members import LgbmMember, LrMember

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Param:
    """One dimension of the search (``tuning.space``): a float or int range (``log``: sampled on a log scale) or a
    categorical."""

    kind: Literal["float", "int", "categorical"]
    low: float = 0.0
    high: float = 0.0
    log: bool = False
    choices: tuple[float, ...] = ()

    @classmethod
    def of(cls, p: TuningParamCfg) -> Param:
        return cls(p.kind, p.low or 0.0, p.high or 0.0, p.log, tuple(p.choices))

    def suggest(self, trial: optuna.Trial, name: str) -> float | int:
        if self.kind == "float":
            return trial.suggest_float(name, self.low, self.high, log=self.log)
        if self.kind == "int":
            return trial.suggest_int(name, int(self.low), int(self.high), log=self.log)
        return float(trial.suggest_categorical(name, list(self.choices)))

    def describe(self) -> str:
        if self.kind == "categorical":
            return "{" + ", ".join(f"{c:g}" for c in self.choices) + "}"
        return f"{self.kind} [{self.low:g}, {self.high:g}]" + (", log" if self.log else "")


LR_C = TUNING_LR_C  # the Optuna name of LR's C (the others are LightGBM's members.lgbm keys)


def search_space(cfg: MlConfig) -> dict[str, Param]:
    """``tuning.space``, in its config order."""
    return {name: Param.of(p) for name, p in cfg.tuning.space.items()}


def lgbm_searched(space: Mapping[str, Param]) -> tuple[str, ...]:
    return tuple(k for k in space if k != LR_C)


PROPOSAL_NAME = "tuning_proposal.yaml"
TRIALS_NAME = "trials.csv"
IMPORTANCES_NAME = "importances.json"
PLOT_NAMES = ("optimization_history.png", "param_importances.png", "parallel_coordinate.png")

_PROPOSAL_HEADER = """\
# Tuning proposal (docs/ML_PIPELINE_SPEC.md §13, ML-19). NOT the recipe: these parameters reach configs/ml.yaml
# (members.lgbm, members.lr) only through a reviewed PR, which changes the recipe_hash and so triggers RETRAIN and
# the gates. cv_pr_auc is the mean held-out PR-AUC of the ensemble over facility-grouped folds of eval_train (never
# the test frame); n_estimators is the median early-stopping best iteration over those folds. Early stopping
# watches the same held-out fold it is scored on, so cv_pr_auc is slightly optimistic: compare proposals with each
# other, not with an OOF PR-AUC of the current recipe (which uses a fixed n_estimators).
"""


@dataclass(frozen=True)
class TuneResult:
    best_params: dict[str, Any]  # members.lgbm, ready to paste into ml.yaml
    best_lr_C: float
    best_value: float  # mean CV PR-AUC (ensemble) of the best trial
    trials: pd.DataFrame
    importances: dict[str, float]  # fANOVA; {} with fewer than 2 complete trials
    proposal_yaml: str
    plots: dict[str, bytes]  # PNG bytes by file name; a plot that failed is missing


@dataclass(frozen=True)
class _Fold:
    x_train: pd.DataFrame
    y_train: np.ndarray
    x_valid: pd.DataFrame
    y_valid: np.ndarray


def _folds(frame: pl.DataFrame, features: Sequence[FeatureSpec], label: str, k: int) -> list[_Fold]:
    ccns = frame["ccn"].to_list()
    mapping = {c: fold_of(c, k) for c in set(ccns)}
    fold = np.array([mapping[c] for c in ccns], dtype=np.int64)
    x = to_model_frame(frame, features)
    y = frame[f"y_{label}"].cast(pl.Float64).to_numpy()
    out = []
    for i in range(k):
        held = fold == i
        for side, mask in (("held-out", held), ("training", ~held)):
            ys = y[mask]
            if ys.size == 0 or ys.min() == ys.max():
                raise MlInputError(
                    f"tuning fold {i}: the {side} rows need both outcomes ({ys.size} rows, {int(ys.sum())} positive)"
                )
        out.append(
            _Fold(
                x_train=x[~held].reset_index(drop=True),
                y_train=y[~held],
                x_valid=x[held].reset_index(drop=True),
                y_valid=y[held],
            )
        )
    return out


def _best_iteration(member: LgbmMember) -> int:
    best = int(member.booster.best_iteration)
    return best if best > 0 else int(member.booster.current_iteration())


def _objective(
    folds: list[_Fold],
    features: Sequence[FeatureSpec],
    cfg: MlConfig,
    rounds: int,
) -> Callable[[optuna.Trial], float]:
    weights = {str(m): float(w) for m, w in cfg.ensemble.weights.items()}
    space = search_space(cfg)
    searched = lgbm_searched(space)
    lr_cache: dict[tuple[int, float], np.ndarray] = {}  # LR doesn't depend on LightGBM's parameters

    def lr_logit(i: int, c: float) -> np.ndarray:
        if (i, c) not in lr_cache:
            f = folds[i]
            params = LrParams(
                C=c,
                max_iter=cfg.members.lr.max_iter,
                onehot_min_frequency=cfg.members.lr.onehot_min_frequency,
            )
            lr_cache[(i, c)] = LrMember.fit(f.x_train, f.y_train, features, params).logit(f.x_valid)
        return lr_cache[(i, c)]

    def objective(trial: optuna.Trial) -> float:
        suggested = {name: space[name].suggest(trial, name) for name in searched}
        c = float(space[LR_C].suggest(trial, LR_C)) if LR_C in space else cfg.members.lr.C
        params = LgbmParams.model_validate(
            {**cfg.members.lgbm.as_dict(), **suggested, "n_estimators": rounds}
        )
        iterations: list[int] = []
        ap_lgbm: list[float] = []
        ap_lr: list[float] = []
        ap_ens: list[float] = []
        for i, f in enumerate(folds):
            member = LgbmMember.fit(
                f.x_train,
                f.y_train,
                features,
                params,
                valid=(f.x_valid, f.y_valid),
                early_stopping_rounds=cfg.tuning.early_stopping_rounds,
            )
            iterations.append(_best_iteration(member))
            lg = member.logit(f.x_valid)
            lr = lr_logit(i, c)
            ap_lgbm.append(average_precision(f.y_valid, lg))
            ap_lr.append(average_precision(f.y_valid, lr))
            ap_ens.append(average_precision(f.y_valid, combine({"lgbm": lg, "lr": lr}, weights)))
        trial.set_user_attr("best_iterations", iterations)
        trial.set_user_attr("fold_pr_auc", ap_ens)
        trial.set_user_attr("lgbm_pr_auc", float(np.mean(ap_lgbm)))
        trial.set_user_attr("lr_pr_auc", float(np.mean(ap_lr)))
        return float(np.mean(ap_ens))

    return objective


def _importances(study: optuna.Study, seed: int) -> dict[str, float]:
    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if len(complete) < 2:
        return {}
    try:
        evaluator = optuna.importance.FanovaImportanceEvaluator(seed=seed)
        return {
            k: float(v)
            for k, v in optuna.importance.get_param_importances(study, evaluator=evaluator).items()
        }
    except Exception as exc:  # e.g. every trial sampled the same values
        log.warning("fANOVA importances failed: %s", exc)
        return {}


def _plots(study: optuna.Study, seed: int) -> dict[str, bytes]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from optuna.visualization import matplotlib as ovm

    makers: dict[str, Callable[[], Any]] = {
        "optimization_history.png": lambda: ovm.plot_optimization_history(study),
        "param_importances.png": lambda: ovm.plot_param_importances(
            study, evaluator=optuna.importance.FanovaImportanceEvaluator(seed=seed)
        ),
        "parallel_coordinate.png": lambda: ovm.plot_parallel_coordinate(study),
    }
    out: dict[str, bytes] = {}
    for name, make in makers.items():
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # optuna's ExperimentalWarning
                ax = make()
            fig = ax.flat[0].figure if isinstance(ax, np.ndarray) else ax.figure
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=72, bbox_inches="tight")
            out[name] = buf.getvalue()
        except Exception as exc:  # a plot never fails the task
            log.warning("tuning plot %s failed: %s", name, exc)
        finally:
            plt.close("all")
    return out


def _plain(v: Any) -> Any:
    """numpy scalars -> Python, so the YAML is plain."""
    return v.item() if isinstance(v, np.generic) else v


def _proposal(
    best_params: Mapping[str, Any],
    best_c: float,
    best_value: float,
    n_complete: int,
    cfg: MlConfig,
    source: Mapping[str, Any] | None,
) -> str:
    doc = {
        "members": {
            "lgbm": {k: _plain(v) for k, v in best_params.items()},
            "lr": {
                "C": best_c,
                "max_iter": cfg.members.lr.max_iter,
                "onehot_min_frequency": cfg.members.lr.onehot_min_frequency,
            },
        },
        "cv_pr_auc": best_value,
        "objective": "mean held-out PR-AUC of the ensemble (ensemble.weights mean of log-odds)",
        "n_trials": n_complete,
        "folds": cfg.tuning.folds,
        "seed": cfg.tuning.seed,
        "source": dict(source or {}),
    }
    return _PROPOSAL_HEADER + yaml.safe_dump(doc, sort_keys=False)


def tune(
    frame: pl.DataFrame,
    features: Sequence[FeatureSpec],
    cfg: MlConfig,
    *,
    label: str,
    n_trials: int | None = None,
    timeout_minutes: float | None = None,
    max_rounds: int | None = None,
    source: Mapping[str, Any] | None = None,
) -> TuneResult:
    """The search on ``frame`` (``eval_train``). ``max_rounds`` caps the boosting rounds early stopping runs within
    (default: ``tuning.max_rounds``); ``source`` (run ids, notes) is recorded in the proposal. A frame that holds
    test or purge rows is a data refusal (``MlInputError``, exit 2)."""
    split = f"split_{label}"
    if split not in frame.columns:
        raise MlInputError(f"tune needs {split} to prove it holds no test rows (spec §13, §20)")
    if frame.filter(pl.col(split).is_in(["test", "purge"])).height:
        raise MlInputError(
            "tune refuses rows of the test or purge split: it runs on eval_train only (spec §13, §20)"
        )
    rounds = int(max_rounds if max_rounds is not None else cfg.tuning.max_rounds)
    folds = _folds(frame, features, label, cfg.tuning.folds)

    sampler = optuna.samplers.TPESampler(seed=cfg.tuning.seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    timeout = (
        timeout_minutes if timeout_minutes is not None else cfg.tuning.timeout_minutes
    ) * 60.0
    study.optimize(
        _objective(folds, features, cfg, rounds),
        n_trials=n_trials if n_trials is not None else cfg.tuning.n_trials,
        timeout=timeout,
        n_jobs=1,
        gc_after_trial=True,
    )
    # optimize runs at least one trial before the timeout is checked, and a failing trial raises (catch=()), so
    # there is always a complete trial here
    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    best = study.best_trial
    iterations = best.user_attrs["best_iterations"]
    space = search_space(cfg)
    # keeps seed, deterministic, threads, subsample_freq; validated, so the proposal is a valid members.lgbm
    best_params: dict[str, Any] = LgbmParams.model_validate(
        {
            **cfg.members.lgbm.as_dict(),
            **{k: best.params[k] for k in lgbm_searched(space)},
            "n_estimators": max(1, int(round(float(np.median(iterations))))),
        }
    ).as_dict()
    best_c = float(best.params[LR_C]) if LR_C in space else cfg.members.lr.C
    if max(iterations) >= rounds:
        log.warning(
            "tuning: a fold of the best trial used all %d rounds without early stopping: raise tuning.max_rounds",
            rounds,
        )
    best_value = float(best.value) if best.value is not None else float("nan")

    return TuneResult(
        best_params=best_params,
        best_lr_C=best_c,
        best_value=best_value,
        trials=study.trials_dataframe(),
        importances=_importances(study, cfg.tuning.seed),
        proposal_yaml=_proposal(best_params, best_c, best_value, len(complete), cfg, source),
        plots=_plots(study, cfg.tuning.seed),
    )


def run_tune(
    store: ObjectStore,
    *,
    ml_run_id: str,
    cfg: MlConfig,
    registry: FeatureRegistry,
    n_trials: int | None = None,
    max_rounds: int | None = None,
) -> TuneResult:
    """The ``tune`` task on a prepared run folder: reads ``eval_train`` (hash-checked), writes ``tuning/`` and logs
    one MLflow run ``tune`` when tracking is on."""
    paths = RunPaths.of(cfg, ml_run_id)
    manifest = read_input_manifest(store, paths)
    if manifest.label != cfg.label:
        raise MlInputError(f"prepare ran for label {manifest.label}, ml.yaml says {cfg.label}")
    features = feature_list(registry, cfg)
    if [f.name for f in features] != manifest.features:
        raise MlInputError(
            "the feature list differs from the one prepare recorded in input/manifest.json: rerun prepare"
        )
    frame = read_frame(store, paths, "eval_train")
    source = {
        "ml_run_id": ml_run_id,
        "gold_run_id": manifest.gold_run_id,
        "dataset_version": manifest.dataset_version,
        "v": manifest.v,
        "v_labels": manifest.v_labels,
        "frame": "eval_train",
        "note": "proposal only; apply through a reviewed PR to configs/ml.yaml (ML-19)",
    }
    result = tune(
        frame,
        features,
        cfg,
        label=cfg.label,
        n_trials=n_trials,
        max_rounds=max_rounds,
        source=source,
    )

    files: dict[str, bytes] = {
        PROPOSAL_NAME: result.proposal_yaml.encode("utf-8"),
        TRIALS_NAME: result.trials.to_csv(index=False).encode("utf-8"),
        IMPORTANCES_NAME: json.dumps(result.importances, indent=2, sort_keys=True).encode("utf-8"),
        **result.plots,
    }
    for name, data in files.items():
        key = paths.tuning_proposal if name == PROPOSAL_NAME else f"{paths.tuning_dir}{name}"
        store.put_bytes(key, data, overwrite=True)

    _log_mlflow(cfg, manifest, ml_run_id, features, result, files, n_trials)
    return result


def _log_mlflow(
    cfg: MlConfig,
    manifest: Any,
    ml_run_id: str,
    features: Sequence[FeatureSpec],
    result: TuneResult,
    files: Mapping[str, bytes],
    n_trials: int | None,
) -> None:
    from . import mlflow_log

    if not mlflow_log.tracking_enabled():
        return
    import mlflow

    tags = mlflow_log.base_tags(
        manifest,
        ml_run_id=ml_run_id,
        recipe_hash=mlflow_log.recipe_hash(features),
        decision="RETUNE",
        mode="retune",
    )
    with mlflow_log.task_run(cfg, "tune", tags) as run_id:
        if run_id is None:
            return
        params: dict[str, Any] = {f"space.{k}": p.describe() for k, p in search_space(cfg).items()}
        params.update(
            {
                "n_trials": n_trials if n_trials is not None else cfg.tuning.n_trials,
                "folds": cfg.tuning.folds,
                "timeout_minutes": cfg.tuning.timeout_minutes,
                "early_stopping_rounds": cfg.tuning.early_stopping_rounds,
                "max_rounds": cfg.tuning.max_rounds,
                "seed": cfg.tuning.seed,
            }
        )
        mlflow_log.log_params(params)
        mlflow_log.log_metrics(
            {"cv_pr_auc": result.best_value, "n_trials_run": float(len(result.trials))}
        )
        mlflow_log.log_json_artifact(result.importances, IMPORTANCES_NAME)
        with tempfile.TemporaryDirectory() as tmp:
            for name, data in files.items():
                if name == IMPORTANCES_NAME:
                    continue
                path = Path(tmp) / name
                path.write_bytes(data)
                mlflow.log_artifact(str(path), artifact_path="tuning")
