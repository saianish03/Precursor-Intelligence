"""The ``evaluate`` task (docs/ML_PIPELINE_SPEC.md §6, §7): the eval fit on the test frame, against the baselines and
the champion's eval fit, by slice, with the sanity checks; writes ``evaluation.json`` for ``gate``.

The test frame is scored only by eval-fit packages (trained on the eval window) and, for the paired comparisons,
the champion's **eval fit** (never ``@champion``'s refit, which may have seen the test months; leakage test 10).
The classification threshold comes from the OOF predictions, never from test. A package whose training months
reach the test window minus the horizon is refused (leakage test 3).
"""

from __future__ import annotations

import io
import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import polars as pl

from ..common.config import FeatureRegistry, MlConfig, load_feature_registry
from ..common.io.metadata_store import MetadataStore
from ..common.io.object_store import ObjectStore
from ..gold.splits import shift
from ..training import MlInputError
from ..training.contracts import RunPaths, Selection, TrainResult, Windows
from ..training.data import (
    average_precision,
    read_frame,
    read_input_manifest,
    read_windows,
)
from ..training.ensemble import PrecursorModel, read_package
from ..training.train import read_decision, resolve_features
from . import baselines, metrics, sanity, slices
from .bootstrap import facility_bootstrap
from .gates import Interval
from .mitigation import tradeoff as mitigation_tradeoff
from .registry import Registry
from .report import (
    ChampionRef,
    EvaluationReport,
    Sanity,
    ScoreReport,
    comparison_key,
)

log = logging.getLogger(__name__)

CHAMPION = "champion"
RULE = "prior_gplus_rule"
HEADLINE = ("pr_auc", "pr_auc_lift", "roc_auc", "brier")
TOP_FEATURES = 20
GATED_ACTIONS = ("promote", "hold")  # ml_runs actions written by gate: the recipe was evaluated


def _sigmoid(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return np.asarray(1.0 / (1.0 + np.exp(-x)), dtype=np.float64)


def _headline(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    base = float(y.mean())
    pr = average_precision(y, p)
    from sklearn.metrics import roc_auc_score

    return {
        "pr_auc": pr,
        "pr_auc_lift": pr / base,
        "roc_auc": float(roc_auc_score(y, p)),
        "brier": float(np.mean((p - y) ** 2)),
    }


def check_training_months(model: PrecursorModel, name: str, windows: Windows) -> None:
    """Leakage test 3: every month an eval-fit package trained on, plus the horizon, is before the test window."""
    months = (model.meta.training_months.get("eval") if model.meta else None) or []
    if not months:
        raise MlInputError(f"eval package {name} records no training months")
    first_test = min(windows.test.months)
    last = max(months)
    if not shift(last, windows.horizon_months) < first_test:
        raise MlInputError(
            f"eval package {name} trained on {last}: + {windows.horizon_months} months reaches the test window "
            f"({first_test})"
        )


def _read_json_model(store: ObjectStore, key: str, model: Any) -> Any:
    if not store.exists(key):
        raise MlInputError(f"{key} is missing: run `ml train` first")
    with store.open_read(key) as f:
        return model.model_validate_json(f.read())


def _read_parquet(store: ObjectStore, key: str) -> pl.DataFrame:
    with store.open_read(key) as f:
        return pl.read_parquet(io.BytesIO(f.read()))


def recipe_evaluated(meta: MetadataStore | None, recipe_hash: str) -> bool:
    """Whether a gate already judged this recipe (an ``ml_runs`` promote or hold row with its hash)."""
    if meta is None:
        return False
    return any(
        r.get("recipe_hash") == recipe_hash and r.get("action") in GATED_ACTIONS
        for r in meta.read("ml_runs")
    )


def champion_ref(registry: Registry | None, cfg: MlConfig) -> ChampionRef | None:
    """The champion version and its eval-fit model URI (the tag ``gate`` wrote), or None without one."""
    if registry is None:
        return None
    v = registry.by_alias(cfg.registry.champion_alias)
    if v is None:
        return None
    uri = v.tags.get("eval_model_uri", "")
    if not uri:
        raise MlInputError(
            f"champion version {v.version} has no eval_model_uri tag: it cannot be compared on this test frame"
        )
    return ChampionRef(version=v.version, eval_model_uri=uri, candidate=v.tags.get("candidate"))


def _score_report(
    y: np.ndarray,
    p: np.ndarray,
    cfg: MlConfig,
    *,
    probability: bool,
    threshold: float | None = None,
    states: np.ndarray | None = None,
    intervals: Mapping[str, Interval] | None = None,
) -> ScoreReport:
    ev = cfg.evaluation
    return ScoreReport(
        ranking=metrics.ranking_metrics(y, p, ev.precision_at),
        probability=metrics.probability_metrics(y, p, ev.ece_bins) if probability else None,
        classification=(
            metrics.classification_metrics(y, p, threshold) if threshold is not None else None
        ),
        within_state=metrics.within_group(y, p, states, 0.10) if states is not None else None,
        intervals=dict(intervals or {}),
    )


def _per_month(
    test: pl.DataFrame, y: np.ndarray, p: np.ndarray, cfg: MlConfig
) -> list[dict[str, Any]]:
    out = []
    months = test["t"].to_numpy()
    for m in sorted(set(months.tolist())):
        mask = months == m
        ym, pm = y[mask], p[mask]
        base = float(ym.mean())
        pr = average_precision(ym, pm) if 0 < base < 1 else float("nan")
        out.append(
            {
                "month": m,
                "development": m in cfg.evaluation.development_months,
                "rows": int(mask.sum()),
                "positives": int(ym.sum()),
                "base_rate": base,
                "pr_auc": pr,
                "pr_auc_lift": pr / base if base else float("nan"),
                "brier": float(np.mean((pm - ym) ** 2)),
                "ece": metrics.ece(ym, pm, cfg.evaluation.ece_bins),
            }
        )
    return out


def top_features(
    model: PrecursorModel, frame: pl.DataFrame, n: int | None = TOP_FEATURES
) -> pl.DataFrame:
    """Mean |contribution| per feature (calibrated log-odds) on ``frame``, largest first (§8.2)."""
    _, contrib = model.explain(frame)
    mean_abs = np.abs(contrib.to_numpy()).mean(axis=0)
    table = pl.DataFrame({"feature": list(contrib.columns), "mean_abs": mean_abs}).sort(
        "mean_abs", descending=True
    )
    return table.head(n) if n is not None else table


@dataclass(frozen=True)
class EvaluateOutput:
    report: EvaluationReport
    test_scores: pl.DataFrame  # ccn, t, y, then p_<score> per candidate, baseline and the champion
    slice_table: pl.DataFrame
    shap_mean_abs: pl.DataFrame
    raw_selected: (
        np.ndarray
    )  # the selected candidate's probability before Platt, for the reliability plot


def evaluate_run(
    store: ObjectStore,
    *,
    ml_run_id: str,
    cfg: MlConfig,
    registry: Registry | None = None,
    feature_registry: FeatureRegistry | None = None,
    meta: MetadataStore | None = None,
    shuffled_label: bool | None = None,
    mitigation: bool = True,
) -> EvaluateOutput:
    """Compute everything ``evaluation.json`` holds, without writing it. ``shuffled_label`` None runs the check
    only for a recipe no gate judged yet (§6.4)."""
    paths = RunPaths.of(cfg, ml_run_id)
    manifest = read_input_manifest(store, paths)
    windows = read_windows(store, paths)
    train: TrainResult = _read_json_model(store, paths.train_result, TrainResult)
    selection: Selection = _read_json_model(store, paths.selection, Selection)
    if manifest.label != cfg.label:
        raise MlInputError(f"prepare ran for label {manifest.label}, ml.yaml says {cfg.label}")
    fregistry = feature_registry or load_feature_registry()
    features = resolve_features(fregistry, cfg, manifest)
    test = read_frame(store, paths, "test")
    if test.is_empty():
        raise MlInputError("the test frame is empty")
    label = f"y_{cfg.label}"
    y = test[label].cast(pl.Float64).to_numpy()
    if not 0 < y.mean() < 1:
        raise MlInputError("the test frame holds one class only")
    groups = test["ccn"].to_numpy()
    states = test["state"].to_numpy() if "state" in test.columns else None
    chosen = train.selected

    # ---- the eval fits on the test frame
    models = {c: read_package(store, prefix) for c, prefix in train.eval_packages.items()}
    for c, model in models.items():
        check_training_months(model, c, windows)
    preds = {c: model.predict_frame(test) for c, model in models.items()}
    probs: dict[str, np.ndarray] = {c: preds[c]["p"].to_numpy() for c in models}
    raw_selected = _sigmoid(preds[chosen]["logit_raw"].to_numpy())

    # ---- threshold on the selected candidate's OOF predictions (never on test)
    oof = _read_parquet(store, train.oof["eval"])
    threshold = metrics.max_f1_threshold(oof["y"].to_numpy(), oof[f"p_{chosen}"].to_numpy())

    # ---- baselines and the champion's eval fit, on the same rows
    base_scores = baselines.baseline_scores(test)
    champ = champion_ref(registry, cfg)
    if champ is not None and registry is not None:
        champ_model = registry.load_model(champ.eval_model_uri)
        probs[CHAMPION] = champ_model.predict_frame(test)["p"].to_numpy()

    # ---- paired bootstrap on shared facility draws
    boot = facility_bootstrap(
        y,
        {**probs, **base_scores},
        groups,
        _headline,
        replicates=cfg.evaluation.bootstrap.replicates,
        seed=cfg.evaluation.bootstrap.seed,
    )

    def intervals(name: str) -> dict[str, Interval]:
        out = {}
        for m in HEADLINE:
            lo, hi = boot.ci(name, m)
            out[m] = Interval(estimate=boot.estimates[name][m], ci_low=lo, ci_high=hi)
        return out

    candidates = {
        c: _score_report(
            y,
            probs[c],
            cfg,
            probability=True,
            threshold=threshold,
            states=states,
            intervals=intervals(c),
        )
        for c in models
    }
    baseline_reports = {
        b: _score_report(y, s, cfg, probability=False, intervals=intervals(b))
        for b, s in base_scores.items()
    }
    comparisons = {
        comparison_key("pr_auc", chosen, RULE): Interval(**boot.difference(chosen, RULE, "pr_auc"))
    }
    for c in models:
        if c != chosen:
            comparisons[comparison_key("pr_auc", chosen, c)] = Interval(
                **boot.difference(chosen, c, "pr_auc")
            )
    champion_eval = None
    if CHAMPION in probs:
        champion_eval = _score_report(
            y,
            probs[CHAMPION],
            cfg,
            probability=True,
            threshold=threshold,
            intervals=intervals(CHAMPION),
        )
        for m in ("pr_auc", "brier"):
            comparisons[comparison_key(m, chosen, CHAMPION)] = Interval(
                **boot.difference(chosen, CHAMPION, m)
            )

    # ---- slices (§7)
    bias_cfg = slices.slice_config(cfg)
    report_slices = list(dict.fromkeys([*bias_cfg.slices.single, *cfg.bias.gate_slices]))
    table = slices.slice_table(
        test, y, probs[chosen], slices=report_slices, bias_cfg=bias_cfg, registry=fregistry
    )
    failures = slices.calibration_failures(table, cfg)
    vs_champion = None
    if CHAMPION in probs:
        gated_values = slices.gated(table, cfg).filter(pl.col("slice").is_in(cfg.bias.gate_slices))
        vs_champion = slices.slice_comparisons(
            test,
            y,
            probs[chosen],
            probs[CHAMPION],
            values=gated_values,
            bias_cfg=bias_cfg,
            registry=fregistry,
            replicates=cfg.evaluation.bootstrap.replicates,
            seed=cfg.evaluation.bootstrap.seed,
        )

    # ---- sanity (§6.4)
    additivity = sanity.shap_additivity(models[chosen], test)
    run_shuffled = (
        shuffled_label
        if shuffled_label is not None
        else not recipe_evaluated(meta, train.recipe_hash)
    )
    eval_frame = read_frame(store, paths, "eval_train") if (run_shuffled or mitigation) else None
    shuffled_roc = (
        sanity.shuffled_label_roc(eval_frame, test, features, cfg)
        if eval_frame is not None and run_shuffled
        else None
    )

    # ---- bias mitigation trade-off on the eval window's OOF predictions (§7), never on test
    tradeoff = None
    if mitigation and eval_frame is not None:
        keyed = eval_frame.select("ccn", "t").join(
            oof.select("ccn", "t", f"p_{chosen}"),
            on=["ccn", "t"],
            how="left",
            maintain_order="left",
        )
        if keyed[f"p_{chosen}"].null_count():
            raise MlInputError("oof/eval.parquet does not cover the eval frame's rows")
        tradeoff = mitigation_tradeoff(
            eval_frame,
            features,
            cfg,
            chosen,
            keyed[f"p_{chosen}"].to_numpy(),
            bias_cfg=bias_cfg,
            registry=fregistry,
        )
    uncleared = sorted({f.feature for f in manifest.leakage_flags if not f.cleared})

    oof_pr: dict[str, float] = {r.candidate: r.oof_pr_auc for r in selection.candidates}
    shap_table = top_features(models[chosen], test, n=None)
    report = EvaluationReport(
        ml_run_id=ml_run_id,
        gold_run_id=manifest.gold_run_id,
        dataset_version=manifest.dataset_version,
        recipe_hash=train.recipe_hash,
        label=cfg.label,
        selected=chosen,
        gold_verified=manifest.verified,
        allow_unverified=cfg.allow_unverified,
        test_months=list(windows.test.months),
        development_months=[
            m for m in windows.test.months if m in cfg.evaluation.development_months
        ],
        rows=test.height,
        facilities=test["ccn"].n_unique(),
        positives=int(y.sum()),
        threshold=threshold,
        candidates=candidates,
        baselines=baseline_reports,
        champion=champ,
        champion_eval=champion_eval,
        oof_pr_auc=oof_pr,
        oof_test_gap={
            c: oof_pr[c] - (candidates[c].ranking["pr_auc"] or float("nan"))
            for c in models
            if c in oof_pr
        },
        per_month=_per_month(test, y, probs[chosen], cfg),
        reliability={
            "raw": metrics.reliability_table(y, raw_selected, cfg.evaluation.ece_bins),
            "calibrated": metrics.reliability_table(y, probs[chosen], cfg.evaluation.ece_bins),
        },
        comparisons=comparisons,
        slices=table.to_dicts(),
        calibration_failures=[s.strip() for s in slices.describe(failures).split(";") if s.strip()],
        slices_vs_champion=vs_champion,
        top_features=shap_table.head(TOP_FEATURES).to_dicts(),
        sanity=Sanity(
            uncleared_leakage_flags=uncleared,
            shap_additive=additivity.holds,
            shap_max_abs_error=additivity.max_abs_error,
            shap_rows=additivity.rows,
            shuffled_label_roc=shuffled_roc,
        ),
        bootstrap_replicates=boot.replicates,
        mitigation=tradeoff,
        flags=list(train.flags),
    )
    scores = test.select("ccn", "t").with_columns(
        pl.Series("y", y.astype(np.int8)),
        *[pl.Series(f"p_{k}", v) for k, v in {**probs, **base_scores}.items()],
    )
    return EvaluateOutput(report, scores, table, shap_table, raw_selected)


def run_evaluate(
    store: ObjectStore,
    *,
    ml_run_id: str,
    cfg: MlConfig,
    registry: Registry | None = None,
    feature_registry: FeatureRegistry | None = None,
    meta: MetadataStore | None = None,
    shuffled_label: bool | None = None,
    mitigation: bool = True,
) -> EvaluateOutput:
    """The ``evaluate`` task: ``evaluate_run``, then ``evaluation.json`` and the report files in the run folder
    and, with tracking on, one MLflow run with the metrics, plots and files (§14.2)."""
    from ..training import mlflow_log
    from . import plots

    started = time.perf_counter()
    out = evaluate_run(
        store,
        ml_run_id=ml_run_id,
        cfg=cfg,
        registry=registry,
        feature_registry=feature_registry,
        meta=meta,
        shuffled_label=shuffled_label,
        mitigation=mitigation,
    )
    paths = RunPaths.of(cfg, ml_run_id)
    report_dir = report_prefix(paths)
    files = {
        "slices.csv": _csv(out.slice_table),
        "shap_mean_abs.csv": _csv(out.shap_mean_abs),
        "per_month.csv": _csv(pl.DataFrame(out.report.per_month)),
        **plots.render_all(out),
    }
    for name, data in files.items():
        store.put_bytes(f"{report_dir}{name}", data, overwrite=True)
    store.put_bytes(
        paths.evaluation, out.report.model_dump_json(indent=2).encode("utf-8"), overwrite=True
    )

    manifest = read_input_manifest(store, paths)
    tags = mlflow_log.base_tags(
        manifest,
        ml_run_id=ml_run_id,
        recipe_hash=out.report.recipe_hash,
        decision=read_decision(store, paths),
    )
    with mlflow_log.task_run(cfg, "evaluate", tags) as run_id:
        if run_id is not None:
            mlflow_log.log_metrics(evaluation_metrics(out.report))
            mlflow_log.log_json_artifact(out.report.model_dump(mode="json"), "evaluation.json")
            plots.log_files(files)
    log.info(
        "evaluate %s: %s test PR-AUC %.4f in %.0fs",
        ml_run_id,
        out.report.selected,
        out.report.candidates[out.report.selected].ranking["pr_auc"] or float("nan"),
        time.perf_counter() - started,
    )
    return out


def report_prefix(paths: RunPaths) -> str:
    return f"{paths.run_dir}report/"


def _csv(df: pl.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.write_csv(buf)
    return buf.getvalue()


def evaluation_metrics(report: EvaluationReport) -> dict[str, float]:
    """The MLflow metrics (§14.2): the selected candidate's test metrics with CI bounds, every candidate's and
    baseline's PR-AUC, the comparisons, and the slice summary."""
    out: dict[str, float] = {}
    sel = report.candidates[report.selected]
    for group in (sel.ranking, sel.probability or {}, sel.classification or {}):
        for k, v in group.items():
            if v is not None:
                out[f"test_{k}"] = float(v)
    for k, iv in sel.intervals.items():
        out[f"test_{k}_ci_low"], out[f"test_{k}_ci_high"] = iv.ci_low, iv.ci_high
    for c, r in report.candidates.items():
        if r.ranking.get("pr_auc") is not None:
            out[f"candidate_{c}_pr_auc"] = float(r.ranking["pr_auc"])
    for b, r in report.baselines.items():
        if r.ranking.get("pr_auc") is not None:
            out[f"baseline_{b}_pr_auc"] = float(r.ranking["pr_auc"])
    for key, iv in report.comparisons.items():
        name = key.replace(":", "_diff_").replace("-", "_vs_")
        out[name], out[f"{name}_ci_low"], out[f"{name}_ci_high"] = (
            iv.estimate,
            iv.ci_low,
            iv.ci_high,
        )
    out["slices_miscalibrated"] = float(len(report.calibration_failures))
    if report.mitigation is not None:
        for v in report.mitigation.variants:
            out[f"mitigation_{v.variant}_oof_pr_auc"] = v.pr_auc
            out[f"mitigation_{v.variant}_oof_brier"] = v.brier
            if v.max_abs_oe is not None:
                out[f"mitigation_{v.variant}_max_abs_oe"] = v.max_abs_oe
    if report.sanity.shuffled_label_roc is not None:
        out["shuffled_label_roc"] = report.sanity.shuffled_label_roc
    return out
