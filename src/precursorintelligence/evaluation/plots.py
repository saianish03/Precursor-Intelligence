"""The evaluation's plots (docs/ML_PIPELINE_SPEC.md §6.5), as PNG bytes for the run folder and MLflow: PR and ROC
curves, the reliability diagram before and after calibration, the lift chart, the confusion matrix and the
candidate comparison.

matplotlib comes with the ``bias`` extra; without it the plots are skipped (logged), never the evaluation.
"""

from __future__ import annotations

import io
import logging
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .evaluate import EvaluateOutput

log = logging.getLogger(__name__)

TOP_FRACTIONS = np.linspace(0.01, 1.0, 100)


def _png(fig: Any) -> bytes:
    import matplotlib.pyplot as plt

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def _curves(out: EvaluateOutput) -> tuple[bytes, bytes]:
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, roc_curve

    s = out.test_scores
    y = s["y"].to_numpy()
    names = [c[2:] for c in s.columns if c.startswith("p_")]
    pr, ax_pr = plt.subplots(figsize=(6, 5))
    roc, ax_roc = plt.subplots(figsize=(6, 5))
    for name in names:
        p = s[f"p_{name}"].to_numpy()
        prec, rec, _ = precision_recall_curve(y, p)
        fpr, tpr, _ = roc_curve(y, p)
        width = 2.2 if name == out.report.selected else 1.0
        ax_pr.plot(rec, prec, label=name, linewidth=width)
        ax_roc.plot(fpr, tpr, label=name, linewidth=width)
    ax_pr.axhline(y.mean(), color="grey", linestyle=":", label="base rate")
    ax_pr.set(xlabel="recall", ylabel="precision", title="Precision-recall (test)")
    ax_roc.plot([0, 1], [0, 1], color="grey", linestyle=":")
    ax_roc.set(xlabel="false positive rate", ylabel="true positive rate", title="ROC (test)")
    ax_pr.legend(fontsize=8)
    ax_roc.legend(fontsize=8)
    return _png(pr), _png(roc)


def _reliability(out: EvaluateOutput) -> bytes:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    for key, style in (("raw", "o--"), ("calibrated", "o-")):
        rows = out.report.reliability[key]
        ax.plot(
            [float(r["mean_predicted"]) for r in rows],
            [float(r["observed_rate"]) for r in rows],
            style,
            label="before Platt" if key == "raw" else "after Platt",
        )
    ax.plot([0, 1], [0, 1], color="grey", linestyle=":")
    ax.set(
        xlabel="mean predicted probability",
        ylabel="observed rate",
        title=f"Reliability, {out.report.selected} (test, equal-mass bins)",
    )
    ax.legend()
    return _png(fig)


def _lift(out: EvaluateOutput) -> bytes:
    import matplotlib.pyplot as plt

    s = out.test_scores
    y = s["y"].to_numpy().astype(float)
    base = y.mean()
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for name in [c[2:] for c in s.columns if c.startswith("p_")]:
        order = np.argsort(-s[f"p_{name}"].to_numpy(), kind="stable")
        hits = np.cumsum(y[order])
        ks = np.maximum(1, np.round(TOP_FRACTIONS * len(y)).astype(int))
        ax.plot(TOP_FRACTIONS * 100, hits[ks - 1] / ks / base, label=name)
    ax.axhline(1, color="grey", linestyle=":")
    ax.set(
        xlabel="top % of homes flagged", ylabel="lift (precision / base rate)", title="Lift (test)"
    )
    ax.legend(fontsize=8)
    return _png(fig)


def _confusion(out: EvaluateOutput) -> bytes:
    import matplotlib.pyplot as plt

    c = out.report.candidates[out.report.selected].classification or {}
    m = np.array([[c.get("tn") or 0, c.get("fp") or 0], [c.get("fn") or 0, c.get("tp") or 0]])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(m, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{int(m[i, j]):,}", ha="center", va="center")
    ax.set_xticks([0, 1], ["not flagged", "flagged"])
    ax.set_yticks([0, 1], ["no G+ citation", "G+ citation"])
    ax.set_title(f"Confusion matrix at p >= {out.report.threshold:.3f} (OOF max-F1)")
    return _png(fig)


def _candidates(out: EvaluateOutput) -> bytes:
    import matplotlib.pyplot as plt

    r = out.report
    rows = [(f"{k}", v.intervals["pr_auc"]) for k, v in r.candidates.items()]
    rows += [(f"{k} (baseline)", v.intervals["pr_auc"]) for k, v in r.baselines.items()]
    if r.champion_eval is not None:
        rows.append(("champion (eval fit)", r.champion_eval.intervals["pr_auc"]))
    fig, ax = plt.subplots(figsize=(6.5, 0.5 * len(rows) + 1.2))
    ys = np.arange(len(rows))
    est = np.array([iv.estimate for _, iv in rows])
    lo = est - np.array([iv.ci_low for _, iv in rows])
    hi = np.array([iv.ci_high for _, iv in rows]) - est
    ax.errorbar(est, ys, xerr=[lo, hi], fmt="o", capsize=3)
    ax.set_yticks(ys, [n for n, _ in rows])
    ax.invert_yaxis()
    ax.set(xlabel="test PR-AUC (95% facility-bootstrap CI)", title="Candidates and baselines")
    return _png(fig)


def render_all(out: EvaluateOutput) -> dict[str, bytes]:
    """Every plot as ``{file name: PNG bytes}``; empty without matplotlib."""
    try:
        import matplotlib

        matplotlib.use("Agg")
    except ImportError:
        log.warning("matplotlib is not installed (the bias extra): no evaluation plots")
        return {}
    pr, roc = _curves(out)
    return {
        "pr_curve.png": pr,
        "roc_curve.png": roc,
        "reliability.png": _reliability(out),
        "lift.png": _lift(out),
        "confusion_matrix.png": _confusion(out),
        "candidates.png": _candidates(out),
    }


def log_files(files: Mapping[str, bytes], artifact_path: str = "report") -> None:
    """Log ``{name: bytes}`` as MLflow artifacts of the active run (a no-op without one)."""
    import mlflow

    from ..training.mlflow_log import tracking_enabled

    if not files or not tracking_enabled() or mlflow.active_run() is None:
        return
    with tempfile.TemporaryDirectory() as tmp:
        for name, data in files.items():
            (Path(tmp) / name).write_bytes(data)
        mlflow.log_artifacts(tmp, artifact_path=artifact_path)
