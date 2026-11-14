"""``model_card.md`` (docs/ML_PIPELINE_SPEC.md §6.5): what a reviewer reads at Checkpoint ML-A (§17.3).

Data (gold run, windows, evaluation gap, development months), the recipe, the selected candidate and why, test
metrics with CIs against the baselines and the champion's eval fit, calibration, the bias summary, the top global
features, the regime and drift flags, the known limitations (§21) and the gate outcome. Every number comes from
``evaluation.json`` and ``train.json``; nothing is recomputed here.
"""

from __future__ import annotations

from typing import Any

from ..training.contracts import InputManifest, TrainResult, Windows
from .gates import GateReport, Interval
from .report import EvaluationReport, comparison_key
from .slices import GEOGRAPHIC

LIMITATIONS = (
    "Label lag: outcomes mature about 20 months after `t`, so the newest training data is always ~20 months old.",
    "Regime changes (2025-07 ratings, 2025-10 fines) are not in the training windows yet; the affected flags are "
    "excluded and skew is monitored.",
    "2025-02 was used by PR #49 to choose the recipe (a development month): honest numbers start with 2025-03.",
    "Empty CMS months (2024-12, 2025-01) shrink the test window to one month; the intervals reflect it.",
    "Within-state ranking is weak (state is a large share of the signal); within-state metrics are reported.",
    "Per-state miscalibration is reported and gated against regression only.",
    "About 68 % of positives come from complaint surveys alone, which public data predicts poorly.",
)


def _f(v: Any, digits: int = 3) -> str:
    if v is None:
        return "–"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def _ci(iv: Interval | None, digits: int = 3, sign: bool = False) -> str:
    if iv is None:
        return "–"
    fmt = f"{{:{'+' if sign else ''}.{digits}f}}"
    return f"{fmt.format(iv.estimate)} [{fmt.format(iv.ci_low)}, {fmt.format(iv.ci_high)}]"


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    return [
        "| " + " | ".join(header) + " |",
        "|" + "|".join("---" for _ in header) + "|",
        *("| " + " | ".join(r) + " |" for r in rows),
    ]


def _mitigation(r: EvaluationReport) -> list[str]:
    """§7: reweighting vs none on the eval window's OOF predictions (never test)."""
    m = r.mitigation
    if m is None:
        return [
            "",
            "## Bias mitigation",
            "- The reweighting trade-off was not run for this evaluation.",
        ]
    out = [
        "",
        f"## Bias mitigation: reweighting ({m.scheme} on `{m.weight_slice}`), eval-window OOF, {m.candidate}",
        f"- {m.rows:,} rows; effective sample size with the weights {m.effective_sample_size:,.0f}. Adopting it is a "
        "recipe change (`ml.yaml: bias.mitigation: reweight`) that goes through the gates.",
        *_table(
            [
                "variant",
                "OOF PR-AUC",
                "OOF Brier",
                "OOF ECE",
                "max abs. O−E (gated slices)",
                "miscalibrated",
            ],
            [
                [
                    v.variant,
                    _f(v.pr_auc),
                    _f(v.brier, 4),
                    _f(v.ece, 4),
                    _f(v.max_abs_oe),
                    str(v.miscalibrated),
                ]
                for v in m.variants
            ],
        ),
    ]
    rows = [x for x in m.by_value if x["slice"] == m.weight_slice]
    if rows:
        out.append(
            f"- O−E by `{m.weight_slice}` (none → reweight): "
            + ", ".join(
                f"{x['value']} {_f(x.get('oe_none'))} → {_f(x.get('oe_reweight'))}" for x in rows
            )
        )
    return out


def render_model_card(
    report: EvaluationReport,
    outcome: GateReport,
    train: TrainResult,
    windows: Windows,
    manifest: InputManifest,
    *,
    version: int | None,
    action: str,
) -> str:
    r, sel = report, report.selected
    s = r.candidates[sel]
    prob = s.probability or {}
    lines = [
        f"# Model card: precursor_y_{r.label}, {sel}"
        + (f" (version {version})" if version is not None else " (not registered)"),
        "",
        f"**Gate outcome: {action.upper()}**"
        + (f" (failed: {', '.join(outcome.failed)})" if outcome.failed else " (no gate failed)"),
        "",
        "## Data",
        f"- Gold run `{manifest.gold_run_id}`, dataset_version `{manifest.dataset_version}`, "
        f"verified: {'yes' if manifest.verified else 'no'}; V = {windows.v}, V_L = {windows.v_labels}",
        f"- Eval window {windows.eval.first}..{windows.eval.last} ({len(windows.eval.months)} months, "
        f"gap `{windows.eval_gap}`); refit window {windows.refit.first}..{windows.refit.last}",
        f"- Test months {', '.join(r.test_months)}: {r.rows:,} rows, {r.facilities:,} homes, "
        f"{r.positives:,} positives (base rate {r.positives / r.rows:.3f})",
    ]
    if r.development_months:
        lines.append(
            f"- **Development months in the test window: {', '.join(r.development_months)}** (chosen on by "
            "earlier work; not an honest hold-out)"
        )
    lines += [
        f"- {len(train.features)} features; excluded: {', '.join(train.excluded_features) or 'none'}",
        "",
        "## Recipe and selection",
        f"- recipe_hash `{r.recipe_hash[:16]}`; candidates selected on out-of-fold PR-AUC after the OOF bias check",
        *(
            f"- {c}: OOF PR-AUC {_f(v)}, test PR-AUC {_f(r.candidates[c].ranking.get('pr_auc'))}"
            f" (OOF − test {_f(r.oof_test_gap.get(c), 3)})"
            for c, v in r.oof_pr_auc.items()
            if c in r.candidates
        ),
        "",
        "## Test metrics (95 % facility-bootstrap CIs)",
        *_table(
            ["score", "PR-AUC", "lift", "ROC-AUC", "Brier"],
            [
                [
                    name + (" (selected)" if name == sel else ""),
                    _ci(rep.intervals.get("pr_auc")),
                    _ci(rep.intervals.get("pr_auc_lift"), 2),
                    _ci(rep.intervals.get("roc_auc")),
                    _ci(rep.intervals.get("brier")) if rep.probability else "–",
                ]
                for name, rep in [
                    *r.candidates.items(),
                    *((f"{b} (baseline)", v) for b, v in r.baselines.items()),
                    *([("champion eval fit", r.champion_eval)] if r.champion_eval else []),
                ]
            ],
        ),
        "",
        f"- PR-AUC vs the prior-G+ rule: {_ci(r.comparisons.get(comparison_key('pr_auc', sel, 'prior_gplus_rule')), sign=True)}",
    ]
    if r.champion is not None:
        lines += [
            f"- vs the champion (version {r.champion.version}, its eval fit on the same rows): PR-AUC "
            f"{_ci(r.comparison('pr_auc', sel, 'champion'), sign=True)}, Brier "
            f"{_ci(r.comparison('brier', sel, 'champion'), 4, sign=True)}",
        ]
    else:
        lines.append("- No champion yet: G5 and G7(a) were skipped.")
    top = [f"{pct}%" for pct in ("1", "2", "5", "10", "20")]
    lines += [
        "- Top-k: "
        + ", ".join(f"precision@{t} {_f(s.ranking.get(f'precision_at_{t[:-1]}'))}" for t in top),
        f"- Within-state (mean over states): PR-AUC {_f((s.within_state or {}).get('pr_auc'))}, "
        f"precision@10% {_f((s.within_state or {}).get('precision_at_10'))}",
        f"- At the OOF max-F1 threshold p ≥ {r.threshold:.3f}: precision "
        f"{_f((s.classification or {}).get('precision'))}, recall {_f((s.classification or {}).get('recall'))}, "
        f"F1 {_f((s.classification or {}).get('f1'))}, MCC {_f((s.classification or {}).get('mcc'))}",
        "",
        "## Calibration",
        f"- Test ECE {_f(prob.get('ece'), 4)}, slope {_f(prob.get('calibration_slope'))}, intercept "
        f"{_f(prob.get('calibration_intercept'))}; mean p {_f(prob.get('mean_predicted'))} vs observed "
        f"{_f(prob.get('observed_rate'))}; Brier skill {_f(prob.get('brier_skill'))}",
        *(
            f"- Platt {k}: a = {v.a:.3f}, b = {v.b:.3f}{' (identity fallback)' if v.identity else ''}"
            for k, v in train.calibration.items()
            if k.endswith(f"/{sel}")
        ),
        "",
        "## Bias (calibration within groups; non-geographic slices, largest |O−E| first)",
        f"- {len(r.calibration_failures)} gated slice value(s) clearly miscalibrated (O−E CI outside ±0.05)"
        + (": " + "; ".join(r.calibration_failures) if r.calibration_failures else ""),
    ]
    rows = [x for x in r.slices if x["slice"] not in GEOGRAPHIC and x.get("reliable")]
    worst = sorted(rows, key=lambda x: -abs(x.get("oe") or 0))[:8]
    lines += _table(
        ["slice", "value", "homes", "base rate", "mean p", "O−E [CI]", "PR-AUC"],
        [
            [
                x["slice"],
                str(x["value"]),
                f"{x['facilities']:,}",
                _f(x.get("base_rate")),
                _f(x.get("mean_p")),
                f"{_f(x.get('oe'))} [{_f(x.get('oe_ci_low'))}, {_f(x.get('oe_ci_high'))}]",
                _f(x.get("pr_auc")),
            ]
            for x in worst
        ],
    )
    states = sorted(
        (x for x in r.slices if x["slice"] == "state" and x.get("reliable")),
        key=lambda x: -abs(x.get("oe") or 0),
    )[:5]
    if states:
        lines.append(
            "- States with the largest |O−E| (reported, not gated; empirical-Bayes shrinkage is Phase 2): "
            + ", ".join(f"{x['value']} {x['oe']:+.3f}" for x in states)
        )
    lines += _mitigation(r)
    lines += [
        "",
        "## Top features (mean absolute contribution, calibrated log-odds, test frame)",
        *_table(
            ["#", "feature", "mean abs. contribution"],
            [
                [str(i + 1), f"`{t['feature']}`", _f(t["mean_abs"], 4)]
                for i, t in enumerate(r.top_features)
            ],
        ),
        "",
        "## Flags",
        f"- Sanity: SHAP additivity {'holds' if r.sanity.shap_additive else 'FAILS'} "
        f"(max error {_f(r.sanity.shap_max_abs_error, 8)} on {r.sanity.shap_rows} rows); shuffled-label "
        f"ROC-AUC {_f(r.sanity.shuffled_label_roc) if r.sanity.shuffled_label_roc is not None else 'not run (recipe already evaluated)'}; "
        f"uncleared leakage flags: {', '.join(r.sanity.uncleared_leakage_flags) or 'none'}",
        f"- Run flags: {', '.join(r.flags) or 'none'}",
        *(f"- Regime flag `{f.feature}`: {f.detail}" for f in manifest.regime_flags),
        "",
        "## Gates",
        *_table(
            ["gate", "status", "reason"],
            [[g.gate, g.status.upper(), g.reason.replace("|", "/")] for g in outcome.results],
        ),
        "",
        "## Known limitations",
        *(f"- {x}" for x in LIMITATIONS),
        "",
    ]
    return "\n".join(lines)
