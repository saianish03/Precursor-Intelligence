"""Run every check and write ``reports/bias/``: CSVs, summary.json, sample_weights.parquet, charts and
bias_report.md. Outputs are deterministic (sorted rows, fixed float format); only ``generated_at`` in
summary.json changes between identical runs.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from ..common.config import FeatureRegistry
from ..gold.dataset import git_state
from . import BiasInputError
from .config import BiasConfig
from .disparity import label_rates, status_totals
from .io import GoldData, apply_scope, require_columns, scope_any
from .missingness import missingness_by_slice, scan_features
from .mitigation import ess_report, sample_weights
from .rater_bias import rater_bias_by_state
from .representation import monthly_counts, representation
from .skew import TIME_THIRDS, skew_by_slice
from .slicefinder import find_problematic, n_tests, top_k
from .slices import build_slice_columns, enumerate_slices

log = logging.getLogger(__name__)

FLOAT_PRECISION = 6
CSV_FILES = (
    "slice_summary.csv",
    "slice_by_month.csv",
    "label_rates.csv",
    "problematic_slices.csv",
    "rater_bias_by_state.csv",
    "missingness_by_slice.csv",
    "skew_by_slice.csv",
)
SECTIONS = (
    "## 1. Summary",
    "## 2. Representation",
    "## 3. Label-rate disparity",
    "## 4. Top problematic slices (SliceFinder)",
    "## 5. Inspector (rater) effects by state",
    "## 6. Missingness disparity",
    "## 7. Skew over time per slice",
    "## 8. Mitigation decisions",
    "## 9. Limitations",
    "## 10. Future work (ML phase)",
)


@dataclass
class BiasResult:
    out_dir: Path
    summary: dict[str, Any]
    tables: dict[str, pl.DataFrame]


def _round(x: Any) -> Any:
    if isinstance(x, float):
        return None if math.isnan(x) else round(x, FLOAT_PRECISION)
    if isinstance(x, dict):
        return {k: _round(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_round(v) for v in x]
    return x


def _write_csv(df: pl.DataFrame, path: Path) -> None:
    df.write_csv(path, float_precision=FLOAT_PRECISION, line_terminator="\n")


def run_bias_report(
    gold: GoldData,
    cfg: BiasConfig,
    *,
    labels: Sequence[str],
    scope: str,
    out_dir: Path,
    registry: FeatureRegistry,
    config_path: Path,
    charts: bool = True,
) -> BiasResult:
    """Run the checks on ``gold`` in ``scope`` and write every output under ``out_dir``."""
    labels = list(labels)
    require_columns(gold.df, cfg, labels)
    df = build_slice_columns(gold.df, cfg, registry)
    # every label check reads its own scope rows (scope train: split_<label> == train); the label-free checks
    # (representation, missingness, the slice list) read the rows in scope for any of the labels
    rows = scope_any(df, cfg, scope, labels)
    if rows.height == 0:
        raise BiasInputError(f"scope {scope} leaves no rows")
    scoped = {lb: apply_scope(df, cfg, scope, lb) for lb in labels}
    slices = enumerate_slices(rows, cfg)
    singles = list(cfg.slices.single)
    totals = {lb: status_totals(scoped[lb], cfg, [lb])[lb] for lb in labels}
    for label, c in totals.items():
        log.info(
            "bias %s: %d rows in scope %s, labeled %d (positive %d, negative %d), excluded immature %d, censored %d",
            label,
            scoped[label].height,
            scope,
            c["used"],
            c["positive"],
            c["negative"],
            c["immature"],
            c["censored"],
        )

    summary_tbl = representation(rows, slices, cfg)
    by_month = monthly_counts(rows, singles)
    no_labels = all(c["used"] == 0 for c in totals.values())
    if no_labels:
        log.warning("bias: no labeled rows in scope %s: representation and missingness only", scope)
    rates = pl.concat([label_rates(scoped[lb], slices, cfg, [lb]) for lb in labels])
    ranked = find_problematic(rates, cfg)
    tested = n_tests(rates)
    log.info("bias: %d slices, %d (label, slice) tests", len(slices), tested)
    rater_parts = [rater_bias_by_state(scoped[lb], cfg, [lb]) for lb in labels]
    rater = pl.concat([t for t, _ in rater_parts])
    rater_summary = {k: v for _, s in rater_parts for k, v in s.items()}
    features = scan_features(rows, registry)
    missing = missingness_by_slice(rows, singles, features, cfg)
    # skew compares the configured train and test splits itself, so it reads every row, not the scope
    skew_features = [f for f in features if not f.endswith("_fn")]
    skew, skew_method = skew_by_slice(df, singles, skew_features, cfg)
    # weights read only split_<weight_label> == train rows, whatever the scope (mitigation.py)
    weights = sample_weights(df, cfg)
    ess = ess_report(weights, df, cfg)

    out_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "slice_summary.csv": summary_tbl,
        "slice_by_month.csv": by_month,
        "label_rates.csv": rates,
        "problematic_slices.csv": top_k(ranked, cfg.thresholds.top_k),
        "rater_bias_by_state.csv": rater,
        "missingness_by_slice.csv": missing,
        "skew_by_slice.csv": skew,
    }
    for name, t in tables.items():
        _write_csv(t, out_dir / name)
    weights.write_parquet(out_dir / "sample_weights.parquet")

    commit, dirty = git_state()
    summary: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "gold": {"source": gold.source, **gold.pointer},
        "config_path": config_path.as_posix(),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "git_commit": commit,
        "git_dirty": dirty,
        "scope": scope,
        "labels": labels,
        "rows_in_scope": rows.height,
        "facilities_in_scope": rows["ccn"].n_unique(),
        "months": {"first": rows["t"].min(), "last": rows["t"].max(), "n": rows["t"].n_unique()},
        "label_rows": totals,
        "no_labeled_rows": no_labels,
        "inference": "by home: cluster-robust (ccn) variances; Fisher's exact on rows for observed cells < 5",
        "n_slices": len(slices),
        "n_tests": tested,
        "thresholds": cfg.thresholds.model_dump(),
        "skew_method": skew_method,
        "rater_bias": rater_summary,
        "mitigation": {
            **ess,
            "gold_v": gold.pointer.get("v"),
            "gold_run_id": gold.pointer.get("run_id"),
        },
        "flagged": {
            "small_slices": int(summary_tbl["small_slice"].sum()),
            "label_rate_ratio": int(rates["flag_ratio"].sum()),
            "problematic": int(ranked["problematic"].sum()) if ranked.height else 0,
            "rater_states": int(rater["flag"].sum()),
            "missingness": int(missing["flag"].sum()),
            "skew": int(skew["flag"].sum()),
        },
    }
    (out_dir / "summary.json").write_text(
        json.dumps(_round(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if charts:
        write_charts(out_dir / "charts", rates, summary_tbl, rater, cfg, labels)
    decisions = propose_decisions(summary_tbl, rates, ranked, rater_summary, missing, skew, cfg)
    _write_csv(decisions, out_dir / "mitigation_decisions.csv")
    (out_dir / "bias_report.md").write_text(
        render_markdown(summary, tables, ranked, decisions, cfg, charts), encoding="utf-8"
    )
    return BiasResult(out_dir=out_dir, summary=summary, tables=tables)


# --------------------------------------------------------------------------- mitigation decisions

DECISIONS = {
    "reweight": "re-weight (sample_weights.parquet)",
    "merge": "merge the slice (too small to estimate)",
    "within_state": "state-relative features / within-state evaluation",
    "more_history": "collect more history",
    "no_action": "documented, no action",
}


def propose_decisions(
    summary_tbl: pl.DataFrame,
    rates: pl.DataFrame,
    ranked: pl.DataFrame,
    rater_summary: dict[str, Any],
    missing: pl.DataFrame,
    skew: pl.DataFrame,
    cfg: BiasConfig,
) -> pl.DataFrame:
    """One proposed decision per finding. These are proposals: the team signs them off in the report."""
    rows: list[dict[str, str]] = []
    weight_slice = cfg.mitigation.weight_slice
    if ranked.height:
        for r in ranked.filter(pl.col("problematic")).iter_rows(named=True):
            finding = f"{r['label']}: {r['slice']} = {r['value']} (h = {r['cohens_h']:+.2f})"
            if r["slice"].split(" x ")[0] in ("state", "cms_region"):
                d, why = (
                    "within_state",
                    "inspection practice differs by state; compare homes within a state",
                )
            elif weight_slice in r["slice"].split(" x "):
                d, why = "reweight", f"{weight_slice} is the configured weight slice"
            else:
                d, why = (
                    "no_action",
                    "a real risk difference between kinds of homes is not unfairness; check model per-slice metrics in the ML phase",
                )
            rows.append(
                {
                    "check": "slicefinder",
                    "finding": finding,
                    "decision": DECISIONS[d],
                    "reason": why,
                }
            )
    small_flagged = rates.join(
        summary_tbl.filter(pl.col("small_slice")).select("slice", "value"), on=["slice", "value"]
    ).filter(pl.col("flag_ratio"))
    for r in (
        small_flagged.select("slice", "value").unique().sort("slice", "value").iter_rows(named=True)
    ):
        rows.append(
            {
                "check": "label_rate (small slice)",
                "finding": f"{r['slice']} = {r['value']}",
                "decision": DECISIONS["merge"],
                "reason": "too few homes for a stable rate",
            }
        )
    for label, s in sorted(rater_summary.items()):
        for state in s["flagged_states"]:
            rows.append(
                {
                    "check": "rater_bias",
                    "finding": f"{label}: {state}",
                    "decision": DECISIONS["within_state"],
                    "reason": "high label rate together with high inspection exposure",
                }
            )
    for r in missing.filter(pl.col("flag")).iter_rows(named=True):
        rows.append(
            {
                "check": "missingness",
                "finding": f"{r['slice']} = {r['value']}: {r['feature']} ({r['kind']} gap {r['gap']:+.2f})",
                "decision": DECISIONS["no_action"],
                "reason": "CMS does not report it for these homes; the model sees nulls as missing",
            }
        )
    if skew.height:
        for r in skew.filter(pl.col("flag")).iter_rows(named=True):
            rows.append(
                {
                    "check": "skew",
                    "finding": f"{r['slice']} = {r['value']}: {r['feature']} ({r['metric']} {r['distance']:.2f})",
                    "decision": DECISIONS["more_history"],
                    "reason": "the slice's feature moved between train and test; monitor drift",
                }
            )
    return pl.DataFrame(
        rows,
        schema={
            "check": pl.String,
            "finding": pl.String,
            "decision": pl.String,
            "reason": pl.String,
        },
    )


# --------------------------------------------------------------------------- markdown


def _fmt(x: Any) -> str:
    if x is None:
        return "–"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, float):
        return "–" if math.isnan(x) else f"{x:.3f}"
    return str(x)


def _table(df: pl.DataFrame, cols: Sequence[str], limit: int | None = None) -> str:
    if df.height == 0:
        return "_None._\n"
    shown = df.head(limit) if limit else df
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in shown.select(cols).iter_rows():
        lines.append("| " + " | ".join(_fmt(v).replace("|", "/") for v in r) + " |")
    more = (
        f"\n_{df.height - shown.height} more rows in the CSV._\n"
        if df.height > shown.height
        else ""
    )
    return "\n".join(lines) + "\n" + more


def render_markdown(
    summary: dict[str, Any],
    tables: dict[str, pl.DataFrame],
    ranked: pl.DataFrame,
    decisions: pl.DataFrame,
    cfg: BiasConfig,
    charts: bool,
) -> str:
    thr = cfg.thresholds
    rep = tables["slice_summary.csv"]
    rates = tables["label_rates.csv"]
    top = tables["problematic_slices.csv"]
    rater = tables["rater_bias_by_state.csv"]
    missing = tables["missingness_by_slice.csv"]
    skew = tables["skew_by_slice.csv"]
    f = summary["flagged"]
    g = summary["gold"]
    out: list[str] = ["# Data bias report: gold facility × month dataset", ""]
    out += [
        SECTIONS[0],
        "",
        f"- Gold: `{g['source']}` (run `{g.get('run_id')}`, V = {g.get('v')}, dataset sha256 `{str(g.get('dataset_sha256'))[:16]}…`).",
        f"- Scope: `{summary['scope']}`; {summary['rows_in_scope']:,} facility-months from {summary['facilities_in_scope']:,} homes, "
        f"{summary['months']['first']} to {summary['months']['last']} ({summary['months']['n']} months).",
        f"- {summary['n_slices']} slice values ({len(cfg.slices.single)} single attributes, {len(cfg.slices.crossings)} two-way crossings); "
        f"{summary['n_tests']} label × slice tests.",
        "- Label rows used (positive + negative); immature and censored rows are excluded and counted:",
        "",
        "| label | used | positive | negative | immature (excluded) | censored (excluded) |",
        "|---|---|---|---|---|---|",
    ]
    for label, c in summary["label_rows"].items():
        out.append(
            f"| {label} | {c['used']:,} | {c['positive']:,} | {c['negative']:,} | {c['immature']:,} | {c['censored']:,} |"
        )
    out += ["", "**Headline findings**", ""]
    out += headline(summary, rep, ranked, rater)
    out += [""]

    out += [
        SECTIONS[1],
        "",
        f"A slice is **small** when it has fewer than {thr.min_slice_facilities} homes or {thr.min_slice_rows} rows: "
        f"its rates are reported but flagged unreliable. {f['small_slices']} slice values are small.",
        "",
    ]
    singles = rep.filter(~pl.col("slice").str.contains(" x ")).filter(pl.col("slice") != "state")
    out.append(
        _table(
            singles,
            [
                "slice",
                "value",
                "n_rows",
                "n_facilities",
                "share_rows",
                "small_slice",
                "first_t",
                "last_t",
            ],
        )
    )
    small = rep.filter(pl.col("small_slice") & (pl.col("n_rows") > 0))
    out += [
        "",
        "Small slice values with rows:",
        "",
        _table(small, ["slice", "value", "n_rows", "n_facilities"], 40),
    ]
    if charts:
        out += ["", "![slice sizes](charts/slice_sizes.png)", ""]

    out += [
        SECTIONS[2],
        "",
        f"Rate = positives / labeled rows in the slice; complement = all other labeled rows. Flagged when the ratio is "
        f"outside [{thr.ratio_low}, {thr.ratio_high}] (four-fifths rule). Intervals are Wilson at {thr.ci_level:.0%}, sized on the effective number of "
        "independent rows (`n_eff`): inference is by home, since a home's labels are correlated across months.",
        "",
    ]
    for label in summary["labels"]:
        flagged = rates.filter((pl.col("label") == label) & pl.col("flag_ratio")).sort(
            "ratio", descending=True
        )
        out += [f"### {label}", "", f"{flagged.height} flagged slice values.", ""]
        out.append(
            _table(
                flagged,
                ["slice", "value", "n", "rate", "ci_low", "ci_high", "complement_rate", "ratio"],
                25,
            )
        )
        if charts:
            out += [f"![{label} by ownership](charts/rate_{label}_ownership_group.png)", ""]

    out += [
        SECTIONS[3],
        "",
        f"Cohen's h against the complement; z-test with a cluster-robust (by home) variance (Fisher's exact on rows when an observed cell has < 5); "
        f"Benjamini–Hochberg across all tests of a label. **Problematic** = adj p < {thr.fdr_alpha}, |h| ≥ {thr.effect_size_T}, "
        f"and not small. {f['problematic']} problematic (label, slice) pairs. Top {thr.top_k} per label:",
        "",
    ]
    out.append(
        _table(
            top,
            [
                "label",
                "rank",
                "slice",
                "value",
                "n",
                "rate",
                "complement_rate",
                "cohens_h",
                "adj_p",
                "test_used",
                "problematic",
            ],
        )
    )

    out += [
        "",
        SECTIONS[4],
        "",
        "Labels come from state survey agencies. These numbers describe **inspection-intensity effects** "
        "(measurement context), not the quality of the homes in a state.",
        "",
    ]
    for label, s in summary["rater_bias"].items():
        rho = ", ".join(f"{k}: ρ = {_fmt(v['rho'])}" for k, v in s["spearman"].items())
        out.append(
            f"- **{label}**: Spearman across states, label rate vs {rho}. Flagged states "
            f"(label rate and exposure above the {s['quantile']:.0%} quantile): {', '.join(s['flagged_states']) or 'none'}."
        )
    out += [""]
    first = summary["labels"][0]
    out.append(
        _table(
            rater.filter(pl.col("label") == first).sort("label_rate", descending=True),
            [
                "state",
                "n",
                "label_rate",
                *[c for c in rater.columns if c.startswith("mean_")],
                cfg.rater_bias.rate_column,
                "flag",
            ],
            15,
        )
    )
    if charts:
        out += ["", f"![state rate vs exposure](charts/state_rate_vs_exposure_{first}.png)", ""]

    out += [
        SECTIONS[5],
        "",
        f"Null rate in the slice minus the overall null rate (for `*_fn` footnotes: the rate of rows carrying a CMS "
        f"reason code). Flagged at a gap of {thr.missingness_gap} or more: {f['missingness']} slice × feature pairs.",
        "",
    ]
    out.append(
        _table(
            missing.filter(pl.col("flag")).sort("gap", descending=True),
            ["slice", "value", "feature", "kind", "null_rate", "overall_null_rate", "gap"],
            25,
        )
    )

    note = (
        " (no split columns: earliest vs latest third of months)"
        if summary["skew_method"].startswith(TIME_THIRDS)
        else ""
    )
    out += [
        "",
        SECTIONS[6],
        "",
        f"Reference vs comparison: `{summary['skew_method']}`{note}. Jensen–Shannon distance (numeric, flag > {thr.skew_js}); "
        f"L∞ (categorical, flag > {thr.skew_linf}). Slices small in either scope are skipped. {f['skew']} flagged.",
        "",
    ]
    out.append(
        _table(
            skew.filter(pl.col("flag")).sort("distance", descending=True),
            ["slice", "value", "feature", "metric", "distance", "n_ref", "n_cmp"],
            25,
        )
    )

    m = summary["mitigation"]
    out += [
        "",
        SECTIONS[7],
        "",
        f"Sample weights: `{m['scheme']}` on `{m['weight_slice']}`, clipped to {list(cfg.mitigation.clip)}; "
        f"fit on `{m['fit_on']}` rows only; effective sample size {m['ess']:,.0f} of {m['n']:,} rows. The weights are a separate artifact "
        "(`sample_weights.parquet`); gold is unchanged.",
        "",
        "**Status: proposed.** Each finding below has one proposed decision; the team confirms or changes them "
        "before the ML phase uses them (full list in `mitigation_decisions.csv`).",
        "",
    ]
    grouped = (
        decisions.group_by("check", "decision", "reason")
        .agg(
            pl.len().alias("findings"),
            pl.col("finding").sort().head(5).str.join("; ").alias("examples"),
        )
        .sort("check", "decision")
    )
    out.append(_table(grouped, ["check", "findings", "decision", "reason", "examples"]))

    out += [
        "",
        SECTIONS[8],
        "",
        "- No person-level demographics exist in CMS facility data; slices are facility attributes and state inspection "
        "practice, not protected classes. Nothing here claims discrimination.",
        "- Small slices are noisy: their rates are shown but never called problematic.",
        "- Label measurement bias (how hard a state inspects) cannot be fully separated from real differences in risk.",
        "- Data level only: no model has been trained. For an insurer, the harms to watch are opportunity denial "
        "(coverage refused for being small, government-run, or in a strict state) and disproportionate product "
        "failure (the score works worse for some kinds of homes).",
        "",
    ]
    out += [
        SECTIONS[9],
        "",
        "- Per-slice precision at the top 5%, recall, PR-AUC and calibration via Fairlearn `MetricFrame` on model "
        "outputs, reusing these slice definitions (`configs/bias_slices.yaml`).",
        "- Train with and without `sample_weights.parquet` and compare per-slice metrics.",
        "- Within-state evaluation for the states flagged in section 5.",
        "",
    ]
    return "\n".join(out)


def headline(
    summary: dict[str, Any], rep: pl.DataFrame, ranked: pl.DataFrame, rater: pl.DataFrame
) -> list[str]:
    """3-6 bullets, computed from the results (no fixed text)."""
    out = []
    f = summary["flagged"]
    if ranked.height:
        prob = ranked.filter(pl.col("problematic"))
        out.append(
            f"- {prob.height} (label, slice) pairs are problematic after FDR control and the effect-size threshold."
        )
        for label in summary["labels"]:
            p = prob.filter(pl.col("label") == label)
            if p.height:
                r = p.row(0, named=True)
                out.append(
                    f"- {label}: the largest effect is {r['slice']} = {r['value']} "
                    f"(rate {r['rate']:.3f} vs {r['complement_rate']:.3f}, h = {r['cohens_h']:+.2f})."
                )
                break
        states = prob.filter(pl.col("slice") == "state")
        if states.height:
            out.append(
                f"- {states['value'].n_unique()} states differ meaningfully from the rest on at least one label."
            )
    flagged_states = sorted(
        {s for v in summary["rater_bias"].values() for s in v["flagged_states"]}
    )
    out.append(
        f"- Possible inspection-intensity effects in: {', '.join(flagged_states) or 'no state'}."
    )
    out.append(
        f"- {f['small_slices']} small slice values; {f['missingness']} missingness gaps; {f['skew']} skewed slice × feature pairs."
    )
    return out[:6]


# --------------------------------------------------------------------------- charts


def write_charts(
    out: Path,
    rates: pl.DataFrame,
    summary_tbl: pl.DataFrame,
    rater: pl.DataFrame,
    cfg: BiasConfig,
    labels: Sequence[str],
) -> None:
    """Headless matplotlib PNGs: label rate with CI per single slice and label, a heatmap per crossing,
    slice sizes, and state label rate vs exposure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out.mkdir(parents=True, exist_ok=True)
    meta = {"Software": None}  # no version stamp, so reruns write the same bytes
    for label in labels:
        lab = rates.filter(pl.col("label") == label)
        for name in cfg.slices.single:
            part = lab.filter((pl.col("slice") == name) & (pl.col("n") > 0))
            if part.height == 0:
                continue
            fig, ax = plt.subplots(figsize=(max(6.0, 0.28 * part.height + 2), 4))
            x = list(range(part.height))
            rate = part["rate"].to_list()
            lo = [r - c for r, c in zip(rate, part["ci_low"].to_list(), strict=True)]
            hi = [c - r for r, c in zip(rate, part["ci_high"].to_list(), strict=True)]
            ax.bar(x, rate, color="#4C78A8")
            ax.errorbar(x, rate, yerr=[lo, hi], fmt="none", ecolor="#333333", capsize=2, lw=0.8)
            overall = part["positives"][0] + part["complement_positives"][0]
            total = part["n"][0] + part["complement_n"][0]
            if total:
                ax.axhline(overall / total, color="#E45756", lw=1, ls="--", label="overall")
                ax.legend(frameon=False)
            ax.set_xticks(
                x, part["value"].to_list(), rotation=90 if part.height > 8 else 0, fontsize=7
            )
            ax.set_ylabel(f"{label} rate")
            ax.set_title(f"{label} rate by {name} (Wilson {cfg.thresholds.ci_level:.0%} CI)")
            fig.tight_layout()
            fig.savefig(out / f"rate_{label}_{name}.png", dpi=110, metadata=meta)
            plt.close(fig)
        for a, b in cfg.slices.crossings:
            part = lab.filter((pl.col("slice") == f"{a} x {b}") & (pl.col("n") > 0))
            if part.height == 0:
                continue
            split = part.with_columns(
                pl.col("value").str.split_exact(" | ", 1).struct.rename_fields(["a", "b"])
            ).unnest("value")
            rows_, cols_ = (
                sorted(split["a"].unique().to_list()),
                sorted(split["b"].unique().to_list()),
            )
            grid = [[float("nan")] * len(cols_) for _ in rows_]
            for ra, rb_, r in split.select("a", "b", "rate").iter_rows():
                grid[rows_.index(ra)][cols_.index(rb_)] = r
            fig, ax = plt.subplots(figsize=(1.0 * len(cols_) + 3, 0.5 * len(rows_) + 2))
            im = ax.imshow(grid, cmap="viridis", aspect="auto")
            ax.set_xticks(range(len(cols_)), cols_, rotation=45, ha="right", fontsize=7)
            ax.set_yticks(range(len(rows_)), rows_, fontsize=7)
            ax.set_xlabel(b)
            ax.set_ylabel(a)
            ax.set_title(f"{label} rate: {a} x {b}")
            fig.colorbar(im, ax=ax)
            fig.tight_layout()
            fig.savefig(out / f"heatmap_{label}_{a}_x_{b}.png", dpi=110, metadata=meta)
            plt.close(fig)
    singles = summary_tbl.filter(
        pl.col("slice").is_in([s for s in cfg.slices.single if s != "state"])
    )
    if singles.height:
        fig, ax = plt.subplots(figsize=(max(6.0, 0.3 * singles.height + 2), 4))
        colors = ["#E45756" if s else "#4C78A8" for s in singles["small_slice"].to_list()]
        names = [f"{s}={v}" for s, v in singles.select("slice", "value").iter_rows()]
        ax.bar(range(singles.height), singles["n_facilities"].to_list(), color=colors)
        ax.set_xticks(range(singles.height), names, rotation=90, fontsize=6)
        ax.set_ylabel("homes")
        ax.set_title("Homes per slice value (red = small slice)")
        fig.tight_layout()
        fig.savefig(out / "slice_sizes.png", dpi=110, metadata=meta)
        plt.close(fig)
    exposure = f"mean_{cfg.rater_bias.exposure_columns[0]}"
    for label in labels:
        part = rater.filter((pl.col("label") == label) & pl.col("label_rate").is_not_null())
        if part.height == 0:
            continue
        fig, ax = plt.subplots(figsize=(6, 5))
        colors = ["#E45756" if f else "#4C78A8" for f in part["flag"].to_list()]
        ax.scatter(part[exposure].to_list(), part["label_rate"].to_list(), c=colors, s=18)
        for s, xv, yv in part.select("state", exposure, "label_rate").iter_rows():
            ax.annotate(s, (xv, yv), fontsize=6, xytext=(2, 2), textcoords="offset points")
        ax.set_xlabel(exposure)
        ax.set_ylabel(f"{label} rate")
        ax.set_title(f"States: {label} rate vs inspection exposure (red = flagged)")
        fig.tight_layout()
        fig.savefig(out / f"state_rate_vs_exposure_{label}.png", dpi=110, metadata=meta)
        plt.close(fig)
