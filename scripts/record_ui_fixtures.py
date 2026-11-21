"""Record the frontend's fixtures from real data: the API responses the React app reads (frontend/src/api/openapi.yaml),
built from one scored month in GCS. Run on a laptop with read access to the data bucket:

    uv run --extra pipeline python scripts/record_ui_fixtures.py --root precursor-intel-mlops-data/ml-dev --v 2026-10 \
        --eval-run phase1-retrain-1

It only reads GCS and writes frontend/fixtures/. The ids follow the evidence tools (agents/tools/evidence.py) so a
fixture's evidence ids are the ones the live API returns. Agent responses are not recorded here: they come from the
agents' fallback templates once those exist (docs/AGENTS_PLAN.md B9).
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any

import gcsfs
import polars as pl

OUT = Path(__file__).resolve().parents[1] / "frontend" / "fixtures"
LIST_ROWS = 500  # the facility list fixture keeps the riskiest rows only; the README says so
MONTHS = 36  # the citation and penalty window, as configs/agents.yaml evidence.citation_months


def months_before(d: date, months: int) -> date:
    y, m = divmod(d.year * 12 + d.month - 1 - months, 12)
    return date(y, m + 1, 1)


def read(fs: gcsfs.GCSFileSystem, key: str) -> pl.DataFrame:
    with fs.open(key, "rb") as f:
        return pl.read_parquet(f)


def read_json(fs: gcsfs.GCSFileSystem, key: str) -> Any:
    with fs.open(key, "rb") as f:
        return json.load(f)


def write(name: str, value: Any) -> None:
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, default=str, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        required=True,
        help="bucket/prefix that holds scores/, silver/, gold/, ml/, status/",
    )
    ap.add_argument("--v", required=True)
    # The run that trained the scoring model; build-evidence finds it in ml_runs instead (E2).
    ap.add_argument("--eval-run", required=True)
    ap.add_argument(
        "--ccn", action="append", default=[], help="extra facilities to record a profile for"
    )
    a = ap.parse_args()
    fs, root, v = gcsfs.GCSFileSystem(), a.root.rstrip("/"), a.v

    manifest = read_json(fs, f"{root}/scores/v={v}/scores_manifest.json")
    status = read_json(fs, f"{root}/status/latest.json")
    label = manifest["label"]
    scores = read(fs, f"{root}/scores/v={v}/facility_scores.parquet").rename({f"p_{label}": "p"})
    reasons = read(fs, f"{root}/scores/v={v}/facility_reasons.parquet")
    chains = read(fs, f"{root}/scores/v={v}/chain_scores.parquet")
    ident = read(fs, f"{root}/gold/dim_facility_latest/part-0.parquet")
    cutoff = date.fromisoformat(f"{v}-01")
    start = months_before(cutoff, MONTHS)
    cites = read(fs, f"{root}/silver/citations_by_vintage/vintage={v}/part-0.parquet").filter(
        (pl.col("survey_date") >= start) & (pl.col("survey_date") < cutoff)
    )
    pens = read(fs, f"{root}/silver/penalties_by_vintage/vintage={v}/part-0.parquet").filter(
        (pl.col("penalty_date") >= start) & (pl.col("penalty_date") < cutoff)
    )
    evaluation = read_json(fs, f"{root}/ml/runs/{a.eval_run}/evaluation.json")

    rows = scores.join(
        ident.select("ccn", "provider_name", "provider_address", "city", "zip", "chain_name"),
        on="ccn",
        how="left",
    ).join(read_beds(fs, root, v), on="ccn", how="left")
    rows = rows.sort("p", descending=True)

    data_as_of = {
        "v": v,
        "score_month": f"{v}-01",
        "model_version": manifest["model_version"],
        "gold_run_id": manifest["gold_run_id"],
    }
    write(
        "status.json",
        {
            **data_as_of,
            "model_name": manifest["model_name"],
            "candidate": manifest["candidate"],
            "scored_at": manifest["scored_at"],
            "gold_verified": manifest["gold_verified"],
            "stale_refresh": manifest["stale_refresh"],
            "n_facilities": scores.height,
            "flags": status.get("flags", []),
            "decision": status.get("decision"),
            "base_rate": evaluation["candidates"][evaluation["selected"]]["ranking"]["base_rate"],
        },
    )

    def item(r: dict[str, Any]) -> dict[str, Any]:
        return {
            "ccn": r["ccn"],
            "name": r["provider_name"],
            "city": r["city"],
            "state": r["state"],
            "chain_id": r["chain_id"],
            "chain_name": r["chain_name"],
            "beds": r["certified_beds"],
            "p": r["p"],
            "risk_percentile": r["risk_percentile"],
            "risk_decile": r["risk_decile"],
        }

    listed = [item(r) for r in rows.head(LIST_ROWS).iter_rows(named=True)]
    write("facilities.json", {"total_in_month": rows.height, "items": listed})
    deciles = scores.group_by("risk_decile").agg(
        pl.len().alias("n"), pl.col("p").mean().alias("mean_p")
    )
    states = scores.group_by("state").agg(
        pl.len().alias("n"),
        (pl.col("risk_decile") == 1).sum().alias("n_top_decile"),
        pl.col("p").mean().alias("mean_p"),
    )
    write(
        "overview.json",
        {
            "data_as_of": data_as_of,
            "deciles": deciles.sort("risk_decile").to_dicts(),
            "states": states.sort("n_top_decile", descending=True).to_dicts(),
            "riskiest": listed[:25],
            "chains": chains.sort("expected_facilities_with_event", descending=True)
            .head(10)
            .join(
                ident.select("chain_id", "chain_name").unique("chain_id"), on="chain_id", how="left"
            )
            .drop("model_version", "gold_run_id", "ml_run_id", "score_month")
            .to_dicts(),
        },
    )

    median = rows.filter(pl.col("risk_decile") == 5).row(0, named=True)["ccn"]
    low = rows.filter(pl.col("risk_decile") == 10).row(0, named=True)["ccn"]
    no_chain = rows.filter(pl.col("chain_id").is_null()).row(0, named=True)["ccn"]
    for ccn in dict.fromkeys([*rows["ccn"].head(3).to_list(), median, low, no_chain, *a.ccn]):
        r = rows.filter(pl.col("ccn") == ccn).row(0, named=True)
        write(
            f"facility/{ccn}.json", profile(r, item(r), reasons, cites, pens, chains, data_as_of, v)
        )

    write("ops.json", {"status": status, "evaluation": summarize(evaluation), "gates": None})
    print(f"wrote {OUT} for {v}: {len(listed)} list rows of {rows.height}")


def read_beds(fs: gcsfs.GCSFileSystem, root: str, v: str) -> pl.DataFrame:
    p = read(fs, f"{root}/silver/provider_by_vintage/vintage={v}/part-0.parquet")
    return p.select("ccn", "certified_beds")


def profile(
    r: dict[str, Any],
    row: dict[str, Any],
    reasons: pl.DataFrame,
    cites: pl.DataFrame,
    pens: pl.DataFrame,
    chains: pl.DataFrame,
    data_as_of: dict[str, Any],
    v: str,
) -> dict[str, Any]:
    ccn = r["ccn"]
    rs = reasons.filter(pl.col("ccn") == ccn).sort(["direction", "rank"], descending=[True, False])
    cs = (
        cites.filter(pl.col("ccn") == ccn)
        .sort(["survey_date", "ftag"], descending=[True, False])
        .unique(["survey_date", "ftag"], keep="first", maintain_order=True)
    )
    ps = (
        pens.filter(pl.col("ccn") == ccn)
        .sort(
            ["penalty_date", "penalty_type", "fine_amount", "payment_denial_start_date", "dup_seq"],
            nulls_last=True,
        )
        .with_columns(
            pl.int_range(1, pl.len() + 1).over(["penalty_date", "penalty_type"]).alias("_n")
        )
        .sort(["penalty_date", "penalty_type", "_n"], descending=[True, False, False])
    )
    chain = chains.filter(pl.col("chain_id") == r["chain_id"]).to_dicts() if r["chain_id"] else []
    return {
        "data_as_of": data_as_of,
        "facility": {**row, "address": r.get("provider_address"), "zip": r["zip"]},
        "score": {
            "evidence_id": f"score:{ccn}:{v}",
            "p": r["p"],
            "risk_percentile": r["risk_percentile"],
            "risk_decile": r["risk_decile"],
        },
        "reasons": [
            {
                "evidence_id": f"reason:{ccn}:{v}:{x['direction']}:{x['rank']}",
                "direction": x["direction"],
                "rank": x["rank"],
                "feature": x["feature"],
                "label": x["label"],
                "feature_value": x["feature_value"],
                "contribution_logodds": x["contribution_logodds"],
            }
            for x in rs.iter_rows(named=True)
        ],
        "citations": [
            {
                "evidence_id": f"citation:{ccn}:{x['survey_date'].isoformat()}:{x['ftag']}",
                "survey_date": x["survey_date"],
                "ftag": x["ftag"],
                "deficiency_category": x["deficiency_category"],
                "scope_severity": x["scope_severity"],
                "is_standard": x["is_standard"],
                "is_complaint": x["is_complaint"],
                "correction_date": x["correction_date"],
            }
            for x in cs.iter_rows(named=True)
        ],
        "penalties": [
            {
                "evidence_id": f"penalty:{ccn}:{x['penalty_date'].isoformat()}:"
                f"{str(x['penalty_type']).lower().replace(' ', '_')}:{x['_n']}",
                "penalty_date": x["penalty_date"],
                "penalty_type": x["penalty_type"],
                "fine_amount": x["fine_amount"],
                "payment_denial_start_date": x["payment_denial_start_date"],
                "payment_denial_length_days": x["payment_denial_length_days"],
            }
            for x in ps.iter_rows(named=True)
        ],
        "chain": (
            {
                "evidence_id": f"chain:{r['chain_id']}:{v}",
                **{
                    k: chain[0][k]
                    for k in (
                        "chain_id",
                        "n_facilities",
                        "beds",
                        "expected_facilities_with_event",
                        "p_any_event",
                        "bed_weighted_mean_p",
                        "max_p",
                        "worst_ccn",
                        "share_top_decile",
                    )
                },
            }
            if chain
            else None
        ),
    }


def summarize(e: dict[str, Any]) -> dict[str, Any]:
    c = e["candidates"][e["selected"]]
    return {
        "ml_run_id": e["ml_run_id"],
        "selected": e["selected"],
        "test_months": e["test_months"],
        "rows": e["rows"],
        "positives": e["positives"],
        "ranking": {
            k: c["ranking"][k]
            for k in (
                "base_rate",
                "pr_auc",
                "pr_auc_lift",
                "roc_auc",
                "precision_at_5",
                "precision_at_10",
                "lift_at_10",
            )
        },
        "probability": {
            k: c["probability"][k] for k in ("brier", "brier_skill", "ece", "calibration_slope")
        },
        "intervals": c["intervals"],
        "baselines": {k: v["ranking"]["pr_auc"] for k, v in e["baselines"].items()},
        "reliability": e["reliability"]["calibrated"],
        "sanity": {
            "shuffled_label_roc": e["sanity"]["shuffled_label_roc"],
            "uncleared_leakage_flags": e["sanity"]["uncleared_leakage_flags"],
        },
        "flags": e["flags"],
    }


if __name__ == "__main__":
    main()
