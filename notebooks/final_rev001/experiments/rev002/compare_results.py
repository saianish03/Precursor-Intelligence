"""Collect every full-run result JSON (outputs/results/*.json, plus any extra folders given, e.g. the downloaded
Colab `precursor_exp_results/results`) into one comparison table, ranked by test PR-AUC.

    uv run --no-sync python compare_results.py [extra_results_dir ...]

Writes outputs/comparison.csv and outputs/comparison.md. Smoke results (`*__smoke.json`) are ignored.
Test is the honest number: calib was used for early stopping / tuning (and sits right after train), so it is optimistic.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
# every local results folder: results/ (gap 0, the original gold split) and results_gap_*/ (train-calib gap)
dirs = [HERE / "outputs" / "results", *sorted((HERE / "outputs").glob("results_gap_*")), *map(Path, sys.argv[1:])]

rows = []
for d in dirs:
    for f in sorted(d.glob("*.json")):
        if f.stem.endswith("__smoke"):
            continue
        r = json.loads(f.read_text(encoding="utf-8"))
        ex = r.get("extra", {})
        rows.append({
            "experiment": r["experiment"], "model": r["model"], "calib_gap": str(r.get("calib_gap", "0")),
            "test_pr_auc": r["test"]["pr_auc"], "test_roc_auc": r["test"]["roc_auc"],
            "test_p_at_5": r["test"]["p_at_5"], "test_p_at_10": r["test"]["p_at_10"], "test_brier": r["test"]["brier"],
            "calib_pr_auc": r["calib"]["pr_auc"], "test_n": r["test"]["n"], "test_base_rate": r["test"]["base_rate"],
            "train_rows": ex.get("train_rows"), "train_time_s": ex.get("train_time_s"), "device": ex.get("device") or ("colab_gpu" if "precursor_exp_results" in str(d) else "cpu"),
            "source": d.name, "written": r.get("written"),
        })

if not rows:
    sys.exit("no full-run results found yet")
df = pd.DataFrame(rows).drop_duplicates(["experiment", "model", "device", "calib_gap"], keep="last")
df = df.sort_values("test_pr_auc", ascending=False).reset_index(drop=True)
ref = df.loc[(df.experiment == "exp3") & (df.model == "lgbm_default") & (df.calib_gap == "0"), "test_pr_auc"]
if len(ref):
    df.insert(4, "delta_vs_lgbm_default", df.test_pr_auc - float(ref.iloc[0]))

out = HERE / "outputs"
df.to_csv(out / "comparison.csv", index=False)
show = df[[c for c in ["experiment", "model", "calib_gap", "test_pr_auc", "delta_vs_lgbm_default", "test_roc_auc", "test_p_at_5",
                       "test_p_at_10", "calib_pr_auc", "train_rows", "train_time_s", "device"] if c in df.columns]]
(out / "comparison.md").write_text(show.round(4).to_markdown(index=False), encoding="utf-8")
print(show.round(4).to_string(index=False))
