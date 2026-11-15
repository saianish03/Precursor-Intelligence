"""The toy ML smoke run (DVC stage `toy_ml`, ML spec §16.2): the whole `ml` chain on the synthetic ML gold.

    uv run python scripts/dvc_toy_ml.py data/toy_ml metrics/toy_ml.json

Rebuilds `<dir>` from the fixture, then runs decide, prepare, train, evaluate, gate, score, publish and monitor from
`<dir>` with a
SQLite MLflow store there (never a shared server, whatever the shell sets; a SQLite store puts its artifacts in
./mlruns). Python, not shell, so the stage runs the same under DVC on Windows (cmd.exe).

It fails if a step fails, if decide does not retrain (a fresh store must), if the gate does not promote (a validation
or bias hold) or if a step skips: the smoke run must reach publish. The metrics file keeps one entry per step with only the facts that are the same on every machine
(counts, choices, actions). The floats and hashes differ between a laptop and the CI runner, so they stay out of it:
each step's full output (the gate's G lines, every float) goes to stdout after the toy gate limits, so CI's log shows
how close each gate is.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

STEPS = ("decide", "prepare", "train", "evaluate", "gate", "score", "publish", "monitor")
RUN_ID = "toy-ml"
FACILITIES = 2000
FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "ml" / "make_ml_gold.py"
# plus every float; psi and headline are floats one level down, scored_at is a clock time
UNSTABLE = {
    "mlflow_run_id",
    "seconds_total",
    "dataset_version",
    "recipe_hash",
    "scored_at",
    "psi",
    "headline",
}


def last_json_object(text: str) -> dict:
    for line in reversed(text.splitlines()):
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("no JSON object in the command's output")


def check(step: str, summary: dict) -> None:
    if step == "decide" and summary.get("decision") != "RETRAIN":
        raise SystemExit(
            f"decide said {summary.get('decision')} ({summary.get('reason')}), not RETRAIN"
        )
    if summary.get("skipped"):
        raise SystemExit(
            f"{step} skipped ({summary.get('reason')}): the smoke run must score and publish"
        )
    if step == "gate" and not (summary.get("passed") and summary.get("action") == "promote"):
        raise SystemExit(
            f"the gate did not promote ({summary.get('failed_gates')}): see the G lines above"
        )


def stable(summary: dict) -> dict:
    return {k: v for k, v in summary.items() if k not in UNSTABLE and not isinstance(v, float)}


def run(out_dir: Path) -> dict:
    shutil.rmtree(out_dir, ignore_errors=True)
    subprocess.run(
        [sys.executable, str(FIXTURE), str(out_dir), "--facilities", str(FACILITIES)], check=True
    )
    env = {**os.environ, "MLFLOW_TRACKING_URI": "sqlite:///mlflow.db"}
    args = ["--env-file", "env.yaml", "--ml-config", "ml.yaml", "--run-id", RUN_ID]
    print(
        "toy gate limits:", json.dumps(yaml.safe_load((out_dir / "ml.yaml").read_text())["gates"])
    )
    metrics = {}
    for step in STEPS:
        done = subprocess.run(
            ["precursor-pipeline", "ml", step, *args],
            cwd=out_dir,
            env=env,
            capture_output=True,
            text=True,
        )
        (out_dir / f"{step}.out").write_text(done.stdout)
        sys.stderr.write(done.stderr)
        sys.stdout.write(done.stdout)  # the gate's G lines and every float, for CI's log
        if done.returncode != 0:
            raise SystemExit(f"ml {step} exited {done.returncode}")
        summary = last_json_object(done.stdout)
        check(step, summary)
        metrics[step] = stable(summary)
    return metrics


if __name__ == "__main__":
    out_dir, metrics_file = map(Path, sys.argv[1:])
    metrics_file.write_text(json.dumps(run(out_dir), indent=2, sort_keys=True) + "\n")
