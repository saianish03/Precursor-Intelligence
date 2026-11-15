"""The toy ML smoke run's metrics file: the JSON summary each `ml` command prints as its last JSON line.

    uv run python scripts/dvc_ml_summary.py metrics/toy_ml.json data/toy_ml/prepare.out [data/toy_ml/train.out ...]

One key per output file (its stem: `prepare`, `train`, ...), so `dvc metrics diff` shows which step moved. A log line
printed after the JSON is skipped; an output with no JSON object is an error, not an empty metric. Keys that change
on every run (the MLflow run id, timings) are dropped, so the committed file only moves when a result does. A step
that skipped (no champion, nothing scored) fails the smoke run: it must reach publish.
"""

import json
import sys
from pathlib import Path

VOLATILE = {"mlflow_run_id", "seconds_total"}


def last_json_object(text: str) -> dict:
    for line in reversed(text.splitlines()):
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("no JSON object in the command's output")


def summarize(paths: list[Path]) -> dict:
    summary = {}
    for p in paths:
        step = {k: v for k, v in last_json_object(p.read_text()).items() if k not in VOLATILE}
        if step.get("skipped"):
            raise SystemExit(
                f"{p.stem} skipped ({step.get('reason')}): the smoke run must score and publish"
            )
        summary[p.stem] = step
    return summary


if __name__ == "__main__":
    out, *inputs = map(Path, sys.argv[1:])
    out.write_text(json.dumps(summarize(inputs), indent=2, sort_keys=True) + "\n")
