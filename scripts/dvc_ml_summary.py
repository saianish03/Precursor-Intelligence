"""The toy ML smoke run's metrics file: the JSON summary each `ml` command prints as its last JSON line.

    uv run python scripts/dvc_ml_summary.py metrics/toy_ml.json data/toy_ml/prepare.out [data/toy_ml/train.out ...]

One key per output file (its stem: `prepare`, `train`, ...), so `dvc metrics diff` shows which step moved. A log line
printed after the JSON is skipped; an output with no JSON object is an error, not an empty metric.
"""

import json
import sys
from pathlib import Path


def last_json_object(text: str) -> dict:
    for line in reversed(text.splitlines()):
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("no JSON object in the command's output")


if __name__ == "__main__":
    out, *inputs = map(Path, sys.argv[1:])
    summary = {p.stem: last_json_object(p.read_text()) for p in inputs}
    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
