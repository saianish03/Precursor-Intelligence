"""The DVC stages whose inputs or command changed since dvc.lock was written (CI, #73 B14).

    uv run dvc status --json | python scripts/dvc_stale_deps.py

Prints the stale stages, one per line, and exits 1 when there are any: the PR changed what a toy stage depends on
without running `uv run dvc repro` and committing dvc.lock. Missing or changed outputs alone don't count (a fresh
checkout has no data/; CI rebuilds it, and metrics drift is only a warning).
"""

import json
import sys

STALE = ("changed deps", "changed command")


def stale_stages(status: dict) -> list[str]:
    return sorted(
        stage
        for stage, changes in status.items()
        if any(isinstance(c, dict) and any(k in c for k in STALE) for c in changes)
    )


def main() -> None:
    raw = sys.stdin.read().strip()
    stages = stale_stages(json.loads(raw) if raw else {})
    for stage in stages:
        print(stage)
    sys.exit(1 if stages else 0)


if __name__ == "__main__":
    main()
