"""The DVC stages whose inputs or command changed since dvc.lock was written (CI, #73 B14).

    uv run dvc status --json | python scripts/dvc_stale_deps.py

Prints the stale stages, one per line, and exits 1 when there are any: the PR changed what a toy stage depends on
without running `uv run dvc repro` and committing dvc.lock. Missing or changed outputs alone don't count (a fresh
checkout has no data/; CI rebuilds it, and metrics drift is only a warning), and neither does a dependency that is
missing: on a fresh checkout that is another stage's output not built yet (toy_summary reads toy_gold's gold). A repo
file a stage lists but that is gone fails tests/contracts/test_dvc_config.py instead.
"""

import json
import sys


def _stale(change: object) -> bool:
    if not isinstance(change, dict):
        return False
    deps = change.get("changed deps") or {}
    return bool(change.get("changed command")) or any(state != "deleted" for state in deps.values())


def stale_stages(status: dict) -> list[str]:
    return sorted(stage for stage, changes in status.items() if any(_stale(c) for c in changes))


def main() -> None:
    raw = sys.stdin.read().strip()
    stages = stale_stages(json.loads(raw) if raw else {})
    for stage in stages:
        print(stage)
    sys.exit(1 if stages else 0)


if __name__ == "__main__":
    main()
