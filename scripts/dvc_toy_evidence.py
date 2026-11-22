"""The toy evidence run (DVC stage `toy_evidence`, docs/AGENTS_PLAN.md E2): build-evidence on the toy ML chain's
published month, then a read-back through the agents' tools.

    uv run python scripts/dvc_toy_evidence.py data/toy_ml metrics/toy_evidence.json

Needs `data/toy_ml` from the stage `toy_ml` (CI forces both). The toy chain has no silver, so this writes the month's
citations and penalties first (tests/agents/evidence_fakes.py: rows inside the window and rows build-evidence must
drop). It fails if build-evidence fails, keeps other rows than the fixture expects, or the agents' reader refuses the
result. The metrics file keeps the build's summary (counts, files, provenance: the same on every machine).
"""

import asyncio
import json
import shutil
import subprocess
import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests" / "agents"))

from evidence_fakes import write_toy_silver  # noqa: E402

from precursorintelligence.agents.tools.evidence import EvidenceTools  # noqa: E402
from precursorintelligence.agents.tools.source import ParquetEvidenceSource  # noqa: E402
from precursorintelligence.common.config import load_agents_config  # noqa: E402
from precursorintelligence.common.io.object_store import LocalObjectStore  # noqa: E402

RUN_ID = "toy-evidence"


def last_json_object(text: str) -> dict:
    for line in reversed(text.splitlines()):
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("no JSON object in the command's output")


def run(toy_dir: Path) -> dict:
    months = sorted((toy_dir / "scores").glob("v=*"))
    if not months:
        raise SystemExit(
            f"{toy_dir}/scores is empty: run `uv run dvc repro --force --single-item toy_ml`"
        )
    v = months[-1].name[2:]
    prefix = load_agents_config().evidence.prefix
    for scratch in ("silver", prefix.split("/")[0]):  # a fresh month every run
        shutil.rmtree(toy_dir / scratch, ignore_errors=True)
    store = LocalObjectStore(toy_dir, immutable=None)
    ccns = pl.read_parquet(months[-1] / "facility_scores.parquet")["ccn"].sort().to_list()
    kept = write_toy_silver(store, v, ccns)

    done = subprocess.run(
        [
            "precursor-pipeline",
            "agents",
            "build-evidence",
            "--v",
            v,
            "--env-file",
            "env.yaml",
            "--ml-config",
            "ml.yaml",
            "--run-id",
            RUN_ID,
        ],
        cwd=toy_dir,
        capture_output=True,
        text=True,
    )
    sys.stderr.write(done.stderr)
    sys.stdout.write(done.stdout)
    if done.returncode != 0:
        raise SystemExit(f"agents build-evidence exited {done.returncode}")
    summary = last_json_object(done.stdout)
    for table in ("citations", "penalties"):
        if summary["rows"][table] != kept[table]:
            raise SystemExit(f"{table}: kept {summary['rows'][table]} rows, expected {kept[table]}")

    tools = EvidenceTools(ParquetEvidenceSource.open(store, prefix), timeout_s=30)
    score = asyncio.run(tools.get_facility_score(ccns[0]))
    citations = asyncio.run(tools.get_citations(ccns[0]))
    if score is None or score.v != v or not citations:
        raise SystemExit(f"the agents' tools could not read {ccns[0]} back from the evidence")
    return summary


if __name__ == "__main__":
    toy_dir, metrics_file = map(Path, sys.argv[1:])
    metrics_file.write_text(json.dumps(run(toy_dir), indent=2, sort_keys=True) + "\n")
