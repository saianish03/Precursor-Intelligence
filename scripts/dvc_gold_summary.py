"""Row and file counts per gold table, written as the metrics file the DVC pipeline versions.

    uv run python scripts/dvc_gold_summary.py data/toy/gold metrics/toy_gold.json

`dvc metrics show` prints it and `dvc metrics diff <rev>` compares it with an older data version.
"""

import json
import sys
from pathlib import Path

import polars as pl


def summarize(gold_dir: Path) -> dict[str, dict[str, int]]:
    """Counts for each table folder; `_staging`, `_current.json` and other `_` entries are run bookkeeping."""
    tables = sorted(p for p in gold_dir.iterdir() if p.is_dir() and not p.name.startswith("_"))
    summary = {}
    for table in tables:
        files = sorted(table.rglob("*.parquet"))
        rows = sum(pl.scan_parquet(f).select(pl.len()).collect().item() for f in files)
        summary[table.name] = {"files": len(files), "rows": rows}
    return summary


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    gold_dir, out_file = Path(argv[0]), Path(argv[1])
    if not gold_dir.is_dir():
        print(f"no gold folder at {gold_dir}", file=sys.stderr)
        return 1
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(
        json.dumps(summarize(gold_dir), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
