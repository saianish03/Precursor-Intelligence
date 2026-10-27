"""scripts/dvc_gold_summary.py: row and file counts per gold table, the metrics file the DVC pipeline versions."""

import importlib.util
import json
from pathlib import Path

import polars as pl

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "dvc_gold_summary.py"


def load_script():
    spec = importlib.util.spec_from_file_location("dvc_gold_summary", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_table(gold: Path, name: str, *partitions: int) -> None:
    for i, rows in enumerate(partitions):
        part = gold / name / f"t=2025-0{i + 1}"
        part.mkdir(parents=True)
        pl.DataFrame({"x": list(range(rows))}).write_parquet(part / "part-0.parquet")


def test_counts_rows_and_files_per_table_across_partitions(tmp_path):
    write_table(tmp_path, "spine", 3)
    write_table(tmp_path, "facility_month_dataset", 2, 4)
    assert load_script().summarize(tmp_path) == {
        "facility_month_dataset": {"files": 2, "rows": 6},
        "spine": {"files": 1, "rows": 3},
    }


def test_run_markers_and_loose_files_are_not_tables(tmp_path):
    write_table(tmp_path, "labels", 5)
    (tmp_path / "_staging" / "run=1").mkdir(parents=True)
    pl.DataFrame({"x": [1]}).write_parquet(tmp_path / "_staging" / "run=1" / "part-0.parquet")
    (tmp_path / "_current.json").write_text("{}")
    assert set(load_script().summarize(tmp_path)) == {"labels"}


def test_main_writes_stable_sorted_json_and_fails_without_a_gold_folder(tmp_path):
    write_table(tmp_path / "gold", "labels", 2)
    out = tmp_path / "metrics" / "toy_gold.json"
    module = load_script()
    assert module.main([str(tmp_path / "gold"), str(out)]) == 0
    assert json.loads(out.read_text()) == {"labels": {"files": 1, "rows": 2}}
    assert out.read_text().endswith("}\n")
    assert module.main([str(tmp_path / "missing"), str(out)]) == 1
