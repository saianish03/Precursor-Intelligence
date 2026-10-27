"""The gold column contract (configs/contracts/facility_month_dataset_spec_columns.csv): CMS star
ratings never reach the model (spec §14; SILVER_CONTRACT.md)."""

import csv
from pathlib import Path

CONTRACT = (
    Path(__file__).parents[2] / "configs" / "contracts" / "facility_month_dataset_spec_columns.csv"
)
NEVER_MODEL = {
    "overall_rating",
    "overall_rating_fn",
    "health_inspection_rating",
    "health_inspection_rating_fn",
}


def _rows() -> list[dict[str, str]]:
    with CONTRACT.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_column_names_are_unique():
    names = [r["column"] for r in _rows()]
    assert len(names) == len(set(names))


def test_cms_ratings_are_never_model_features():
    use = {r["column"]: r["model_use"] for r in _rows()}
    assert set(use) >= NEVER_MODEL, NEVER_MODEL - set(use)
    assert {c: use[c] for c in NEVER_MODEL} == dict.fromkeys(NEVER_MODEL, "ablation_only")
