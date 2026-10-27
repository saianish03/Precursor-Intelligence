"""The feature registry names only allowed inputs and no label (spec §8, §11.3; GOLD_HANDOFF open item 5)."""

import csv

from precursorintelligence.common.config import load_feature_registry, project_root
from precursorintelligence.gold.features.reader import FEATURE_INPUTS

CONTRACT = project_root() / "configs" / "contracts" / "facility_month_dataset_spec_columns.csv"
NOT_A_TABLE = {"vintage_month (t)"}  # era flags come from t itself


def test_every_registered_source_is_a_feature_input():
    for spec in load_feature_registry().features:
        if spec.source_table in NOT_A_TABLE:
            continue
        tables = {s.strip() for s in spec.source_table.split("+")}
        assert tables <= set(FEATURE_INPUTS), (spec.name, tables - set(FEATURE_INPUTS))


def test_no_label_split_or_badge_column_is_a_feature():
    with CONTRACT.open(encoding="utf-8") as f:
        labels = {r["column"] for r in csv.DictReader(f) if r["family"] in ("labels", "key")}
    names = {f.name for f in load_feature_registry().features}
    assert not names & labels
    assert "high_performing_icon" not in names  # a CMS badge, never a feature (open item 12)
