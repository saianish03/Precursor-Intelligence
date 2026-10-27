"""A gold run reads only its allowed silver inputs (SILVER_CONTRACT §6; spec §0.4, §8).

The spine may read ``facility_universe``; the labels the two ledgers and ``facility_presence``; the features
(through the feature reader) only ``*_by_vintage`` partitions and ``citation_lookup`` up to V. Nothing else: no
``citation_lookup_versions`` (its valid_to looks into the future), no ``gold/dim_facility_latest``
(``is_active`` is as of V). The feature reader and the registry have their own contract tests.
"""

import dataclasses
import re
import sys
from pathlib import Path

import yaml

from precursorintelligence.common.config import (
    load_labels_config,
    load_settings,
    load_splits_config,
    project_root,
)
from precursorintelligence.common.io.object_store import LocalObjectStore
from precursorintelligence.gold.build import SILVER_INPUTS, run_gold
from precursorintelligence.gold.features.reader import FEATURE_INPUTS
from precursorintelligence.ingestion.context import build_context

sys.path.insert(0, str(Path(__file__).parents[1] / "gold"))  # the toy world lives in tests/gold
from make_toy_silver import TOY_SPLITS, make_toy, no_gate_splits  # noqa: E402

GOLD_SRC = project_root() / "src" / "precursorintelligence" / "gold"
ALLOWED = {"facility_universe", "facility_presence", "ledger_citations", "ledger_penalties"}
STAR_RATINGS = ("overall_rating", "health_inspection_rating", "qm_rating", "staffing_rating")
# CMS badges handled like the star ratings (SILVER_CONTRACT §10: never a model feature).
CMS_BADGES = ("high_performing_icon",)


def _star_rating_columns() -> list[str]:
    keep = yaml.safe_load(
        (project_root() / "configs" / "keep_columns.yaml").read_text(encoding="utf-8")
    )
    cols = keep["provider_by_vintage"]["columns"]
    return [c for c in cols if c.removesuffix("_fn") in STAR_RATINGS or c in CMS_BADGES]


def test_keep_columns_still_lists_the_star_ratings_this_test_guards():
    # If silver renames them, this test must be updated rather than silently guard nothing.
    assert sorted(_star_rating_columns()) == sorted(
        [*STAR_RATINGS, *(f"{c}_fn" for c in STAR_RATINGS), *CMS_BADGES]
    )


# The column contract (configs/contracts/facility_month_dataset_spec_columns.csv) keeps the four ratings as
# facility_staffing features, tagged model / ablation_only in feature_registry.yaml (open item 6, settled by
# following the contract). That family is the only gold code that may name them; no gold code names the badge.
RATINGS_ALLOWED_IN = {GOLD_SRC / "features" / "facility_staffing.py"}


def test_gold_code_names_a_star_rating_column_only_in_facility_staffing():
    paths = sorted(GOLD_SRC.rglob("*.py"))
    assert GOLD_SRC / "spine.py" in paths and set(paths) >= RATINGS_ALLOWED_IN
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for col in _star_rating_columns():
            if path in RATINGS_ALLOWED_IN and col not in CMS_BADGES:
                continue
            assert not re.search(rf"\b{col}\b", text), f"{path.relative_to(GOLD_SRC)} names {col}"


def test_gold_declares_only_the_allowed_silver_inputs():
    assert set(SILVER_INPUTS) == ALLOWED


class RecordingStore(LocalObjectStore):
    """A local store that remembers every object it opens for reading."""

    def __init__(self, root: Path) -> None:
        super().__init__(root, immutable=None)
        self.reads: list[str] = []

    def open_read(self, key: str):  # type: ignore[no-untyped-def]
        self.reads.append(key)
        return super().open_read(key)


def test_a_gold_run_reads_nothing_but_the_allowed_inputs(tmp_path):
    env = make_toy(tmp_path / "toy")
    ctx = build_context(load_settings(env_path=env))
    store = RecordingStore(tmp_path / "toy")
    ctx = dataclasses.replace(ctx, object_store=store)
    result = run_gold(
        ctx, labels_cfg=load_labels_config(), splits_cfg=load_splits_config(env.parent / TOY_SPLITS)
    )
    assert result.exit_code == 0, result.failures

    silver_reads = {k for k in store.reads if not k.startswith(f"gold/_staging/run={ctx.run_id}/")}
    whole_tables = {
        k for k in silver_reads if k.endswith("/part-0.parquet") and "/vintage=" not in k
    }
    assert whole_tables == {f"silver/{name}/part-0.parquet" for name in ALLOWED}
    partitions = silver_reads - whole_tables
    pattern = re.compile(r"silver/([a-z_]+)/vintage=(\d{4}-\d{2})/part-0\.parquet")
    for key in partitions:
        m = pattern.fullmatch(key)
        assert m and m[1] in FEATURE_INPUTS and m[2] <= result.v, key
    assert not any("dim_facility_latest" in k for k in store.reads)


def test_a_run_with_an_earlier_v_reads_no_feature_input_after_it(tmp_path):
    env = make_toy(tmp_path / "toy")
    ctx = build_context(load_settings(env_path=env))
    store = RecordingStore(tmp_path / "toy")
    ctx = dataclasses.replace(ctx, object_store=store)
    result = run_gold(
        ctx,
        labels_cfg=load_labels_config(),
        splits_cfg=load_splits_config(
            no_gate_splits(env.parent)
        ),  # no toy training month at 2024-06
        v="2024-06",
    )
    assert result.exit_code == 0, result.failures
    vintages = {k.split("vintage=")[1][:7] for k in store.reads if "/vintage=" in k}
    assert vintages and max(vintages) == "2024-06"
