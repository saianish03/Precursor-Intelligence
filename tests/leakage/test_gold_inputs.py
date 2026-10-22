"""Gold's spine and labels read only their allowed silver inputs (SILVER_CONTRACT §6; spec §0.4, §8).

The spine may read ``facility_universe``; the labels the two ledgers and ``facility_presence``. Nothing else:
no ``*_by_vintage`` table (so no star ratings), no ``citation_lookup_versions`` (its valid_to looks into the
future), no ``gold/dim_facility_latest`` (``is_active`` is as of V). Feature inputs and the feature registry
get their own contract tests with the features.
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
from precursorintelligence.ingestion.context import build_context

sys.path.insert(0, str(Path(__file__).parents[1] / "gold"))  # the toy world lives in tests/gold
from make_toy_silver import make_toy  # noqa: E402

GOLD_SRC = project_root() / "src" / "precursorintelligence" / "gold"
ALLOWED = {"facility_universe", "facility_presence", "ledger_citations", "ledger_penalties"}
STAR_RATINGS = ("overall_rating", "health_inspection_rating", "qm_rating", "staffing_rating")


def _star_rating_columns() -> list[str]:
    keep = yaml.safe_load(
        (project_root() / "configs" / "keep_columns.yaml").read_text(encoding="utf-8")
    )
    cols = keep["provider_by_vintage"]["columns"]
    return [c for c in cols if c.removesuffix("_fn") in STAR_RATINGS]


def test_keep_columns_still_lists_the_star_ratings_this_test_guards():
    # If silver renames them, this test must be updated rather than silently guard nothing.
    assert sorted(_star_rating_columns()) == sorted(
        [*STAR_RATINGS, *(f"{c}_fn" for c in STAR_RATINGS)]
    )


def test_gold_code_never_names_a_star_rating_column():
    for path in GOLD_SRC.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        for col in _star_rating_columns():
            assert not re.search(rf"\b{col}\b", text), f"{path.name} names {col}"


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
    result = run_gold(ctx, labels_cfg=load_labels_config(), splits_cfg=load_splits_config())
    assert result.exit_code == 0, result.failures

    silver_reads = {k for k in store.reads if not k.startswith(f"gold/_staging/run={ctx.run_id}/")}
    assert silver_reads == {f"silver/{name}/part-0.parquet" for name in ALLOWED}
    assert not any("_by_vintage" in k or "dim_facility_latest" in k for k in store.reads)
