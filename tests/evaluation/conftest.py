"""Fixtures for the evaluation tests that need a trained run folder: the synthetic gold of tests/ml/make_ml_gold.py,
prepared and trained once per module with small LightGBM fits (training is the slow part)."""

import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "ml"))  # the synthetic gold lives in tests/ml
from make_ml_gold import MlGold, make_ml_gold, ml_config  # noqa: E402

from precursorintelligence.common.config import (  # noqa: E402
    MlConfig,
    load_feature_registry,
    load_labels_config,
    load_splits_config,
)
from precursorintelligence.common.io.metadata_store import SqlAlchemyMetadataStore  # noqa: E402
from precursorintelligence.common.io.object_store import LocalObjectStore  # noqa: E402
from precursorintelligence.training.data import run_prepare  # noqa: E402
from precursorintelligence.training.train import TrainOutput, run_train  # noqa: E402

RUN_ID = "manual__eval"


def small_config() -> MlConfig:
    """The fixture-sized ml config: small LightGBM fits and few bootstrap replicates (fast tests)."""
    cfg = ml_config()
    lgbm = cfg.members.lgbm.model_validate(
        {
            **cfg.members.lgbm.as_dict(),
            "n_estimators": 40,
            "min_child_samples": 20,
            "num_threads": 1,
        }
    )
    ev = cfg.evaluation.model_copy(
        update={
            "bootstrap": cfg.evaluation.bootstrap.model_copy(update={"replicates": 50}),
            "shuffled_label": cfg.evaluation.shuffled_label.model_copy(update={"n_estimators": 40}),
        }
    )
    # the fixture has hundreds of homes, not 15k: gate the slices it can fill
    bias = cfg.bias.model_copy(update={"gate_min_facilities": 20, "gate_min_positives": 5})
    return cfg.model_copy(
        update={
            "members": cfg.members.model_copy(update={"lgbm": lgbm}),
            "evaluation": ev,
            "bias": bias,
        }
    )


@dataclass(frozen=True)
class TrainedRun:
    gold: MlGold
    cfg: MlConfig
    run_id: str
    train: TrainOutput

    def store(self) -> LocalObjectStore:
        return LocalObjectStore(self.gold.root, immutable=None)

    def meta(self) -> SqlAlchemyMetadataStore:
        m = SqlAlchemyMetadataStore(
            f"sqlite:///{(self.gold.root / 'ops' / 'metadata.db').as_posix()}"
        )
        m.ensure_schema()
        return m


@pytest.fixture(scope="module")
def trained_template(tmp_path_factory) -> TrainedRun:
    gold = make_ml_gold(tmp_path_factory.mktemp("eval") / "store")
    cfg = small_config()
    store = LocalObjectStore(gold.root, immutable=None)
    meta = SqlAlchemyMetadataStore(f"sqlite:///{(gold.root / 'ops' / 'metadata.db').as_posix()}")
    meta.ensure_schema()
    run_prepare(
        store,
        meta,
        ml_run_id=RUN_ID,
        cfg=cfg,
        labels_cfg=load_labels_config(),
        splits_cfg=load_splits_config(),
        registry=load_feature_registry(),
    )
    mp = pytest.MonkeyPatch()
    mp.delenv("MLFLOW_TRACKING_URI", raising=False)
    try:
        out = run_train(
            store, ml_run_id=RUN_ID, cfg=cfg, registry=load_feature_registry(), recipe_hash="test"
        )
    finally:
        mp.undo()
    return TrainedRun(gold, cfg, RUN_ID, out)


@pytest.fixture
def trained(trained_template, tmp_path, monkeypatch) -> TrainedRun:
    """A private copy of the module's trained run folder (tests may write into it)."""
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    src = trained_template.gold
    root = tmp_path / "store"
    shutil.copytree(src.root, root)
    env = root / "env.yaml"  # the copy still points at the template's root
    env.write_text(
        env.read_text(encoding="utf-8").replace(src.root.as_posix(), root.as_posix()),
        encoding="utf-8",
    )
    gold = MlGold(root, env, src.run_id, src.v, src.v_labels, src.merged_sha256)
    return TrainedRun(gold, trained_template.cfg, trained_template.run_id, trained_template.train)
