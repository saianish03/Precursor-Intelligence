"""Helpers shared by the evaluation tests (tests/evaluation): the fixture-sized ml config and an in-memory model
registry. Here rather than in a conftest, so test modules import them by a unique name."""

from make_ml_gold import ml_config

from precursorintelligence.common.config import MlConfig
from precursorintelligence.evaluation.registry import RegisteredVersion


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


class FakeRegistry:
    """An in-memory ``evaluation.registry.Registry``: versions are numbered from 1, models kept by URI."""

    def __init__(self, models=None):
        self._rv = RegisteredVersion
        self.versions: dict[int, dict[str, str]] = {}
        self.sources: dict[int, str] = {}
        self.aliases: dict[str, int] = {}
        self.models = dict(models or {})  # model URI -> PrecursorModel

    def by_alias(self, alias):
        v = self.aliases.get(alias)
        return self.get(v) if v is not None else None

    def get(self, version):
        tags = self.versions.get(version)
        return self._rv(version, dict(tags)) if tags is not None else None

    def find_by_ml_run_id(self, ml_run_id):
        hits = [v for v, t in self.versions.items() if t.get("ml_run_id") == ml_run_id]
        return self.get(min(hits)) if hits else None

    def register(self, model_uri, tags):
        v = max(self.versions, default=0) + 1
        self.versions[v], self.sources[v] = dict(tags), model_uri
        return self.get(v)

    def set_tags(self, version, tags):
        self.versions[version].update(tags)

    def set_alias(self, alias, version):
        self.aliases[alias] = version

    def delete_alias(self, alias):
        self.aliases.pop(alias, None)

    def load_model(self, model_uri):
        return self.models[model_uri]
