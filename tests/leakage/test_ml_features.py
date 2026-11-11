"""Leakage test 5 (docs/ML_PIPELINE_SPEC.md §17.1, §1.4): the model input is the registry's model list minus
ml.yaml features.exclude, in order, and never holds a label, a key, a forbidden star rating or the chain id."""

import re
import sys
from pathlib import Path

from precursorintelligence.common.config import load_feature_registry, load_ml_config
from precursorintelligence.training.data import feature_list, to_model_frame

sys.path.insert(0, str(Path(__file__).parents[1] / "ml"))  # the synthetic gold lives in tests/ml
from make_ml_gold import build_frame  # noqa: E402

FORBIDDEN = re.compile(
    r"^(y_|status_|split_|slice__)|_fwd|overall_rating|health_inspection_rating|^(chain_id|t|cutoff_date|ccn)$"
)


def test_the_model_input_is_the_registry_model_list_minus_exclude_in_order():
    registry, cfg = load_feature_registry(), load_ml_config()
    expected = [
        f.name
        for f in registry.features
        if f.model_use == "model" and f.name not in cfg.features.exclude
    ]
    names = [f.name for f in feature_list(registry, cfg)]
    assert names == expected
    assert not [n for n in names if FORBIDDEN.search(n)]
    assert not {"era", "penalty_regime_2025_10"} & set(names)  # ML-21
    # ablation_only (the star ratings and their footnotes) and product_only (chain_id) never get in
    not_model = {f.name for f in registry.features if f.model_use != "model"}
    assert not not_model & set(names)
    assert {"overall_rating_fn", "health_inspection_rating_fn"} <= not_model


def test_the_model_frame_has_exactly_those_columns():
    registry, cfg = load_feature_registry(), load_ml_config()
    features = feature_list(registry, cfg)
    df = build_frame(30, "2026-10", "2026-10", seed=5)
    assert {"y_g12", "status_g12", "split_g12", "chain_id", "overall_rating"} <= set(df.columns)
    out = to_model_frame(df, features)
    assert list(out.columns) == [f.name for f in features]
    assert not [c for c in out.columns if FORBIDDEN.search(c)]
