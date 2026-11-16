"""The monitoring tests reuse the training tests' synthetic gold and stores (tests/training/conftest.py) and the
evaluation tests' trained run folder (tests/evaluation/conftest.py)."""

import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "ml"))  # make_ml_gold, ml_eval_fixtures


def _load(name: str):
    spec = importlib.util.spec_from_file_location(
        f"{name}_conftest", Path(__file__).parents[1] / name / "conftest.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_eval, _training = _load("evaluation"), _load("training")
trained_template, trained = _eval.trained_template, _eval.trained
gold, store, meta = _training.gold, _training.store, _training.meta
