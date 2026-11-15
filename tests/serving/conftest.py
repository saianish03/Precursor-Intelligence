"""The serving tests reuse the evaluation tests' trained run folder (tests/evaluation/conftest.py)."""

import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "ml"))  # make_ml_gold, ml_eval_fixtures

_spec = importlib.util.spec_from_file_location(
    "evaluation_conftest", Path(__file__).parents[1] / "evaluation" / "conftest.py"
)
assert _spec is not None and _spec.loader is not None
_eval = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_eval)

trained_template = _eval.trained_template
trained = _eval.trained
tracking = _eval.tracking
