"""Repository-wide pytest setup.

* Makes ``src/`` importable without installing the package (until pyproject.toml is added).
* Live tests marked ``network`` are skipped unless selected explicitly: ``pytest -m network``.
"""

import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def pytest_configure(config):
    config.addinivalue_line("markers", "network: hits the live CMS API (run with -m network)")


def pytest_collection_modifyitems(config, items):
    if "network" in (config.getoption("markexpr") or ""):
        return
    skip = pytest.mark.skip(reason="live network test; run with -m network")
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip)
