"""``common.config`` and ``common.obs`` must import with only the base dependencies installed, because the
api / ui / agents images do not install the ``pipeline`` extra. Pipeline-only libraries are blocked in a
fresh interpreter to simulate a base-only install."""

import subprocess
import sys
import textwrap

PIPELINE_ONLY = [
    "requests",
    "tenacity",
    "certifi",
    "sqlalchemy",
    "polars",
    "fsspec",
    "gcsfs",
    "duckdb",
    "pandas",
    "pyarrow",
    "psycopg",
    "typer",
]


def test_common_imports_without_pipeline_extra():
    code = textwrap.dedent(f"""
        import importlib.abc, sys
        BLOCKED = {PIPELINE_ONLY!r}
        class Block(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0] in BLOCKED:
                    raise ImportError(f"blocked for this test: {{name}}")
        sys.meta_path.insert(0, Block())
        import precursorintelligence.common.config
        import precursorintelligence.common.obs.logging
    """)
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
