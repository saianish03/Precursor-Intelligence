"""Write the toy silver world (silver_fakes.py) to a folder, laid out like real silver.

uv run python tests/gold/make_toy_silver.py data/toy
"""

import sys
from pathlib import Path

from silver_fakes import build_silver, write_silver

from precursorintelligence.common.io.object_store import LocalObjectStore


def main(root: str) -> None:
    store = LocalObjectStore(root, immutable=None)
    tables = build_silver()
    write_silver(store, tables)
    print(f"wrote {len(tables)} toy tables under {Path(root).resolve()}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python tests/gold/make_toy_silver.py <folder>")
    main(sys.argv[1])
