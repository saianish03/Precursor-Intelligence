"""Shared building blocks: config (``config``), logging (``obs``) and storage ports (``io``).

Import rules (keep every image working):
  * ``common.config`` and ``common.obs`` need only the base dependencies, so any image may import them.
  * ``common.io`` needs the ``pipeline`` extra (sqlalchemy, fsspec, polars); only pipeline images import it.
  * ``common`` never imports a stage package. The ingestion run context lives in ``ingestion.context``.
"""
