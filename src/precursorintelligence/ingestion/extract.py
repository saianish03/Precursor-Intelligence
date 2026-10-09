"""Task ``extract`` (fan-out: one call per stored zip).

Streams in-scope members out of the stored zip and back into the ObjectStore under
``landing/cms_nh/extracted/zip_date=<date>/<zip_stem>/<file_name>``. Works identically on local disk
and object storage because it only uses ``open_read`` / ``put_stream``. The archive is re-validated
(names, symlinks, bomb limits) before any member is read; junk entries are skipped; sub-folders are
flattened (the original member path is kept in ``extracted_files``).
"""

from __future__ import annotations

import logging
import posixpath
import zipfile
from dataclasses import dataclass

from ..common.context import Context, utc_now
from ..common.io.object_store import ObjectExistsError
from .layout import OTHER, TableClassifier
from .sources.base import CatalogEntry
from .zipsafe import inspect_archive, is_allowed_extension

log = logging.getLogger(__name__)

EXTRACT_PREFIX = "landing/cms_nh/extracted"
CHUNK = 8 * 1024 * 1024


@dataclass
class ExtractResult:
    entry_id: str
    n_extracted: int
    n_skipped_existing: int
    skipped: bool


def extracted_prefix(entry: CatalogEntry, zip_key: str) -> str:
    stem = posixpath.splitext(posixpath.basename(zip_key))[0]
    return f"{EXTRACT_PREFIX}/zip_date={entry.zip_date.isoformat()}/{stem}"


def extract_entry(ctx: Context, entry: CatalogEntry, zip_key: str, zip_sha256: str) -> ExtractResult:
    store, limits = ctx.object_store, ctx.cfg.security.zip
    done = [r for r in ctx.metadata.read("extracted_files", {"entry_id": entry.entry_id})
            if r["zip_sha256"] == zip_sha256]
    if done and all(store.exists(r["object_key"]) for r in done):
        log.info("extract %s: already extracted (%d files)", entry.zip_name, len(done))
        return ExtractResult(entry.entry_id, 0, len(done), True)

    classifier = TableClassifier(ctx.cfg.tables)
    only_in_scope = ctx.cfg.extract.only_in_scope
    prefix = extracted_prefix(entry, zip_key)
    rows, used_names, n_new, n_existing = [], set(), 0, 0
    with store.open_read(zip_key) as f:
        members = inspect_archive(f, limits)
        f.seek(0)
        with zipfile.ZipFile(f) as zf:
            for m in members:
                if m.is_junk or not is_allowed_extension(m, limits):
                    continue
                table = classifier.classify(m.file_name)
                if only_in_scope and table == OTHER:
                    continue
                name = m.file_name
                if name in used_names:  # flattening collision: keep both
                    stem, ext = posixpath.splitext(name)
                    name = f"{stem}__{m.crc32}{ext}"
                used_names.add(name)
                key = f"{prefix}/{name}"
                info = store.stat(key)
                if info and info.size == m.size:
                    n_existing += 1
                else:
                    with zf.open(m.path) as src:
                        try:
                            store.put_stream(key, iter(lambda: src.read(CHUNK), b""))
                        except ObjectExistsError:
                            raise RuntimeError(f"extracted object {key} exists with a different size") from None
                    store.make_immutable(key)
                    n_new += 1
                rows.append({
                    "entry_id": entry.entry_id, "member_path": m.path, "zip_sha256": zip_sha256,
                    "table_name": table, "object_key": key, "size_bytes": m.size, "crc32": m.crc32,
                    "run_id": ctx.run_id, "extracted_at": utc_now(),
                })
    ctx.metadata.upsert("extracted_files", rows)
    log.info("extract %s: %d files written, %d already present", entry.zip_name, n_new, n_existing)
    return ExtractResult(entry.entry_id, n_new, n_existing, False)
