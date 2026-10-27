"""End-to-end Step 1 against a fake CMS server: layouts, duplicates, gaps, idempotency, re-publication,
quarantine, URL rejection, and identical results on two different storage backends."""

import io
import json
import zipfile

import pytest
from cms_fakes import ARCH, FakeResponse, csv_bytes, make_zip

from precursorintelligence.common.io.object_store import sha256_of
from precursorintelligence.ingestion.catalog import discover, select
from precursorintelligence.ingestion.download import run_download
from precursorintelligence.ingestion.pipeline import EXIT_VALIDATION, run_step1


def build_archive(fake):
    # L2 flat, Jan 2021
    fake.add("2021-01-27", make_zip("2021-01", "Jan2021"))
    # L2 in a sub-folder; zip dated Feb but holding Jan data again -> duplicate vintage (later wins)
    fake.add(
        "2021-02-27",
        make_zip("2021-01", "Jan2021", subfolder="nursing_homes_including_rehab_services_02_2021"),
    )
    # 2021-02 is missing entirely (gap); L3 with junk and manifest for 2021-03
    fake.add("2021-03-29", make_zip("2021-03", "Mar2021", layout="L3", junk=True))
    # partial corrective release: only provider_info for 2021-03
    fake.add("2021-04-06", make_zip("2021-03", "Mar2021", layout="L3", tables=["provider_info"]))
    # out-of-scope old zip (date filter)
    fake.add("2020-12-21", make_zip("2020-11", "Nov2020"))
    # current zip, same data month as the newest archive
    fake.add("2021-04-06", make_zip("2021-03", "Mar2021"), kind="current")


def _inv(ctx):
    return sorted(
        (
            r["zip_date"],
            r["file_name"],
            r["table_name"],
            r["vintage_month"],
            r["vintage_source"],
            r["row_count"],
            r["status"],
        )
        for r in ctx.metadata.read("inventory")
    )


def test_full_step1_offline(fake_cms, make_ctx):
    build_archive(fake_cms)
    ctx = make_ctx(fake_cms)
    res = run_step1(ctx)
    assert not res.failures, res.failures
    assert res.selected == 5  # 2020-12-21 excluded by min_zip_date
    assert res.downloads == {"downloaded": 5}

    inv = ctx.metadata.read("inventory")
    prov = {(r["zip_date"], r["zip_type"]): r for r in inv if r["table_name"] == "provider_info"}
    assert prov[("2021-01-27", "theme")]["status"] == "duplicate_vintage"
    assert prov[("2021-02-27", "theme")]["status"] == "selected"  # later zip wins for 2021-01
    assert (
        prov[("2021-04-06", "theme")]["status"] == "selected"
    )  # archive beats current on same date
    assert prov[("2021-04-06", "current")]["status"] == "duplicate_vintage"
    assert all(r["vintage_source"] == "processing_date" for r in prov.values())
    assert all(r["row_count"] == 3 for r in prov.values())
    assert any(r["table_name"] == "junk" for r in inv)
    assert not ctx.object_store.list("quarantine"), (
        "successful downloads must leave quarantine empty"
    )

    kinds = {a["kind"] for a in ctx.metadata.read("anomalies", {"run_id": ctx.run_id})}
    assert {"partial_zip", "missing_vintage", "duplicate_vintage", "junk_members"} <= kinds

    summary = (
        ctx.object_store.open_read("reports/step1/latest/01_inventory_summary.md").read().decode()
    )
    assert "V = **`2021-03`**" in summary and "2021-02" in summary
    assert "2018-03-04" in summary  # earliest survey date in the 2021-01 health citations
    csv_text = ctx.object_store.open_read("reports/step1/latest/01_inventory.csv").read().decode()
    assert csv_text.splitlines()[0].startswith("zip_name,zip_sha256,file_name,table,vintage_month")


def test_rerun_is_idempotent(fake_cms, make_ctx, tmp_path):
    build_archive(fake_cms)
    first = make_ctx(fake_cms, root=tmp_path)
    run_step1(first)
    second = make_ctx(fake_cms, root=tmp_path)
    res = run_step1(second)
    assert res.downloads == {"skipped": 5}
    assert res.extracted == 0
    gets = [u for m, u, h in fake_cms.requests if m == "GET" and u.endswith(".zip")]
    assert len(gets) == 5, "second run must not re-download"
    assert _inv(first) == _inv(second)


def test_republished_zip_is_kept_side_by_side(fake_cms, make_ctx, tmp_path):
    rel = fake_cms.add("2021-01-27", make_zip("2021-01", "Jan2021"))
    run_step1(make_ctx(fake_cms, root=tmp_path))
    fake_cms.replace(rel, make_zip("2021-01", "Jan2021", extra={"readme.txt": b"changed"}))
    ctx = make_ctx(fake_cms, root=tmp_path)
    res = run_step1(ctx)
    assert res.downloads == {"republished": 1}
    keys = ctx.object_store.list("landing/cms_nh/zips")
    assert len(keys) == 2 and any("__" in k for k in keys)  # original + <stem>__<sha8>.zip
    new_sha = sha256_of(ctx.object_store, next(k for k in keys if "__" in k))
    selected = ctx.metadata.read("inventory", {"status": "selected"})
    assert len(selected) == 7, "the republished copy is what downstream loads"
    assert {(r["zip_name"], r["zip_sha256"]) for r in selected} == {
        (f"nursing-homes_2021-01-27__{new_sha[:8]}.zip", new_sha)
    }


def _republish_once(fake_cms, ctx):
    original = make_zip("2021-01", "Jan2021")
    rel = fake_cms.add("2021-01-27", original)
    discover(ctx)
    entries = select(ctx)
    run_download(ctx, entries)
    fake_cms.replace(rel, make_zip("2021-01", "Jan2021", extra={"NH_extra.txt": b"x"}))
    discover(ctx)
    entries = select(ctx)
    assert run_download(ctx, entries)[0].status == "republished"
    return rel, original, entries


def test_republished_then_header_only_change_keeps_correct_object(fake_cms, make_ctx):
    ctx = make_ctx(fake_cms)
    rel, _, entries = _republish_once(fake_cms, ctx)
    fake_cms.last_modified[rel] = "Mon, 01 Jan 2029 00:00:00 GMT"  # same bytes, new header
    r = run_download(ctx, entries)[0]
    assert r.status == "unchanged"
    assert "__" in r.object_key  # the republished copy, not the original
    assert sha256_of(ctx.object_store, r.object_key) == r.sha256


def test_republished_then_reverted_points_at_original(fake_cms, make_ctx):
    ctx = make_ctx(fake_cms)
    rel, original, entries = _republish_once(fake_cms, ctx)
    fake_cms.replace(rel, original)  # CMS goes back to the first version
    discover(ctx)
    r = run_download(ctx, select(ctx))[0]
    assert r.status == "unchanged"
    assert "__" not in r.object_key
    assert sha256_of(ctx.object_store, r.object_key) == r.sha256


def test_corrupt_zip_is_quarantined(fake_cms, make_ctx):
    good = make_zip("2021-01", "Jan2021")
    bad = bytearray(good)
    bad[60] ^= 0xFF  # corrupt member data, keep the PK header
    fake_cms.add("2021-01-27", bytes(bad))
    ctx = make_ctx(fake_cms)
    res = run_step1(ctx)
    assert res.downloads == {"quarantined": 1}
    assert not ctx.object_store.list("landing/cms_nh/zips")
    assert ctx.object_store.list("quarantine/cms_nh")
    # a quarantined zip fails the download step in run_log, like a failed download
    (step,) = [r for r in ctx.metadata.read("run_log") if r["task"] == "download"]
    assert step["status"] == "failed" and json.loads(step["detail"])["n_failures"] == 1


def test_unsafe_catalog_url_is_rejected_without_a_request(fake_cms, make_ctx):
    fake_cms.add(
        "2021-01-27",
        make_zip("2021-01", "Jan2021"),
        path=f"{ARCH}/../../evil/nursing-homes_2021-01-27.zip",
    )
    ctx = make_ctx(fake_cms)
    res = run_step1(ctx)
    assert res.downloads == {"rejected": 1}
    assert not [u for m, u, h in fake_cms.requests if "evil" in u]


@pytest.mark.parametrize("other_backend", ["memory"])
def test_backend_swap_gives_identical_inventory(fake_cms, make_ctx, tmp_path, other_backend):
    build_archive(fake_cms)
    local = make_ctx(fake_cms, backend="local", root=tmp_path / "a")
    run_step1(local)
    other = make_ctx(fake_cms, backend=other_backend, root=tmp_path / "b")
    run_step1(other)
    assert _inv(local) == _inv(other)
    assert other.object_store.root_uri.startswith("memory://")


def test_failed_metastore_lookup_does_not_abort_discover(fake_cms, make_ctx):
    fake_cms.add("2021-01-27", make_zip("2021-01", "Jan2021"))
    ctx = make_ctx(fake_cms)
    url = ctx.http.api_url(ctx.cfg.source.endpoints.metastore_item.format(dataset_id="4pq5-n9py"))
    fake_cms.overrides[url] = FakeResponse(404, b"gone", {"Content-Type": "text/html"})
    out = discover(ctx)
    assert out.n_entries == 1
    ids = {r["dataset_id"] for r in ctx.metadata.read("source_datasets")}
    assert "4pq5-n9py" not in ids and len(ids) == 6


def test_current_entry_without_theme_is_ignored(fake_cms, make_ctx):
    fake_cms.add("2021-01-27", make_zip("2021-01", "Jan2021"))
    fake_cms.add("2021-01-28", make_zip("2021-01", "Jan2021"), kind="current")
    fake_cms.current[0].pop("theme")
    ctx = make_ctx(fake_cms)
    assert discover(ctx).n_entries == 1  # only the archive entry


SEP = "2026-09"


def test_suffixed_release_selects_all_seven_tables(fake_cms, make_ctx):
    """The 2026-10-08 release names files `..._Sep2026_<N>.csv`; before the fix all of them were
    classified `other` and the month was dropped silently."""
    fake_cms.add(
        "2026-10-08",
        make_zip(SEP, "Sep2026", layout="L3", version_suffix=3, real_lookup=True),
    )
    ctx = make_ctx(fake_cms)
    res = run_step1(ctx)
    assert not res.failures, res.failures
    sel = {
        r["table_name"]: r
        for r in ctx.metadata.read("inventory")
        if r["status"] == "selected" and r["vintage_month"] == SEP
    }
    assert set(sel) == set(ctx.cfg.tables)
    assert all(r["file_name"].endswith("_Sep2026_3.csv") for r in sel.values())
    assert sel["citation_lookup"]["vintage_source"] == "filename_token"
    assert sel["provider_info"]["vintage_source"] == "processing_date"
    assert all(r["in_manifest"] for r in sel.values())


def test_zip_with_no_recognised_table_exits_2(fake_cms, make_ctx):
    """A zip whose CSVs match no table pattern (the next file-name change) is an error, not a
    silent empty month."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for t in ("ProviderInfo", "HealthCitations"):
            zf.writestr(f"NH-{t}-Sep2026.csv", csv_bytes(SEP))
    fake_cms.add("2026-10-08", buf.getvalue())
    ctx = make_ctx(fake_cms)
    res = run_step1(ctx)
    assert res.exit_code == EXIT_VALIDATION
    (a,) = [
        a
        for a in ctx.metadata.read("anomalies", {"run_id": ctx.run_id})
        if a["kind"] == "no_tables_recognised"
    ]
    assert a["severity"] == "error" and "2 CSV files" in a["detail"]
    (step,) = [r for r in ctx.metadata.read("run_log") if r["task"] == "resolve"]
    assert step["status"] == "failed" and json.loads(step["detail"])["n_failures"] == 1


def test_zip_with_only_known_out_of_scope_files_is_not_an_error(fake_cms, make_ctx):
    """Known CMS files we don't ingest (e.g. FireSafety) are not "unrecognised"."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("NH_FireSafetyCitations_Sep2026.csv", csv_bytes(SEP))
        zf.writestr("NH_Ownership_Sep2026_3.csv", csv_bytes(SEP))
    fake_cms.add("2026-10-08", buf.getvalue())
    ctx = make_ctx(fake_cms)
    res = run_step1(ctx)
    assert res.exit_code == 0, res.failures
    kinds = {a["kind"] for a in ctx.metadata.read("anomalies", {"run_id": ctx.run_id})}
    assert "no_tables_recognised" not in kinds


def test_the_higher_file_version_wins_numerically(fake_cms, make_ctx):
    """`_10` beats `_9` in the same zip, and the duplicate anomaly names both files."""
    v9 = "4pq5-n9py_2026-09-01_NH_ProviderInfo_Sep2026_9.csv"
    fake_cms.add(
        "2026-10-08",
        make_zip(SEP, "Sep2026", layout="L3", version_suffix=10, extra={v9: csv_bytes(SEP)}),
    )
    ctx = make_ctx(fake_cms)
    run_step1(ctx)
    prov = {
        r["file_name"]: r["status"]
        for r in ctx.metadata.read("inventory")
        if r["table_name"] == "provider_info"
    }
    assert prov == {
        "4pq5-n9py_2026-09-01_NH_ProviderInfo_Sep2026_10.csv": "selected",
        v9: "duplicate_vintage",
    }
    (dup,) = [
        a["detail"]
        for a in ctx.metadata.read("anomalies", {"run_id": ctx.run_id})
        if a["kind"] == "duplicate_vintage"
    ]
    assert "Sep2026_10.csv" in dup and "Sep2026_9.csv" in dup
