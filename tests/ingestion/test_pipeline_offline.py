"""End-to-end Step 1 against a fake CMS server: layouts, duplicates, gaps, idempotency, re-publication,
quarantine, URL rejection, and identical results on two different storage backends."""

import pytest
from cms_fakes import ARCH, make_zip

from precursorintelligence.ingestion.pipeline import run_step1


def build_archive(fake):
    # L2 flat, Jan 2021
    fake.add("2021-01-27", make_zip("2021-01", "Jan2021"))
    # L2 in a sub-folder; zip dated Feb but holding Jan data again -> duplicate vintage (later wins)
    fake.add("2021-02-27", make_zip("2021-01", "Jan2021", subfolder="nursing_homes_including_rehab_services_02_2021"))
    # 2021-02 is missing entirely (gap); L3 with junk and manifest for 2021-03
    fake.add("2021-03-29", make_zip("2021-03", "Mar2021", layout="L3", junk=True))
    # partial corrective release: only provider_info for 2021-03
    fake.add("2021-04-06", make_zip("2021-03", "Mar2021", layout="L3", tables=["provider_info"]))
    # out-of-scope old zip (date filter)
    fake.add("2020-12-21", make_zip("2020-11", "Nov2020"))
    # current zip, same data month as the newest archive
    fake.add("2021-04-06", make_zip("2021-03", "Mar2021"), kind="current")


def _inv(ctx):
    return sorted((r["zip_date"], r["file_name"], r["table_name"], r["vintage_month"], r["vintage_source"],
                   r["row_count"], r["status"]) for r in ctx.metadata.read("inventory"))


def test_full_step1_offline(fake_cms, make_ctx):
    build_archive(fake_cms)
    ctx = make_ctx(fake_cms)
    res = run_step1(ctx)
    assert not res.failures, res.failures
    assert res.selected == 5                      # 2020-12-21 excluded by min_zip_date
    assert res.downloads == {"downloaded": 5}

    inv = ctx.metadata.read("inventory")
    prov = {(r["zip_date"], r["zip_type"]): r for r in inv if r["table_name"] == "provider_info"}
    assert prov[("2021-01-27", "theme")]["status"] == "duplicate_vintage"
    assert prov[("2021-02-27", "theme")]["status"] == "selected"          # later zip wins for 2021-01
    assert prov[("2021-04-06", "theme")]["status"] == "selected"          # archive beats current on same date
    assert prov[("2021-04-06", "current")]["status"] == "duplicate_vintage"
    assert all(r["vintage_source"] == "processing_date" for r in prov.values())
    assert all(r["row_count"] == 3 for r in prov.values())
    assert any(r["table_name"] == "junk" for r in inv)
    assert not ctx.object_store.list("quarantine"), "successful downloads must leave quarantine empty"

    kinds = {a["kind"] for a in ctx.metadata.read("anomalies", {"run_id": ctx.run_id})}
    assert {"partial_zip", "missing_vintage", "duplicate_vintage", "junk_members"} <= kinds

    summary = ctx.object_store.open_read("reports/step1/latest/01_inventory_summary.md").read().decode()
    assert "V = **`2021-03`**" in summary and "2021-02" in summary
    assert "2018-03-04" in summary                  # earliest survey date in the 2021-01 health citations
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
    assert len(keys) == 2 and any("__" in k for k in keys)            # original + <stem>__<sha8>.zip


def test_corrupt_zip_is_quarantined(fake_cms, make_ctx):
    good = make_zip("2021-01", "Jan2021")
    bad = bytearray(good)
    bad[60] ^= 0xFF                                                     # corrupt member data, keep the PK header
    fake_cms.add("2021-01-27", bytes(bad))
    ctx = make_ctx(fake_cms)
    res = run_step1(ctx)
    assert res.downloads == {"quarantined": 1}
    assert not ctx.object_store.list("landing/cms_nh/zips")
    assert ctx.object_store.list("quarantine/cms_nh")


def test_unsafe_catalog_url_is_rejected_without_a_request(fake_cms, make_ctx):
    fake_cms.add("2021-01-27", make_zip("2021-01", "Jan2021"), path=f"{ARCH}/../../evil/nursing-homes_2021-01-27.zip")
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
