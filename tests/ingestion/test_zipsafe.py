import io
import stat
import zipfile

import pytest

from precursorintelligence.common.config import ZipLimits
from precursorintelligence.ingestion.zipsafe import UnsafeArchiveError, inspect_archive, is_junk, verify_crc


def _zip(entries: dict[str, bytes], *, symlink: str | None = None) -> io.BytesIO:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
        if symlink:
            zi = zipfile.ZipInfo(symlink)
            zi.external_attr = (stat.S_IFLNK | 0o777) << 16
            zf.writestr(zi, "/etc/passwd")
    buf.seek(0)
    return buf


def test_good_archive_and_junk_flags():
    members = inspect_archive(_zip({"a/NH_ProviderInfo_Jan2021.csv": b"x,y\n1,2\n", "__MACOSX/._a": b"j",
                                    ".DS_Store": b"j"}), ZipLimits())
    assert {m.file_name: m.is_junk for m in members} == {"NH_ProviderInfo_Jan2021.csv": False, "._a": True,
                                                         ".DS_Store": True}


@pytest.mark.parametrize("name", ["../evil.csv", "a/../../evil.csv", "/abs/evil.csv", "C:/evil.csv", "a\\..\\evil.csv"])
def test_zip_slip_names_are_rejected(name):
    with pytest.raises(UnsafeArchiveError):
        inspect_archive(_zip({name: b"x"}), ZipLimits())


def test_symlink_member_is_rejected():
    with pytest.raises(UnsafeArchiveError, match="symlink"):
        inspect_archive(_zip({"ok.csv": b"x"}, symlink="link.csv"), ZipLimits())


def test_zip_bomb_ratio_is_rejected():
    limits = ZipLimits(max_compression_ratio=50, ratio_check_min_bytes=1024 * 1024)
    with pytest.raises(UnsafeArchiveError, match="compression ratio"):
        inspect_archive(_zip({"bomb.csv": b"\x00" * (20 * 1024 * 1024)}), limits)


def test_member_count_and_total_size_limits():
    with pytest.raises(UnsafeArchiveError, match="members"):
        inspect_archive(_zip({f"f{i}.csv": b"x" for i in range(6)}), ZipLimits(max_members=5))
    with pytest.raises(UnsafeArchiveError, match="total uncompressed"):
        inspect_archive(_zip({"a.csv": b"x" * 5000}), ZipLimits(max_total_uncompressed_bytes=1000))


def test_nested_zip_rejected_by_default():
    with pytest.raises(UnsafeArchiveError, match="nested"):
        inspect_archive(_zip({"inner.zip": b"PK"}), ZipLimits())


def test_not_a_zip_and_crc():
    with pytest.raises(UnsafeArchiveError):
        inspect_archive(io.BytesIO(b"not a zip at all"), ZipLimits())
    good = _zip({"a.csv": b"hello world" * 100})
    verify_crc(good)
    raw = bytearray(good.getvalue())
    idx = raw.find(b"a.csv") + len(b"a.csv")   # first byte of compressed data of the only member
    raw[idx + 5] ^= 0xFF
    with pytest.raises(UnsafeArchiveError):
        verify_crc(io.BytesIO(bytes(raw)))


def test_is_junk():
    assert is_junk("__MACOSX/x/._y.csv") and is_junk("a/.DS_Store") and is_junk("a/._x")
    assert not is_junk("a/NH_ProviderInfo_Jan2021.csv")
