"""Decoding rules of docs/DATA_PIPELINE_SPEC.md §3.1 step 3: strict UTF-8, then cp1252, never lossy."""

import pytest

from precursorintelligence.harmonize.reader import DecodeError, decode_csv


def test_utf8_file_is_decoded_as_utf8_with_multibyte_characters_intact():
    text, encoding = decode_csv("Provider Name\nCafé Résidence – Oakwood\n".encode())
    assert encoding == "utf-8"
    assert "Café Résidence – Oakwood" in text


def test_cp1252_curly_apostrophe_0x92_falls_back_to_cp1252():
    raw = b"Provider Name\nSt. Mary\x92s Nursing Home\n"
    text, encoding = decode_csv(raw)
    assert encoding == "cp1252"
    assert "St. Mary’s Nursing Home" in text


def test_a_utf8_file_is_never_read_as_cp1252():
    # "â€™" is what a UTF-8 curly apostrophe turns into when wrongly decoded as cp1252.
    text, _ = decode_csv("Mary’s\n".encode())
    assert "â" not in text and "’" in text


def test_pure_ascii_counts_as_utf8():
    assert decode_csv(b"a,b\n1,2\n") == ("a,b\n1,2\n", "utf-8")


def test_bytes_undefined_in_cp1252_fail_instead_of_being_replaced():
    # 0x81 is not valid UTF-8 here and is undefined in cp1252: nothing may be silently replaced.
    with pytest.raises(DecodeError):
        decode_csv(b"Provider Name\nBad\x81Name\n")
