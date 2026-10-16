"""All-text parsing and the blank-row drop of docs/DATA_PIPELINE_SPEC.md §3.1 steps 4 and 5."""

import polars as pl

from precursorintelligence.harmonize.reader import parse_csv_text

HEADER = '"CMS Certification Number (CCN)","Provider Name","Overall Rating"\r\n'


def test_every_column_is_text_and_leading_zeros_and_letters_survive():
    text = HEADER + '"015009","Oak Manor","5"\r\n"15E123","Pine Court","3"\r\n'
    parsed = parse_csv_text(text)
    assert set(parsed.frame.schema.values()) == {pl.String}
    assert parsed.frame["CMS Certification Number (CCN)"].to_list() == ["015009", "15E123"]
    assert parsed.frame["Overall Rating"].to_list() == ["5", "3"]


def test_original_header_names_and_order_are_kept():
    parsed = parse_csv_text(HEADER + '"015009","Oak Manor","5"\r\n')
    assert parsed.frame.columns == [
        "CMS Certification Number (CCN)",
        "Provider Name",
        "Overall Rating",
    ]


def test_quoted_commas_and_newlines_stay_inside_one_field():
    text = (
        HEADER + '"015009","Oak Manor, East Wing","line one\nline two"\r\n"015010","Pine","1"\r\n'
    )
    parsed = parse_csv_text(text)
    assert parsed.frame.height == 2
    assert parsed.frame["Provider Name"][0] == "Oak Manor, East Wing"
    assert parsed.frame["Overall Rating"][0] == "line one\nline two"


def test_the_trailing_blank_line_is_dropped_and_counted():
    parsed = parse_csv_text(HEADER + '"015009","Oak Manor","5"\r\n\r\n')
    assert parsed.frame.height == 1
    assert parsed.blank_rows == 1


def test_a_row_with_every_field_empty_is_blank_but_a_partly_empty_row_is_data():
    parsed = parse_csv_text(HEADER + ',,\r\n"015009",,\r\n')
    assert parsed.frame["CMS Certification Number (CCN)"].to_list() == ["015009"]
    assert parsed.frame["Provider Name"].to_list() == [None]  # an empty field is null, not ""
    assert parsed.blank_rows == 1


def test_a_file_with_no_blank_line_reports_zero_blank_rows():
    assert parse_csv_text(HEADER + '"015009","Oak Manor","5"\r\n').blank_rows == 0


def test_a_header_only_file_is_an_empty_frame_with_its_columns():
    parsed = parse_csv_text(HEADER)
    assert parsed.frame.height == 0 and parsed.blank_rows == 0
    assert parsed.frame.columns[0] == "CMS Certification Number (CCN)"
