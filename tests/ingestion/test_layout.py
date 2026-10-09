"""Member names below are real names read from the CMS archive (one sample per layout)."""

import io

import pytest
from cms_fakes import make_settings

from precursorintelligence.ingestion.layout import (
    OTHER,
    TableClassifier,
    detect_layout,
    filename_month,
    header_hash,
    manifest_filenames,
    parse_date,
    processing_date_from,
    read_header_and_first_row,
)

CASES = {
    # L1 (2019)
    "ProviderInfo_Download.csv": "provider_info",
    "HealthDeficiencies_Download.csv": "health_citations",
    "Penalties_Download.csv": "penalties",
    "QualityMsrMDS_Download.csv": "mds_qm",
    "QualityMsrClaims_Download.csv": "claims_qm",
    "FireSafetyDeficiencies_Download.csv": OTHER,
    "SurveySummary_Download.csv": OTHER,
    # L2 (2021)
    "NH_ProviderInfo_Jan2021.csv": "provider_info",
    "NH_HealthCitations_Nov2021.csv": "health_citations",
    "NH_SurveyDates_Nov2021.csv": "survey_dates",
    "NH_Penalties_Nov2021.csv": "penalties",
    "NH_QualityMsr_MDS_Nov2021.csv": "mds_qm",
    "NH_QualityMsr_Claims_Nov2021.csv": "claims_qm",
    "NH_CitationDescriptions_Nov2021.csv": "citation_lookup",
    "NH_FireSafetyCitations_Nov2021.csv": OTHER,
    "NH_SurveySummary_Nov2021.csv": OTHER,
    "NH_QMDataCollection_Complaint_Periods_Jan2021.csv": OTHER,
    "NH_CovidVaxProvider_20211212.csv": OTHER,
    "Swing_Bed_SNF_data_Nov2021.csv": OTHER,
    # L3 (2026-07 onward)
    "4pq5-n9py_2026-07-01_NH_ProviderInfo_Jul2026.csv": "provider_info",
    "r5ix-sfxw_2026-08-01_NH_HealthCitations_Aug2026.csv": "health_citations",
    "svdt-c123_2026-08-01_NH_SurveyDates_Aug2026.csv": "survey_dates",
    "tagd-9999_2026-08-01_NH_CitationDescriptions_Aug2026.csv": "citation_lookup",
    "ifjz-ge4w_2026-07-01_NH_FireSafetyCitations_Jul2026.csv": OTHER,
}


@pytest.fixture(scope="module")
def classifier(tmp_path_factory):
    return TableClassifier(make_settings(tmp_path_factory.mktemp("c")).sources.tables)


@pytest.mark.parametrize("name,table", CASES.items())
def test_classify_real_names(classifier, name, table):
    assert classifier.classify(name) == table


def test_detect_layout():
    assert detect_layout(["ProviderInfo_Download.csv"]) == "L1"
    assert detect_layout(["NH_ProviderInfo_Jan2021.csv", "readme.txt"]) == "L2"
    assert detect_layout(["4pq5-n9py_2026-07-01_NH_ProviderInfo_Jul2026.csv", "manifest.json"]) == "L3"


def test_filename_month_and_dates():
    assert filename_month("NH_ProviderInfo_Nov2021.csv") == "2021-11"
    assert filename_month("4pq5-n9py_2026-07-29_NH_ProviderInfo_Jul2026.csv") == "2026-07"
    assert filename_month("ProviderInfo_Download.csv") is None
    assert str(parse_date("2021-11-01")) == "2021-11-01"
    assert str(parse_date("11/01/2021")) == "2021-11-01"
    assert parse_date("garbage") is None


def test_processing_date_read_from_first_row_with_bom():
    data = "﻿CMS Certification Number (CCN),Provider Name,Processing Date\n015009,\"A, B\",2021-11-01\n"
    header, row = read_header_and_first_row(io.BytesIO(data.encode("utf-8")))
    assert header[0] == "CMS Certification Number (CCN)"
    assert str(processing_date_from(header, row)) == "2021-11-01"
    assert header_hash(header) == header_hash(list(header))


def test_manifest_filenames():
    raw = b'[{"dataset_id":"x","resources":[{"filename":"NH_ProviderInfo_Aug2026.csv"}]}]'
    assert manifest_filenames(raw) == {"NH_ProviderInfo_Aug2026.csv"}
