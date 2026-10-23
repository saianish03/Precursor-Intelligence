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
    # L3 with a version suffix (the 2026-10-08 release, Sep 2026 data)
    "4pq5-n9py_2026-09-30_NH_ProviderInfo_Sep2026_3.csv": "provider_info",
    "r5ix-sfxw_2026-09-30_NH_HealthCitations_Sep2026_2.csv": "health_citations",
    "svdt-c123_2026-09-30_NH_SurveyDates_Sep2026_2.csv": "survey_dates",
    "g6vv-u9sr_2026-09-30_NH_Penalties_Sep2026_2.csv": "penalties",
    "djen-97ju_2026-09-30_NH_QualityMsr_MDS_Sep2026_2.csv": "mds_qm",
    "ijh5-nb2v_2026-09-30_NH_QualityMsr_Claims_Sep2026_2.csv": "claims_qm",
    "tagd-9999_2026-09-30_NH_CitationDescriptions_Sep2026_2.csv": "citation_lookup",
    "ifjz-ge4w_2026-09-30_NH_FireSafetyCitations_Sep2026_2.csv": OTHER,
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
    assert (
        detect_layout(["4pq5-n9py_2026-07-01_NH_ProviderInfo_Jul2026.csv", "manifest.json"]) == "L3"
    )


def test_filename_month_and_dates():
    assert filename_month("NH_ProviderInfo_Nov2021.csv") == "2021-11"
    assert filename_month("4pq5-n9py_2026-07-29_NH_ProviderInfo_Jul2026.csv") == "2026-07"
    assert filename_month("tagd-9999_2026-09-30_NH_CitationDescriptions_Sep2026_2.csv") == "2026-09"
    assert filename_month("ProviderInfo_Download.csv") is None
    assert str(parse_date("2021-11-01")) == "2021-11-01"
    assert str(parse_date("11/01/2021")) == "2021-11-01"
    assert parse_date("garbage") is None


def test_processing_date_read_from_first_row_with_bom():
    data = (
        '﻿CMS Certification Number (CCN),Provider Name,Processing Date\n015009,"A, B",2021-11-01\n'
    )
    header, row = read_header_and_first_row(io.BytesIO(data.encode("utf-8")))
    assert header[0] == "CMS Certification Number (CCN)"
    assert str(processing_date_from(header, row)) == "2021-11-01"
    assert header_hash(header) == header_hash(list(header))


def test_manifest_filenames():
    raw = b'[{"dataset_id":"x","resources":[{"filename":"NH_ProviderInfo_Aug2026.csv"}]}]'
    assert manifest_filenames(raw) == {"NH_ProviderInfo_Aug2026.csv"}


# One real name per known out-of-scope family (all 826 such CSVs in the 67 archive zips match).
OUT_OF_SCOPE = [
    "NH_StateUSAverages_Jan2021.csv",
    "NH_SurveySummary_Jan2021.csv",
    "NH_Ownership_Jan2021.csv",
    "NH_HlthInspecCutpointsState_Jan2021.csv",
    "NH_FireSafetyCitations_Jan2021.csv",
    "NH_DataCollectionIntervals_Jan2021.csv",
    "NH_QMDataCollection_Complaint_Periods_Jan2021.csv",
    "NH_CovidVaxProvider_20230917.csv",
    "Skilled_Nursing_Facility_Quality_Reporting_Program_National_Data_Jan2021.csv",
    "5sqm-2qku_2026-07-01_Skilled_Nursing_Facility_Quality_Reporting_Program_National_Data_Jul2026.csv",
    "6uyb-waub_2026-07-01_Swing_Bed_SNF_data_Jul2026.csv",
    "FY_2023_SNF VBP_Aggregate_Performance.csv",
    "pdc_s3_nhc_data_yf87_tv6k.csv",
]


def test_out_of_scope_patterns_cover_known_files_and_never_an_in_scope_table(tmp_path, classifier):
    import re

    patterns = [re.compile(p) for p in make_settings(tmp_path).sources.out_of_scope_patterns]
    for name in OUT_OF_SCOPE:
        assert classifier.classify(name) == OTHER and any(p.search(name) for p in patterns), name
    for name, table in CASES.items():
        if table != OTHER:
            assert not any(p.search(name) for p in patterns), name
