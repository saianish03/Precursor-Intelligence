"""A small silver ``citation_lookup_versions``: real CMS deficiency-tag rows (NH_CitationDescriptions, public), with
two reworded tags (F0605, K0311), a legacy F-tag that repeats a current one's wording (F0323 / F0689), and CMS's
double space in F0690. Newest vintage 2026-10."""

from __future__ import annotations

import io

import polars as pl

from precursorintelligence.agents.corpus import LOOKUP_VERSIONS_KEY
from precursorintelligence.common.io.object_store import ObjectStore

NEWEST = "2026-10"

EP = "Emergency Preparedness Deficiencies"
RR = "Resident Rights Deficiencies"
ABUSE = "Freedom from Abuse, Neglect, and Exploitation Deficiencies"
CARE_PLAN = "Resident Assessment and Care Planning Deficiencies"
QOL = "Quality of Life and Care Deficiencies"
NURSING = "Nursing and Physician Services Deficiencies"
PHARMACY = "Pharmacy Service Deficiencies"
ADMIN = "Administration Deficiencies"

# (prefix, tag, category, description, valid_from, valid_to)
ROWS: list[tuple[str, str, str, str, str, str]] = [
    ("E", "0001", EP, "Establish an Emergency Preparedness Program (EP).", "2021-01", NEWEST),
    ("E", "0039", EP, "Conduct testing and exercise requirements.", "2021-01", NEWEST),
    (
        "F",
        "0550",
        RR,
        "Honor the resident's right to a dignified existence, self-determination, communication, "
        "and to exercise his or her rights.",
        "2021-01",
        NEWEST,
    ),
    (
        "F",
        "0600",
        ABUSE,
        "Protect each resident from all types of abuse such as physical, mental, sexual abuse, "
        "physical punishment, and neglect by anybody.",
        "2021-01",
        NEWEST,
    ),
    (
        "F",
        "0605",
        ABUSE,
        "Ensure that each resident is free from medications that restrain them, unless needed for "
        "medical treatment.",
        "2021-01",
        "2025-03",
    ),
    (
        "F",
        "0605",
        ABUSE,
        "Prevent the use of unnecessary psychotropic medications or use medications that may "
        "restrain a resident's ability to function.",
        "2025-04",
        NEWEST,
    ),
    (
        "F",
        "0609",
        ABUSE,
        "Timely report suspected abuse, neglect, or theft and report the results of the "
        "investigation to proper authorities.",
        "2021-01",
        NEWEST,
    ),
    (
        "F",
        "0656",
        CARE_PLAN,
        "Develop and implement a complete care plan that meets all the resident's needs, with "
        "timetables and actions that can be measured.",
        "2021-01",
        NEWEST,
    ),
    (
        "F",
        "0686",
        QOL,
        "Provide appropriate pressure ulcer care and prevent new ulcers from developing.",
        "2021-01",
        NEWEST,
    ),
    (
        "F",
        "0689",
        QOL,
        "Ensure that a nursing home area is free from accident hazards and provides adequate "
        "supervision to prevent accidents.",
        "2021-01",
        NEWEST,
    ),
    (
        "F",
        "0323",
        QOL,
        "Make sure that the nursing home area is free from accident hazards and risks and provides "
        "supervision to prevent avoidable accidents",
        "2021-01",
        NEWEST,
    ),
    (
        "F",
        "0690",
        QOL,
        "Provide appropriate care for residents who are continent or incontinent of bowel/bladder, "
        "appropriate catheter care,  and appropriate care  to prevent urinary tract infections.",
        "2021-01",
        NEWEST,
    ),
    (
        "F",
        "0697",
        QOL,
        "Provide safe, appropriate pain management for a resident who requires such services.",
        "2021-01",
        NEWEST,
    ),
    (
        "F",
        "0725",
        NURSING,
        "Provide enough nursing staff every day to meet the needs of every resident; and have a "
        "licensed nurse in charge on each shift.",
        "2021-01",
        NEWEST,
    ),
    (
        "F",
        "0758",
        PHARMACY,
        "Implement gradual dose reductions(GDR) and non-pharmacological interventions, unless "
        "contraindicated, prior to initiating or instead of continuing psychotropic medication; and PRN orders for "
        "psychotropic medications are only used when the medication is necessary and PRN use is limited.",
        "2021-01",
        NEWEST,
    ),
    (
        "F",
        "0760",
        PHARMACY,
        "Ensure that residents are free from significant medication errors.",
        "2021-01",
        NEWEST,
    ),
    (
        "F",
        "0761",
        PHARMACY,
        "Ensure drugs and biologicals used in the facility are labeled in accordance with "
        "currently accepted professional principles; and all drugs and biologicals must be stored in locked "
        "compartments, separately locked, compartments for controlled drugs.",
        "2021-01",
        NEWEST,
    ),
    (
        "F",
        "0812",
        "Nutrition and Dietary Deficiencies",
        "Procure food from sources approved or considered "
        "satisfactory and store, prepare, distribute and serve food in accordance with professional standards.",
        "2021-01",
        NEWEST,
    ),
    (
        "F",
        "0867",
        ADMIN,
        "Set up an ongoing quality assessment and assurance group to review quality deficiencies  "
        "and develop corrective plans of action.",
        "2021-01",
        NEWEST,
    ),
    (
        "F",
        "0880",
        "Infection Control Deficiencies",
        "Provide and implement an infection prevention and control program.",
        "2021-01",
        NEWEST,
    ),
    (
        "F",
        "0947",
        NURSING,
        "Ensure nurse aides have the skills they need to care for residents, and give nurse "
        "aides education in dementia care and abuse prevention.",
        "2021-01",
        NEWEST,
    ),
    (
        "K",
        "0311",
        "Smoke Deficiencies",
        "Have protected exits that allow the resident to escape the building.",
        "2021-01",
        "2021-03",
    ),
    (
        "K",
        "0311",
        "Smoke Deficiencies",
        "Have an enclosure around a vertical opening shaft.",
        "2021-04",
        NEWEST,
    ),
    (
        "K",
        "0353",
        "Smoke Deficiencies",
        "Inspect, test, and maintain automatic sprinkler systems.",
        "2021-01",
        NEWEST,
    ),
    (
        "K",
        "0712",
        "Miscellaneous Deficiencies",
        "Have simulated fire drills held at unexpected times.",
        "2021-01",
        NEWEST,
    ),
    (
        "K",
        "0918",
        "Gas, Vacuum, and Electrical Systems Deficiencies",
        "Have generator or other power source capable of supplying service within 10 seconds.",
        "2021-01",
        NEWEST,
    ),
]


def lookup_frame(rows: list[tuple[str, str, str, str, str, str]] | None = None) -> pl.DataFrame:
    rows = ROWS if rows is None else rows
    return pl.DataFrame(
        {
            "deficiency_prefix": [r[0] for r in rows],
            "deficiency_tag": [r[1] for r in rows],
            "deficiency_prefix_and_number": [f"{r[0]}-{r[1]}" for r in rows],
            "deficiency_description": [r[3] for r in rows],
            "deficiency_category": [r[2] for r in rows],
            "valid_from_vintage": [r[4] for r in rows],
            "valid_to_vintage": [r[5] for r in rows],
        }
    )


def write_lookup(store: ObjectStore, frame: pl.DataFrame | None = None) -> None:
    buf = io.BytesIO()
    (lookup_frame() if frame is None else frame).write_parquet(buf)
    store.put_bytes(LOOKUP_VERSIONS_KEY, buf.getvalue(), overwrite=True)
