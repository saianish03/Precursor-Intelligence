# Deep research round 2: why the model is stuck at ROC 0.71 and what the way out is (2026-11-07)

Inputs: 4 deep web-research passes (CMS data-source verification, survey process and drift, inspection-outcome ML
literature, alternative non-CMS data) + error analysis and quick signal diagnostics on our own test month (2025-02,
exp12 old-ensemble scores, 14,643 rows, 3,750 positives). Round 1 shortlist: `RESEARCH_SOLUTIONS.md`.

## 1. What our own data says (error analysis)

Label reconstructed from the silver citation ledger (G+ F-tag, survey date 2025-02-01..2026-01-31; 99.85 % agreement
with `y_g12`).

**Where the positives come from**

| G+ came from | Share of positives | Recall in top 20 % | Mean score |
|---|---|---|---|
| complaint survey only | **68 %** | 39 % | 0.36 |
| both complaint and standard | 18 % | 45 % | 0.40 |
| standard survey only | 14 % | 30 % | 0.32 |

**Prior history of the positives**

| Prior G+ in 36 months | Share of positives | Recall in top 20 % |
|---|---|---|
| none ("first-timers") | **32 %** | **6 %** |
| 1–2 | 33 % | 34 % |
| 3+ | 35 % | 74 % |

**How well the model ranks within each kind of row**

| Rows | n | Base rate | ROC | PR-AUC |
|---|---|---|---|---|
| had a standard survey in the window | 10,126 | 0.280 | 0.702 | 0.497 |
| had ≥ 1 complaint survey in the window | 8,770 | 0.406 | **0.632** | 0.557 |
| no health survey at all | 2,261 | 0.000 | — | — |

**Oracle test: what if we knew the future?** These features peek at the future and can't be used; they show where the
missing information is.

| Added to the current score | ROC | PR-AUC | Best accuracy |
|---|---|---|---|
| nothing | 0.707 | 0.472 | 76.1 % |
| knew whether a standard survey would happen | 0.709 | 0.475 | 76.1 % |
| **knew how many complaint surveys would happen** | **0.819** | **0.584** | **78.2 %** |

**Conclusion:** the ceiling is **complaint-survey arrival**: who gets complained about in the next 12 months.
Standard-survey timing is already solved. Even perfect knowledge of complaint arrivals only reaches ~78 % accuracy,
because the label is 74 % negative and surveyor judgement is noisy. **80 % accuracy is not reachable honestly.**

**Quick diagnostics on observable proxies** (5-fold logistic stacking on the test month; indicative, not a formal
experiment):

| Added to the current score | ROC | PR-AUC |
|---|---|---|
| nothing | 0.7092 | 0.4800 |
| facility-reported incidents + substantiated complaints (Provider Info, **dropped in silver today**) | 0.7120 | 0.4834 |
| sibling homes' G+ rate through shared owners (87 % coverage) | 0.7093 | 0.4803 |
| state strictness (G+ share of health surveys, last 3 / 6 / 12 months) | 0.7112 | 0.4816 |
| sibling + state strictness | 0.7114 | 0.4820 |

Each observable proxy adds +0.001 to +0.003 ROC, the same scale as PBJ (exp12) and case-mix (exp7).

**Other facts we checked**
- **Complaint surveys with no citation are invisible.** Inspection Dates lists a complaint survey only if it produced a
  citation (`survey_timing.py`). Only 0.1 % of complaint surveys in 2024–26 have no citation. Complaint intake can't be
  rebuilt from survey data.
- **Chain ID / Affiliated Entity appear in Provider Info only from 2026.** That is too late for 2023 training. Owner
  names are the only chain proxy before then.
- **State miscalibration is large.** The standard deviation of the per-state gap between actual and predicted rate is
  7.4 points; examples are state code 37 (+17 points under-predicted) and 50 (−19).

## 2. What the literature says

- **No published ML model predicts CMS G+ citations.** Comparable "inspector judgement" labels top out low:

  | Study | Result |
  |---|---|
  | Ontario nursing homes, next inspection (3 classes) | 40 % accuracy, called "weak" |
  | England CQC (GP practices) | κ ≈ 0.1; accuracy at or below the majority class |
  | England CQC (acute hospitals) | 24 % out of sample |
  | US nursing-home COVID-outbreak GBM | AUC 0.72 |

  High AUCs (0.93–0.97, customs and tax) come from document-determined labels. Our 0.71 is at or above the published
  state of the art.
- **The label is noisy.**
  - Federal comparison surveys found G+ deficiencies that state surveyors missed in ~12–15 % of surveys; above 25 % in
    9 states (GAO-10-70, GAO-08-517).
  - 20 % of disputed deficiencies are deleted and 7 % downgraded (GAO, FY2007).
  - G+ share of deficiencies by state, 2022–24: Kentucky 17.1 %, Tennessee 15.3 %, Nevada 1.5 %. Alabama went from 5.3 % to
    18.3 % between 2023 and 2024.
  - Estimated ceiling: **ROC ~0.75–0.78, PR-AUC ~0.52–0.56**. We are within ~0.05 of it.
- **Drift drivers between 2023 and 2025:**
  - States were catching up on the COVID survey backlog. Only 56 % of facilities had a standard survey in 2022 and 65 %
    in 2024; California went from 38 % to 72 %.
  - New complaint deadlines took effect on 2023-10-01 (QSO-22-19-NH).
  - Phase-3 surveyor guidance took effect on 2022-10-24, and revised guidance on 2025-04-28.
  - The CMP rule changed on 2024-08-06.
  - New regime break: the risk-based survey (QSO-26-14-NH) started on 2026-09-08.
- **The biggest lift in comparable domains came from the inspector side.**
  - Chicago food inspections: inspector cluster was the top feature (odds ratio ~4.7). The JHU/APL replication got
    almost the same result with inspector ID alone.
  - Customs fraud (DATE, KDD 2020): entity embeddings were the largest ablation. GraphFC (CIKM 2023) added +252 %
    recall from a graph linking shared entities.
  - CMS publishes no surveyor IDs. Proxies are state × recent-period strictness and team clusters (same state, same
    week, nearby). Our state strictness proxy gave +0.002 ROC.
- **Alternative data:**
  - CDC NHSN weekly shortage and outbreak flags (CCN-keyed, weekly) were never tested against citations.
  - HCRIS cost reports have mixed evidence.
  - Online reviews were weak in all studies (IRR 0.93; Yelp disagreed with Care Compare 75 % of the time) and are
    hard to get with timestamps.
  - County labour market: +10 points nurse turnover → +19 % deficiencies, but PBJ turnover already covers most of it.
- **Field wins are measured as hit rate at a fixed budget, not accuracy.**

  | Domain | Hit-rate gain |
  |---|---|
  | EPA hazardous waste (randomized field test) | +79–82 % |
  | Clean Water Act | 2×–7× |
  | Boston restaurants (Yelp) | +25 % more violations found |

  Our top 2 % is 80 % precise vs a 26 % base rate (3.1× lift).

## 3. The way: three tracks

### Track A — stack every small real gain (low risk, ~1 day)

One combined experiment, **exp13**, using exp12's design (train 2023-01..11, test 2025-02 + 2024-11, 3-fold keep rule,
clustered bootstrap CIs). Add the families one at a time, then together:

1. Complaint / incident intake:
   - facility-reported incidents and substantiated complaints (bring from bronze into silver/gold);
   - their 6-month and 12-month change;
   - a regime flag for the 2025-07 iQIES definition change.
2. Inspector-side proxies:
   - state strictness in the last 3 / 6 / 12 months, by survey type (complaint vs standard) and for J+;
   - state backlog index (share of facilities over 15.9 months since last standard survey) and its 6-month change;
   - state survey throughput.
3. Entity: leave-one-out sibling G+ rate through shared owners (and county).
4. PBJ (exp12) and case-mix HPRD (exp7) features.
5. Exponential recency weights (τ chosen on a 2024 month).
6. Per-state recalibration of the final score using only matured labels.

Expected: **ROC ~0.715–0.725, PR-AUC ~0.49–0.50**. Accuracy stays ~76 %.

### Track B — reframe how the model is judged (no compute)

- Lead with **lift and precision in the top 2 / 5 / 10 %**, how inspection-targeting work is judged (EPA, Chicago, Clean
  Water Act). Report accuracy honestly next to the 74.4 % majority baseline.
- Split the score into P(survey) × P(harm | survey), and add a separate complaint-risk score. Insurers can then see why
  a facility is flagged.
- Explicitly state the ceiling with the evidence in §2: label noise and surveyor variance cap ROC around 0.75–0.78.

### Track C — the only path to a real jump: new information about complaints (weeks, outside our data)

The oracle shows that complaint arrival is worth +0.11 ROC. Sources that carry part of it:

- **CDC NHSN weekly LTC data** (staff/supply shortage flags, outbreaks, reporting gaps): CCN-keyed, weekly, free. This
  is the most promising untested source. **Next step: a feasibility check like exp12.**
- **State complaint-intake data** (CMS ACTS is not public; some states publish facility complaints, e.g. California
  CHHS). This could be a FOIA request or a state pilot.
- **Deficiency narrative text:** near-harm wording in D/E citations. Expected gain is modest; effort is high.

## Recommendation

1. Do **Track A (exp13)** now. It is cheap, uses data we already have, and gives an honest final number for the
   project.
2. Write **Track B** into the report and model card at the same time.
3. Run the **NHSN feasibility check** (Track C) as exp14. It is the only candidate with a plausible path to a
   noticeable gain.

## Sources (beyond `RESEARCH_SOLUTIONS.md`)
- GAO-10-70 (severity understatement): https://gao.gov/assets/a298958.html
- OIG complaint deadlines OEI-01-19-00421: https://oig.hhs.gov/reports/all/2020/states-continued-to-fall-short-in-meeting-required-timeframes-for-investigating-nursing-home-complaints-2016-2018
- LTCCC "The Accountability Gap" 2026: https://nursinghome411.org/wp-content/uploads/2026/09/LTCCC-Report-The-Accountability-Gap-2026.pdf
- QSO-26-14-NH (risk-based survey): https://www.cms.gov/files/document/qso-26-14-nh-revised-2026-09-29.pdf
- Gandhi, Olenski & Shi (inspection timing): https://mshi311.github.io/website2/2026_05_01_SNFinspections.pdf
- Chen & Dillender (government monitoring): https://populationanalytics.nd.edu/assets/625318/chen_dillender_government_monitoring_of_health_care_quality.pdf
- KFF deficiencies analysis: https://www.kff.org/medicaid/a-closer-look-at-deficiencies-in-nursing-homes/
- Kannan et al. (Chicago hindsight): https://arxiv.org/abs/1910.04906 ; JHU/APL replication: https://github.com/iscoe/restaurant_inspections
- CQC prediction studies: https://pmc.ncbi.nlm.nih.gov/articles/PMC6917361/ ; https://pmc.ncbi.nlm.nih.gov/articles/PMC7184898/
- DATE customs (KDD 2020): https://dl.acm.org/doi/10.1145/3394486.3403339 ; GraphFC: https://arxiv.org/abs/2305.11377
- EPA inspection field test: https://michaelgreenstone.com/?p=1828 ; Clean Water Act: https://link.springer.com/article/10.1038/s41893-018-0142-9
- Antwi & Bowblis (turnover → deficiencies): https://research.upjohn.org/up_workingpapers/249/
- NHSN staff-shortage analyses: https://pmc.ncbi.nlm.nih.gov/articles/PMC7598889
- Online reviews vs deficiencies: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11692201/ ; https://qualitysafety.bmj.com/content/27/8/619

## Addendum: verified data sources (CMS data dictionary rev. Oct 2026, QSSAM-26-01-NH, data.cms.gov)

- **"Number of Facility Reported Incidents" / "Number of Substantiated Complaints"** (Provider Info): defined as the
  number of incidents / complaints **in the past 3 years that resulted in a citation**. They are derived from citation
  history, not complaint intake, which is why they added only +0.003 ROC.
  - **Definition break:** from the iQIES cutover (2025-07-14) they counted all allegations, including unverified ones.
  - **Removal:** both columns were deleted from the 202602 release.
  - **Conclusion:** no public, facility-level complaint-intake measure exists.
- **Inspection Dates:** "for complaint inspections, dates are included only if the inspection resulted in one or more
  citations." Uncited complaint-survey volume is not observable.
- **Complaint intake:**
  - CMS ACTS / iQIES intake data is not public.
  - New York publishes record-level LTC ombudsman complaints (data.ny.gov fxkb-sqzm, 2019-10 to 2025-09; state facility
    ID, needs a CCN crosswalk). It covers one state only.
  - No usable 2022–26 intake data found for CA, TX, IL, WA or OH.
- **Chains:**
  - Provider Info `Affiliated Entity ID` from 202306, renamed `Chain ID` in 202507; 69.6 % of CCNs had one in Oct 2026.
  - PECOS SNF Enrollments `AFFILIATION ENTITY ID` from 2022-09, 65.9 % populated, 47 monthly versions.
  - SNF CHOW files are quarterly from 2022Q1.
- **Deficiency narrative text:** monthly CMS zips `Full-Statement-of-Deficiencies-<Month>-<YYYY>.zip` (~600 MB, includes
  `inspection_text`, IDR flags). Most months from 2021-01 exist.
