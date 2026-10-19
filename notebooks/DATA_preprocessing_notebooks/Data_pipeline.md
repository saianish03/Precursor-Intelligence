# Data Pipeline

How we turn CMS's monthly nursing-home files into one clean, trustworthy dataset for training a model.

---

## The idea in one paragraph

Every month, the US government agency CMS (Centers for Medicare & Medicaid Services) publishes a snapshot of every nursing home in the country: its inspections, the problems inspectors found ("citations"), fines, staffing levels, and quality measures. We collect many of these monthly snapshots and, for each nursing home and each month, write down **what was known about the home at that moment** (the *features*) and **what happened to it over the following year** (the *label*). The result is one big table where every row is "one nursing home in one month". A model can learn from it: *homes that looked like this went on to have serious problems.*

---

## 1. Where the data comes from

CMS publishes the data free on its **Provider Data Catalog** (data.cms.gov).

- **Archived snapshots:** one zip file per month, going back several years. We use these for history.
- **Current snapshot:** available through an API. We use this to add each new month as it is released.

Each monthly zip contains about 20 CSV files. We use six of them:

| File | What one row is | What we use it for |
|---|---|---|
| **Provider Information** | one nursing home | the list of homes, plus size, ownership, staffing, staff turnover, status flags, star ratings |
| **Health Citations** | one problem found by an inspector | the home's history of problems, and the thing we predict |
| **Inspection Dates** | one inspection visit | how recently and how often the home was inspected |
| **Penalties** | one fine or payment suspension | the home's enforcement history |
| **Quality Measures (MDS and Claims)** | one home × one measure | resident outcomes: falls, pressure ulcers, hospitalizations, and so on |
| **Citation Descriptions** | one regulation code | readable names for citation codes |

Every file has the same ID for each nursing home: the **CCN** (CMS Certification Number), a 6-character code such as `015009`. The CCN is the glue that connects all the files.

### What we collected

- **28 monthly snapshots** covering **January 2024 → August 2026**. CMS skipped a few months (2024-12, 2025-01, 2025-08, 2026-01).
- About **14,700 nursing homes** per month (15,022 different homes overall; 332 closed during the period).

**Why we didn't use 2020–2022:** during COVID, CMS paused routine inspections, and states spent 2021–2022 catching up on the backlog. Inspection and citation rates in those years don't reflect normal conditions, so learning from them would teach the model the wrong patterns.

---

## 2. The key terms

**Citation severity (A to L).** When inspectors find a problem, they give it a letter. Severity runs from "minimal potential for harm" up to "immediate jeopardy", and scope runs from one resident to the whole facility:

| | One resident | Several | Widespread |
|---|---|---|---|
| **Immediate jeopardy** (serious harm or death likely) | J | K | L |
| **Actual harm** (a resident was hurt) | G | H | I |
| Potential for more than minimal harm | D | E | F |
| Minimal potential for harm | A | B | C |

A **serious citation** means **G to L**: a resident was actually harmed or put in immediate danger. That is what we predict.

**Data month (t) and cutoff date.** Each snapshot records the date CMS pulled the data, for example 1 August 2026. That snapshot is "month t", and that date is its **cutoff**. Everything before the cutoff is the past (allowed as input). Everything after it is the future (what we predict).

---

## 3. The pipeline, step by step

We run ten steps. After five of them, a person reads a short report and approves it before the pipeline continues (🛑 checkpoints). That way, a data problem is caught early instead of silently ending up in the model.

### Step 1: Take inventory 🛑
Open every zip, including yearly bundles that contain monthly zips, and list what's inside. Read the real data month from the date inside each file, not from the zip's file name. If two zips contain the same month, keep the complete, later copy.

*What it caught:* one re-released July 2026 zip had files with only a header row and no data. The pipeline kept the good July copy instead.

### Step 2: Clean and standardize 🛑
CMS sometimes renames columns between releases, for example "Affiliated Entity" became "Chain". A mapping file translates every column name into one standard name. **If CMS adds a column we haven't seen, the pipeline stops and asks**, rather than guessing. Then each value gets the right type: dates become dates, Y/N becomes True/False, and the CCN stays as text so leading zeros are never lost.

*What it caught:* an extra unnamed index column in one release, junk files that macOS had added to the zips, and 2,209 exact duplicate penalty rows.

### Step 3: Build event histories ("ledgers")
Stack all the monthly snapshots of citations, penalties, and inspections into one history each. For every citation, we record **when it first appeared**, **when it last appeared**, its **final severity** (severities sometimes change after appeals), and whether **CMS later deleted it**.

These histories are used **only to work out what happened in the future** (the labels), never as model inputs. That separation is what stops future information from leaking into the inputs.

*Result:* 702,050 citations tracked. 0.58% were later deleted by CMS, and 1,089 changed severity.

### Step 4: Measure how late CMS posts things 🛑
CMS doesn't publish a citation the day it happens. We measured the delay:

| | Typical delay | 95% published within |
|---|---|---|
| Citations | ~2 months | ~6 months |
| Fines and penalties | ~6 months | ~18 months |

So we only trust a month's outcome once enough time has passed for the events to show up: about 6 months after the window ends for citations, and 18 months for penalties. That waiting time is the **buffer**.

This also shows why we predict **citations rather than fines**: fines mostly follow citations, but appear about a year later.

### Step 5: List the homes and choose the quality measures 🛑
Count the homes in each month, and pick **17 quality measures** that CMS published every month. In January 2025, CMS changed the definition of 3 of them (pressure ulcers, incontinence, mobility decline); we link the old and new versions so the history stays continuous.

### Step 6: Build the skeleton ("spine")
Create one row for every home that was active in each month: **CCN + month + cutoff date**. This gives **413,729 rows**. Every other piece of information is attached to these rows.

### Step 7: Calculate the features
For each month, look **only at that month's snapshot** and calculate 134 facts about each home. For example:
- "How many serious citations in the last 24 months?" (count rows in the citations file)
- "How many dollars of fines in the last 24 months?" (add up fine amounts)
- "How many days since the last standard inspection?" (latest date, compared with the cutoff)
- "Did RN staffing go up or down compared with 6 months ago?" (compare with the snapshot from 6 months earlier)

Because each month's features use only that month's snapshot, they contain exactly what someone would have known at the time. Nothing from the future can sneak in.

### Step 8: Work out the labels
Using the event histories from step 3, check each home and month: **did a serious (G–L) citation happen in the 12 months after the cutoff?** That gives a yes/no answer. We also record a 6-month version and a penalty version.

Some rows can't be answered yet:
- **immature:** the 12 months, plus the buffer, haven't passed yet, so we don't know the outcome;
- **censored:** the home closed before the 12 months were up.

These rows are kept for scoring but never used for training.

### Step 9: Put it all together
Attach every feature and label to the skeleton, matching on **CCN + month**. After each join, the pipeline checks that the row count hasn't changed: one row in, one row out.

### Step 10: Check everything, then freeze 🛑
Run 17 automatic checks. The most important:

| Check | What it proves |
|---|---|
| **Time-travel test:** rebuild a past month using only files that existed back then | features never use future information. The result was **identical** |
| Hand traces: pick 20 yes-rows and 20 no-rows per label and re-check them against the raw history | labels are correct (0 mismatches) |
| Every "yes" has its event after the cutoff, never before | no peeking at the outcome (81,048 of 81,048 passed) |
| Shuffle the labels and train a quick model: it should score like a coin flip | the inputs carry no hidden answer (0.497 ≈ 0.5) |
| Known rates match earlier analysis | parsing is right: 13.7% of inspections find harm (expected ~13%) |

**All 17 checks passed.** The dataset was then **frozen** as `training_dataset_v1` and never changed after that point.

---

## 4. What comes out

**`facility_month_dataset.parquet`**: **413,729 rows × 153 columns**. One row is one nursing home in one month.

| Group | Columns | Examples |
|---|---|---|
| Identity | 3 | CCN, month, cutoff date |
| Inspection timing | 9 | days since last inspection, complaint inspections this year |
| Citation history | 26 | serious citations in 24 months, worst severity on the last inspection, repeat violations |
| Enforcement | 7 | number and dollars of fines, payment suspensions |
| Facility and staffing | 47 | beds, occupancy, nurse hours per resident, staff turnover, staffing changes |
| Quality measures | 40 | falls with injury, pressure ulcers, hospitalizations, antipsychotic use |
| State context | 5 | how this home compares with others in its state |
| Labels, status, splits | 16 | serious citation in 12 or 6 months (yes/no), status, train/test assignment |

Two companion spreadsheets describe every column in detail:
- `column_directory.xlsx`: what each column means and where it comes from;
- `column_lineage.xlsx`: how each raw CMS column is counted, summed, or joined to produce each feature.

A manifest file (`dataset_manifest.json`) records exactly which zips (with fingerprints) and settings produced the dataset, so it can be rebuilt identically.

---

## 5. The rules that keep it honest

1. **Inputs only from the past.** Features for a month use only that month's CMS snapshot.
2. **Outcomes only from the future.** Labels use only events after the cutoff.
3. **Wait for the data to settle.** A month is labeled only after the window plus the posting buffer has passed.
4. **No silent guessing.** Unknown columns, empty files, or row-count changes stop the pipeline.
5. **A person signs off** at five checkpoints.
6. **Star ratings are never inputs.** CMS builds them from the same citations we predict, so using them would be circular. We keep them only to compare against.

---

## 6. How to run it

1. Put the CMS zips in `My Drive/nursing dataset/archive/`. Leave them zipped and unopened: Excel deletes the leading zeros in the CCN.
2. Open `NH_Full_Pipeline.ipynb` in Google Colab and run the setup cell.
3. Run steps 1–10 in order. At each 🛑, read the report, then run the approve cell.
4. The frozen dataset is saved to Drive and restored automatically in later sessions.

**Each new month:** `fetch_current_release.py` downloads the latest release from the CMS API into the archive folder. Rerun the pipeline, and every new month adds one more month of labeled history.

---

## 7. Known limits

- **Short history.** We currently have 2024 → 2026, so there is only a little labeled history (labels need 12 months plus a buffer to mature). Adding the 2023 archive would roughly double it.
- **Penalties settle slowly.** With an 18-month posting delay, the penalty label needs more history before it can be trained.
- **Closed homes before 2024 are missing.**
- **State differences.** States inspect with different strictness; the state-context features help the model account for that.
