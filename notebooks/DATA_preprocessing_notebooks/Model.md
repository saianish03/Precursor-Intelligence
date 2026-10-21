# Model

How the risk model was built, what it is compared against and why, how well it works, and what goes in and comes out.

---

## 1. The question the model answers

> **"What is the chance this nursing home gets a serious citation in the next 12 months?"**

A **serious citation** means scope/severity **G to L**: an inspector found that residents were actually harmed or put in immediate jeopardy. We chose this target for three reasons:

- it is the public signal closest to **resident harm**, which is what drives nursing-home liability claims;
- **12 months** matches the length of a liability insurance policy;
- it becomes known about a year sooner than fines, which mostly follow these citations and are posted months later.

We trained three versions:

| Model | Predicts | Status |
|---|---|---|
| **A: `y_g12`** | serious citation within **12 months** | trained (primary) |
| `y_g6` | serious citation within **6 months** | trained (early warning between renewals) |
| B: `y_pen12` | fine or payment denial within 12 months | **not trainable yet**: penalties take ~18 months to be posted, so too few months have known outcomes |

---

## 2. How it was built

### Step 1: Training data
We used the frozen dataset `training_dataset_v1` (see `Data_pipeline.md`): one row per nursing home per month, with **129 input features** and the yes/no outcome.

### Step 2: Splitting by time, not at random
A model for insurers must predict the **future**, so we test it on months that come **after** the training months:

| Model | Train on | Gap (unused) | Test on |
|---|---|---|---|
| `y_g12` | Jan 2024 | Feb 2024 → Jan 2025 | **Feb 2025** |
| `y_g6` | Jan → Jul 2024 | Aug 2024 → Jan 2025 | **Feb → Aug 2025** |

**Why the gap?** A training row from January 2024 has an outcome window that runs to January 2025. If the test started before then, the training outcomes would overlap the test period, which is a subtle form of cheating. The gap ("purge") is one full horizon long.

### Step 3: The model: LightGBM
LightGBM is a **gradient-boosted decision tree** model: hundreds of small decision trees, each correcting the mistakes of the ones before it. We chose it because:

- it is the strongest standard choice for **tabular data** like ours (rows and columns of numbers and categories);
- it handles **missing values** natively. Many features are legitimately blank, such as "days since last fine" for a home that was never fined;
- it handles **categories** (state, ownership type) directly;
- it learns **non-linear patterns and interactions**, for example "many complaints *and* falling staffing";
- it is fast, and it supports **exact SHAP explanations**, which we need for the reasons shown to users.

Settings: 600 trees, learning rate 0.03, 63 leaves per tree, at least 100 rows per leaf (prevents fitting tiny, noisy groups), and 80% row and column sampling per tree.

### Step 4: Calibration: making the percentage mean what it says
A raw model score ranks homes well, but "0.6" doesn't automatically mean "60% chance". We fix that with **isotonic calibration**: a second, simple step that maps raw scores to real observed rates.

The calibration step must be fit on homes the model has **never seen**. Each home appears every month with nearly identical data, so the model partly "remembers" homes it trained on, and calibrating on those homes makes the percentages overconfident. So **20% of homes are held out**: the trees never see them, and they are used only to calibrate. In early tests, this reduced calibration error from 0.25 to well under 0.05.

### Step 5: Explanations
For every score, **SHAP** breaks the prediction into how much each feature pushed the risk up or down. The three biggest upward pushes become the plain-language reasons (see section 6).

---

## 3. The baselines, and why they are the right ones

A model is only useful if it beats what people would otherwise do. We compare against three baselines, from the simplest to the toughest:

### Baseline 0: Random guessing (the base rate)
Picking homes at random finds serious-citation homes at the overall rate: **25.4%** for 12 months. Any model must clear this floor.

### Baseline 1: CMS star ratings (the status quo)
**Why it's the most important baseline:** it is what insurers, families, and regulators use today. Underwriters look up a home's star rating on Care Compare. If our model can't beat stars, there is no product.

We test both the **overall star rating** and the **health-inspection star rating** (fewer stars = riskier), using the rating **as of the same month** so the comparison is fair.

Star ratings are a tough baseline, because CMS builds them from the same inspection history. But they are coarse (only 5 levels), curved by state, and they summarize the past rather than forecasting.

### Baseline 2: A simple severity rule (the "smart spreadsheet")
```
score = (actual-harm citations in 24 months) + 2 × (immediate-jeopardy citations in 24 months) + 0.1 × (all citations in 24 months)
```

**Why it's a baseline:** it is what a careful analyst could build in Excel in an afternoon, with no machine learning. If LightGBM didn't beat this rule, the extra complexity wouldn't be worth it. It is the honest check on whether the model learned anything beyond "past problems predict future problems".

### Ablation: the model plus star ratings
We also train the model **with** star ratings added as inputs. If that improved it, the model would be missing information the stars contain. It didn't (PR-AUC 0.435 vs 0.433), so the model already captures what the stars know.

---

## 4. Results

Measured on the test months the model never saw.

### Model A: serious citation within 12 months (test: Feb 2025, 14,682 homes, base rate 25.4%)

| Metric | **Model** | Star rating (overall) | Star rating (health inspection) | Severity rule |
|---|---|---|---|---|
| ROC-AUC | **0.693** | 0.627 | 0.623 | 0.660 |
| PR-AUC | **0.433** | 0.327 | 0.325 | 0.400 |
| Precision, top 5% | **64.0%** | 34.7% | 37.4% | 54.4% |
| Precision, top 10% | **54.3%** | 35.3% | 36.8% | 49.2% |
| Top-10% lift | **2.14×** | 1.40× | 1.46× | 1.94× |
| Within-state precision, top 5% | **47.5%** | 39.8% | 38.6% | 43.1% |
| Calibration error (ECE) | **0.016** | | | |

### `y_g6`: serious citation within 6 months (test: Feb → Aug 2025, 88,279 rows, base rate 14.7%)

| Metric | **Model** | Star rating (overall) | Severity rule |
|---|---|---|---|
| ROC-AUC | **0.699** | 0.631 | 0.659 |
| PR-AUC | **0.291** | 0.200 | 0.249 |
| Precision, top 5% | **42.6%** | 24.7% | 34.8% |
| Top-10% lift | **2.49×** | 1.63× | 2.14× |
| Calibration error (ECE) | **0.013** | | |

### Risk within each star level (Model A)

| Stars | All homes | Model's riskiest 20% | Model's safest 20% |
|---|---|---|---|
| 1★ | 37.1% | **62.0%** | 16.8% |
| 2★ | 29.4% | 46.1% | 16.7% |
| 3★ | 23.1% | 35.0% | 13.6% |
| 4★ | 18.9% | 32.9% | 12.3% |
| 5★ | 12.7% | **22.2%** | 5.0% |

### What the metrics mean

| Metric | Meaning |
|---|---|
| **ROC-AUC** | Pick one home that got a serious citation and one that didn't: how often the model ranks the first as riskier (0.5 = coin flip) |
| **PR-AUC** | Ranking quality focused on the risky homes (random = base rate) |
| **Precision, top 5%** | Of the 5% of homes flagged as riskiest, the share that actually had a serious citation |
| **Lift** | How many times riskier the flagged group is than average |
| **Within-state precision** | The same, but ranking inside each state, which shows the model isn't just picking strict states |
| **Calibration error (ECE)** | The average gap between predicted and actual rates (0.016 = 1.6 points) |

### In plain words

- If an insurer reviews the **riskiest 5%** of homes, the model finds a serious-citation home **64% of the time**, against **35%** using star ratings and 25% at random.
- **Homes with the same star rating differ in risk by 2.5–4.5×**, and the model finds that difference. The riskiest 20% of 5-star homes are about as risky as an average 3-star home.
- **The percentages can be trusted:** when the model says ~40%, about 38–40% of those homes have a serious citation.

### What drives the model (top features by importance)

`state`, staff turnover, complaint inspections in 24 months, days since the last complaint inspection, ED visits and hospitalizations per 1,000 resident-days, new antipsychotic use, days since last penalty, rehospitalization rate, and falls with major injury.

---

## 5. Input and output

### What you give it

**Just a facility ID (CCN) and a month.** You never type the 129 features yourself; the pipeline builds them from that month's CMS data.

```
Input:   ccn = "145764"     month = "2026-08"
```

Behind the scenes, that becomes one row of 129 features. An illustrative subset:

| Feature | Value |
|---|---|
| `state` | IL |
| `certified_beds` | 294 |
| `n_complaint_surveys_24m` | 24 |
| `n_gplus_surveys_36m` | 13 |
| `n_cit_24m` | 61 |
| `sev_points_24m` | 572 |
| `days_since_last_std` | 210 |
| `n_fines_24m` | 4 |
| `fine_dollars_24m` | log(1 + $96,500) |
| `hprd_total` | 3.42 |
| `turnover_total` | 58.0 |
| `d6_hprd_rn` | −0.12 |
| `qm_ls_falls_major_injury` | 3.1 |
| … | … (129 in total) |

*(Values other than state, beds, complaint inspections, and harm-level inspections are illustrative.)*

### What you get back

**A risk card:**

```
MORGAN PARK HEALTHCARE (145764), CHICAGO IL · data month 2026-08
Risk of serious citation (G–L) within 12 months: 90.3%
  top 0.1% nationally · #1 of 666 in IL · tier: High
  Main drivers:
   • 294 certified beds (state median 112)
   • 24 complaint inspections in the last 24 months (state median 5)
   • 13 inspections with harm-level citations in 36 months
```

**The same result as data** (one row of `scores_2026-08.csv`, or an API response):

```json
{
  "ccn": "145764",
  "provider_name": "MORGAN PARK HEALTHCARE",
  "state": "IL",
  "data_month": "2026-08",
  "label": "y_g12",
  "risk": 0.903,
  "national_percentile": 100.0,
  "state_rank": 1,
  "state_n": 666,
  "tier": "High",
  "reasons": [
    "294 certified beds (state median 112)",
    "24 complaint inspections in the last 24 months (state median 5)",
    "13 inspections with harm-level citations in 36 months"
  ],
  "top_contributions": {"n_complaint_surveys_24m": 0.41, "certified_beds": 0.38, "n_gplus_surveys_36m": 0.33, "...": "..."}
}
```

*(The `top_contributions` values shown are illustrative; real values are in `shap_2026-08.csv`.)*

| Field | Meaning |
|---|---|
| `risk` | Calibrated probability (0.903 = 90.3%) |
| `national_percentile` | Position among all ~14,700 homes that month |
| `state_rank` / `state_n` | Rank within the home's state |
| `tier` | **High** = top 5% · **Elevated** = next 15% · **Standard** = the rest |
| `reasons` | The 3 features that pushed risk up the most, with the state median for comparison |
| `top_contributions` | The raw SHAP values behind the reasons (log-odds), for auditing |

**For the whole country:** `scores_2026-08.csv` holds one such row for each of the 14,690 homes. In August 2026: 728 High (risk 61.5–90.3%), 2,203 Elevated (34.6–61.5%), and 11,759 Standard (average 18.8%).

---

## 6. How the reasons are written

1. SHAP gives every feature a contribution for this home: positive = pushed risk up, negative = pushed it down.
2. Take the largest positive contributions, skipping context that isn't actionable (`state`, `era`) and showing only one of any near-duplicates (for example, "days since inspection" and "inspection due date").
3. Fill a template with the home's own value and its state median: *"24 complaint inspections in the last 24 months (state median 5)"*.

There's no free-written text, so every number traces back to CMS data. One caution: reasons explain **the model**, not the home's care. A feature can push risk up in combination with others even when its value looks good on its own.

---

## 7. How to run it

In the notebook (`NH_Full_Pipeline.ipynb`), **Models** section:

```python
import train_evaluate as TE, sh_explain as XP
fits   = TE.main(PROJECT, labels=("g12", "pen12", "g6"))   # train + evaluate (saved to models/<label>/)
scores = XP.score_month(P, "g12")                          # every home, latest month
XP.brief(P, scores, "145764")                              # one risk card
```

Saved per model, in `models/<label>/`: `model.joblib` (model + calibrator), `test_metrics.csv`, `double_lift.csv`, `calibration_deciles.csv`, `feature_importance.csv`, `test_predictions.csv`, `scores_<month>.csv`, `shap_<month>.parquet`, and charts.

---

## 8. Limitations and next steps

| Limitation | Next step |
|---|---|
| Model A trains on only one month (data starts in 2024) | Add the 2023 archive: about 7 training and 7 test months |
| Penalty model (B) not trainable yet | More months of history (each new release adds one) |
| `state` is the top feature: Illinois holds 328 of the 728 High-tier homes | Replace raw state with "compared to state" features; keep reporting within-state metrics |
| Citations are a proxy for claims | Insurer pilot: validate scores against real claims |
| Single train/test split | Rolling backtests over several test periods once more history exists |
| Tuning not yet done | Hyperparameter search (Optuna) grouped by facility |
