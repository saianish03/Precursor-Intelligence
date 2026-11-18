# State bias and fairness audit: Precursor main model

**Model audited:** `old_ens`, the mean of LightGBM and logistic regression over 133 features, trained on 2023-01..2023-11.
**Target:** P(serious G+ health citation within 12 months) per nursing home and month.
**Data:** test month **2025-02** (primary, 14,643 facilities, base rate 25.6 %) and **2024-11** (stability check, 14,659 facilities, base rate 26.2 %).
**State:** taken from the gold `state` column. All rows matched, so the CCN fallback was never needed. There are 53 jurisdictions (50 states, DC, PR, GU).
**Re-run:** from the repo root, `uv run --no-sync python notebooks/final_rev001/state_bias/state_bias_audit.py`. Add `--stage retrain` first to run the retraining part, which needs at least 2.5 GB of free RAM.

---

## 1. Summary of findings

1. **There is real, systematic state-level miscalibration.** National calibration is good: observed 25.6 % against predicted 26.2 %. Per state, the gap between observed and expected (O−E) has a standard deviation of **7.6 points** across the 34 states with enough data. Sampling noise alone would give about 2.3 points. A random-intercept model estimates the true between-state spread at τ ≈ **5.7–6.4 points** (I² = 86 %). After Benjamini–Hochberg correction, **18 of 34** eligible states are significantly miscalibrated. A likelihood-ratio test of state intercepts on top of the model's logit gives χ²(52) = 342, p ≈ 0.
2. **The gap is stable, not noise.** The per-state O−E in 2024-11 and 2025-02 has Spearman ρ = **0.96**. Fourteen states are significant in both months with the same sign.
   - **Under-predicted** (more G+ citations than the model expects): OK +17 pts, MS +14, CT +11, MN +10, NY +7, NJ/MD +5.
   - **Over-predicted:** WA −19 pts, LA −14, OR −12, KS −9, MA −9, IA −6.
3. **This is mostly label regime, and the model trails behind it.** The `state_gplus_rate_12m` context feature tracks the model's mean state score almost perfectly (r = 0.97). It tracks the state's actual base rate much less well (r = 0.79). The per-state calibration gap correlates **r = 0.91** with "state base rate in the label window minus the state's trailing G+ rate".
   - In other words, the model anchors on each state's past surveyor strictness. It is wrong where a state's survey regime moved: WA, LA and OR became more lenient; OK, MS and CT became stricter, or their survey backlog cleared.
   - The bias does **not** compress high- and low-base-rate states toward the mean. The correlation between base rate and O−E is 0.04, and the slope of mean prediction on base rate is 0.97.
4. **Error-rate gaps are large but mostly mechanical.** At the accuracy threshold (0.505), the model flags 75 % of Illinois facilities and none in 13 eligible states. TPR ranges from 0.00 to 0.87 and FPR from 0.00 to 0.59. Demographic parity, the 4/5ths rule (33 of 34 states fail it relative to Illinois) and equalized odds all fail badly.
   - With base rates from 9 % (KY) to 58 % (IL) and a single high threshold, this is what even a perfectly calibrated score produces (the impossibility result: calibration and equal error rates cannot hold together when base rates differ).
   - The part that *is* bias: OK, CT, NY, NJ, MD and AZ get zero flags partly *because* they are under-predicted. Of all 3,750 positives, 739 (20 %) sit in states where the model raises no flag at all.
5. **Predictive parity largely holds.** When the model does flag a facility, precision is similar across states: only 4 states differ significantly after BH correction, and the permutation-test ratio is 2.7, compared with 6–84 for the other criteria.
6. **Ranking within states is uneven but moderate.** Within-state ROC-AUC ranges from 0.52 (AR) to 0.76 (IL). The n-weighted mean is 0.67 against 0.71 overall, with true spread τ = 0.036.
   - Only **3.6 %** of the positive–negative pairs in the overall AUC are within the same state. So the overall AUC (0.709) is almost entirely a between-state comparison.
   - Each state's mean score alone gives AUC 0.64. Removing the state component from the score lowers PR-AUC from 0.480 to 0.406.
7. **Facility size and ownership slices add no bias beyond state.** Across 174 slices (state × beds tercile × ownership; minimum 100 rows and 15 of each class), 57 are miscalibrated. However, **0 of 125** state intersections deviate significantly from their own state after BH correction. Nationally, size and ownership groups are calibrated within 1.5 points.
8. **Mitigation helps only a little when done honestly.**
   - The per-state recalibration that can actually be deployed is fitted on 2024-01, the newest month whose 12-month labels are complete before 2025-02. It cuts the per-state O−E SD only from 7.6 to **6.5 points** (EB-shrunk: 7.1). Significantly miscalibrated states drop from 18 to 15. PR-AUC moves 0.480 → 0.481 and ROC 0.709 → 0.713.
   - Fitting on 2024-11 instead looks far better (SD 2.5, 0 significant states, PR-AUC 0.494). But those labels overlap the test window and would not exist at scoring time, so that result is an upper bound only.
   - The drift between a state's past and current strictness happens within about a year, faster than matured labels arrive.
9. **Removing state information does not remove the bias, it only costs accuracy** (counterfactual retrain, section 5.4).
   - Without `state`: ROC 0.708 → 0.703, PR-AUC 0.477 → 0.476, per-state O−E SD unchanged (7.79 → 7.79).
   - Without all 5 state-derived features: ROC 0.698, PR-AUC 0.463 (Δ PR-AUC CI −0.020..−0.008). O−E SD is 6.4, but 17 states are still significantly miscalibrated (16 with full features).
   - Other features carry state through as proxies, and the label itself is state-dependent.

**Bottom line:** the model is not "unfair" in the sense of applying a different rule to similar facilities. Within each state its ranking is similar, and its precision when it flags is similar. But it carries a **stable, state-specific calibration error of about ±6 points (SD)**. That error comes from relying on each state's *historical* survey strictness, which drifts. For insurer use, absolute probabilities in about a dozen states are off by 9–19 points.
- Neither removing state features nor recalibrating on matured labels fixes this in time: deployable recalibration only takes the spread from 7.6 to 6.5 points.
- Per-state monitoring (the funnel plot) and caution with absolute probabilities by state must therefore be part of the scoring pipeline. Error-rate parity across states cannot be achieved with a single national threshold, and the label itself makes it undesirable: a G+ in Nevada is not the same event as a G+ in Kentucky.

---

## 2. Methods, from basic to advanced

| Level | Method | What it asks | Notes |
|---|---|---|---|
| Basic | Per-state n, base rate, mean score, O−E, ROC-AUC, PR-AUC (and lift = PR-AUC / base rate), Brier | How the model performs in each state | |
| Basic | Confusion-matrix rates at three cut-offs: the accuracy threshold 0.5046 (from exp12 out-of-fold), the national top-10 % and the national top-20 % | Who gets flagged and what errors result | flag rate (selection), TPR/recall, FPR, FNR, precision (PPV), FOR (false omission), accuracy, FN/FP |
| Group fairness | Demographic parity (flag-rate gap and ratio), disparate impact (4/5ths rule, against the top state and against national), equal opportunity (TPR gap), equalized odds (max of TPR and FPR gaps), predictive parity (precision gap), FNR/FOR parity, treatment equality (FN/FP) | Standard group-fairness criteria, with states as groups | Reported as national value, min/max state, max−min and max \|state − national\|. Only **eligible states** count: n ≥ 100, ≥ 20 positives and ≥ 20 negatives (34 of 53). Also flag rate ÷ base rate as a base-rate-adjusted parity. |
| Calibration | Calibration-in-the-large (O−E and O/E, z-test with Var = Σp(1−p)), ECE (equal-mass bins), logistic recalibration slope and intercept (intercept fitted with logit(p) as offset), Hosmer–Lemeshow (external-validation form, df = groups) | Does a score of 30 % mean 30 % in every state? | The most meaningful test when base rates differ |
| Rigor | 500-replicate bootstrap CIs per state. Within one month every facility has exactly one row, so facility-clustered resampling equals row resampling; a two-month pooled O−E resamples facilities as clusters. | Uncertainty of each state estimate | |
| Rigor | Two-proportion z-tests, state against the rest of the nation, for base rate, flag rate, TPR, FPR, precision and FOR; z-test for O−E; BH FDR across eligible states within each family | Which individual states differ | |
| Rigor | Permutation tests (1,000 shuffles of state labels; for TPR only among positives, for FPR only among negatives) on the between-state sum of squares | Is there *any* state heterogeneity beyond chance? | |
| Rigor | Empirical-Bayes (DerSimonian–Laird) shrinkage of O−E, TPR, FPR and within-state AUC; funnel plot | Stops tiny states (DC n = 16, AK n = 20, GU n = 1) from looking extreme by chance | |
| Advanced 1 | Multilevel models: a linear mixed model of the residual y − p with a random state intercept (variance component, ICC); a Bayesian random-intercept logistic model y ~ logit(p) + (1\|state) (latent ICC); a likelihood-ratio test of state fixed intercepts | How much of the residual is "state", and which states deviate | |
| Advanced 2 | AUC decomposition into within-state and between-state pairs; AUC of state-mean-only scores; AUC and PR-AUC of the score centred within each state | Does ranking power come from facilities or from the state? | |
| Advanced 3 | Automated slice discovery over state × beds tercile (cut-offs 80 and 120 beds) × ownership, at orders 1–3, with minimum support, BH over all 522 tests, replication on 2024-11, and an "excess over parent state" test | Hidden intersectional problems | |
| Advanced 4 | State-feature attribution (TreeSHAP `pred_contrib` on a retrained LightGBM) plus a counterfactual retrain without `state`, and without all 5 state-derived features | Is the bias caused by the state features? | See section 5.4 |
| Advanced 5 | Stability: rank correlation of per-state metrics between 2024-11 and 2025-02 | Systematic or noise? | The two label windows overlap by 9 months, so the months are not independent |
| Advanced 6 | Mitigation: per-state logit intercepts (raw or EB-shrunk), global Platt scaling, Platt plus state, and per-state thresholds that equalise TPR. Fitted on an earlier month, evaluated on 2025-02. | What it costs to fix | |

State-derived features in the 133: `state` (categorical), `state_gplus_rate_12m`, `state_median_hprd_total`, `rel_hprd_total` and `rel_sev_points_24m`. The last two are facility values relative to the state.

---

## 3. Results

### 3.1 National reference (2025-02)

| n | base rate | mean pred | ROC | PR-AUC | Brier | ECE | flag rate @0.505 | TPR | FPR | precision | accuracy |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 14,643 | 0.256 | 0.262 | 0.709 | 0.480 | 0.169 | 0.028 | 0.106 | 0.235 | 0.062 | 0.566 | 0.758 |

The national calibration slope is 0.83 (SE 0.02), so the model is slightly over-confident overall. The national Hosmer–Lemeshow test rejects (χ² = 84). Part of this is drift over time, and the global Platt step in section 6 corrects it.

### 3.2 Most mis-calibrated states (eligible, 2025-02; O−E in points, EB = empirically shrunk; 95 % bootstrap CI)

| state | n | base rate | mean pred | O−E | 95 % CI | EB | BH q | O−E 2024-11 | trailing state G+ rate (feature) |
|---|---|---|---|---|---|---|---|---|---|
| **OK** | 283 | 30.4 % | 13.1 % | **+17.3** | +12.0, +22.7 | +15.4 | <0.001 | +18.1 | 16.1 % |
| **MS** | 200 | 35.0 % | 21.2 % | **+13.8** | +7.3, +20.4 | +11.0 | <0.001 | +10.3 | 20.3 % |
| **CT** | 192 | 29.2 % | 18.6 % | **+10.6** | +4.2, +16.4 | +8.6 | 0.001 | +9.8 | 8.2 % |
| **MN** | 338 | 35.8 % | 26.0 % | **+9.8** | +4.3, +15.2 | +8.4 | <0.001 | +8.6 | 28.4 % |
| **NY** | 593 | 16.0 % | 8.8 % | **+7.2** | +4.5, +10.1 | +7.0 | <0.001 | +6.5 | 10.8 % |
| AZ | 140 | 18.6 % | 12.2 % | +6.4 | 0.0, +12.0 | +5.2 | 0.038 | +2.5 | 11.4 % |
| NJ | 346 | 20.2 % | 15.2 % | +5.0 | +0.8, +8.7 | +4.5 | 0.021 | +5.9 | 10.1 % |
| MD | 220 | 18.2 % | 13.1 % | +5.0 | +0.6, +10.5 | +4.4 | 0.045 | +5.5 | 14.4 % |
| … | | | | | | | | | |
| IA | 390 | 23.6 % | 30.0 % | −6.4 | −10.2, −2.6 | −5.6 | 0.012 | −5.7 | 29.4 % |
| MA | 340 | 17.6 % | 26.2 % | **−8.6** | −12.3, −4.9 | −7.4 | 0.001 | −8.0 | 19.7 % |
| KS | 296 | 24.3 % | 33.3 % | **−8.9** | −13.5, −4.0 | −7.4 | 0.003 | −8.5 | 31.0 % |
| OR | 127 | 20.5 % | 32.0 % | **−11.5** | −17.7, −5.2 | −7.8 | 0.012 | −12.5 | 30.5 % |
| LA | 264 | 18.9 % | 33.0 % | **−14.0** | −18.6, −9.2 | −11.4 | <0.001 | −10.9 | 28.6 % |
| **WA** | 191 | 29.8 % | 49.1 % | **−19.2** | −25.7, −12.4 | −14.2 | <0.001 | −17.8 | 42.5 % |

Large, well-calibrated states: PA −0.7, OH +0.8, MI +0.8, FL +1.6, IN +1.7 (all q > 0.27). CA (−4.2) and TX (−3.7) are significant only because they are big; the gaps are small in practice. Small states such as NM (−12.9, n = 68), HI, DC, VT, AK and WY have wide CIs and are shrunk strongly by EB. The full table is in `tables/per_state_metrics_2025-02.csv`.

**Per-state calibration slope:** the median is 0.94. Hosmer–Lemeshow rejects in 13 of 34 states, or 8 after BH. AR is the outlier (slope 0.09, within-state ROC 0.53): the score barely ranks facilities there.

### 3.3 Group-fairness criteria (eligible states, 2025-02)

| criterion | cut-off | national | min (state) | max (state) | max−min |
|---|---|---|---|---|---|
| Flag rate (demographic parity) | acc. thr | 0.106 | 0.00 (13 states) | 0.75 (IL) | 0.75 |
| Disparate impact, 4/5ths vs top state | acc. thr | | 33 of 34 states < 0.8 × IL | | |
| Disparate impact vs national | acc. thr | | 24 of 34 states < 0.8 × national | | |
| TPR (equal opportunity) | acc. thr | 0.235 | 0.00 | 0.87 (IL) | 0.87 |
| FPR | acc. thr | 0.062 | 0.00 | 0.59 (IL) | 0.59 |
| Equalized odds (max \|ΔTPR\|, \|ΔFPR\|) | acc. thr | | | 0.64 (IL) | |
| Precision (predictive parity) | acc. thr | 0.566 | 0.33 (KS) | 0.69 (NC) | 0.35 (14 states with ≥ 20 flags) |
| FOR (false omission) | acc. thr | 0.219 | 0.087 (KY) | 0.363 (MI) | 0.28 |
| Treatment equality FN/FP | acc. thr | 4.2 | 0.30 (IL) | 69 (MS) | |
| TPR | top-20 % | 0.389 | 0.00 (8 states) | 0.96 (IL) | 0.96 |
| Precision | top-20 % | 0.498 | 0.26 (MA) | 0.64 (IL) | 0.38 (17 states with ≥ 20 flags) |
| Within-state ROC-AUC | | 0.709 | 0.525 (AR) | 0.759 (IL) | 0.23 |
| PR-AUC lift over base rate | | 1.88 | 1.08 (AR) | 2.59 (NE) | |

States with **no flags at all** at the accuracy threshold: TN, KY, FL, IN, GA, AR, VA, NJ, MD, AZ, NY, CT, OK. The first six have low base rates and are calibrated, so this follows from the threshold. The last seven are **under-predicted**, so the model's bias suppresses their flags. Permutation tests (all p = 0.001) confirm heterogeneity for every criterion. The size of the between-state dispersion relative to chance is: flag rate ×84, FPR ×45, TPR ×27, base rate ×19, residual ×6.3, precision ×2.7. The 2024-11 figures are almost identical (`tables/fairness_gap_summary.csv`).

### 3.4 Multilevel model, variance decomposition

| | 2025-02 | 2024-11 |
|---|---|---|
| LMM on residual y−p: state SD τ | 6.4 pts | 6.1 pts |
| ICC (residual) | 0.024 | 0.022 |
| Random-intercept logistic: state SD (logit) | 0.38 | 0.35 |
| Latent-scale ICC | 0.042 | 0.036 |
| LRT state intercepts, χ²(52) | 342 (p ≈ 0) | 315 (p ≈ 0) |
| Spearman of state random effects across months | 0.90 | |

State explains only about 2–4 % of the residual variance. Most error is facility-level noise, which is expected for a 12-month binary event. Even so, a state offset of 0.38 on the logit scale is about ±7 points at a 25 % base rate. That is material for absolute-risk pricing.

### 3.5 Where the ranking power comes from (AUC decomposition, 2025-02)

| | value |
|---|---|
| Overall ROC-AUC | 0.709 |
| Share of positive–negative pairs within the same state | 3.6 % |
| AUC on within-state pairs | 0.681 |
| AUC on between-state pairs | 0.710 |
| AUC of the state's mean score only (no facility information) | 0.638 |
| AUC of the previous month's state base rate only | 0.658 |
| AUC of `state_gplus_rate_12m` only | 0.634 |
| AUC of the score centred within each state | 0.661 |
| PR-AUC: full score vs centred within state | 0.480 vs 0.406 |

A large share of the headline AUC is "which state is the facility in". That is legitimate signal, because surveyor regimes really do differ, but it is also the channel through which regime drift becomes miscalibration.

### 3.6 Intersectional slices (state × beds tercile × ownership)

- 174 slices met the minimum support and 522 tests were run. 57 slices are significantly miscalibrated after BH correction, and 55 of them have the same sign in 2024-11.
- Every top slice is a subset of a biased state, for example OK × for-profit (+15.9 pts), NY × large (+9.8), WA × for-profit (−21.3), LA × large × for-profit (−17.4) and TX × small (−11.6).
- **Excess over the parent state:** 0 of 125 state intersections are significant after BH. Size and ownership do not add calibration bias on top of state.
- Nationally, calibration-in-the-large is good for every size and ownership group: large +0.1, medium −0.7, small −1.1 pts; for-profit −1.2, non-profit +1.5, government −0.1. The error-rate gaps by size (FNR: small 0.92, large 0.63) are again a threshold effect of different base rates (small 19.5 % vs large 31.6 %).

Full list: `tables/slices_state_beds_ownership_2025-02.csv`.

### 3.7 Stability, 2024-11 vs 2025-02 (34 eligible states)

| per-state metric | Spearman ρ |
|---|---|
| O−E (calibration gap) | **0.96** |
| base rate | 0.98 |
| mean prediction | 0.997 |
| TPR / FPR at the threshold | 0.96 / 0.95 |
| within-state ROC-AUC | 0.77 |

The calibration bias is systematic. Caveat: the label windows (Dec 2024–Nov 2025 and Mar 2025–Feb 2026) share 9 months, so this partly measures persistence within one survey period. The direction did not change between two prediction dates three months apart.

---

## 4. Charts (`figures/`)

- `funnel_oe_2025-02.png`: O−E against state size, with 95 % and 99.8 % control limits. WA, LA, OK, MS, CT, MN and NY sit outside the limits.
- `forest_oe_by_state_2025-02.png`: every state's O−E with its bootstrap CI and EB-shrunk estimate.
- `base_rate_vs_mean_pred.png`: state base rate against mean prediction.
- `reliability_extreme_states.png`: reliability curves for the 3 most under-predicted states (OK, MS, CT) and the 3 most over-predicted (WA, LA, OR). The curves are shifted but have roughly the right slope, so the problem is the intercept, not the ranking.
- `tpr_fpr_by_state.png`: error rates by state at the accuracy threshold.
- `stability_oe_2024-11_vs_2025-02.png`: per-state O−E in the two months.
- `mitigation_recalibration_oe.png`: per-state O−E before and after state recalibration.
- `counterfactual_no_state_oe.png`: per-state O−E for the full LightGBM, the version without `state`, and the version without all state information (section 5.4).

---

## 5. Interpretation: model bias or label regime?

### 5.1 What is genuinely label or regime difference (not model unfairness)

- Base rates range from 8.7 % (KY) to 58 % (IL) in 2025-02, consistent with the known spread in state G+ share (about 1.5 % NV to about 17 % KY of deficiencies) and differences in survey backlogs. A "G+ within 12 months" outcome means different things in different states.
- So the large demographic-parity, disparate-impact, TPR, FPR and equalized-odds gaps at one national threshold are **expected** and mostly reflect true base-rate differences. Forcing them equal would mean flagging facilities in lenient-survey states that are not more likely to be cited.

### 5.2 What is a model problem

- **Calibration by state.** The model predicts each state's *past* strictness, through the `state` category and `state_gplus_rate_12m`, not its current strictness. The residual tracks the drift (r = 0.91), and the drift persists across months.
- **Consequence:** under-predicted states (OK, MS, CT, NY, NJ, MD, AZ, MN) get too few flags and too-low prices. Over-predicted states (WA, LA, OR, KS, MA, IA) get too many flags, with FPR of 0.37 in WA against 0.06 nationally.
- **Uneven ranking quality.** AR (AUC 0.53), NJ (0.57) and CO (0.58) are near-uninformative within the state. The facility features carry less signal there. A cause may be more random survey timing, but this audit cannot prove it.

### 5.3 What is not a problem

- Precision among flagged facilities (predictive parity) holds across states.
- Facility size and ownership add no calibration bias beyond state.

### 5.4 Feature attribution and counterfactual (retrain without state information)

**How it was run:** `state_bias_audit.py --stage retrain` (5.4 min, free RAM 2.4–2.7 GB), then the default audit.

- LightGBM was retrained with the exp12 params on 2023-01..2023-11 in three arms: all 133 features, without `state` (132), and without all 5 state-derived features (128).
- The LR half was refit to rebuild `old_ens`. The reproduction check matches the original scores to within 5e-06.
- 2024-01 was scored for the deployable recalibration in section 6.
- Tables: `tables/retrain_counterfactual_summary.csv`, `tables/retrain_per_state_*.csv`, `tables/retrain_state_feature_gain.csv`, `tables/shap_state_contrib.csv`.

**Attribution: how much the model uses state.**
- `state` is the **#1 feature by gain** (24.8 % of total gain) in the full LightGBM, and `state_gplus_rate_12m` is #3 (6.9 %).
- When `state` is removed, `state_gplus_rate_12m` takes over as #1 (18.9 % of gain).
- TreeSHAP: state features make up **25.5 %** of the total absolute attribution on average (n-weighted, 2025-02). Per state, they shift the logit by about −0.98 (AL), −0.91 (NY) up to +1.05 (SD) and +1.11 (IL).

**Counterfactual (LightGBM only, same params and train window):**

| arm | month | ROC | PR-AUC | Brier | accuracy | state O−E SD (pts) | n-wtd mean \|O−E\| | states sig. miscal. (BH) | TPR gap (thr) | within-state AUC |
|---|---|---|---|---|---|---|---|---|---|---|
| full (133) | 2025-02 | 0.708 | 0.477 | 0.1703 | 0.758 | 7.8 | 5.5 | 16 | 0.84 | 0.666 |
| no `state` (132) | 2025-02 | 0.703 | 0.476 | 0.1719 | 0.760 | 7.8 | 6.1 | 15 | 0.79 | 0.666 |
| no state info (128) | 2025-02 | 0.698 | 0.463 | 0.1712 | 0.755 | 6.4 | 5.2 | 17 | 0.68 | 0.659 |
| full (133) | 2024-11 | 0.710 | 0.488 | 0.1719 | 0.755 | 7.4 | 5.3 | 21 | 0.85 | 0.665 |
| no `state` (132) | 2024-11 | 0.703 | 0.486 | 0.1733 | 0.755 | 7.6 | 5.9 | 23 | 0.80 | 0.663 |
| no state info (128) | 2024-11 | 0.695 | 0.467 | 0.1740 | 0.750 | 6.4 | 5.2 | 17 | 0.70 | 0.655 |

Δ (no state info − full), facility-clustered bootstrap 95 % CI:

| month | PR-AUC | ROC |
|---|---|---|
| 2025-02 | −0.020..−0.008 | −0.015..−0.005 |
| 2024-11 | −0.027..−0.014 | −0.020..−0.009 |

**Reading:**
- Dropping `state` alone changes almost nothing; `state_gplus_rate_12m` replaces it.
- Dropping all state information:
  - costs about 0.01–0.02 PR-AUC, significantly;
  - lowers within-state AUC slightly;
  - narrows the TPR gap a little, because the model can no longer push whole states up or down;
  - **does not fix calibration**: still 17 and 17 states are significantly miscalibrated. The label is state-dependent and other features (staffing relative to the state, inspection history) act as proxies.
- The bias is in the label regime, not in the use of a state feature. Removing state is not a valid mitigation.

**Feature-level evidence** (gold feature values and the existing predictions, eligible states, 2025-02):

| check | value |
|---|---|
| corr(state mean of `state_gplus_rate_12m`, state mean prediction) | **0.97** |
| corr(state mean of `state_gplus_rate_12m`, state observed base rate) | 0.79 |
| corr(state O−E, base rate − `state_gplus_rate_12m`) | **0.91** |
| AUC of `state_gplus_rate_12m` alone | 0.634 |
| AUC of the state-mean score alone / previous-month state base rate alone | 0.638 / 0.658 |

The state component of the score follows the trailing state G+ rate almost one-for-one. Its errors are exactly the states where the label period departs from that history: WA's feature is 42.5 % against a base rate of 29.8 %, and OK's is 16.1 % against 30.4 %.

The counterfactual above confirms what this evidence suggested beforehand. Removing state information costs accuracy and leaves the calibration gaps in place, because nothing then encodes the state regime and the label remains state-dependent.

---

## 6. Mitigation options and trade-offs

Fitted on **2024-11** and evaluated on **2025-02**. Scores come from `old_ens`; thresholds are the original 0.5046 or re-chosen on the fitting month. Table: `tables/mitigation_recalibration.csv`.

| option | ROC | PR-AUC | Brier | accuracy | flag rate | state O−E SD (pts) | mean \|O−E\| | states sig. miscal. (BH) | TPR gap | FPR gap |
|---|---|---|---|---|---|---|---|---|---|---|
| M0 original | 0.709 | 0.480 | 0.1690 | 0.758 | 10.6 % | 7.6 | 4.8 | 18 | 0.87 | 0.59 |
| M4 global Platt only | 0.709 | 0.480 | 0.1683 | 0.762 | 7.2 % | 6.8 | 3.8 | 12 | 0.81 | 0.46 |
| M1 state intercept, unshrunk | 0.724 | 0.495 | 0.1655 | 0.764 | 9.0 % | **2.0** | 1.5 | **0** | 0.83 | 0.51 |
| **M2 state intercept, EB-shrunk** | **0.724** | **0.494** | 0.1656 | 0.763 | 9.1 % | 2.5 | 1.8 | **0** | 0.83 | 0.51 |
| M3 global Platt + EB state | 0.722 | 0.493 | 0.1658 | 0.763 | 7.6 % | 2.5 | 1.7 | 0 | 0.85 | 0.54 |
| M5 per-state thresholds equalising TPR | (unchanged) | (unchanged) | | 0.733 | 13.0 % | 7.6 | | 18 | **0.21** | **0.15** |

**Caution: the table above is an upper bound.** The 2024-11 label window (Dec 2024–Nov 2025) overlaps 2025-02's (Mar 2025–Feb 2026) by 9 months and would not be available at 2025-02 scoring time.

**Deployable variant, the honest estimate.** Fitted on **2024-01**, whose 12-month labels are complete before 2025-02, and evaluated on 2025-02. Table: `tables/mitigation_recalibration_fit_2024-01.csv`.

| option | ROC | PR-AUC | Brier | accuracy | flag rate | state O−E SD (pts) | mean \|O−E\| | states sig. miscal. (BH) | TPR gap | FPR gap |
|---|---|---|---|---|---|---|---|---|---|---|
| M0 original | 0.709 | 0.480 | 0.1690 | 0.758 | 10.6 % | 7.6 | 4.8 | 18 | 0.87 | 0.59 |
| M4 global Platt only | 0.709 | 0.480 | 0.1689 | 0.759 | 10.4 % | 7.5 | 4.6 | 14 | 0.87 | 0.57 |
| M1 state intercept, unshrunk | 0.713 | 0.481 | 0.1685 | 0.759 | 10.8 % | **6.5** | 4.2 | **15** | 0.85 | 0.53 |
| M2 state intercept, EB-shrunk | 0.712 | 0.482 | 0.1685 | 0.758 | 10.8 % | 7.1 | 4.5 | 16 | 0.86 | 0.56 |
| M3 global Platt + EB state | 0.712 | 0.482 | 0.1685 | 0.759 | 10.4 % | 7.1 | 4.4 | 15 | 0.86 | 0.56 |
| M5 per-state thresholds equalising TPR | (unchanged) | (unchanged) | | 0.739 | 11.9 % | 7.6 | | 18 | **0.38** | **0.20** |

**Reading:** with a 13-month-old fitting window, state recalibration removes only about 15 % of the calibration spread (7.6 → 6.5 points) and leaves 15 states significantly miscalibrated. The ±6-point state errors come from **regime changes within the last year**. Those changes are not yet visible in any matured label, so no post-hoc recalibration on matured labels can fix most of them in real time. The stable rank correlation between the two test months (ρ = 0.96) reflects their 9-month label overlap, not long-run stability.

**Trade-offs**

- **State recalibration (M1/M2)**, if fresh labels were available, would remove the calibration bias and raise PR-AUC and ROC by about 0.015 (upper bound, fit on 2024-11). Deployably (fit on 2024-01) it gives only about +0.004 ROC and +0.001 PR-AUC and narrows the spread from 7.6 to 6.5–7.1 points. It re-orders facilities across states, while within-state ranking is untouched.
  - Use EB shrinkage (M2) so that small states (fewer than about 100 facilities) do not get noisy offsets. Unshrunk offsets help large states slightly more but are unstable for VT, WY, DC and AK.
  - **Cost:** it needs labels that matured recently, and it must be refitted as survey regimes drift. In deployment, the newest month with a complete 12-month label is about 13 months old (see the deployable variant above). It also makes the score explicitly "state-adjusted", which is the opposite of removing state from the model.
- **Global Platt (M4)** fixes the national over-confidence but leaves 12 states miscalibrated.
- **Per-state thresholds (M5)** almost equalise TPR and FPR, but cost 2.5 points of accuracy and raise the flag rate. In lenient-survey states they flag facilities whose actual risk is low. That fits "equal recall of citations per state", not "rank by true risk". It suits a regulator-style audit list better than insurance pricing.
- **Removing state features** (measured in section 5.4) costs 0.01–0.02 PR-AUC and does not reduce the number of miscalibrated states. The label stays state-dependent and other features act as proxies for state.
- **Recommended:**
  - Keep state information as features.
  - Add an EB-shrunk per-state intercept recalibration step, refitted monthly on the latest matured labels. It helps a little and costs nothing.
  - Make **per-state O−E monitoring with the funnel plot** the main control, because the deployable recalibration cannot catch regime changes in time. Use early signals as drift alerts: a state's partial-window G+ rate in the last 3–6 months, and its survey backlog.
  - Use national rank cut-offs for lists and treat absolute probabilities in flagged states with caution.
  - Report per-state flag rates alongside base rates so users see the regime effect.

---

## 7. Limitations

- **Small states.** 19 of 53 jurisdictions fail the support rule (n ≥ 100, ≥ 20 positives and ≥ 20 negatives), for example AK 20, DC 16, VT 34, WY 35, HI 42, DE 44, PR 7 and GU 1. Their raw metrics are unreliable. EB-shrunk values are reported, and they are excluded from max–min gap summaries.
- **The label is regime-dependent and noisy.** A G+ citation needs an inspection to happen. Survey frequency, backlogs and the strictness of the state survey agency all enter the label. No test here can separate "facility is riskier" from "state surveyors cite more"; the audit can only show that the model's state component lags the regime.
- **State is a feature.** Fairness across states is not a protected-attribute question in the usual sense. Including state is defensible and also predictive. The concern is calibration drift, not disparate treatment.
- **Two test months with overlapping label windows** (9 of 12 months shared). Stability and the 2024-11 → 2025-02 mitigation are therefore optimistic. The deployable variant fitted on 2024-01 (section 6) is the honest estimate.
- **One row per facility per month**, so facility-clustered bootstrap equals row bootstrap within a month. Clustering matters only in the two-month pooled O−E (`tables/oe_pooled_two_months_cluster_bootstrap.csv`, which gives the same ranking).
- **Thresholds.** At 0.505 only about 10 % of facilities are flagged, so many states have zero flags. TPR, FPR and precision are therefore undefined or extreme there. The top-10 % and top-20 % cut-offs are reported for this reason.
- Hosmer–Lemeshow and ECE use 3–10 equal-mass bins depending on state size. Their power is low for small states.

---

## 8. Files

| file | contents |
|---|---|
| `state_bias_audit.py` | the re-runnable script (`--stage retrain` and the default `--stage audit`) |
| `results.json` | every summary statistic quoted here |
| `tables/per_state_metrics_2025-02.csv` / `_2024-11.csv` | all per-state metrics, BH q-values, bootstrap CIs, EB estimates, random-effect BLUPs |
| `tables/fairness_gap_summary.csv` | group-fairness gaps at 3 cut-offs, both months |
| `tables/permutation_tests_2025-02.csv` | global heterogeneity tests |
| `tables/oe_pooled_two_months_cluster_bootstrap.csv` | facility-clustered two-month O−E |
| `tables/slices_state_beds_ownership_2025-02.csv` | intersectional slices with replication and excess-over-state |
| `tables/mitigation_recalibration*.csv`, `tables/state_logit_intercepts_fit_*.csv` | mitigation results and fitted state offsets |
| `tables/retrain_*`, `tables/shap_state_contrib.csv` | counterfactual retrain outputs (only if the retrain stage ran) |
| `figures/*.png` | charts listed in section 4 |
