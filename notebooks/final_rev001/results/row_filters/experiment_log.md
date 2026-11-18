| ID | Change (one only) | Model | #Features | PR-AUC (CV mean ± std) | ROC-AUC | Recall@10% | Brier | Δ PR-AUC vs baseline (± std) | Test PR-AUC (Δ) | Keep? |
|---|---|---|---|---|---|---|---|---|---|---|
| R0 | baseline (train rows 168,181) | LightGBM tuned | 75 | 0.5246 ± 0.0112 | 0.7319 | 0.2299 | 0.1705 | +0.0000 ± 0.0000 | 0.4870 (+0.0000) | baseline |
| R1 | drop_2022H1 (train rows 108,541) | LightGBM tuned | 75 | 0.5258 ± 0.0128 | 0.7320 | 0.2302 | 0.1705 | +0.0011 ± 0.0017 | 0.4873 (+0.0002) | no |
| R2 | drop_2022 (train rows 59,085) | LightGBM tuned | 75 | 0.5251 ± 0.0139 | 0.7309 | 0.2300 | 0.1707 | +0.0004 ± 0.0032 | 0.4879 (+0.0009) | no |
| R4 | drop_stale_insp (train rows 128,711) | LightGBM tuned | 75 | 0.5244 ± 0.0113 | 0.7315 | 0.2294 | 0.1707 | -0.0003 ± 0.0002 | 0.4857 (-0.0013) | no |
| R5 | drop_missing_staffing (train rows 162,629) | LightGBM tuned | 75 | 0.5242 ± 0.0114 | 0.7318 | 0.2294 | 0.1706 | -0.0005 ± 0.0005 | 0.4876 (+0.0006) | no |
| R6 | drop_missing_qm (train rows 163,709) | LightGBM tuned | 75 | 0.5248 ± 0.0111 | 0.7322 | 0.2303 | 0.1705 | +0.0002 ± 0.0001 | 0.4872 (+0.0001) | yes |
| R7 | drop_tiny_or_hospital (train rows 161,657) | LightGBM tuned | 75 | 0.5252 ± 0.0102 | 0.7325 | 0.2301 | 0.1704 | +0.0006 ± 0.0011 | 0.4868 (-0.0002) | no |
| R8 | drop_new_facility (train rows 167,060) | LightGBM tuned | 75 | 0.5248 ± 0.0111 | 0.7318 | 0.2304 | 0.1706 | +0.0002 ± 0.0005 | 0.4866 (-0.0005) | no |
| R9 | quarterly_thin (train rows 59,389) | LightGBM tuned | 75 | 0.5244 ± 0.0116 | 0.7319 | 0.2292 | 0.1706 | -0.0003 ± 0.0007 | 0.4867 (-0.0003) | no |
| R10 | noise_2pct (train rows 164,817) | LightGBM tuned | 75 | 0.5240 ± 0.0114 | 0.7298 | 0.2305 | 0.1721 | -0.0007 ± 0.0006 | 0.4866 (-0.0004) | no |
| R11 | noise_5pct (train rows 159,771) | LightGBM tuned | 75 | 0.5223 ± 0.0102 | 0.7287 | 0.2294 | 0.1756 | -0.0023 ± 0.0010 | 0.4860 (-0.0011) | no |
