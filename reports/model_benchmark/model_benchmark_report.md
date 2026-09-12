# ETAFlow — Baseline ML Development & Benchmarking Report (M6-A)

**Generated:** 2026-09-12T08:26:52.991667+00:00
**Random Seed:** 42
**Total Dataset Records:** 100,000
**Prediction Horizons:** BOOKING, DISPATCH

---

## 1. Executive Summary & Best-Performing Baselines

Milestone 6-A establishes the foundational machine learning baseline performance across both operational horizons:
- **Point A (Order Booking)**: Evaluates ETA and delay risk using solely preorder attributes.
- **Point B (Vehicle Dispatch)**: Recalibrates predictions using realized warehouse processing, loading durations, and real-time transit telematics.

### Top Performing Candidates (Test Set Evaluation)

| Operational Horizon | Modeling Task | Winning Model Family | Primary Metric | Secondary Metric |
| :--- | :--- | :--- | :--- | :--- |
| **Booking** | Delivery ETA (Regression) | **Lightgbm** | **MAE: 0.517 days** | RMSE: 0.806 days (R²: 0.738) |
| **Booking** | Delay Risk (Classification) | **Xgboost** | **F1: 0.8245** | PR-AUC: 0.9274 (ROC-AUC: 0.9632) |
| **Dispatch** | Delivery ETA (Regression) | **Lightgbm** | **MAE: 0.153 days** | RMSE: 0.278 days (R²: 0.969) |
| **Dispatch** | Delay Risk (Classification) | **Lightgbm** | **F1: 0.9476** | PR-AUC: 0.9888 (ROC-AUC: 0.9939) |

---

## 2. Dataset & Chronological Split Structure

To reflect real-world logistics deployment and avoid temporal lookahead bias, records are split strictly chronologically by `order_date`:

| Partition | Sample Count | Ratio | Start Timestamp | End Timestamp | Delay Incidence | Mean Delivery Days |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** | 70,000 | 70.0% | 2023-01-01 00:01:22 | 2025-02-05 14:20:04 | 25.96% | 2.48 days |
| **Validation** | 15,000 | 15.0% | 2025-02-05 14:36:48 | 2025-07-18 13:47:30 | 24.80% | 2.41 days |
| **Test** | 15,000 | 15.0% | 2025-07-18 14:38:43 | 2025-12-30 23:50:07 | 29.17% | 2.63 days |

**Strict Boundary Integrity Verification:**
- max(Train Date) < min(Validation Date): `PASSED` (`2025-02-05 14:20:04` < `2025-02-05 14:36:48`)
- max(Validation Date) < min(Test Date): `PASSED` (`2025-07-18 13:47:30` < `2025-07-18 14:38:43`)
- ID Overlap Across Partitions: `0 Records (Disjoint)`
- Transformation Leakage Prevention: `ETAPreprocessingPipeline` fitted strictly on Train only.

---

## 3. Delivery-Time Regression Benchmark Results

Target variable: `actual_delivery_days` (continuous duration in days).

| Horizon | Model Family | Level | MAE (days) | RMSE (days) | R² | Within ±0.5d | Within ±1.0d | Within ±2.0d | Train Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Booking | Lightgbm | Level 3 (Boosting) | 0.517 | 0.806 | 0.738 | 64.6% | 87.6% | 97.2% | 0.46s |
| Booking | Xgboost | Level 3 (Boosting) | 0.519 | 0.810 | 0.736 | 64.4% | 87.7% | 97.1% | 0.65s |
| Booking | Random Forest | Level 2 (Bagging) | 0.545 | 0.847 | 0.711 | 61.7% | 86.3% | 96.9% | 10.45s |
| Booking | Ridge | Level 1 (Linear) | 0.605 | 0.913 | 0.664 | 55.5% | 86.0% | 96.4% | 0.44s |
| Booking | Dummy | Level 0 (Heuristic) | 1.176 | 1.582 | -0.010 | 26.8% | 49.5% | 89.0% | 0.00s |
| Dispatch | Lightgbm | Level 3 (Boosting) | 0.153 | 0.278 | 0.969 | 94.8% | 98.7% | 99.8% | 0.55s |
| Dispatch | Xgboost | Level 3 (Boosting) | 0.157 | 0.285 | 0.967 | 94.6% | 98.6% | 99.7% | 0.69s |
| Dispatch | Random Forest | Level 2 (Bagging) | 0.222 | 0.365 | 0.946 | 90.6% | 97.7% | 99.6% | 13.37s |
| Dispatch | Ridge | Level 1 (Linear) | 0.345 | 0.563 | 0.872 | 79.9% | 93.7% | 98.8% | 0.45s |
| Dispatch | Dummy | Level 0 (Heuristic) | 1.176 | 1.582 | -0.010 | 26.8% | 49.5% | 89.0% | 0.00s |

---

## 4. Delay Risk Classification Benchmark Results

Target variable: `is_delayed` (binary 0/1 indicator). Natural class imbalance: ~26.3% positive incidence.

| Horizon | Model Family | Level | Precision | Recall | F1 Score | ROC-AUC | PR-AUC | Accuracy | Train Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Booking | Xgboost | Level 3 (Boosting) | 0.8737 | 0.7806 | 0.8245 | 0.9632 | 0.9274 | 0.9031 | 0.60s |
| Booking | Lightgbm | Level 3 (Boosting) | 0.8657 | 0.7852 | 0.8235 | 0.9635 | 0.9278 | 0.9018 | 0.48s |
| Booking | Logistic Regression | Level 1 (Linear) | 0.8519 | 0.7861 | 0.8177 | 0.9557 | 0.9167 | 0.8977 | 2.88s |
| Booking | Random Forest | Level 2 (Bagging) | 0.9256 | 0.6508 | 0.7643 | 0.9532 | 0.9130 | 0.8829 | 0.96s |
| Booking | Dummy | Level 0 (Heuristic) | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.2917 | 0.7083 | 0.00s |
| Dispatch | Lightgbm | Level 3 (Boosting) | 0.9623 | 0.9333 | 0.9476 | 0.9939 | 0.9888 | 0.9699 | 0.59s |
| Dispatch | Xgboost | Level 3 (Boosting) | 0.9640 | 0.9298 | 0.9466 | 0.9937 | 0.9879 | 0.9694 | 0.72s |
| Dispatch | Logistic Regression | Level 1 (Linear) | 0.9458 | 0.9294 | 0.9375 | 0.9913 | 0.9827 | 0.9639 | 2.67s |
| Dispatch | Random Forest | Level 2 (Bagging) | 0.9702 | 0.7948 | 0.8738 | 0.9857 | 0.9712 | 0.9330 | 1.21s |
| Dispatch | Dummy | Level 0 (Heuristic) | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.2917 | 0.7083 | 0.00s |

---

## 5. Booking vs. Dispatch Comparative Analysis

### Operational Question:
> *Does the additional information available at dispatch improve delivery-time and delay prediction?*

1. **Delivery-Time Estimation**: Dispatch horizon features achieve an absolute MAE reduction of **0.364 days** (+70.37% relative error reduction, moving from 0.517 days at Booking to 0.153 days at Dispatch).
2. **Delay Risk Discrimination**: Incorporating realized warehouse processing, loading durations, and transit telematics increases the classification F1 score by **+0.1230** (+14.92% relative improvement, from 0.8245 to 0.9476).
3. **Conclusion**: Physical departure data provides substantial incremental signal over preorder booking static data, confirming the dual-horizon architectural premise.

### Quantitative Horizon Deltas (Best Models):

| Metric | Booking Horizon | Dispatch Horizon | Absolute Delta | Relative Improvement |
| :--- | :--- | :--- | :--- | :--- |
| **Regression MAE** (lower better) | 0.517 days | 0.153 days | -0.364 days | **+70.37%** |
| **Regression RMSE** (lower better) | 0.806 days | 0.278 days | -0.529 days | **+65.57%** |
| **Within ±1 Day** (higher better) | 87.6% | 98.7% | +11.2% | +12.76% |
| **Classification F1** (higher better) | 0.8245 | 0.9476 | +0.1230 | **+14.92%** |
| **Classification PR-AUC** (higher better) | 0.9274 | 0.9888 | +0.0613 | **+6.61%** |

---

## 6. Known Limitations & Roadmap to Milestone 6-B

1. **Default Hyperparameters**: All models evaluated in M6-A used reasonable default baselines without optimization. M6-B will execute Bayesian hyperparameter sweeps (Optuna).
2. **Fixed Decision Thresholds**: Delay classification currently assumes a default 0.5 decision cutoff. In M6-B, operational cost-sensitive threshold tuning will optimize recall vs. false alarms.
3. **Probability Calibration**: Tree ensemble probabilities exhibit mild uncalibrated confidence skew. M6-B will implement Platt scaling / isotonic regression.
4. **Subgroup Performance & Feature Attribution**: Granular error analysis across carrier tiers, distance buckets, and weather zones, alongside SHAP attributions, will be conducted in M6-B.

---

## 7. Artifact Manifest

- Model weights serialized in `models/baseline/` (joblib format).
- Tabular results exported to `reports/model_benchmark/regression_results.csv` and `classification_results.csv`.
- Machine-readable benchmark summary stored in `reports/model_benchmark/benchmark_summary.json`.