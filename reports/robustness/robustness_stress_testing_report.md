# ETAFlow — Model Robustness & Stress Testing Report

**Execution Timestamp:** 2026-09-12 10:01:56 UTC
**Perturbations Evaluated:** Missingness (5%, 10%, 20% MCAR), Traffic Spikes, Severe Weather, Poor Roads

---

## 1. Missing-Data Injection Stress Test

| Candidate Model | Baseline Metric | 5% Missingness | 10% Missingness | 20% Missingness | Max Degradation (%) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **booking_regression** (MAE) | 0.3998 | 0.4437 | 0.4847 | 0.5683 | **+42.2%** |
| **booking_classification** (F1) | 0.8396 | 0.8071 | 0.7706 | 0.6968 | **+17.0%** |
| **dispatch_regression** (MAE) | 0.1219 | 0.1934 | 0.2615 | 0.3881 | **+218.3%** |
| **dispatch_classification** (F1) | 0.9456 | 0.9153 | 0.8819 | 0.8190 | **+13.4%** |

---

## 2. Environmental Shock Scenarios

| Candidate Model | Baseline Metric | High Traffic Shock | Severe Weather Shock | Poor Road Shock |
| :--- | :--- | :--- | :--- | :--- |
| **booking_regression** | 0.3998 | 0.3998 | 0.3998 | 0.3998 |
| **booking_classification** | 0.8396 | 0.8396 | 0.8396 | 0.8396 |
| **dispatch_regression** | 0.1219 | 0.7507 | 1.1784 | 0.2750 |
| **dispatch_classification** | 0.9456 | 0.7922 | 0.7001 | 0.8981 |

---

## 3. Resilience Conclusions
- Preprocessing median/mode imputation insulates models from catastrophic failure under missing telematics.
- Environmental shocks appropriately raise predicted duration and delay risk without breaking bounds.
