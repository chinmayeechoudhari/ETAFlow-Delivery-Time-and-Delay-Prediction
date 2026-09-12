# ETAFlow — Probability Calibration & Reliability Report

**Execution Timestamp:** 2026-09-12 10:01:48 UTC
**Calibration Methods Evaluated:** Platt Scaling (Sigmoid) and Isotonic Regression
**Evaluation Metric:** Brier Score Loss & Expected Calibration Error (ECE)

---

## 1. Calibration Performance Summary

| Horizon | Model | Uncalibrated Brier | Platt Sigmoid Brier | Isotonic Brier | Recommended Calibration |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Booking** | XGBOOST | 0.0559 | 0.0575 | 0.0552 | **Isotonic** |
| **Dispatch** | LIGHTGBM | 0.0215 | 0.0209 | 0.0201 | **Isotonic** |

---

## 2. Decision on Practical Reliability
- Probability calibration aligns model risk scores with empirical real-world frequency.
- Lower Brier score directly translates to more reliable downstream alert routing in dispatch operations.
