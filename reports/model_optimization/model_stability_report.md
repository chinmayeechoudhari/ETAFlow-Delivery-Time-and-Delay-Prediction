# ETAFlow — Multi-Seed Model Stability Report

**Execution Timestamp:** 2026-09-12 10:02:23 UTC
**Seeds Evaluated:** [42, 123, 2024, 3407, 999]
**Validation Partition:** Chronological Holdout (15,000 records)

---

## 1. Stability Audit Across Seeds

| Task | Model | Metric | Mean | Std Dev | Min | Max | Range | Stability Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Booking Regression** | LIGHTGBM | MAE | 0.4002 | **0.00037** | 0.3998 | 0.4008 | 0.0011 | **PASS (Highly Stable)** |
| **Booking Classification** | XGBOOST | F1 | 0.8391 | **0.00055** | 0.8382 | 0.8397 | 0.0016 | **PASS (Highly Stable)** |
| **Dispatch Regression** | LIGHTGBM | MAE | 0.1226 | **0.00037** | 0.1219 | 0.1229 | 0.0010 | **PASS (Highly Stable)** |
| **Dispatch Classification** | LIGHTGBM | F1 | 0.9461 | **0.00061** | 0.9452 | 0.9468 | 0.0016 | **PASS (Highly Stable)** |

---

## 2. Conclusion
- All candidates exhibit negligible variance across random seeds (Std Dev < 0.005).
- Models are completely robust to stochastic tree-splitting variations.
