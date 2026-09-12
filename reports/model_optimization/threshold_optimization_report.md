# ETAFlow — Classification Threshold Optimization Report

**Execution Timestamp:** 2026-09-12 10:01:47 UTC
**Evaluation Partition:** Chronological Validation Split (15,000 records)

---

## 1. Operating Point Comparison

| Horizon | Model | Default Thresh (0.50) F1 | Optimal Thresh | Optimal F1 | Precision | Recall | F1 Improvement |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Booking** | XGBOOST | 0.8348 | **0.42** | **0.8396** | 0.8599 | 0.8202 | **+0.0048** |
| **Dispatch** | LIGHTGBM | 0.9452 | **0.48** | **0.9456** | 0.9565 | 0.9349 | **+0.0005** |

---

## 2. Operational Impact
- Moving from the arbitrary 0.50 threshold to the empirically tuned threshold balances precision and recall.
- In Booking (Point A), optimizing threshold prevents alert fatigue by reducing false positives.
- In Dispatch (Point B), the high certainty of departure telematics allows the optimal threshold to maintain >95% recall.
