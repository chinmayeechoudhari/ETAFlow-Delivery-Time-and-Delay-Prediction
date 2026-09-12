# ETAFlow — Bayesian Hyperparameter Optimization Report

**Execution Timestamp:** 2026-09-12 10:01:41 UTC
**Tuning Framework:** Optuna (Tree-structured Parzen Estimator / TPE)
**Validation Partition:** Chronological Holdout (15,000 records; test partition strictly untouched)

---

## 1. Optimization Summary

| Task | Candidate Model | Objective | Best Val Score | Total Trials | Runtime (s) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Booking Regression | LIGHTGBM | Minimize MAE | **0.3998** | 20 | 20.5s |
| Booking Classification | XGBOOST | Maximize F1 | **0.8348** | 20 | 26.4s |
| Dispatch Regression | LIGHTGBM | Minimize MAE | **0.1219** | 20 | 28.4s |
| Dispatch Classification | LIGHTGBM | Maximize F1 | **0.9452** | 20 | 24.4s |

---

## 2. Optimal Parameter Configurations

### BOOKING REGRESSION — LIGHTGBM
```yaml
learning_rate: 0.03527
n_estimators: 300
num_leaves: 55
max_depth: 8
min_child_samples: 13
subsample: 0.67583
colsample_bytree: 0.67846
reg_alpha: 4.52190
reg_lambda: 0.29832
```

### BOOKING CLASSIFICATION — XGBOOST
```yaml
learning_rate: 0.07277
n_estimators: 150
max_depth: 6
min_child_weight: 6
subsample: 0.67394
colsample_bytree: 0.98783
gamma: 3.87566
reg_alpha: 5.72790
reg_lambda: 3.79585
```

### DISPATCH REGRESSION — LIGHTGBM
```yaml
learning_rate: 0.04605
n_estimators: 250
num_leaves: 97
max_depth: 9
min_child_samples: 13
subsample: 0.74670
colsample_bytree: 0.61278
reg_alpha: 0.00745
reg_lambda: 0.13774
```

### DISPATCH CLASSIFICATION — LIGHTGBM
```yaml
learning_rate: 0.15644
n_estimators: 300
num_leaves: 60
max_depth: 4
min_child_samples: 25
subsample: 0.73666
colsample_bytree: 0.69941
reg_alpha: 0.01497
reg_lambda: 0.03730
```
