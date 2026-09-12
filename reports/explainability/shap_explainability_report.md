# ETAFlow — SHAP Model Explainability Report

**Execution Timestamp:** 2026-09-12 10:02:25 UTC
**Explainer Technique:** TreeExplainer (Exact Shapley Values)
**Evaluated Models:** Tree-based final candidate models

---

## 1. Global Feature Importance (Top 10 Drivers)

### Booking Regression

| Rank | Feature Name | Mean |SHAP| Value |
| :--- | :--- | :--- |
| 1 | `distance_km` | **0.38761** |
| 2 | `transport_mode_Air` | **0.36122** |
| 3 | `month` | **0.21083** |
| 4 | `log_distance_km` | **0.19375** |
| 5 | `route_historical_delay_rate` | **0.12631** |
| 6 | `transport_mode_Rail` | **0.08032** |
| 7 | `route_type_Urban_Last_Mile` | **0.06835** |
| 8 | `vehicle_type_Commercial_Belly_Cargo` | **0.05731** |
| 9 | `transport_mode_Road` | **0.05133** |
| 10 | `transport_mode_Sea` | **0.03866** |

### Dispatch Regression

| Rank | Feature Name | Mean |SHAP| Value |
| :--- | :--- | :--- |
| 1 | `number_of_stops` | **0.40309** |
| 2 | `distance_km` | **0.23754** |
| 3 | `congestion_index` | **0.17932** |
| 4 | `transport_mode_Air` | **0.15586** |
| 5 | `log_distance_km` | **0.14002** |
| 6 | `route_complexity_score` | **0.12467** |
| 7 | `traffic_weather_interaction` | **0.10611** |
| 8 | `transport_mode_Road` | **0.08424** |
| 9 | `transport_mode_Rail` | **0.07297** |
| 10 | `pickup_to_dispatch_hours` | **0.06450** |

### Dispatch Classification

| Rank | Feature Name | Mean |SHAP| Value |
| :--- | :--- | :--- |
| 1 | `promised_delivery_days` | **2.69132** |
| 2 | `service_level_Standard` | **1.91460** |
| 3 | `service_level_Economy_Surface` | **1.09614** |
| 4 | `transport_mode_Air` | **0.96503** |
| 5 | `congestion_index` | **0.88206** |
| 6 | `distance_km` | **0.65013** |
| 7 | `number_of_stops` | **0.47913** |
| 8 | `transport_mode_Rail` | **0.47407** |
| 9 | `weather_condition_Clear` | **0.42710** |
| 10 | `pickup_to_dispatch_hours` | **0.36799** |

---

## 2. Representative Local Explanations with Natural Language Narratives

### Local Inspection: Booking Regression
```text
Shipment: SHP00079204
Predicted ETA (days): 1.28 (Baseline: 2.48)
ETA increased because:
  - transport_mode_Air = 0.0 (+0.13 days)
  - vehicle_type_Commercial_Belly_Cargo = 0.0 (+0.02 days)
  - carrier_BlueDart = 0.0 (+0.02 days)
  - vehicle_type_Multi_Axle_Container_Trailer = 1.0 (+0.02 days)
ETA decreased because:
  - distance_km = 690.7 (-0.48 days)
  - log_distance_km = 6.5392 (-0.24 days)
  - month = 2.0 (-0.15 days)
  - route_historical_delay_rate = 0.184 (-0.14 days)
```

### Local Inspection: Dispatch Regression
```text
Shipment: SHP00079204
Predicted ETA (days): 1.32 (Baseline: 2.48)
ETA increased because:
  - transport_mode_Air = 0.0 (+0.08 days)
  - carrier_BlueDart = 0.0 (+0.02 days)
  - traffic_level_Low = 0.0 (+0.01 days)
  - complexity_per_stop = 0.6725 (+0.01 days)
ETA decreased because:
  - distance_km = 690.7 (-0.25 days)
  - number_of_stops = 3.0 (-0.20 days)
  - log_distance_km = 6.5392 (-0.13 days)
  - route_complexity_score = 2.69 (-0.12 days)
```

### Local Inspection: Dispatch Classification
```text
Shipment: SHP00079204
Predicted Delay Probability: -12.11 (Baseline: -4.37)
Delay risk increased because:
  - service_level_Economy_Surface = 0.0 (+0.466 log-odds)
  - transport_mode_Air = 0.0 (+0.435 log-odds)
  - weather_risk_score = 0.017 (+0.393 log-odds)
  - route_historical_delay_rate = 0.184 (+0.230 log-odds)
Delay risk reduced because:
  - service_level_Standard = 1.0 (-2.336 log-odds)
  - promised_delivery_days = 4.3 (-1.863 log-odds)
  - priority_level_Low = 1.0 (-0.895 log-odds)
  - weather_condition_Clear = 1.0 (-0.471 log-odds)
```
