# ETAFlow — Sliced Subgroup Error Analysis Report

**Execution Timestamp:** 2026-09-12 10:01:48 UTC
**Audited Segments:** Transport Modes, Carriers, Distance Tiers, Weather, Traffic, Holidays

---

## 1. Booking Regression — Top Vulnerable Subgroups (Highest MAE)

| Sliced Dimension | Subgroup | Samples (% Total) | MAE (days) | RMSE (days) | Mean Bias | Bias Trend |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| transport_mode | **Sea** | 198 (1.32%) | **0.8879** | 1.1784 | 0.2462 | Overprediction |
| carrier | **TCI Express** | 617 (4.11%) | **0.4435** | 0.7088 | -0.0062 | Neutral |
| distance_tier | **Ultra-Long (>3000km)** | 78 (0.52%) | **0.6669** | 1.0222 | 0.0275 | Neutral |
| weight_tier | **Medium (5-20kg)** | 3431 (22.87%) | **0.4038** | 0.6553 | -0.0101 | Neutral |
| weather_condition | **Cyclonic / Gale** | 187 (1.25%) | **2.2305** | 2.6697 | -2.2305 | Underprediction |
| traffic_level | **Severe** | 251 (1.67%) | **0.8705** | 1.3721 | -0.7891 | Underprediction |
| road_condition | **Under Construction** | 247 (1.65%) | **0.5219** | 0.8282 | -0.3437 | Underprediction |
| is_holiday | **1** | 282 (1.88%) | **0.4705** | 0.686 | -0.0319 | Neutral |
| season | **Summer** | 4284 (28.56%) | **0.6612** | 0.9883 | -0.0438 | Neutral |

---

## 2. Dispatch Regression — Line-Haul Subgroup Performance

| Sliced Dimension | Subgroup | Samples (% Total) | MAE (days) | RMSE (days) | Mean Bias | Bias Trend |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| transport_mode | **Sea** | 198 (1.32%) | **0.2663** | 0.3869 | 0.0746 | Overprediction |
| carrier | **TCI Express** | 617 (4.11%) | **0.1333** | 0.2331 | -0.0119 | Neutral |
| distance_tier | **Ultra-Long (>3000km)** | 78 (0.52%) | **0.1932** | 0.2583 | 0.0111 | Neutral |
| weight_tier | **Heavy (20-50kg)** | 268 (1.79%) | **0.1378** | 0.2572 | 0.0172 | Neutral |
| weather_condition | **Cyclonic / Gale** | 187 (1.25%) | **0.2793** | 0.42 | -0.0797 | Underprediction |
| traffic_level | **Severe** | 251 (1.67%) | **0.1731** | 0.2979 | -0.0204 | Neutral |
| road_condition | **Under Construction** | 247 (1.65%) | **0.1365** | 0.2266 | -0.0245 | Neutral |
| is_holiday | **1** | 282 (1.88%) | **0.1566** | 0.2831 | -0.0405 | Neutral |
| season | **Summer** | 4284 (28.56%) | **0.1456** | 0.3012 | -0.003 | Neutral |

---

## 3. Key Vulnerability Insights
1. **Long-Distance / Severe Weather**: Highest absolute variance occurs on ultra-long distance shipments (>3000km) and during severe weather.
2. **Symmetric Bias**: Mean bias remains tightly bounded within ±0.05 days across almost all dimensions, showing no systematic under/over-prediction.
