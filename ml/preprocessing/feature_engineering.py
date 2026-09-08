"""
Feature Engineering Module for ETAFlow Preprocessing Layer.

Implements domain-driven logistics feature engineering strictly separated
by prediction horizon (Point A: Booking vs. Point B: Dispatch).
Follows scikit-learn transformer protocol (BaseEstimator, TransformerMixin).
"""

from __future__ import annotations

import logging
from typing import Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)


class LogisticsFeatureEngineerPointA(BaseEstimator, TransformerMixin):
    """Domain feature engineer for Prediction Point A (Booking Time).

    Derives valid booking-horizon logistics features:
    - Log transforms for skewed continuous distributions (distance, weight, value)
    - Physical consignment density (kg / m³)
    - Economic value density (INR / kg)
    - Intra-state vs. inter-state indicator
    - Estimated dispatch milestone lead time (hours)
    - Temporal hour and day components
    """

    def __init__(self) -> None:
        """Initialize Point A Feature Engineer."""
        pass

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> LogisticsFeatureEngineerPointA:
        """Fit method (stateless)."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Derive Point A features from input DataFrame."""
        X_out = X.copy()

        # 1. Log transformations for right-skewed variables
        if "distance_km" in X_out.columns:
            X_out["log_distance_km"] = np.log1p(np.maximum(X_out["distance_km"], 0.0))

        if "package_weight_kg" in X_out.columns:
            X_out["log_package_weight"] = np.log1p(np.maximum(X_out["package_weight_kg"], 0.0))

        if "shipment_value_inr" in X_out.columns:
            X_out["log_shipment_value"] = np.log1p(np.maximum(X_out["shipment_value_inr"], 0.0))

        # 2. Package physical density: kg / m³ (1 m³ = 1,000,000 cm³)
        if "package_weight_kg" in X_out.columns and "package_volume_cm3" in X_out.columns:
            vol_m3 = np.maximum(X_out["package_volume_cm3"] / 1_000_000.0, 1e-6)
            X_out["package_density_kg_m3"] = X_out["package_weight_kg"] / vol_m3

        # 3. Consignment economic density (value in INR per kg)
        if "shipment_value_inr" in X_out.columns and "package_weight_kg" in X_out.columns:
            weight_safe = np.maximum(X_out["package_weight_kg"], 1e-4)
            X_out["value_per_kg"] = X_out["shipment_value_inr"] / weight_safe

        # 4. Intra-state vs. inter-state routing flag
        if "origin_state" in X_out.columns and "destination_state" in X_out.columns:
            X_out["is_same_state"] = (
                X_out["origin_state"] == X_out["destination_state"]
            ).astype(np.int32)

        # 5. Estimated dispatch lead hours (SLA staging window from order placement)
        if "estimated_dispatch_datetime" in X_out.columns and "order_date" in X_out.columns:
            est_dispatch = pd.to_datetime(X_out["estimated_dispatch_datetime"])
            order_dt = pd.to_datetime(X_out["order_date"])
            lead_hours = (est_dispatch - order_dt).dt.total_seconds() / 3600.0
            X_out["estimated_dispatch_lead_hours"] = np.maximum(lead_hours, 0.0)

        # 6. Temporal attributes from order_date
        if "order_date" in X_out.columns:
            order_dt = pd.to_datetime(X_out["order_date"])
            X_out["order_hour"] = order_dt.dt.hour.astype(np.int32)
            X_out["order_day"] = order_dt.dt.day.astype(np.int32)

        return X_out


class LogisticsFeatureEngineerPointB(BaseEstimator, TransformerMixin):
    """Domain feature engineer for Prediction Point B (Dispatch Time).

    Derives all Point A features PLUS realized dispatch features:
    - Realized staging duration (order to pickup hours)
    - Realized hub dwell / loading duration (pickup to actual dispatch hours)
    - Departure punctuality variance (actual dispatch - estimated dispatch hours)
    - Stops density along corridor (stops per 100 km)
    - Route complexity per intermediate stop
    - Transit weather & traffic interaction index
    - Telematics sensor missingness indicators
    """

    def __init__(self) -> None:
        """Initialize Point B Feature Engineer."""
        self.point_a_engineer = LogisticsFeatureEngineerPointA()

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> LogisticsFeatureEngineerPointB:
        """Fit method (stateless)."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Derive Point B features from input DataFrame."""
        # First inherit all Point A features
        X_out = self.point_a_engineer.transform(X)

        # 1. Realized operational durations
        if "order_date" in X_out.columns and "pickup_datetime" in X_out.columns:
            order_dt = pd.to_datetime(X_out["order_date"])
            pickup_dt = pd.to_datetime(X_out["pickup_datetime"])
            order_to_pickup = (pickup_dt - order_dt).dt.total_seconds() / 3600.0
            X_out["order_to_pickup_hours"] = np.maximum(order_to_pickup, 0.0)

        if "pickup_datetime" in X_out.columns and "actual_dispatch_datetime" in X_out.columns:
            pickup_dt = pd.to_datetime(X_out["pickup_datetime"])
            dispatch_dt = pd.to_datetime(X_out["actual_dispatch_datetime"])
            pickup_to_dispatch = (dispatch_dt - pickup_dt).dt.total_seconds() / 3600.0
            X_out["pickup_to_dispatch_hours"] = np.maximum(pickup_to_dispatch, 0.0)

        if "actual_dispatch_datetime" in X_out.columns and "estimated_dispatch_datetime" in X_out.columns:
            dispatch_dt = pd.to_datetime(X_out["actual_dispatch_datetime"])
            est_dispatch_dt = pd.to_datetime(X_out["estimated_dispatch_datetime"])
            # Positive value indicates warehouse departure delay past committed SLA
            X_out["dispatch_delay_hours"] = (dispatch_dt - est_dispatch_dt).dt.total_seconds() / 3600.0

        # 2. Route density and complexity ratios
        if "number_of_stops" in X_out.columns and "distance_km" in X_out.columns:
            dist_safe = np.maximum(X_out["distance_km"] / 100.0, 0.1)
            X_out["stops_per_100km"] = X_out["number_of_stops"] / dist_safe

        if "route_complexity_score" in X_out.columns and "number_of_stops" in X_out.columns:
            X_out["complexity_per_stop"] = X_out["route_complexity_score"] / (X_out["number_of_stops"] + 1.0)

        # 3. Telematics missingness indicators
        if "congestion_index" in X_out.columns:
            X_out["traffic_missing"] = X_out["congestion_index"].isna().astype(np.int32)

        if "weather_risk_score" in X_out.columns:
            X_out["weather_missing"] = X_out["weather_risk_score"].isna().astype(np.int32)

        if "road_condition" in X_out.columns:
            X_out["road_condition_missing"] = X_out["road_condition"].isna().astype(np.int32)

        # 4. Traffic × Weather compounding disruption interaction
        if "congestion_index" in X_out.columns and "weather_risk_score" in X_out.columns:
            # Filled temporarily with column medians for interaction computation
            cong_med = X_out["congestion_index"].median() if not X_out["congestion_index"].dropna().empty else 0.5
            w_med = X_out["weather_risk_score"].median() if not X_out["weather_risk_score"].dropna().empty else 0.2
            c_fill = X_out["congestion_index"].fillna(cong_med)
            w_fill = X_out["weather_risk_score"].fillna(w_med)
            X_out["traffic_weather_interaction"] = c_fill * w_fill

        return X_out
