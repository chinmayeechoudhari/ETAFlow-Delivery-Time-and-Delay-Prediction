"""
Data Validation Module for ETAFlow Preprocessing Layer.

Validates raw dataset integrity, ensures zero target leakage, and validates
post-transformation feature matrices.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Mandatory target and milestone exclusions
STRICT_TARGET_COLUMNS: Set[str] = {
    "actual_delivery_days",
    "delivery_delay_days",
    "is_delayed",
    "delivery_datetime",
}

# Realized dispatch and operational features forbidden at Point A (Booking)
POINT_B_OPERATIONAL_COLUMNS: Set[str] = {
    "pickup_datetime",
    "actual_dispatch_datetime",
    "warehouse_processing_hours",
    "loading_time_hours",
    "handling_time_hours",
    "customs_clearance_hours",
    "number_of_stops",
    "route_complexity_score",
    "traffic_level",
    "congestion_index",
    "weather_condition",
    "weather_risk_score",
    "road_condition",
}


class DataValidationError(Exception):
    """Raised when dataset validation checks fail."""
    pass


class PreprocessingValidator:
    """Validator for raw datasets, feature isolation, and transformed matrices."""

    @staticmethod
    def validate_raw_data(
        df: pd.DataFrame,
        expected_rows: Optional[int] = 100000,
        expected_cols: Optional[int] = 55,
        id_col: str = "shipment_id",
    ) -> Dict[str, Any]:
        """Validate the raw dataset structure and critical field invariants.

        Args:
            df: Raw DataFrame.
            expected_rows: Expected number of rows (default 100,000). None skips check.
            expected_cols: Expected number of columns (default 55). None skips check.
            id_col: Identifier column name.

        Returns:
            Dict containing validation summary.
        """
        logger.info("Validating raw dataset: shape=(%d, %d)...", len(df), len(df.columns))

        if expected_rows is not None and len(df) != expected_rows:
            raise DataValidationError(
                f"Raw dataset row count mismatch: expected {expected_rows}, got {len(df)}"
            )

        if expected_cols is not None and len(df.columns) != expected_cols:
            raise DataValidationError(
                f"Raw dataset column count mismatch: expected {expected_cols}, got {len(df.columns)}"
            )

        if id_col not in df.columns:
            raise DataValidationError(f"Identifier column '{id_col}' not found in dataset")

        if df[id_col].duplicated().any():
            dup_count = df[id_col].duplicated().sum()
            raise DataValidationError(f"Identifier column '{id_col}' contains {dup_count} duplicate IDs")

        # Validate target invariants
        if "actual_delivery_days" in df.columns:
            if (df["actual_delivery_days"] <= 0).any():
                invalid_count = (df["actual_delivery_days"] <= 0).sum()
                raise DataValidationError(f"'actual_delivery_days' has {invalid_count} non-positive values")

        if "is_delayed" in df.columns:
            unique_delayed = set(df["is_delayed"].dropna().unique())
            if not unique_delayed.issubset({0, 1}):
                raise DataValidationError(f"'is_delayed' contains non-binary values: {unique_delayed}")

        if "delivery_delay_days" in df.columns:
            if (df["delivery_delay_days"] < 0).any():
                invalid_count = (df["delivery_delay_days"] < 0).sum()
                raise DataValidationError(f"'delivery_delay_days' contains negative values: {invalid_count}")

        logger.info("Raw dataset successfully passed all validation checks.")
        return {
            "rows": len(df),
            "columns": len(df.columns),
            "unique_shipments": df[id_col].nunique(),
            "status": "VALID",
        }

    @staticmethod
    def validate_feature_isolation(
        feature_columns: List[str],
        horizon: str,
    ) -> bool:
        """Ensure strict isolation against target leakage and horizon cross-contamination.

        Args:
            feature_columns: List of feature column names produced for ML matrix.
            horizon: 'booking' or 'dispatch'.

        Returns:
            True if feature isolation passes.
        """
        cols_set = set(feature_columns)

        # 1. Target Leakage Check (Zero tolerance for both horizons)
        leaked_targets = cols_set.intersection(STRICT_TARGET_COLUMNS)
        if leaked_targets:
            raise DataValidationError(
                f"TARGET LEAKAGE DETECTED in {horizon} feature set! Leaked columns: {leaked_targets}"
            )

        # 2. Prediction Horizon A (Booking) Lookahead Check
        if horizon.lower() == "booking":
            point_b_leaks = cols_set.intersection(POINT_B_OPERATIONAL_COLUMNS)
            if point_b_leaks:
                raise DataValidationError(
                    f"LOOKAHEAD LEAKAGE DETECTED in Booking feature set! Point B columns: {point_b_leaks}"
                )

        logger.info("Feature isolation check PASSED for horizon '%s' (0 leakage).", horizon)
        return True

    @staticmethod
    def validate_processed_matrix(
        X: pd.DataFrame,
        expected_rows: int,
        horizon: str,
        expected_id_col: Optional[str] = "shipment_id",
        allow_nan: bool = False,
    ) -> Dict[str, Any]:
        """Validate ML-ready processed feature matrix.

        Args:
            X: Processed feature DataFrame.
            expected_rows: Expected number of rows.
            horizon: 'booking' or 'dispatch'.
            expected_id_col: Identifier column if present in index or columns.
            allow_nan: If False, checks for 0 nulls across the matrix.

        Returns:
            Dict of validation statistics.
        """
        logger.info("Validating %s feature matrix: shape=(%d, %d)...", horizon, len(X), len(X.columns))

        if len(X) != expected_rows:
            raise DataValidationError(
                f"Processed matrix row count changed! Expected {expected_rows}, got {len(X)}"
            )

        # Ensure no unexpected NaN values remain
        if not allow_nan:
            null_counts = X.isna().sum()
            cols_with_nulls = null_counts[null_counts > 0]
            if not cols_with_nulls.empty:
                raise DataValidationError(
                    f"Processed matrix contains unhandled NaN values in columns: {cols_with_nulls.to_dict()}"
                )

        # Validate feature isolation
        feature_cols = [c for c in X.columns if c != expected_id_col]
        PreprocessingValidator.validate_feature_isolation(feature_cols, horizon)

        logger.info("Processed matrix validation PASSED for horizon '%s'.", horizon)
        return {
            "horizon": horizon,
            "row_count": len(X),
            "feature_count": len(feature_cols),
            "null_count": int(X.isna().sum().sum()),
            "status": "VALID",
        }
