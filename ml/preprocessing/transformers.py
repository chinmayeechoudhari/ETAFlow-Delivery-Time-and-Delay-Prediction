"""
Custom Scikit-Learn Compatible Transformers for ETAFlow Preprocessing Layer.

Provides reusable transformers for structural missingness sanitization,
temporal decomposition, and informative missingness indicator generation.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)


class HolidaySanitizer(BaseEstimator, TransformerMixin):
    """Sanitizes holiday names by replacing structural NaN values with an explicit category."""

    def __init__(self, column: str = "holiday_name", fill_value: str = "No_Holiday") -> None:
        """Initialize HolidaySanitizer.

        Args:
            column: Name of the holiday column.
            fill_value: Replacement string for structural nulls (default: 'No_Holiday').
        """
        self.column = column
        self.fill_value = fill_value

    def fit(self, X: pd.DataFrame, y: Optional[Any] = None) -> HolidaySanitizer:
        """Fit method (stateless)."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform method replacing NaN in the holiday column."""
        X_out = X.copy()
        if self.column in X_out.columns:
            X_out[self.column] = X_out[self.column].fillna(self.fill_value).astype(str)
        return X_out


class MissingIndicatorAdder(BaseEstimator, TransformerMixin):
    """Generates binary indicator flags for features subject to operational missingness."""

    def __init__(self, columns: Optional[List[str]] = None, suffix: str = "_missing") -> None:
        """Initialize MissingIndicatorAdder.

        Args:
            columns: Columns to generate indicators for.
            suffix: Suffix for created indicator columns.
        """
        self.columns = columns or []
        self.suffix = suffix

    def fit(self, X: pd.DataFrame, y: Optional[Any] = None) -> MissingIndicatorAdder:
        """Fit method (stateless)."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform method generating binary missingness indicators."""
        X_out = X.copy()
        for col in self.columns:
            if col in X_out.columns:
                indicator_col = f"{col}{self.suffix}"
                X_out[indicator_col] = X_out[col].isna().astype(np.int32)
        return X_out


class TemporalFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extracts temporal components from timestamp columns available at booking time."""

    def __init__(self, date_column: str = "order_date") -> None:
        """Initialize TemporalFeatureExtractor.

        Args:
            date_column: Name of the timestamp column to decompose.
        """
        self.date_column = date_column

    def fit(self, X: pd.DataFrame, y: Optional[Any] = None) -> TemporalFeatureExtractor:
        """Fit method (stateless)."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform method extracting date components."""
        X_out = X.copy()
        if self.date_column in X_out.columns:
            dt_series = pd.to_datetime(X_out[self.date_column])
            X_out["order_hour"] = dt_series.dt.hour.astype(np.int32)
            X_out["order_day"] = dt_series.dt.day.astype(np.int32)
            # order_day_of_week as integer 0=Monday, 6=Sunday
            X_out["order_day_of_week"] = dt_series.dt.dayofweek.astype(np.int32)
        return X_out
